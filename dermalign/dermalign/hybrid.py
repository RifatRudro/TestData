"""DermAlign-X: dual-stream CNN + Transformer model with bidirectional cross-attention fusion.

                    image
                  /       \\
       CNN stream           Transformer stream
   (ConvNeXt, stride 32)   (ViT, patch 16, global context)
       local texture            long-range structure
            |                         |
       1x1 proj + DW-conv       linear proj + DW-conv      -> D channels each
            |                         |
            +--- CNN tokens  <-attends-  ViT tokens (full grid)     "what global context explains this texture?"
            +--- ViT tokens  <-attends-  CNN tokens                 "which local evidence supports this region?"
                       \\             /
                  gated fusion  g * a + (1 - g) * b   (per-cell, per-channel gate)
                              |
              DermAlign spatial attention pooling  (attention map = explanation,
                              |                      aligned to the lesion mask in training)
                         [+ metadata]  ->  classifier

The fused map lives on the CNN grid, so `att` has the same B x 1 x h x w contract as
`DermModel` and the alignment loss, Grad-CAM (w.r.t. the fused map), IG and the sanity
check in xai.py work unchanged.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import _TOKEN_ARCHS, LayerNorm2d, to_bchw


FUSE_MODES = ("xattn", "concat", "xattn_nogate")  # full model, no cross-attention, no learned gate


class _DWConvProj(nn.Module):
    """Project a B x C x h x w map to D channels and add a depth-wise conv positional bias."""

    def __init__(self, c_in: int, d: int):
        super().__init__()
        self.proj = nn.Conv2d(c_in, d, 1)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.norm = LayerNorm2d(d)

    def forward(self, x):
        x = self.proj(x)
        return self.norm(x + self.dw(x))


class _CrossBlock(nn.Module):
    """Pre-norm cross-attention (queries from one stream, keys/values from the other) + MLP."""

    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.nq, self.nkv, self.nm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))

    def forward(self, q, kv):
        kvn = self.nkv(kv)
        q = q + self.attn(self.nq(q), kvn, kvn, need_weights=False)[0]
        return q + self.mlp(self.nm(q))


def _tokens(x):  # B C h w -> B hw C
    return x.flatten(2).transpose(1, 2)


def _grid(t, hw):  # B hw C -> B C h w
    b, _, c = t.shape
    return t.transpose(1, 2).reshape(b, c, *hw)


class HybridDermModel(nn.Module):
    def __init__(self, arch: str, arch2: str, num_classes: int = 7, pretrained: bool = True,
                 drop_rate: float = 0.2, drop_path_rate: float = 0.1, meta_dim: int = 0,
                 att_hidden: int = 256, img_size: int | None = None,
                 fuse_dim: int = 256, fuse_heads: int = 4, fuse_depth: int = 1,
                 fuse_mode: str = "xattn"):
        super().__init__()
        if fuse_mode not in FUSE_MODES:
            raise ValueError(f"fuse_mode must be one of {FUSE_MODES}, got {fuse_mode!r}")
        self.head_type, self.fuse_mode = "attn", fuse_mode
        kw = dict(pretrained=pretrained, num_classes=0, global_pool="")

        def make(name):
            k = dict(kw)
            if img_size and any(t in name for t in _TOKEN_ARCHS):
                k["img_size"] = img_size
            try:
                return timm.create_model(name, drop_path_rate=drop_path_rate, **k)
            except TypeError:
                return timm.create_model(name, **k)

        # both streams live under `backbone.` -> lower LR group, partial re-init in the sanity check
        self.backbone = nn.ModuleDict({"cnn": make(arch), "vit": make(arch2)})
        c1, c2 = self.backbone["cnn"].num_features, self.backbone["vit"].num_features
        self.prefix2 = getattr(self.backbone["vit"], "num_prefix_tokens", 0)
        self.c1, self.c2, self.C = c1, c2, fuse_dim

        d = fuse_dim
        self.proj_cnn, self.proj_vit = _DWConvProj(c1, d), _DWConvProj(c2, d)
        n_cross = 0 if fuse_mode == "concat" else fuse_depth
        self.cnn_from_vit = nn.ModuleList([_CrossBlock(d, fuse_heads, 0.1) for _ in range(n_cross)])
        self.vit_from_cnn = nn.ModuleList([_CrossBlock(d, fuse_heads, 0.1) for _ in range(n_cross)])
        if fuse_mode == "xattn":
            self.gate = nn.Sequential(nn.Conv2d(2 * d, d, 1), nn.Sigmoid())
        elif fuse_mode == "concat":
            self.cat_proj = nn.Conv2d(2 * d, d, 1)
        self.norm = LayerNorm2d(d)

        self.att = nn.Sequential(nn.Conv2d(d, att_hidden, 1), nn.GELU(), nn.Conv2d(att_hidden, 1, 1))
        nn.init.zeros_(self.att[-1].weight)  # start as uniform attention (= GAP)
        nn.init.zeros_(self.att[-1].bias)

        self.meta_dim = meta_dim
        fc_in = d
        if meta_dim > 0:
            self.meta_mlp = nn.Sequential(nn.Linear(meta_dim, 64), nn.GELU(), nn.Dropout(0.1),
                                          nn.Linear(64, 64), nn.GELU())
            fc_in += 64
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(fc_in, num_classes)

    # split so XAI code can take gradients w.r.t. the fused feature map
    def forward_features(self, x):
        a = self.backbone["cnn"].forward_features(x)
        a = to_bchw(a, self.c1)
        v = to_bchw(self.backbone["vit"].forward_features(x), self.c2, self.prefix2)
        a, v = self.proj_cnn(a), self.proj_vit(v)
        hw = a.shape[-2:]
        if len(self.cnn_from_vit):
            vhw = v.shape[-2:]
            ta, tv = _tokens(a), _tokens(v)
            for blk_a, blk_v in zip(self.cnn_from_vit, self.vit_from_cnn):
                ta, tv = blk_a(ta, tv), blk_v(tv, ta)  # simultaneous update: each side reads the other's previous tokens
            a, v = _grid(ta, hw), _grid(tv, vhw)
        if v.shape[-2:] != hw:
            v = F.adaptive_avg_pool2d(v, hw)
        if self.fuse_mode == "concat":
            return self.cat_proj(torch.cat([a, v], 1))
        if self.fuse_mode == "xattn_nogate":
            return 0.5 * (a + v)
        g = self.gate(torch.cat([a, v], 1))
        return g * a + (1 - g) * v

    def forward_head(self, feat, meta=None):
        f = self.norm(feat)
        a = self.att(f)
        b, _, h, w = a.shape
        att = torch.softmax(a.flatten(1).float(), dim=1).view(b, 1, h, w)
        pooled = (f.float() * att).sum((2, 3)).to(f.dtype)
        if self.meta_dim > 0:
            if meta is None:
                raise ValueError("model built with metadata branch but meta=None")
            pooled = torch.cat([pooled, self.meta_mlp(meta.to(pooled.dtype))], 1)
        return {"logits": self.fc(self.drop(pooled)), "att": att}

    def forward(self, x, meta=None):
        return self.forward_head(self.forward_features(x), meta)

    def param_groups(self, lr: float, head_lr_mult: float, wd: float):
        groups = {"bb_decay": [], "bb_no": [], "hd_decay": [], "hd_no": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            part = "bb" if n.startswith("backbone.") else "hd"
            kind = "no" if p.ndim <= 1 or n.endswith(".bias") else "decay"
            groups[f"{part}_{kind}"].append(p)
        return [
            {"params": groups["bb_decay"], "lr": lr, "weight_decay": wd},
            {"params": groups["bb_no"], "lr": lr, "weight_decay": 0.0},
            {"params": groups["hd_decay"], "lr": lr * head_lr_mult, "weight_decay": wd},
            {"params": groups["hd_no"], "lr": lr * head_lr_mult, "weight_decay": 0.0},
        ]

    def head_modules(self):
        mods = [self.proj_cnn, self.proj_vit, self.cnn_from_vit, self.vit_from_cnn, self.norm,
                self.att, self.fc, self.drop]
        mods += [getattr(self, n) for n in ("gate", "cat_proj", "meta_mlp") if hasattr(self, n)]
        return mods
