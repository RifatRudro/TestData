"""Backbones (timm) with three heads:

* gap  - global average pooling + linear (standard baseline)
* cbam - CBAM channel+spatial attention, then GAP (baseline as in Xception+CBAM papers)
* attn - DermAlign spatial attention pooling. The attention map A (sums to 1 over the
         feature grid) is the model's built-in explanation and is aligned to the lesion
         mask during training by the lesion-alignment loss (see losses.py).

Optional metadata branch (age, sex, coarse anatomical site) with late fusion.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import LayerNorm2d
except ImportError:  # older timm
    from timm.models.layers import LayerNorm2d

_TOKEN_ARCHS = ("vit", "deit", "beit", "eva", "swin")


def to_bchw(feat: torch.Tensor, channels: int, prefix_tokens: int = 0) -> torch.Tensor:
    """Bring backbone features to B x C x h x w (handles ViT tokens and NHWC Swin)."""
    if feat.ndim == 3:  # B, N, C tokens
        feat = feat[:, prefix_tokens:, :]
        b, n, c = feat.shape
        s = int(round(n ** 0.5))
        if s * s != n:
            raise ValueError(f"cannot reshape {n} tokens to a square grid")
        return feat.transpose(1, 2).reshape(b, c, s, s)
    if feat.ndim == 4 and feat.shape[1] != channels and feat.shape[-1] == channels:
        return feat.permute(0, 3, 1, 2).contiguous()
    return feat


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        hidden = max(8, c // r)
        self.mlp = nn.Sequential(nn.Conv2d(c, hidden, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, c, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa, sa


class DermModel(nn.Module):
    def __init__(self, arch: str, num_classes: int = 7, head: str = "attn", pretrained: bool = True,
                 drop_rate: float = 0.2, drop_path_rate: float = 0.1, meta_dim: int = 0,
                 att_hidden: int = 256, img_size: int | None = None):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0, global_pool="")
        if img_size and any(k in arch for k in _TOKEN_ARCHS):
            kw["img_size"] = img_size
        try:
            self.backbone = timm.create_model(arch, drop_path_rate=drop_path_rate, **kw)
        except TypeError:
            self.backbone = timm.create_model(arch, **kw)
        C = self.backbone.num_features
        self.C, self.head_type = C, head
        self.prefix = getattr(self.backbone, "num_prefix_tokens", 0)
        self.norm = LayerNorm2d(C)
        if head == "attn":
            self.att = nn.Sequential(nn.Conv2d(C, att_hidden, 1), nn.GELU(), nn.Conv2d(att_hidden, 1, 1))
            nn.init.zeros_(self.att[-1].weight)  # start as uniform attention (= GAP)
            nn.init.zeros_(self.att[-1].bias)
        elif head == "cbam":
            self.cbam = CBAM(C)
        elif head != "gap":
            raise ValueError(head)
        self.meta_dim = meta_dim
        fc_in = C
        if meta_dim > 0:
            self.meta_mlp = nn.Sequential(nn.Linear(meta_dim, 64), nn.GELU(), nn.Dropout(0.1),
                                          nn.Linear(64, 64), nn.GELU())
            fc_in += 64
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(fc_in, num_classes)

    # split so XAI code can take gradients w.r.t. the feature map
    def forward_features(self, x):
        return to_bchw(self.backbone.forward_features(x), self.C, self.prefix)

    def forward_head(self, feat, meta=None):
        f = self.norm(feat)
        att = None
        if self.head_type == "attn":
            a = self.att(f)
            b, _, h, w = a.shape
            att = torch.softmax(a.flatten(1).float(), dim=1).view(b, 1, h, w)
            pooled = (f.float() * att).sum((2, 3)).to(f.dtype)
        elif self.head_type == "cbam":
            f2, sa = self.cbam(f)
            pooled = f2.mean((2, 3))
            att = sa.float() / sa.float().flatten(1).sum(1).clamp_min(1e-8).view(-1, 1, 1, 1)
        else:
            pooled = f.mean((2, 3))
        if self.meta_dim > 0:
            if meta is None:
                raise ValueError("model built with metadata branch but meta=None")
            pooled = torch.cat([pooled, self.meta_mlp(meta.to(pooled.dtype))], 1)
        logits = self.fc(self.drop(pooled))
        return {"logits": logits, "att": att}

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
        mods = [self.norm, self.fc, self.drop]
        for name in ("att", "cbam", "meta_mlp"):
            if hasattr(self, name):
                mods.append(getattr(self, name))
        return mods


def build_model(cfg: dict, pretrained: bool | None = None) -> DermModel:
    from .constants import META_DIM, NUM_CLASSES
    if cfg["head"] == "xfuse":
        from .hybrid import HybridDermModel
        return HybridDermModel(
            arch=cfg["arch"], arch2=cfg.get("arch2", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
            num_classes=NUM_CLASSES, pretrained=cfg["pretrained"] if pretrained is None else pretrained,
            drop_rate=cfg.get("drop_rate", 0.2), drop_path_rate=cfg.get("drop_path", 0.1),
            meta_dim=META_DIM if cfg.get("use_meta") else 0, att_hidden=cfg.get("att_hidden", 256),
            img_size=cfg.get("img_size"), fuse_dim=cfg.get("fuse_dim", 256),
            fuse_heads=cfg.get("fuse_heads", 4), fuse_depth=cfg.get("fuse_depth", 1),
            fuse_mode=cfg.get("fuse_mode", "xattn"),
        )
    return DermModel(
        arch=cfg["arch"], num_classes=NUM_CLASSES, head=cfg["head"],
        pretrained=cfg["pretrained"] if pretrained is None else pretrained,
        drop_rate=cfg.get("drop_rate", 0.2), drop_path_rate=cfg.get("drop_path", 0.1),
        meta_dim=META_DIM if cfg.get("use_meta") else 0, att_hidden=cfg.get("att_hidden", 256),
        img_size=cfg.get("img_size"),
    )
