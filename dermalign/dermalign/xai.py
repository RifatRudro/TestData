"""Explanation methods and quantitative explanation metrics.

Methods: built-in attention (DermAlign head), Grad-CAM, Grad-CAM++, Integrated Gradients,
plus two controls every localisation number must beat: a centred Gaussian prior and a
random map (lesions in dermoscopy are usually central, so the centre prior is a strong
control that most papers omit).

Metrics:
* Localisation: relevance mass accuracy (share of saliency inside the lesion mask),
  relevance rank accuracy (share of the top-K salient pixels inside the mask, K = mask
  area; Arras et al., 2022), pointing game (argmax inside mask, small tolerance).
* Shortcut proxy: share of saliency in the outer border band outside the dilated lesion
  (where rulers, dark vignette corners and ink usually sit).
* Faithfulness: deletion / insertion AUC (Petsiuk et al., 2018) on a coarse cell grid.
* Sanity: model-parameter randomisation (Adebayo et al., 2018): Spearman rank
  correlation between maps of the trained and partially re-initialised model.
* Agreement: Spearman correlation between methods.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- maps
def _norm01(m: torch.Tensor) -> torch.Tensor:
    m = m.clamp_min(0)
    flat = m.flatten(1)
    mn, mx = flat.min(1)[0], flat.max(1)[0]
    return ((flat - mn[:, None]) / (mx - mn).clamp_min(1e-12)[:, None]).view_as(m)


def _up(m: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(m[:, None].float(), size=size, mode="bilinear", align_corners=False)[:, 0]


def attention_map(model, x, meta=None):
    with torch.no_grad():
        out = model(x, meta)
    if out["att"] is None:
        return None
    return _norm01(_up(out["att"][:, 0], x.shape[-2:]))


def grad_cam(model, x, target, meta=None, plus=False):
    feat = model.forward_features(x)
    logits = model.forward_head(feat, meta)["logits"]
    score = logits.gather(1, target[:, None]).sum()
    grads = torch.autograd.grad(score, feat)[0]
    feat, grads = feat.detach().float(), grads.float()
    if plus:
        g2, g3 = grads ** 2, grads ** 3
        sum_a = feat.sum((2, 3), keepdim=True)
        denom = 2 * g2 + sum_a * g3
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w = (alpha * F.relu(grads)).sum((2, 3))
    else:
        w = grads.mean((2, 3))
    cam = F.relu((w[:, :, None, None] * feat).sum(1))
    return _norm01(_up(cam, x.shape[-2:]))


def integrated_gradients(model, x, target, meta=None, steps=32, baseline="blur"):
    if baseline == "blur":
        k = max(3, (x.shape[-1] // 8) | 1)
        x0 = F.avg_pool2d(F.pad(x, [k // 2] * 4, mode="reflect"), k, stride=1)
    else:
        x0 = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (x0 + a * (x - x0)).requires_grad_(True)
        logits = model(xi, meta)["logits"]
        g = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), xi)[0]
        total += g.detach()
    attr = ((x - x0) * total / steps).abs().sum(1)
    attr = F.avg_pool2d(attr[:, None], 5, stride=1, padding=2)[:, 0]  # light smoothing
    return _norm01(attr)


def centre_prior(x, sigma_frac=0.25):
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * (2 * sigma_frac) ** 2))
    return g[None].repeat(b, 1, 1).to(x.device)


def random_map(x, seed=0):
    gen = torch.Generator().manual_seed(seed)
    b, _, h, w = x.shape
    m = torch.rand(b, 1, max(2, h // 32), max(2, w // 32), generator=gen)
    return _norm01(F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[:, 0]).to(x.device)


def compute_maps(model, x, target, methods, meta=None, ig_steps=32):
    out = {}
    for m in methods:
        if m == "attention":
            a = attention_map(model, x, meta)
            if a is not None:
                out[m] = a
        elif m == "gradcam":
            out[m] = grad_cam(model, x, target, meta, plus=False)
        elif m == "gradcampp":
            out[m] = grad_cam(model, x, target, meta, plus=True)
        elif m == "ig":
            out[m] = integrated_gradients(model, x, target, meta, steps=ig_steps)
        elif m == "centre":
            out[m] = centre_prior(x)
        elif m == "random":
            out[m] = random_map(x)
    return {k: v.detach().cpu() for k, v in out.items()}


# --------------------------------------------------------------------------- localisation
def localisation_metrics(sal: np.ndarray, mask: np.ndarray, border_frac=0.1, tol_frac=0.03,
                         dilate_frac=0.05):
    """sal, mask: H x W. Returns dict (NaN if mask empty)."""
    m = mask > 0.5
    if m.sum() == 0:
        return None
    s = np.clip(sal, 0, None).astype(np.float64)
    tot = s.sum() + 1e-12
    h, w = m.shape
    mass = float((s * m).sum() / tot)
    k = int(m.sum())
    topk = np.argpartition(-s.ravel(), k - 1)[:k]
    rank = float(m.ravel()[topk].mean())
    tol = max(1, int(round(tol_frac * h)))
    mt = F.max_pool2d(torch.from_numpy(m[None, None].astype(np.float32)), 2 * tol + 1, 1, tol)[0, 0].numpy() > 0
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    point = float(mt[iy, ix])
    r = max(1, int(round(dilate_frac * h)))
    md = F.max_pool2d(torch.from_numpy(m[None, None].astype(np.float32)), 2 * r + 1, 1, r)[0, 0].numpy() > 0
    band = np.zeros_like(m)
    bh, bw = max(1, int(border_frac * h)), max(1, int(border_frac * w))
    band[:bh], band[-bh:], band[:, :bw], band[:, -bw:] = True, True, True, True
    border = float((s * (band & ~md)).sum() / tot)
    return {"mass_in_mask": mass, "rank_acc": rank, "pointing": point,
            "border_outside": border, "mask_area": float(m.mean())}


# --------------------------------------------------------------------------- faithfulness
@torch.no_grad()
def deletion_insertion(model, x, target, sal, meta=None, grid=14, n_steps=20, batch=32):
    """x: 1 x 3 x H x W (normalised), sal: H x W tensor. Returns (deletion AUC, insertion AUC)
    of the target-class probability as cells are removed / inserted by saliency order.
    Removed cells are replaced by a heavily blurred copy of the image."""
    _, _, h, w = x.shape
    k = max(3, (w // 6) | 1)
    blur = F.avg_pool2d(F.pad(x, [k // 2] * 4, mode="reflect"), k, stride=1)
    cell = F.adaptive_avg_pool2d(sal[None, None].float().to(x.device), grid)[0, 0].flatten()
    order = torch.argsort(cell, descending=True)
    n_cells = grid * grid
    fracs = np.linspace(0, 1, n_steps + 1)
    del_imgs, ins_imgs = [], []
    for f in fracs:
        n = int(round(f * n_cells))
        keep = torch.zeros(n_cells, device=x.device)
        keep[order[:n]] = 1.0
        m = F.interpolate(keep.view(1, 1, grid, grid), size=(h, w), mode="nearest")
        del_imgs.append(x * (1 - m) + blur * m)
        ins_imgs.append(blur * (1 - m) + x * m)
    probs = []
    for imgs in (del_imgs, ins_imgs):
        stack = torch.cat(imgs)
        p = []
        for s in range(0, len(stack), batch):
            chunk = stack[s:s + batch]
            mm = None if meta is None else meta.expand(len(chunk), -1)
            p.append(torch.softmax(model(chunk, mm)["logits"].float(), 1)[:, target])
        probs.append(torch.cat(p).cpu().numpy())
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = lambda y: float(trapz(y, fracs))
    return auc(probs[0]), auc(probs[1])


# --------------------------------------------------------------------------- sanity check
def randomise_top(model: nn.Module, frac: float = 0.25, seed: int = 0) -> nn.Module:
    """Copy of the model with the head and the top `frac` of backbone weight tensors
    re-initialised (cascading randomisation from the top, Adebayo et al. 2018)."""
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in m.head_modules():
            for n, p in mod.named_parameters():
                if p.ndim > 1:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)
                elif n.endswith("bias"):
                    p.zero_()
        weights = [p for n, p in m.backbone.named_parameters() if p.ndim > 1]
        for p in weights[int(len(weights) * (1 - frac)):]:
            std = float(p.std()) if p.numel() > 1 else 0.02
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return m


def map_correlation(a: np.ndarray, b: np.ndarray, size=32) -> float:
    ta = F.adaptive_avg_pool2d(torch.as_tensor(a)[None, None].float(), size).flatten().numpy()
    tb = F.adaptive_avg_pool2d(torch.as_tensor(b)[None, None].float(), size).flatten().numpy()
    if ta.std() < 1e-12 or tb.std() < 1e-12:
        return 0.0
    return float(spearmanr(ta, tb).correlation)
