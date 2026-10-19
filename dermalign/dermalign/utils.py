"""Small utilities: seeding, JSON, temperature scaling, image helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 300) -> float:
    """Temperature scaling (Guo et al., 2017): minimise NLL of softmax(logits / T)."""
    import torch
    import torch.nn.functional as F
    L = torch.tensor(np.array(logits), dtype=torch.float64)
    Y = torch.tensor(np.array(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(L / log_t.exp(), Y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def shades_of_gray(img: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi, 2004), as used by
    Barata et al. for dermoscopy. img: HxWx3 uint8 -> uint8."""
    x = img.astype(np.float32)
    illum = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1.0 / power)
    illum = illum / (np.linalg.norm(illum) + 1e-6)
    x = x / (illum * np.sqrt(3.0) + 1e-6)
    return np.clip(x, 0, 255).astype(np.uint8)


def dhash(img, size: int = 16) -> np.ndarray:
    """Difference hash (size*size bits, packed to uint8) of a PIL image. Used only to find
    exact / near-exact copies (re-saved, resized); different photos of one lesion are
    covered by lesion_id and the confirmed duplicate lists instead."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_pairs(hashes_a: np.ndarray, hashes_b: np.ndarray | None = None,
                  max_dist: int = 8, chunk: int = 256):
    """Return (i, j, dist) with Hamming distance <= max_dist between packed-bit hashes
    (N x nbytes uint8). If hashes_b is None, compares a with itself (i < j)."""
    same = hashes_b is None
    b = hashes_a if same else hashes_b
    out = []
    for s in range(0, len(hashes_a), chunk):
        a = hashes_a[s:s + chunk]
        d = _POPCOUNT8[a[:, None, :] ^ b[None, :, :]].sum(-1, dtype=np.int32)
        ii, jj = np.nonzero(d <= max_dist)
        for i, j in zip(ii, jj):
            gi = s + i
            if same and j <= gi:
                continue
            out.append((gi, int(j), int(d[i, j])))
    return out


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def environment_info() -> dict:
    """Library versions, device and git commit - saved next to every run for reproducibility."""
    import platform
    import subprocess
    import sys

    import timm
    import torch
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "timm": timm.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                            cwd=Path(__file__).resolve().parent, timeout=5).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info
