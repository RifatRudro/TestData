"""Dataset, transforms, metadata encoding, samplers and split helpers."""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2

from .constants import IMAGENET_MEAN, IMAGENET_STD, META_DIM, NUM_CLASSES, SEXES, SITES
from .utils import shades_of_gray


# ----------------------------------------------------------------------------- metadata
def encode_meta(age, sex, site) -> np.ndarray:
    v = np.zeros(META_DIM, dtype=np.float32)
    if age is None or (isinstance(age, float) and math.isnan(age)):
        v[1] = 1.0
    else:
        v[0] = float(age) / 100.0
    v[2 + SEXES.index(sex if sex in SEXES else "unknown")] = 1.0
    v[2 + len(SEXES) + SITES.index(site if site in SITES else "unknown")] = 1.0
    return v


# ----------------------------------------------------------------------------- augmentation
def add_hair(img: Image.Image, rng: random.Random, n_max: int = 10) -> Image.Image:
    """Draw thin dark curved strands (synthetic hair) - image only, mask untouched."""
    img = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(1, n_max)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, math.pi)
        seg = rng.uniform(0.15, 0.5) * max(w, h) / 8
        pts = [(x, y)]
        for _ in range(8):
            ang += rng.uniform(-0.3, 0.3)
            x += math.cos(ang) * seg
            y += math.sin(ang) * seg
            pts.append((x, y))
        c = rng.randint(15, 70)
        draw.line(pts, fill=(c, int(c * 0.8), int(c * 0.7)), width=rng.randint(1, max(1, w // 200)))
    return img


def build_transforms(img_size: int, train: bool):
    norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.333), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.15, 0.02)], p=0.8),
            *norm,
        ])
    return v2.Compose([v2.Resize((img_size, img_size), antialias=True), *norm])


# ----------------------------------------------------------------------------- dataset
class DermDataset(Dataset):
    """Returns dict(image, label, mask, mask_valid, meta, idx).

    * Colour constancy is applied on the fly unless the row was cached with it.
    * Masks (if any) are transformed jointly with the image; mask_valid=0 otherwise.
    """

    def __init__(self, df: pd.DataFrame, img_size: int, train: bool, use_masks: bool = True,
                 color_constancy: bool = True, hair_p: float = 0.0, rot90: bool = True):
        self.df = df.reset_index(drop=True)
        self.paths = self.df.path.tolist()
        self.mpaths = [m if isinstance(m, str) and m else None for m in self.df.mask_path.tolist()]
        self.labels = self.df.label.astype(int).tolist()
        self.cc_done = self.df.get("cc_applied", pd.Series([0] * len(self.df))).fillna(0).astype(int).tolist()
        self.meta = np.stack([encode_meta(a, s, t) for a, s, t in
                              zip(self.df.age, self.df.sex, self.df.site)]) if len(self.df) else \
            np.zeros((0, META_DIM), np.float32)
        self.img_size, self.train = img_size, train
        self.use_masks, self.cc, self.hair_p, self.rot90 = use_masks, color_constancy, hair_p, rot90
        self.tf = build_transforms(img_size, train)
        self.rng = random.Random()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.cc and not self.cc_done[i]:
            img = Image.fromarray(shades_of_gray(np.asarray(img)))
        if self.train and self.hair_p > 0 and self.rng.random() < self.hair_p:
            img = add_hair(img, self.rng)
        W, H = img.size
        mvalid = 0.0
        if self.use_masks and self.mpaths[i] is not None:
            m = Image.open(self.mpaths[i]).convert("L")
            if m.size != (W, H):
                m = m.resize((W, H), Image.NEAREST)
            m = (np.asarray(m) > 127).astype(np.uint8)
            mvalid = float(m.sum() > 0)
        else:
            m = np.zeros((H, W), dtype=np.uint8)
        image = tv_tensors.Image(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(m)[None])
        image, mask = self.tf(image, mask)
        if self.train and self.rot90:
            k = self.rng.randint(0, 3)
            if k:
                image = torch.rot90(image, k, dims=(1, 2))
                mask = torch.rot90(mask, k, dims=(1, 2))
        return {
            "image": image.as_subclass(torch.Tensor),
            "label": self.labels[i],
            "mask": mask.as_subclass(torch.Tensor).float(),
            "mask_valid": mvalid,
            "meta": torch.from_numpy(self.meta[i]),
            "idx": i,
        }


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng.seed(seed + worker_id)


# ----------------------------------------------------------------------------- sampling
def class_counts(labels) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=NUM_CLASSES).astype(np.float64)


def make_sampler(labels, q: float):
    """Sample i with probability proportional to n_{y_i}^{-q}.
    q=0: natural, q=0.5: square-root, q=1: class-balanced.
    Returns (sampler or None, effective class prior seen during training)."""
    counts = class_counts(labels)
    safe = np.maximum(counts, 1)
    eff = counts * safe ** (-q)
    prior = eff / eff.sum()
    if q == 0:
        return None, prior
    w = safe[np.asarray(labels, dtype=int)] ** (-q)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels),
                                 replacement=True), prior


# ----------------------------------------------------------------------------- splits
def oversample_then_split(ham: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Simulate the common (leaky) practice: oversample minority classes by duplicating
    images *before* splitting, so copies of one image land in train and test."""
    rng = np.random.RandomState(seed)
    parts = [ham.assign(copy_id=0)]
    for c, g in ham.groupby("label"):
        extra = target - len(g)
        if extra > 0:
            idx = rng.choice(len(g), extra, replace=True)
            parts.append(g.iloc[idx].assign(copy_id=np.arange(1, extra + 1)))
    rep = pd.concat(parts, ignore_index=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rep["fold_leak"] = -1
    for k, (_, te) in enumerate(skf.split(rep, rep.label)):
        rep.loc[te, "fold_leak"] = k
    return rep


def outer_split(df_ham: pd.DataFrame, protocol: str, fold: int, seed: int, leak_target: int):
    """Return (trainval, outer_test or None) for the chosen protocol and fold (-1 = no outer)."""
    if protocol == "oversample_leak":
        rep = oversample_then_split(df_ham, leak_target, seed)
        if fold < 0:
            return rep, None
        return rep[rep.fold_leak != fold].copy(), rep[rep.fold_leak == fold].copy()
    col = {"lesion": "fold_lesion", "image": "fold_image"}[protocol]
    if fold < 0:
        return df_ham.copy(), None
    return df_ham[df_ham[col] != fold].copy(), df_ham[df_ham[col] == fold].copy()


def inner_split(trainval: pd.DataFrame, protocol: str, seed: int, frac: float = 0.1):
    """Hold out ~frac of trainval for checkpoint selection + temperature fitting,
    using the same grouping rule as the outer protocol."""
    n_splits = max(2, int(round(1 / frac)))
    tv = trainval.reset_index(drop=True)
    if protocol == "lesion":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
        tr, va = next(splitter.split(tv, tv.label, groups=tv.group_id))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
        tr, va = next(splitter.split(tv, tv.label))
    return tv.iloc[tr].copy(), tv.iloc[va].copy()
