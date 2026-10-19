#!/usr/bin/env python
"""Tiny synthetic stand-in for HAM10000 / ISIC 2018 / BCN20000 (pipeline smoke test only).

It mimics the *file layout* and metadata of the real datasets - not their content.
Numbers produced on it mean nothing.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
COLORS = {"akiec": (190, 110, 100), "bcc": (200, 150, 160), "bkl": (140, 100, 70), "df": (160, 120, 110),
          "mel": (60, 40, 40), "nv": (120, 80, 60), "vasc": (170, 40, 60)}
COUNTS = {"akiec": 30, "bcc": 35, "bkl": 50, "df": 25, "mel": 50, "nv": 200, "vasc": 25}
SITES = ["back", "lower extremity", "trunk", "upper extremity", "face", "abdomen", "unknown"]


def draw_lesion(cls, rng, size=(160, 120), jitter=0):
    w, h = size
    skin = tuple(int(c + rng.uniform(-15, 15)) for c in (225, 180, 160))
    img = Image.new("RGB", size, skin)
    mask = Image.new("L", size, 0)
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    cx, cy = w / 2 + rng.uniform(-15, 15) + jitter, h / 2 + rng.uniform(-10, 10)
    rx, ry = rng.uniform(20, 40), rng.uniform(15, 32)
    col = tuple(int(c + rng.uniform(-20, 20)) for c in COLORS[cls])
    box = [cx - rx, cy - ry, cx + rx, cy + ry]
    d.ellipse(box, fill=col)
    dm.ellipse(box, fill=255)
    if cls in ("mel", "bcc"):  # irregular blotches
        for _ in range(4):
            ox, oy = rng.uniform(-rx / 2, rx / 2), rng.uniform(-ry / 2, ry / 2)
            r = rng.uniform(5, 12)
            d.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r], fill=tuple(max(0, c - 40) for c in col))
    if rng.random() < (0.6 if cls == "mel" else 0.15):  # ruler shortcut in the corner
        d.rectangle([5, h - 18, 60, h - 8], fill=(240, 240, 240))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    # dark vignette corners
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    arr *= np.clip(1.25 - 0.45 * r, 0.2, 1)[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ham_dir = os.path.join(a.out, "HAM10000")
    os.makedirs(os.path.join(ham_dir, "HAM10000_images_part_1"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "HAM10000_segmentations"), exist_ok=True)
    rows, img_id, les_id = [], 24306, 0
    for cls, n in COUNTS.items():
        made = 0
        while made < n:
            les_id += 1
            k = 1 if rng.random() < 0.7 else rng.choice([2, 3])  # some lesions with several images
            age, sex, site = rng.choice([np.nan] + list(range(20, 85, 5))), rng.choice(["male", "female"]), rng.choice(SITES)
            for j in range(k):
                if made >= n:
                    break
                im, m = draw_lesion(cls, rng, jitter=3 * j)
                iid = f"ISIC_{img_id:07d}"
                im.save(os.path.join(ham_dir, "HAM10000_images_part_1", iid + ".jpg"), quality=92)
                m.save(os.path.join(a.out, "HAM10000_segmentations", iid + "_segmentation.png"))
                rows.append({"lesion_id": f"HAM_{les_id:07d}", "image_id": iid, "dx": cls,
                             "dx_type": "follow_up" if cls == "nv" else "histo", "age": age, "sex": sex,
                             "localization": site})
                img_id += 1
                made += 1
    pd.DataFrame(rows).to_csv(os.path.join(ham_dir, "HAM10000_metadata.csv"), index=False)

    # ISIC 2018 test layout
    tdir = os.path.join(a.out, "ISIC2018_Task3_Test_Input")
    os.makedirs(tdir, exist_ok=True)
    gt = []
    for cls, n in COUNTS.items():
        for _ in range(max(6, n // 3)):
            im, _ = draw_lesion(cls, rng)
            iid = f"ISIC_{img_id:07d}"
            im.save(os.path.join(tdir, iid + ".jpg"), quality=92)
            gt.append({"image": iid, **{c.upper(): float(c == cls) for c in CLASSES}})
            img_id += 1
    g = pd.DataFrame(gt).rename(columns={"AKIEC": "AKIEC"})
    g[["image", "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]].to_csv(
        os.path.join(a.out, "ISIC2018_Task3_Test_GroundTruth.csv"), index=False)

    # BCN-like external set with free-text diagnoses (incl. SCC to be excluded)
    bdir = os.path.join(a.out, "BCN", "images")
    os.makedirs(bdir, exist_ok=True)
    names = {"nv": "nevus", "mel": "melanoma", "bcc": "basal cell carcinoma", "bkl": "seborrheic keratosis",
             "akiec": "actinic keratosis", "df": "dermatofibroma", "vasc": "vascular lesion"}
    brow = []
    for cls in CLASSES + ["scc"]:
        for _ in range(8):
            im, _ = draw_lesion(cls if cls != "scc" else "akiec", rng)
            iid = f"ISIC_{img_id:07d}"
            im.save(os.path.join(bdir, iid + ".jpg"), quality=92)
            brow.append({"isic_id": iid, "diagnosis": names.get(cls, "squamous cell carcinoma"),
                         "age_approx": rng.choice([35, 50, 65]), "sex": rng.choice(["male", "female"]),
                         "anatom_site_general": rng.choice(["anterior torso", "head/neck", "lower extremity"])})
            img_id += 1
    pd.DataFrame(brow).to_csv(os.path.join(a.out, "BCN", "metadata.csv"), index=False)
    print(f"synthetic data in {a.out}: HAM={len(rows)} test={len(gt)} bcn={len(brow)}")


if __name__ == "__main__":
    main()
