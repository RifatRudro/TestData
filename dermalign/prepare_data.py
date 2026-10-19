#!/usr/bin/env python
"""Build the master CSV for all experiments.

* HAM10000 (= ISIC 2018 Task 3 training set) for development, with
  - lesion groups = lesion_id, merged with the 20 duplicate / near-duplicate pairs
    confirmed by Abhishek et al. (Sci Data 2025) and optional dHash duplicates,
  - 5-fold StratifiedGroupKFold (column fold_lesion) for the leakage-free protocol,
  - 5-fold StratifiedKFold by image (column fold_image) for the inflation experiment.
* ISIC 2018 Task 3 official test set (1,512 images, ISIC_0035068 dropped) - sealed test.
* Optional: ISIC 2018 validation set, external sets (BCN20000, PH2, ...) via CSV.
* Optional: resized cache with Shades-of-Gray colour constancy pre-applied.

Example
-------
python prepare_data.py \
  --ham_dir /data/HAM10000 --ham_masks_dir /data/HAM10000_segmentations \
  --isic18_test_dir /data/ISIC2018_Task3_Test_Input \
  --isic18_test_gt /data/ISIC2018_Task3_Test_GroundTruth.csv \
  --external bcn20000 /data/bcn/metadata.csv /data/bcn/images \
  --cache_dir /data/cache512 --out data/master.csv
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from dermalign.constants import (CLASSES, CLASS_TO_IDX, ISIC18_COLS, KNOWN_BAD_TEST_IMAGES,
                                 coarse_site, map_diagnosis_row, norm_sex)
from dermalign.leakage import hash_images, make_lesion_folds
from dermalign.utils import UnionFind, hamming_pairs, save_json, shades_of_gray

RES = Path(__file__).parent / "dermalign" / "resources"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def index_files(root, suffixes=IMG_EXT, strip=""):
    """Map stem -> path for every image file under root (recursive)."""
    out = {}
    if not root:
        return out
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(suffixes):
                stem = os.path.splitext(f)[0]
                if strip and stem.endswith(strip):
                    stem = stem[: -len(strip)]
                out.setdefault(stem, os.path.join(dirpath, f))
    return out


def find_csv(root, must_have):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".csv"):
                p = os.path.join(dirpath, f)
                try:
                    cols = set(pd.read_csv(p, nrows=2).columns)
                except Exception:
                    continue
                if set(must_have) <= cols:
                    return p
    return None


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")
    return None


def load_ham(args):
    meta_csv = args.ham_metadata or find_csv(args.ham_dir, ["lesion_id", "image_id", "dx"])
    if meta_csv is None:
        raise FileNotFoundError("HAM10000_metadata.csv not found under --ham_dir")
    meta = pd.read_csv(meta_csv)
    imgs = index_files(args.ham_dir)
    masks = index_files(args.ham_masks_dir, strip="_segmentation") if args.ham_masks_dir else {}
    df = pd.DataFrame({
        "image_id": meta["image_id"],
        "lesion_id": meta["lesion_id"],
        "dx": meta["dx"],
        "dx_type": meta.get("dx_type"),
        "age": meta.get("age"),
        "sex": meta.get("sex", pd.Series(["unknown"] * len(meta))).map(norm_sex),
        "site": meta.get("localization", pd.Series([None] * len(meta))).map(coarse_site),
    })
    df["path"] = df.image_id.map(imgs)
    df["mask_path"] = df.image_id.map(masks)
    missing = df.path.isna().sum()
    if missing:
        print(f"[HAM] WARNING: {missing} images listed in metadata not found on disk; dropped")
        df = df[df.path.notna()]
    df["dataset"] = "ham"
    print(f"[HAM] {len(df)} images, {df.lesion_id.nunique()} lesions, "
          f"{df.mask_path.notna().sum()} masks")
    return df.reset_index(drop=True)


def load_isic18(img_dir, gt_csv, split, meta_csv=None, masks_dir=None):
    imgs = index_files(img_dir)
    if gt_csv:
        gt = pd.read_csv(gt_csv)
        idc = pick_col(gt, ["image", "image_id", "isic_id"])
        onehot = gt[list(ISIC18_COLS)].values
        dx = [ISIC18_COLS[list(ISIC18_COLS)[k]] for k in onehot.argmax(1)]
        df = pd.DataFrame({"image_id": gt[idc], "dx": dx})
    else:  # fall back on the DermaMNIST-E label file (official ISIC 2018 val/test labels)
        lab = pd.read_csv(RES / "dermamnist_e_labels.csv")
        lab = lab[lab.split == split]
        df = pd.DataFrame({"image_id": lab.image, "dx": lab.diag})
        print(f"[ISIC18-{split}] using bundled DermaMNIST-E labels ({len(df)} rows)")
    df = df[~df.image_id.isin(KNOWN_BAD_TEST_IMAGES)]
    df["path"] = df.image_id.map(imgs)
    df = df[df.path.notna()].copy()
    df["lesion_id"] = df.image_id  # no lesion ids released for the test set
    df["age"], df["sex"], df["site"] = np.nan, "unknown", "unknown"
    if meta_csv:
        m = pd.read_csv(meta_csv)
        mid = pick_col(m, ["image_id", "isic_id", "image"])
        m = m.set_index(mid)
        agec = pick_col(m, ["age", "age_approx"], required=False)
        sexc = pick_col(m, ["sex"], required=False)
        sitec = pick_col(m, ["localization", "anatom_site_general", "anatom_site_general_challenge"],
                         required=False)
        if agec:
            df["age"] = df.image_id.map(m[agec])
        if sexc:
            df["sex"] = df.image_id.map(m[sexc]).map(norm_sex)
        if sitec:
            df["site"] = df.image_id.map(m[sitec]).map(coarse_site)
    masks = index_files(masks_dir, strip="_segmentation") if masks_dir else {}
    df["mask_path"] = df.image_id.map(masks)
    df["dataset"] = f"isic18_{split}"
    print(f"[ISIC18-{split}] {len(df)} images: {dict(Counter(df.dx))}")
    return df.reset_index(drop=True)


def load_external(name, csv_path, img_dir):
    raw = pd.read_csv(csv_path)
    idc = pick_col(raw, ["image_id", "isic_id", "image", "bcn_filename", "filename", "name"])
    # most specific first; ISIC-archive exports use the hierarchical diagnosis_1..5 fields
    label_cols = [c for c in ["dx", "diagnosis", "diagnosis_5", "diagnosis_4", "diagnosis_3", "diagnosis_2",
                              "label", "clinical_diagnosis", "diag"] if c in raw.columns]
    if not label_cols:
        raise ValueError(f"[{name}] no diagnosis column in {list(raw.columns)}")
    imgs = index_files(img_dir)
    ids = raw[idc].astype(str).map(lambda s: os.path.splitext(os.path.basename(s))[0])
    dx = raw[label_cols].apply(lambda r: map_diagnosis_row(list(r.values)), axis=1)
    dropped = Counter(raw.loc[dx.isna(), label_cols].astype(str).agg(" | ".join, axis=1))
    if dropped:
        print(f"[{name}] excluded labels outside the 7 classes: {dict(dropped)}")
    agec = pick_col(raw, ["age", "age_approx"], required=False)
    sexc = pick_col(raw, ["sex"], required=False)
    sitec = pick_col(raw, ["localization", "anatom_site_general", "anatom_site_general_challenge",
                           "anatom_site_special"], required=False)
    lesc = pick_col(raw, ["lesion_id"], required=False)
    maskc = pick_col(raw, ["mask_path"], required=False)
    df = pd.DataFrame({
        "image_id": ids, "dx": dx,
        "lesion_id": raw[lesc].astype(str) if lesc else ids,
        "age": raw[agec] if agec else np.nan,
        "sex": raw[sexc].map(norm_sex) if sexc else "unknown",
        "site": raw[sitec].map(coarse_site) if sitec else "unknown",
        "mask_path": raw[maskc] if maskc else None,
    })
    df["path"] = df.image_id.map(imgs)
    if "path" in raw.columns:
        df["path"] = df["path"].fillna(raw["path"])
    df = df[df.dx.notna() & df.path.notna()].copy()
    df["dataset"] = name
    print(f"[{name}] {len(df)} images: {dict(Counter(df.dx))}")
    return df.reset_index(drop=True)


def build_ph2_csv(ph2_root, out_csv):
    """Convert the PH2 folder layout into a CSV usable with --external."""
    root = Path(ph2_root)
    txt = next(root.rglob("PH2_dataset.txt"), None)
    if txt is None:
        raise FileNotFoundError("PH2_dataset.txt not found")
    rows, header = [], None
    for line in open(txt, encoding="latin-1"):
        if not line.strip().startswith("||"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("||")]
        if header is None:
            header = cells
            continue
        rows.append(cells)
    name_i = next(i for i, h in enumerate(header) if h.lower().startswith("name"))
    clin_i = next(i for i, h in enumerate(header) if "clinical" in h.lower())
    recs = []
    for r in rows:
        if len(r) <= max(name_i, clin_i):
            continue
        name, clin = r[name_i], r[clin_i]
        dx = {"0": "nv", "1": "nv", "2": "mel"}.get(clin)
        img = next(root.rglob(f"{name}_Dermoscopic_Image/{name}.bmp"), None)
        msk = next(root.rglob(f"{name}_lesion/{name}_lesion.bmp"), None)
        if dx and img:
            recs.append({"image_id": name, "dx": dx, "path": str(img),
                         "mask_path": str(msk) if msk else None})
    pd.DataFrame(recs).to_csv(out_csv, index=False)
    print(f"[PH2] wrote {len(recs)} rows to {out_csv}")


def merge_groups(ham, args):
    """lesion_id groups merged with confirmed duplicate pairs (+ optional dHash)."""
    uf = UnionFind(ham.lesion_id.unique())
    img2les = dict(zip(ham.image_id, ham.lesion_id))
    n_merge = 0
    for f in ["newly_discovered_duplicates.csv", "near_duplicates_different_times.csv"]:
        p = RES / f
        if not p.exists():
            continue
        for _, r in pd.read_csv(p).iterrows():
            a, b = r.from_img.replace(".jpg", ""), r.to_img.replace(".jpg", "")
            if a in img2les and b in img2les and uf.find(img2les[a]) != uf.find(img2les[b]):
                uf.union(img2les[a], img2les[b])
                n_merge += 1
    print(f"[groups] merged {n_merge} lesion pairs from the confirmed duplicate lists")
    hashes = None
    if args.dhash:
        hashes = compute_hashes(ham.path.tolist(), args.workers)
        pairs = hamming_pairs(hashes, max_dist=args.dhash_max_dist)
        n2, kept = 0, []
        for i, j, d in pairs:
            li, lj = ham.lesion_id.iat[i], ham.lesion_id.iat[j]
            if li == lj or ham.dx.iat[i] != ham.dx.iat[j]:  # same lesion already, or not a true copy
                continue
            kept.append((ham.image_id.iat[i], ham.image_id.iat[j], li, lj, ham.dx.iat[i], d))
            if uf.find(li) != uf.find(lj):
                uf.union(li, lj)
                n2 += 1
        pd.DataFrame(kept, columns=["image_a", "image_b", "lesion_a", "lesion_b", "dx", "hamming"]).to_csv(
            Path(args.out).with_name("dhash_pairs_to_review.csv"), index=False)
        print(f"[groups] dHash: {len(kept)} cross-lesion near-identical pairs (same dx) -> merged {n2} groups; "
              f"review {Path(args.out).with_name('dhash_pairs_to_review.csv')}")
        if len(kept) > 0.02 * len(ham):
            print("[groups] WARNING: unusually many dHash matches - lower --dhash_max_dist and review the pairs")
    ham["group_id"] = ham.lesion_id.map(uf.find)
    return ham, hashes, n_merge


def compute_hashes(paths, workers):
    """dHash of every image. Unreadable files get a random hash (never within max_dist of
    anything): an all-zero hash would make every failed file a 'duplicate' of every other."""
    H = hash_images(paths, workers)
    dh = H["dh"]
    if (~H["ok"]).any():
        print(f"[hash] WARNING: {int((~H['ok']).sum())} unreadable images excluded from duplicate detection")
        dh[~H["ok"]] = np.random.RandomState(0).randint(0, 256, (int((~H["ok"]).sum()), dh.shape[1])).astype(np.uint8)
    return dh


def _cache_one(job):
    src, msrc, dst, mdst, short, cc = job
    try:
        if not os.path.exists(dst):
            im = Image.open(src).convert("RGB")
            w, h = im.size
            s = short / min(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
            arr = np.asarray(im)
            if cc:
                arr = shades_of_gray(arr)
            Image.fromarray(arr).save(dst, quality=95)
        if msrc and isinstance(msrc, str) and not os.path.exists(mdst):
            m = Image.open(msrc).convert("L")
            w, h = Image.open(dst).size
            m = m.resize((w, h), Image.NEAREST)
            m.save(mdst)
        return True
    except Exception as e:  # pragma: no cover
        print("cache failed", src, e)
        return False


def build_cache(df, cache_dir, short, workers, cc):
    jobs = []
    for r in df.itertuples():
        d = Path(cache_dir) / r.dataset
        d.mkdir(parents=True, exist_ok=True)
        dst = str(d / f"{r.image_id}.jpg")
        mdst = str(d / f"{r.image_id}_mask.png")
        jobs.append((r.path, r.mask_path, dst, mdst, short, cc))
    with Pool(workers) as p:
        ok = p.map(_cache_one, jobs, chunksize=32)
    df = df.copy()
    df["path"] = [j[2] for j in jobs]
    df["mask_path"] = [j[3] if isinstance(j[1], str) else None for j in jobs]
    df["cc_applied"] = int(cc)
    print(f"[cache] {sum(ok)}/{len(ok)} images cached at short side {short}px in {cache_dir}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ham_dir", required=True)
    ap.add_argument("--ham_metadata", default=None)
    ap.add_argument("--ham_masks_dir", default=None)
    ap.add_argument("--isic18_test_dir", default=None)
    ap.add_argument("--isic18_test_gt", default=None)
    ap.add_argument("--isic18_test_meta", default=None)
    ap.add_argument("--isic18_val_dir", default=None)
    ap.add_argument("--isic18_val_gt", default=None)
    ap.add_argument("--external", nargs=3, action="append", default=[],
                    metavar=("NAME", "CSV", "IMG_DIR"))
    ap.add_argument("--ph2_root", default=None, help="convenience: parse PH2 folder as external 'ph2'")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dhash", action="store_true", help="also merge dHash near-duplicates")
    ap.add_argument("--dhash_max_dist", type=int, default=8, help="bits out of 256")
    ap.add_argument("--check_test_overlap", action="store_true",
                    help="flag test images that are near-identical to HAM images")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--cache_short", type=int, default=512)
    ap.add_argument("--no_color_constancy", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="data/master.csv")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ham = load_ham(args)
    ham, ham_hashes, n_dup_merge = merge_groups(ham, args)
    ham["label"] = ham.dx.map(CLASS_TO_IDX)

    # Leakage-free folds: grouped by (merged) lesion, stratified by class.
    ham["fold_lesion"] = make_lesion_folds(ham, args.n_folds, args.seed)
    # Image-level folds (the common - leaky - protocol), for the inflation experiment.
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    ham["fold_image"] = -1
    for k, (_, te) in enumerate(skf.split(ham, ham.label)):
        ham.loc[ham.index[te], "fold_image"] = k

    # How many image-level test images have a sibling image of the same lesion in training?
    leak = []
    for k in range(args.n_folds):
        te = ham[ham.fold_image == k]
        tr_groups = set(ham.loc[ham.fold_image != k, "group_id"])
        leak.append(te.group_id.isin(tr_groups).mean())
    print(f"[folds] image-level split: {100*np.mean(leak):.1f}% of test images share a lesion "
          f"with training images (lesion-level split: 0%)")

    parts = [ham]
    if args.isic18_test_dir:
        parts.append(load_isic18(args.isic18_test_dir, args.isic18_test_gt, "test", args.isic18_test_meta))
    if args.isic18_val_dir:
        parts.append(load_isic18(args.isic18_val_dir, args.isic18_val_gt, "val"))
    ext = list(args.external)
    if args.ph2_root:
        ph2_csv = Path(args.out).with_name("ph2_index.csv")
        Path(ph2_csv).parent.mkdir(parents=True, exist_ok=True)
        build_ph2_csv(args.ph2_root, ph2_csv)
        ext.append(("ph2", str(ph2_csv), args.ph2_root))
    for name, csv_path, img_dir in ext:
        parts.append(load_external(name, csv_path, img_dir))

    df = pd.concat(parts, ignore_index=True)
    df["label"] = df.dx.map(CLASS_TO_IDX).astype(int)
    df["group_id"] = df["group_id"].fillna(df["dataset"] + ":" + df["lesion_id"].astype(str))
    for c in ["fold_lesion", "fold_image"]:
        df[c] = df[c].fillna(-1).astype(int)
    df["cc_applied"] = 0
    df["test_dup_of_ham"] = 0

    if args.check_test_overlap and len(df) > len(ham):
        if ham_hashes is None:
            ham_hashes = compute_hashes(ham.path.tolist(), args.workers)
        other = df[df.dataset != "ham"]
        oh = compute_hashes(other.path.tolist(), args.workers)
        pairs = hamming_pairs(oh, ham_hashes, max_dist=args.dhash_max_dist)
        flagged = sorted({i for i, _, _ in pairs})
        df.loc[other.index[flagged], "test_dup_of_ham"] = 1
        print(f"[overlap] {len(flagged)} non-HAM images are near-identical to a HAM image "
              f"(column test_dup_of_ham=1; excluded by evaluate.py)")

    if args.cache_dir:
        df = build_cache(df, args.cache_dir, args.cache_short, args.workers,
                         cc=not args.no_color_constancy)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "path", "mask_path", "dataset", "dx", "label", "lesion_id", "group_id",
            "fold_lesion", "fold_image", "age", "sex", "site", "dx_type", "cc_applied",
            "test_dup_of_ham"]
    df[cols].to_csv(args.out, index=False)

    summary = {
        "n_images": int(len(df)),
        "per_dataset": {d: {"n": int(len(g)), "classes": {c: int((g.dx == c).sum()) for c in CLASSES},
                            "masks": int(g.mask_path.notna().sum())}
                        for d, g in df.groupby("dataset")},
        "ham_lesions": int(ham.lesion_id.nunique()),
        "ham_groups_after_merge": int(ham.group_id.nunique()),
        "duplicate_pairs_merged": int(n_dup_merge),
        "image_split_test_share_with_train_lesion": float(np.mean(leak)),
        "fold_sizes_lesion": ham.fold_lesion.value_counts().sort_index().to_dict(),
    }
    save_json(summary, Path(args.out).with_suffix(".summary.json"))
    print(f"[done] wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
