"""Leakage audit and repair for the master CSV and for finished runs.

The protocol claims that no lesion (nor a confirmed / near-exact duplicate of one) is shared
between train, inner validation and the outer fold, and that no test image is a copy of a
HAM10000 image. `audit_master` checks that claim from the files, `audit_runs` replays the
train / validation / outer sets that finished runs actually used, and `fix_master` repairs
what is found (merge duplicate groups, move the merged group to a single fold, flag test copies).
"""
from __future__ import annotations

import hashlib
import io
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .data import inner_split, outer_split
from .utils import UnionFind, dhash, hamming_pairs


# --------------------------------------------------------------------------- folds
def make_lesion_folds(ham: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Class-stratified folds that keep every `group_id` inside one fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=ham.index, dtype=int)
    for k, (_, te) in enumerate(sgkf.split(ham, ham.label, groups=ham.group_id)):
        fold.iloc[te] = k
    return fold


# --------------------------------------------------------------------------- hashing
def _hash_file(path):
    try:
        data = Path(path).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            return hashlib.md5(data).hexdigest(), dhash(im)
    except Exception:
        return None, np.zeros(32, dtype=np.uint8)


def hash_images(paths, workers: int = 1) -> dict:
    """md5 of the file bytes + dHash for every path. Unreadable files get ok=False and are
    excluded from pairing (an all-zero hash would otherwise match every other failure)."""
    paths = list(paths)
    if workers > 1:
        with Pool(workers) as p:
            out = p.map(_hash_file, paths, chunksize=64)
    else:
        out = [_hash_file(p) for p in paths]
    return {"md5": np.array([o[0] or "" for o in out], dtype=object),
            "dh": np.stack([o[1] for o in out]).astype(np.uint8),
            "ok": np.array([o[0] is not None for o in out])}


def find_leak_pairs(df: pd.DataFrame, H: dict, max_dist: int = 8) -> dict:
    """Image pairs (positional row indices into df) that can leak:
    ext: (non-HAM row, HAM row, dist)  - a test / external image that is a copy of a HAM image
    ham: (HAM row, HAM row, dist)      - same diagnosis, different group, near-identical pixels
    exact: number of ext pairs with byte-identical files."""
    is_ham = (df.dataset == "ham").to_numpy()
    ham_pos = np.flatnonzero(is_ham & H["ok"])
    oth_pos = np.flatnonzero(~is_ham & H["ok"])
    ext, hamp = [], []
    if len(ham_pos) and len(oth_pos):
        for i, j, d in hamming_pairs(H["dh"][oth_pos], H["dh"][ham_pos], max_dist=max_dist):
            ext.append((int(oth_pos[i]), int(ham_pos[j]), d))
    if len(ham_pos) > 1:
        gid, dx = df.group_id.to_numpy(), df.dx.to_numpy()
        for i, j, d in hamming_pairs(H["dh"][ham_pos], None, max_dist=max_dist):
            a, b = int(ham_pos[i]), int(ham_pos[j])
            if gid[a] != gid[b] and dx[a] == dx[b]:
                hamp.append((a, b, d))
    exact = sum(H["md5"][a] == H["md5"][b] for a, b, _ in ext)
    return {"ext": ext, "ham": hamp, "exact": int(exact), "n_unreadable": int((~H["ok"]).sum())}


# --------------------------------------------------------------------------- audit
def audit_master(df: pd.DataFrame, pairs: dict | None = None) -> dict:
    """Run every check that can be done from the master CSV (and from `pairs` if given)."""
    checks = []

    def add(name, ok, bad, severity="error", note=""):
        bad = list(bad)
        checks.append({"name": name, "ok": bool(ok), "n_bad": len(bad), "severity": severity,
                       "examples": [str(x) for x in bad[:10]], "note": note})

    ham = df[df.dataset == "ham"]
    oth = df[df.dataset != "ham"]
    n_folds = int(ham.fold_lesion.max()) + 1 if len(ham) else 0

    bad = ham[~ham.fold_lesion.between(0, max(n_folds - 1, 0))].image_id
    add("fold_coverage", len(ham) > 0 and n_folds >= 2 and len(bad) == 0, bad,
        note=f"{n_folds} lesion folds over {len(ham)} HAM10000 images")

    n_f = ham.groupby("group_id").fold_lesion.nunique()
    add("group_single_fold", (n_f <= 1).all(), n_f[n_f > 1].index,
        note="a group in two folds = the same lesion in train and test")
    n_f = ham.groupby("lesion_id").fold_lesion.nunique()
    add("lesion_single_fold", (n_f <= 1).all(), n_f[n_f > 1].index)

    share = float(ham.group_id.value_counts(normalize=True).iloc[0]) if len(ham) else 0.0
    add("largest_group_share", share <= 0.05, [], "warning",
        note=f"largest group holds {100 * share:.1f}% of HAM10000; >5% usually means --max_dist chains "
             f"unrelated images together (unbalanced folds) - lower it and review the pairs")
    dup_ids = df.image_id[df.image_id.duplicated(keep=False)]
    add("unique_image_ids", len(dup_ids) == 0, dup_ids.unique(),
        note="an image_id present twice (also across datasets)")
    shared = (set(ham.lesion_id.dropna().astype(str)) & set(oth.lesion_id.dropna().astype(str)))
    add("no_lesion_id_overlap_with_test", not shared, sorted(shared))

    if pairs is None:
        add("test_not_copy_of_ham", True, [], "skipped", "image checks not run (no hashes)")
        add("no_cross_group_copies_across_folds", True, [], "skipped", "image checks not run")
    else:
        flag = df.test_dup_of_ham.to_numpy() if "test_dup_of_ham" in df else np.zeros(len(df), int)
        ext_pos = sorted({a for a, _, _ in pairs["ext"]})
        unflagged = [df.image_id.iat[a] for a in ext_pos if flag[a] != 1]
        add("test_not_copy_of_ham", not unflagged, unflagged,
            note=f"{len(ext_pos)} test/external images are near-copies of HAM10000 "
                 f"({pairs['exact']} byte-identical); all must carry test_dup_of_ham=1")
        fold = df.fold_lesion.to_numpy()
        straddle = [f"{df.image_id.iat[a]}~{df.image_id.iat[b]}" for a, b, _ in pairs["ham"]
                    if fold[a] != fold[b]]
        add("no_cross_group_copies_across_folds", not straddle, straddle,
            note=f"{len(pairs['ham'])} near-identical same-diagnosis HAM pairs in different groups; "
                 f"{len(straddle)} of them sit in different folds")
        add("images_readable", pairs["n_unreadable"] == 0, [], "warning" if pairs["n_unreadable"] else "error",
            note=f"{pairs['n_unreadable']} unreadable files were left out of the pair search")
        if pairs["n_unreadable"]:
            checks[-1]["ok"] = False
    ok = all(c["ok"] for c in checks if c["severity"] == "error")
    return {"ok": ok, "n_folds": n_folds, "checks": checks}


def audit_runs(df: pd.DataFrame, runs_dir) -> pd.DataFrame:
    """Per finished run, rebuild the sets the run really used (from its exported predictions)
    and test them against the *current* groups of the master CSV."""
    ham = df[df.dataset == "ham"]
    g_of = dict(zip(ham.image_id, ham.group_id))
    all_ids = set(ham.image_id)
    flagged = set(df.image_id[df.test_dup_of_ham == 1]) if "test_dup_of_ham" in df else set()
    rows = []
    for cfg_path in sorted(Path(runs_dir).glob("*/*/config.json")):
        run = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        split = cfg.get("split", "lesion")
        row = {"exp": run.parent.name, "run": run.name, "split": split, "fold": cfg.get("fold"),
               "n_train": np.nan, "n_val": np.nan, "n_outer": np.nan, "overlap_train_val": np.nan,
               "overlap_train_outer": np.nan, "outer_share_with_train_lesion": np.nan,
               "test_copies_predicted": np.nan, "status": "incomplete"}
        pv, po = run / "preds_innerval.csv", run / "preds_outer.csv"
        if pv.exists():
            val = set(pd.read_csv(pv, usecols=["image_id"]).image_id) & all_ids
            outer = (set(pd.read_csv(po, usecols=["image_id"]).image_id) & all_ids) if po.exists() else set()
            train = all_ids - val - outer
            gt, gv, go = ({g_of[i] for i in s} for s in (train, val, outer))
            row.update(n_train=len(train), n_val=len(val), n_outer=len(outer),
                       overlap_train_val=len(gt & gv), overlap_train_outer=len(gt & go))
            if outer:
                row["outer_share_with_train_lesion"] = float(np.mean([g_of[i] in gt for i in outer]))
            if split != "lesion":
                row["status"] = "intentionally leaky"
            elif row["overlap_train_val"] or row["overlap_train_outer"]:
                row["status"] = "LEAK"
            else:
                row["status"] = "ok"
            n_copy = 0
            for f in run.glob("preds_*.csv"):
                if f.name not in ("preds_innerval.csv", "preds_outer.csv"):
                    n_copy += len(set(pd.read_csv(f, usecols=["image_id"]).image_id) & flagged)
            row["test_copies_predicted"] = n_copy
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- repair
def fix_master(df: pd.DataFrame, pairs: dict, n_folds: int | None = None, seed: int = 42,
               refold: bool = False) -> tuple[pd.DataFrame, dict]:
    """Repair what `audit_master` finds.

    1. flag test / external copies of HAM images (test_dup_of_ham=1; evaluate.py drops them);
    2. merge the groups of near-identical cross-group HAM pairs;
    3. put every merged group into one fold: the fold holding most of its images (minimal move,
       other folds - and the runs that used them - stay valid), or, with refold=True, regenerate
       all folds from scratch with the same seed (every existing fold run becomes stale).
    """
    df = df.copy()
    ham_mask = (df.dataset == "ham").to_numpy()
    if "test_dup_of_ham" not in df:
        df["test_dup_of_ham"] = 0
    ext_pos = sorted({a for a, _, _ in pairs["ext"]})
    newly_flagged = int((df.test_dup_of_ham.to_numpy()[ext_pos] != 1).sum()) if ext_pos else 0
    if ext_pos:
        df.iloc[ext_pos, df.columns.get_loc("test_dup_of_ham")] = 1

    uf = UnionFind(df.loc[ham_mask, "group_id"].unique())
    gid = df.group_id.to_numpy()
    n_merge = 0
    for a, b, _ in pairs["ham"]:
        if uf.find(gid[a]) != uf.find(gid[b]):
            uf.union(gid[a], gid[b])
            n_merge += 1
    df.loc[ham_mask, "group_id"] = df.loc[ham_mask, "group_id"].map(uf.find)

    old = df.loc[ham_mask, "fold_lesion"].copy()
    ham = df[ham_mask]
    if refold:
        nf = n_folds or int(old.max()) + 1
        df.loc[ham_mask, "fold_lesion"] = make_lesion_folds(ham, nf, seed)
    else:
        counts = ham.groupby(["group_id", "fold_lesion"]).size().unstack(fill_value=0)
        multi = counts[(counts > 0).sum(axis=1) > 1]
        if len(multi):
            target = multi.idxmax(axis=1)  # first maximum = lowest fold id on ties
            sel = ham.group_id.isin(target.index)
            df.loc[ham.index[sel.to_numpy()], "fold_lesion"] = ham.group_id[sel].map(target).to_numpy()
    new = df.loc[ham_mask, "fold_lesion"]
    moved = old.to_numpy() != new.to_numpy()
    touched = sorted(set(old[moved]) | set(new[moved]))
    info = {"merged_group_pairs": n_merge, "newly_flagged_test_images": newly_flagged,
            "images_moved": int(moved.sum()), "folds_touched": [int(t) for t in touched],
            "refold": bool(refold), "largest_group_share": float(ham.group_id.value_counts(normalize=True).iloc[0]),
            "fold_size_ratio": float(new.value_counts().max() / new.value_counts().mean()), "fold_sizes": {int(k): int(v) for k, v in new.value_counts().sort_index().items()}}
    return df, info


# --------------------------------------------------------------------------- report
def to_markdown(report: dict, runs: pd.DataFrame | None = None, fix_info: dict | None = None) -> str:
    lines = ["# Leakage audit", "", f"Result: **{'PASS' if report['ok'] else 'FAIL'}**", "",
             "| check | result | n | note |", "|---|---|---|---|"]
    for c in report["checks"]:
        res = "pass" if c["ok"] else ("FAIL" if c["severity"] == "error" else "warn")
        if c["severity"] == "skipped":
            res = "skipped"
        lines.append(f"| {c['name']} | {res} | {c['n_bad']} | {c['note']} |")
    for c in report["checks"]:
        if not c["ok"] and c["examples"]:
            lines += ["", f"`{c['name']}` examples: " + ", ".join(c["examples"])]
    if runs is not None and len(runs):
        cols = list(runs.columns)
        lines += ["", "## Finished runs", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in runs.itertuples(index=False):
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    if fix_info:
        lines += ["", "## Repair", "", "```json", json.dumps(fix_info, indent=2), "```"]
    return "\n".join(lines) + "\n"
