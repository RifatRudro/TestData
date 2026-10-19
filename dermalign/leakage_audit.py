#!/usr/bin/env python
"""Audit (and optionally repair) data leakage in the master CSV and in finished runs.

  python leakage_audit.py --csv data/master.csv --runs_dir runs            # audit, exit 1 on leakage
  python leakage_audit.py --csv data/master.csv --fix                      # write master.fixed.csv
  python leakage_audit.py --csv data/master.csv --fix --inplace            # replace master.csv (backup kept)

Checks: every lesion group sits in one fold; no id overlap with the test sets; no test image is a
copy of a HAM10000 image; no near-identical same-diagnosis HAM pair straddles two folds; finished
`lesion` runs never trained on a lesion of their validation / outer set (judged with the *current*
groups). --fix merges duplicate groups, moves each merged group into one fold (minimal change;
--refold regenerates every fold), flags test copies, and lists the runs that must be retrained.
Run it once after prepare_data.py; it is also the first stage of run_all.sh.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from dermalign.leakage import (audit_master, audit_runs, find_leak_pairs, fix_master, hash_images,
                               to_markdown)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--runs_dir", default=None, help="also audit the splits finished runs used")
    ap.add_argument("--out_dir", default="results/leakage")
    ap.add_argument("--max_dist", type=int, default=8, help="dHash Hamming distance (bits of 256) for 'copy'")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no_images", action="store_true", help="skip pixel-level checks (metadata only)")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--inplace", action="store_true", help="with --fix: overwrite --csv (backup .bak kept)")
    ap.add_argument("--refold", action="store_true", help="with --fix: regenerate all folds, not just the conflicting groups")
    ap.add_argument("--quarantine", action="store_true",
                    help="move runs that trained on leaked lesions (or on a repaired fold) to <runs_dir>/_stale/ "
                         "so run_all.sh retrains them instead of skipping them")
    ap.add_argument("--seed", type=int, default=42, help="seed of prepare_data.py (only used by --refold)")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    pairs = None
    if not args.no_images:
        missing = int((~df.path.map(os.path.exists)).sum())
        if missing:
            print(f"[audit] {missing}/{len(df)} image files not found - pixel-level checks skipped "
                  f"(run where the images are, or pass --no_images)")
        else:
            print(f"[audit] hashing {len(df)} images ...")
            pairs = find_leak_pairs(df, hash_images(df.path, args.workers), args.max_dist)
    report = audit_master(df, pairs)
    runs = audit_runs(df, args.runs_dir) if args.runs_dir and Path(args.runs_dir).exists() else None

    fix_info, fixed = None, None
    run_leaks = runs is not None and len(runs) and (runs.status == "LEAK").any()
    if args.fix and (not report["ok"] or run_leaks):
        if pairs is None:
            sys.exit("[audit] --fix needs the pixel-level pairs (images must be readable)")
        fixed, fix_info = fix_master(df, pairs, seed=args.seed, refold=args.refold)
        report_after = audit_master(fixed, pairs)
        runs_after = audit_runs(fixed, args.runs_dir) if runs is not None else None
        stale = []
        if runs_after is not None and len(runs_after):
            touched = set(fix_info["folds_touched"])
            stale = runs_after[(runs_after.status == "LEAK") |
                               ((runs_after.split == "lesion") & runs_after.fold.isin(touched))]
            stale = sorted(f"{r.exp}/{r.run}" for r in stale.itertuples())
        fix_info.update({"audit_after_fix": "PASS" if report_after["ok"] else "FAIL", "runs_to_retrain": stale})
        out = Path(args.csv) if args.inplace else Path(args.csv).with_name(Path(args.csv).stem + ".fixed.csv")
        if args.inplace:
            shutil.copy(args.csv, str(args.csv) + ".bak")
        fixed.to_csv(out, index=False)
        print(f"[fix] wrote {out}: {json.dumps(fix_info)}")
        report, runs = report_after, runs_after
        report["fixed_csv"] = str(out)

    if args.quarantine and runs is not None and len(runs):
        todo = set(fix_info["runs_to_retrain"]) if fix_info else set()
        todo |= {f"{r.exp}/{r.run}" for r in runs.itertuples() if r.status == "LEAK"}
        for name in sorted(todo):
            dst = Path(args.runs_dir) / "_stale" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(Path(args.runs_dir) / name), str(dst))
        if todo:
            print(f"[quarantine] moved {len(todo)} stale runs to {Path(args.runs_dir) / '_stale'} - re-run run_all.sh to retrain them")
            runs = runs[~runs.apply(lambda r: f"{r.exp}/{r.run}" in todo, axis=1)]
            if fix_info is not None:
                fix_info["quarantined_runs"] = sorted(todo)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leakage_report.json").write_text(json.dumps(
        {**report, "runs": None if runs is None else runs.to_dict("records"), "fix": fix_info}, indent=2, default=str))
    md = to_markdown(report, runs, fix_info)
    (out_dir / "leakage_report.md").write_text(md)
    print(md)

    leaky = runs is not None and len(runs) and (runs.status == "LEAK").any()
    ok = report["ok"] and not leaky
    print(f"[audit] {'PASS' if ok else 'FAIL'}  (report in {out_dir})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
