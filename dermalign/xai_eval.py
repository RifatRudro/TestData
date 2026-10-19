#!/usr/bin/env python
"""Quantitative explanation evaluation for one trained run.

python xai_eval.py --run_dir runs/dermalign/fold0_seed42 --csv data/master.csv --dataset outer
python xai_eval.py --run_dir runs/dermalign/fold-1_seed42 --csv data/master.csv --dataset ph2

--dataset outer  -> that run's outer-fold HAM10000 images (masks needed: HAM masks)
--dataset NAME   -> any dataset in the master CSV that has masks (e.g. ph2)

Writes <run_dir>/xai_<dataset>/: per_image.csv, summary.csv, faithfulness.csv,
sanity.csv, agreement.csv, examples.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dermalign.constants import CLASSES, IMAGENET_MEAN, IMAGENET_STD
from dermalign.data import DermDataset
from dermalign.models import build_model
from dermalign.utils import seed_everything
from dermalign.xai import (compute_maps, deletion_insertion, localisation_metrics, map_correlation,
                           randomise_top)

MODEL_METHODS = {"attention", "gradcam", "gradcampp", "ig"}


def load_run(run_dir, device):
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


def select_rows(df, run_dir, dataset, n, seed):
    if dataset == "outer":
        ids = set(pd.read_csv(Path(run_dir) / "preds_outer.csv").image_id)
        rows = df[(df.dataset == "ham") & df.image_id.isin(ids)]
    else:
        rows = df[df.dataset == dataset]
    rows = rows[rows.mask_path.notna()].drop_duplicates("image_id")
    if len(rows) == 0:
        raise SystemExit(f"no images with masks for dataset={dataset}")
    if n and len(rows) > n:  # stratified subsample, at least a few per class
        per = pd.concat([g.sample(min(len(g), max(5, int(round(n * len(g) / len(rows))))), random_state=seed)
                         for _, g in rows.groupby("label")])
        rows = per.sample(min(n, len(per)), random_state=seed) if len(per) > n else per
    return rows.reset_index(drop=True)


def denorm(x):
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", default="outer")
    ap.add_argument("--methods", default="attention,gradcampp,gradcam,ig,centre,random")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_faith", type=int, default=100)
    ap.add_argument("--n_sanity", type=int, default=50)
    ap.add_argument("--n_fig", type=int, default=14)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--target", default="pred", choices=["pred", "true"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(args.run_dir, device)
    methods = [m for m in args.methods.split(",") if m]
    if cfg["head"] != "attn" and "attention" in methods:
        methods.remove("attention")
    out_dir = Path(args.out or Path(args.run_dir) / f"xai_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    rows = select_rows(df, args.run_dir, args.dataset, args.n, args.seed)
    ds = DermDataset(rows, cfg["img_size"], train=False, use_masks=True,
                     color_constancy=not cfg.get("no_color_constancy", False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    use_meta = cfg.get("use_meta", False)
    print(f"[xai] {len(rows)} images, methods={methods}, device={device}")

    per_image, small_maps, keep = [], {m: [] for m in methods}, []
    for b in loader:
        x = b["image"].to(device)
        meta = b["meta"].to(device) if use_meta else None
        with torch.no_grad():
            pred = model(x, meta)["logits"].argmax(1)
        target = pred if args.target == "pred" else torch.as_tensor(b["label"], device=device)
        maps = compute_maps(model, x, target, methods, meta, args.ig_steps)
        for i in range(x.shape[0]):
            idx = int(b["idx"][i])
            mask = b["mask"][i, 0].numpy()
            keep.append((idx, int(target[i]), int(b["label"][i])))
            for m, mp in maps.items():
                loc = localisation_metrics(mp[i].numpy(), mask)
                if loc is None:
                    continue
                per_image.append({"image_id": rows.image_id.iat[idx], "label": CLASSES[int(b["label"][i])],
                                  "pred": CLASSES[int(pred[i])], "method": m, **loc})
                small_maps[m].append(F.adaptive_avg_pool2d(mp[i][None, None], 32)[0, 0].numpy())
    pi = pd.DataFrame(per_image)
    pi.to_csv(out_dir / "per_image.csv", index=False)
    summ = pi.groupby("method")[["mass_in_mask", "rank_acc", "pointing", "border_outside", "mask_area"]] \
        .agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(out_dir / "summary.csv")
    print(summ.round(3).to_string())

    # agreement between methods (Spearman on 32x32 maps)
    ag = []
    n_valid = max((len(v) for v in small_maps.values()), default=0)
    ms = [m for m in methods if len(small_maps[m]) == n_valid and n_valid > 0]
    for i, a in enumerate(ms):
        for b_ in ms[i + 1:]:
            r = [map_correlation(x1, x2) for x1, x2 in zip(small_maps[a], small_maps[b_])]
            ag.append({"method_a": a, "method_b": b_, "spearman_mean": float(np.mean(r)),
                       "spearman_std": float(np.std(r))})
    pd.DataFrame(ag).to_csv(out_dir / "agreement.csv", index=False)

    # faithfulness: deletion / insertion
    fa = []
    n_f = min(args.n_faith, len(rows))
    for j in range(n_f):
        s = ds[j]
        x = s["image"][None].to(device)
        meta = s["meta"][None].to(device) if use_meta else None
        with torch.no_grad():
            t = int(model(x, meta)["logits"].argmax(1)) if args.target == "pred" else int(s["label"])
        maps = compute_maps(model, x, torch.tensor([t], device=device), methods, meta, args.ig_steps)
        for m, mp in maps.items():
            d, ins = deletion_insertion(model, x, t, mp[0], meta)
            fa.append({"image_id": rows.image_id.iat[j], "method": m, "deletion_auc": d, "insertion_auc": ins})
    fa = pd.DataFrame(fa)
    if len(fa):
        fsum = fa.groupby("method")[["deletion_auc", "insertion_auc"]].agg(["mean", "std"])
        fsum.columns = [f"{a}_{b}" for a, b in fsum.columns]
        fsum.to_csv(out_dir / "faithfulness.csv")
        print(fsum.round(3).to_string())

    # sanity: model-parameter randomisation
    rmodel = randomise_top(model.cpu(), frac=0.25).to(device).eval()
    model.to(device)
    sm = [m for m in methods if m in MODEL_METHODS]
    san = []
    n_s = min(args.n_sanity, len(rows))
    for j in range(n_s):
        s = ds[j]
        x = s["image"][None].to(device)
        meta = s["meta"][None].to(device) if use_meta else None
        with torch.no_grad():
            t = int(model(x, meta)["logits"].argmax(1))
        tt = torch.tensor([t], device=device)
        m1 = compute_maps(model, x, tt, sm, meta, args.ig_steps)
        m2 = compute_maps(rmodel, x, tt, sm, meta, args.ig_steps)
        for m in m1:
            if m in m2:
                san.append({"image_id": rows.image_id.iat[j], "method": m,
                            "spearman_trained_vs_random": map_correlation(m1[m][0].numpy(), m2[m][0].numpy())})
    san = pd.DataFrame(san)
    if len(san):
        ssum = san.groupby("method")["spearman_trained_vs_random"].agg(["mean", "std"])
        ssum.to_csv(out_dir / "sanity.csv")
        print("sanity (lower |rho| = explanation depends on learned weights):")
        print(ssum.round(3).to_string())

    # example figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = rows.groupby("label").head(2).index[: args.n_fig]
        show = [m for m in methods if m != "random"]
        fig, axes = plt.subplots(len(order), 1 + len(show), figsize=(2.2 * (1 + len(show)), 2.2 * len(order)),
                                 squeeze=False)
        for r, j in enumerate(order):
            s = ds[int(j)]
            x = s["image"][None].to(device)
            meta = s["meta"][None].to(device) if use_meta else None
            with torch.no_grad():
                t = int(model(x, meta)["logits"].argmax(1))
            maps = compute_maps(model, x, torch.tensor([t], device=device), show, meta, args.ig_steps)
            img = denorm(s["image"])
            axes[r, 0].imshow(img)
            axes[r, 0].contour(s["mask"][0].numpy(), levels=[0.5], colors="lime", linewidths=0.8)
            axes[r, 0].set_title(f"{CLASSES[int(s['label'])]} -> {CLASSES[t]}", fontsize=8)
            for c, m in enumerate(show, start=1):
                axes[r, c].imshow(img)
                if m in maps:
                    axes[r, c].imshow(maps[m][0].numpy(), cmap="jet", alpha=0.45)
                axes[r, c].contour(s["mask"][0].numpy(), levels=[0.5], colors="white", linewidths=0.6)
                if r == 0:
                    axes[r, c].set_title(m, fontsize=8)
        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "examples.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print("figure failed:", e)
    print(f"[xai] wrote {out_dir}")


if __name__ == "__main__":
    main()
