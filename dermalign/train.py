#!/usr/bin/env python
"""Train one model for one (protocol, fold, seed) and export predictions.

Protocols (--split):
  lesion          leakage-free: outer folds grouped by merged lesion id (main protocol)
  image           common practice: outer folds by image (siblings of a lesion leak)
  oversample_leak common mistake: oversample minority classes, then split by image
--fold -1 trains on all HAM10000 (minus the inner validation part) = final model.

For every run, ~10% of the training part (grouped the same way as the outer split) is
held out for checkpoint selection and temperature scaling, so outer-fold and test
numbers are never used for any decision.

Outputs in <runs_dir>/<exp_name>/fold{k}_seed{s}/:
  config.json, log.csv, model.pt, temperature.json,
  preds_outer.csv (if fold >= 0), preds_<dataset>.csv for every non-HAM dataset in the CSV.
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dermalign.constants import CLASSES, NUM_CLASSES
from dermalign.data import (DermDataset, class_counts, inner_split, make_sampler, outer_split,
                            worker_init_fn)
from dermalign.losses import alignment_loss, attention_entropy, build_criterion
from dermalign.metrics import balanced_accuracy, macro_f1
from dermalign.models import build_model
from dermalign.utils import environment_info, fit_temperature, save_json, seed_everything, softmax_np


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--split", default="lesion", choices=["lesion", "image", "oversample_leak"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leak_target", type=int, default=1500,
                    help="per-class target count for --split oversample_leak")
    # model
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--head", default="attn", choices=["gap", "attn", "cbam", "xfuse"],
                    help="xfuse = DermAlign-X dual-stream CNN+ViT cross-attention fusion (see dermalign/hybrid.py)")
    ap.add_argument("--arch2", default="vit_small_patch16_224.augreg_in21k_ft_in1k", help="transformer stream of --head xfuse")
    ap.add_argument("--fuse_dim", type=int, default=256)
    ap.add_argument("--fuse_heads", type=int, default=4)
    ap.add_argument("--fuse_depth", type=int, default=1)
    ap.add_argument("--fuse_mode", default="xattn", choices=["xattn", "concat", "xattn_nogate"],
                    help="DermAlign-X ablations: cross-attention + gate (default) / concat+1x1 conv / cross-attention, fixed 0.5 gate")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--drop_rate", type=float, default=0.2)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--att_hidden", type=int, default=256)
    ap.add_argument("--use_meta", action="store_true")
    # loss / imbalance
    ap.add_argument("--loss", default="la", choices=["la", "ce", "wce", "focal"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--sampler_q", type=float, default=0.0,
                    help="sampling prob ~ n_c^-q (0 natural, 0.5 sqrt, 1 balanced)")
    ap.add_argument("--lambda_align", type=float, default=0.0)
    ap.add_argument("--dilate_frac", type=float, default=0.05)
    ap.add_argument("--att_entropy", type=float, default=0.0,
                    help="weight of an entropy bonus that keeps attention from collapsing")
    # optimisation
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr_mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    ap.add_argument("--patience", type=int, default=0, help="early stopping patience (0 = off)")
    ap.add_argument("--hair_p", type=float, default=0.3)
    ap.add_argument("--no_color_constancy", action="store_true")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation (4 views)")
    ap.add_argument("--inner_frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--max_steps_per_epoch", type=int, default=0, help="debug: cap steps per epoch")
    ap.add_argument("--no_test_pred", action="store_true")
    return ap.parse_args(argv)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def make_loader(df, args, train, sampler=None, shuffle=False):
    ds = DermDataset(df, args.img_size, train=train, use_masks=train and args.lambda_align > 0,
                     color_constancy=not args.no_color_constancy, hair_p=args.hair_p if train else 0.0)
    return DataLoader(ds, batch_size=args.batch_size if train else args.batch_size * 2,
                      sampler=sampler, shuffle=shuffle and sampler is None, drop_last=train,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=args.workers > 0 and train, worker_init_fn=worker_init_fn)


@torch.no_grad()
def predict(model, loader, device, tta=False, use_meta=False, amp=True):
    model.eval()
    out = []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        meta = b["meta"].to(device) if use_meta else None
        views = [x] if not tta else [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
        acc = 0
        for v in views:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                acc = acc + model(v, meta)["logits"].float()
        out.append((acc / len(views)).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, NUM_CLASSES), np.float32)


def export_preds(df, logits, T, path, role):
    probs = softmax_np(logits, T)
    probs_unc = softmax_np(logits, 1.0)
    rec = pd.DataFrame({"image_id": df.image_id.values, "group_id": df.group_id.values,
                        "label": df.label.values, "dataset": df.dataset.values, "role": role})
    if "copy_id" in df.columns:
        rec["copy_id"] = df.copy_id.values
    for i, c in enumerate(CLASSES):
        rec[f"logit_{c}"] = logits[:, i]
        rec[f"prob_{c}"] = probs[:, i]
        rec[f"probu_{c}"] = probs_unc[:, i]
    rec.to_csv(path, index=False)
    return rec


def main(argv=None):
    args = get_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    run_dir = Path(args.runs_dir) / args.exp_name / f"fold{args.fold}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    ham = df[df.dataset == "ham"].copy()
    others = df[df.dataset != "ham"].copy()

    trainval, outer = outer_split(ham, args.split, args.fold, args.seed, args.leak_target)
    inner_protocol = "lesion" if args.split == "lesion" else "image"
    train_df, val_df = inner_split(trainval, inner_protocol, args.seed, args.inner_frac)
    if args.split == "lesion":  # guarantee no lesion overlap anywhere
        assert not set(train_df.group_id) & set(val_df.group_id)
        if outer is not None:
            assert not set(trainval.group_id) & set(outer.group_id)
            assert not set(val_df.group_id) & set(outer.group_id)
            assert not set(train_df.image_id) & set(outer.image_id)

    counts = class_counts(train_df.label)
    sampler, prior = make_sampler(train_df.label.values, args.sampler_q)
    cfg = vars(args).copy()
    cfg.update({"pretrained": not args.no_pretrained, "n_train": len(train_df), "n_val": len(val_df),
                "n_outer": 0 if outer is None else len(outer), "train_counts": counts.tolist(),
                "train_prior": prior.tolist(), "device": str(device), "run_dir": str(run_dir)})
    save_json(cfg, run_dir / "config.json")
    save_json(environment_info(), run_dir / "env.json")
    print(f"[{args.exp_name}] split={args.split} fold={args.fold} seed={args.seed} "
          f"train={len(train_df)} val={len(val_df)} outer={cfg['n_outer']} device={device}")

    train_loader = make_loader(train_df, args, train=True, sampler=sampler, shuffle=True)
    val_loader = make_loader(val_df, args, train=False)

    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = build_criterion(args.loss, counts, prior, args.tau, args.label_smoothing).to(device)
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.head_lr_mult, args.wd))
    steps_per_epoch = len(train_loader) if not args.max_steps_per_epoch else min(len(train_loader), args.max_steps_per_epoch)
    total = max(1, steps_per_epoch * args.epochs)
    warm = int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else None
    ema = EMA(model, args.ema) if args.ema > 0 else None

    best, best_state, bad, log = -1.0, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        t0, n, s_loss, s_cls, s_al, s_ent = time.time(), 0, 0.0, 0.0, 0.0, 0.0
        for step, b in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x = b["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = torch.as_tensor(b["label"], device=device)
            meta = b["meta"].to(device) if args.use_meta else None
            if meta is not None and model.training:  # metadata dropout: robustness to missing meta
                drop = (torch.rand(meta.shape[0], 1, device=device) < 0.1).float()
                unknown = torch.zeros_like(meta)
                unknown[:, 1] = 1.0  # age missing
                unknown[:, 2 + 2] = 1.0  # sex unknown
                unknown[:, -1] = 1.0  # site unknown
                meta = meta * (1 - drop) + unknown * drop
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x, meta)
            loss_cls = criterion(out["logits"].float(), y)
            loss = loss_cls
            l_al = torch.zeros((), device=device)
            if args.lambda_align > 0 and out["att"] is not None:
                l_al = alignment_loss(out["att"], b["mask"].to(device), torch.as_tensor(
                    b["mask_valid"], device=device, dtype=torch.float32), args.dilate_frac)
                loss = loss + args.lambda_align * l_al
            ent = torch.zeros((), device=device)
            if args.att_entropy > 0 and out["att"] is not None:
                ent = attention_entropy(out["att"])
                loss = loss - args.att_entropy * ent
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)
            bs = x.shape[0]
            n += bs
            s_loss += loss.item() * bs
            s_cls += loss_cls.item() * bs
            s_al += l_al.item() * bs
            s_ent += ent.item() * bs
        eval_model = ema.model if ema is not None else model
        vlog = predict(eval_model, val_loader, device, tta=False, use_meta=args.use_meta, amp=amp)
        vpred = vlog.argmax(1)
        vy = val_df.label.values
        v_ba, v_f1 = balanced_accuracy(vy, vpred), macro_f1(vy, vpred)
        rec = {"epoch": epoch + 1, "loss": s_loss / max(n, 1), "cls": s_cls / max(n, 1),
               "align": s_al / max(n, 1), "att_entropy": s_ent / max(n, 1), "val_bal_acc": v_ba,
               "val_macro_f1": v_f1, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        log.append(rec)
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        print(" ".join(f"{k}={v:.2e}" if k == "lr" else f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in rec.items()))
        score = v_ba + 1e-3 * v_f1
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": cfg}, run_dir / "model.pt")

    # temperature scaling on the inner validation set (same TTA as at test time)
    vlog = predict(model, val_loader, device, tta=args.tta, use_meta=args.use_meta, amp=amp)
    T = fit_temperature(vlog, val_df.label.values)
    save_json({"T": T, "best_val_score": best}, run_dir / "temperature.json")
    export_preds(val_df, vlog, T, run_dir / "preds_innerval.csv", "innerval")
    print(f"temperature T={T:.3f}")

    if outer is not None:
        ol = predict(model, make_loader(outer, args, train=False), device, args.tta, args.use_meta, amp)
        rec = export_preds(outer, ol, T, run_dir / "preds_outer.csv", "outer")
        p = rec[[f"prob_{c}" for c in CLASSES]].values
        print(f"outer fold: BA={balanced_accuracy(rec.label.values, p.argmax(1)):.4f} "
              f"macroF1={macro_f1(rec.label.values, p.argmax(1)):.4f}")
    if not args.no_test_pred:
        for name, g in others.groupby("dataset"):
            tl = predict(model, make_loader(g, args, train=False), device, args.tta, args.use_meta, amp)
            rec = export_preds(g, tl, T, run_dir / f"preds_{name}.csv", "test")
            p = rec[[f"prob_{c}" for c in CLASSES]].values
            print(f"{name}: BA={balanced_accuracy(rec.label.values, p.argmax(1)):.4f} "
                  f"macroF1={macro_f1(rec.label.values, p.argmax(1)):.4f}")
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
