#!/usr/bin/env python
"""Parameter count and forward cost of every architecture in the study (no weights downloaded).

  python complexity.py --img_size 384 --out results/complexity.csv

GFLOPs = floating-point operations of one forward pass of a single image (a multiply-add counts
as 2 FLOPs, torch.utils.flop_counter convention); this covers matmuls / convolutions / attention
and ignores element-wise ops. evaluate.py adds the table to the report when the file exists.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from dermalign.models import build_model

ARCH = "convnext_tiny.fb_in22k_ft_in1k"
ARCH2 = "vit_small_patch16_224.augreg_in21k_ft_in1k"


def specs(arch, arch2):
    return [
        ("DenseNet201", dict(arch="densenet201.tv_in1k", head="gap")),
        ("EfficientNet-B3", dict(arch="efficientnet_b3.ra2_in1k", head="gap")),
        ("Xception+CBAM", dict(arch="legacy_xception.tf_in1k", head="cbam")),
        ("ResNet50", dict(arch="resnet50.a1_in1k", head="gap")),
        ("ViT-B/16", dict(arch="vit_base_patch16_224.augreg2_in21k_ft_in1k", head="gap")),
        ("ConvNeXt-T (GAP)", dict(arch=arch, head="gap")),
        ("DermAlign (ConvNeXt-T + attention head)", dict(arch=arch, head="attn")),
        ("DermAlign-X (CNN + ViT, cross-attention)", dict(arch=arch, arch2=arch2, head="xfuse")),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--arch", default=ARCH)
    ap.add_argument("--arch2", default=ARCH2)
    ap.add_argument("--out", default="results/complexity.csv")
    ap.add_argument("--latency", action="store_true", help="also time one CPU forward pass (noisy; indicative only)")
    args = ap.parse_args(argv)

    rows = []
    for name, cfg in specs(args.arch, args.arch2):
        model = build_model({**cfg, "pretrained": False, "img_size": args.img_size}).eval()
        x = torch.randn(1, 3, args.img_size, args.img_size)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        row = {"model": name, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
               "GFLOPs": fc.get_total_flops() / 1e9}
        if args.latency:
            with torch.no_grad():
                t0 = time.time()
                model(x)
            row["cpu_ms"] = (time.time() - t0) * 1e3
        rows.append(row)
        print(f"{name:45s} {row['params_M']:7.1f} M  {row['GFLOPs']:7.1f} GFLOPs", flush=True)
    df = pd.DataFrame(rows).round(2)
    df.insert(1, "input", f"{args.img_size}x{args.img_size}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
