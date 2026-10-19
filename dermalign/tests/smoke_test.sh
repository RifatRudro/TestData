#!/usr/bin/env bash
# CPU smoke test on synthetic data: every stage of run_all.sh with tiny settings.
# Usage: bash tests/smoke_test.sh [workdir]
set -euo pipefail
W=${1:-/tmp/dermalign_smoke}
cd "$(dirname "$0")/.."
rm -rf "$W" && mkdir -p "$W"
python -m pytest -q tests/test_units.py
python tests/make_synthetic.py --out "$W/raw"
python prepare_data.py --ham_dir "$W/raw/HAM10000" --ham_masks_dir "$W/raw/HAM10000_segmentations" \
  --isic18_test_dir "$W/raw/ISIC2018_Task3_Test_Input" --isic18_test_gt "$W/raw/ISIC2018_Task3_Test_GroundTruth.csv" \
  --external bcn20000 "$W/raw/BCN/metadata.csv" "$W/raw/BCN/images" \
  --dhash --check_test_overlap --cache_dir "$W/cache" --cache_short 128 --workers 2 --out "$W/master.csv"
CSV="$W/master.csv" RUNS="$W/runs" OUT="$W/results" FOLDS="0" IMG_FAST=64 IMG_FINAL=64 EP_FAST=1 EP_FINAL=1 \
  WORKERS=0 BS=16 LEAK_TARGET=60 EXTRA="--no_pretrained --max_steps_per_epoch 2" \
  XAI_ARGS="--n 20 --n_faith 3 --n_sanity 3 --ig_steps 4 --n_fig 4" \
  ARCH2=vit_tiny_patch16_224.augreg_in21k_ft_in1k bash run_all.sh
test -f "$W/results/report.md" && echo "SMOKE TEST PASSED (numbers are meaningless)"
