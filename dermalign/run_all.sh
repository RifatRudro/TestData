#!/usr/bin/env bash
# Full experiment plan for the DermAlign-Net paper.
#
#   STAGES="audit infl abl base final hyb hybabl xai eval" bash run_all.sh
#
# Edit the paths/variables below (or export them before calling). Each stage can be
# run on its own; finished runs are skipped, so the script can be re-started after a
# Kaggle/Colab session ends (copy the runs/ folder out and back in).
set -euo pipefail

CSV=${CSV:-data/master.csv}
RUNS=${RUNS:-runs}
OUT=${OUT:-results}
WORKERS=${WORKERS:-4}
BS=${BS:-32}
ARCH=${ARCH:-convnext_tiny.fb_in22k_ft_in1k}
IMG_FAST=${IMG_FAST:-224}     # inflation + ablation stages
IMG_FINAL=${IMG_FINAL:-384}   # final model and baselines compared on the test sets
EP_FAST=${EP_FAST:-25}
EP_FINAL=${EP_FINAL:-30}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS_FINAL=${SEEDS_FINAL:-"42"}      # journal: "42 7 2024"
LAMBDA=${LAMBDA:-0.5}
LEAK_TARGET=${LEAK_TARGET:-1500}      # per-class count for the oversample-then-split protocol
BASE_LOSS=${BASE_LOSS:-ce}            # loss used by the re-benchmarked literature baselines
ARCH2=${ARCH2:-vit_small_patch16_224.augreg_in21k_ft_in1k}  # transformer stream of DermAlign-X
AUTOFIX=${AUTOFIX:-1}                 # audit stage repairs leakage it finds (set 0 to only report and abort)
STAGES=${STAGES:-"audit infl abl base final hyb hybabl xai eval"}
EXTRA=${EXTRA:-}                      # extra args for every train.py call (e.g. "--no_amp")
XAI_ARGS=${XAI_ARGS:-"--n 300 --n_faith 100 --n_sanity 50"}

COMMON="--csv $CSV --runs_dir $RUNS --workers $WORKERS --batch_size $BS $EXTRA"

run() {  # run <exp> <fold> <seed> <args...>  (skips finished runs)
  local exp=$1 fold=$2 seed=$3; shift 3
  if [ -f "$RUNS/$exp/fold${fold}_seed${seed}/model.pt" ]; then
    echo "skip $exp fold$fold seed$seed"; return
  fi
  python train.py $COMMON --exp_name "$exp" --fold "$fold" --seed "$seed" "$@"
}

has() { [[ " $STAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# Stage 0 - leakage audit (leakage_audit.py). Fails the pipeline if a lesion group spans two folds,
# a test image is a copy of a HAM10000 image, or a finished run trained on a lesion of its
# validation / outer set. With AUTOFIX=1 it repairs the master CSV (backup: master.csv.bak) and moves
# the affected runs to $RUNS/_stale/ so that the stages below retrain them instead of skipping them.
if has audit; then
  FIX=""; [ "$AUTOFIX" = "1" ] && FIX="--fix --inplace --quarantine"
  python leakage_audit.py --csv "$CSV" --runs_dir "$RUNS" --out_dir "$OUT/leakage" --workers "$WORKERS" $FIX
fi

# ---------------------------------------------------------------------------
# Stage 1 - protocol inflation (Table A, top). Same plain model, three protocols.
# Shows how much of the 96-99% reported on HAM10000 comes from leakage.
if has infl; then
  for f in $FOLDS; do
    for sp in oversample_leak image lesion; do
      run "infl_${sp}" $f 42 --split $sp --leak_target $LEAK_TARGET --arch $ARCH --head gap --loss ce \
          --img_size $IMG_FAST --epochs $EP_FAST --tta
    done
  done
fi

# ---------------------------------------------------------------------------
# Stage 2 - ablation under the leakage-free protocol (Table A, bottom).
if has abl; then
  F="--split lesion --arch $ARCH --img_size $IMG_FAST --epochs $EP_FAST --tta"
  for f in $FOLDS; do
    run abl0_gap_ce          $f 42 $F --head gap  --loss ce
    run abl1_gap_la          $f 42 $F --head gap  --loss la
    run abl2_attn_la         $f 42 $F --head attn --loss la --lambda_align 0
    run abl3_dermalign       $f 42 $F --head attn --loss la --lambda_align $LAMBDA
    run abl4_dermalign_meta  $f 42 $F --head attn --loss la --lambda_align $LAMBDA --use_meta
    # alternatives for the imbalance component (optional, uncomment for the journal)
    # run abl_x_focal        $f 42 $F --head attn --loss focal --lambda_align $LAMBDA
    # run abl_x_sqrtsampler  $f 42 $F --head attn --loss la --sampler_q 0.5 --lambda_align $LAMBDA
    # lambda sensitivity
    # for L in 0.1 0.3 1.0; do run abl_lambda_$L $f 42 $F --head attn --loss la --lambda_align $L; done
  done
fi

# ---------------------------------------------------------------------------
# Stage 3 - literature baselines re-trained under the same protocol (Table B).
# Trained on all of HAM10000 (fold -1), evaluated once on the sealed test sets.
if has base; then
  B="--split lesion --img_size $IMG_FINAL --epochs $EP_FINAL --loss $BASE_LOSS --tta"
  for s in $SEEDS_FINAL; do
    run base_densenet201      -1 $s $B --arch densenet201.tv_in1k --head gap
    run base_efficientnet_b3  -1 $s $B --arch efficientnet_b3.ra2_in1k --head gap
    run base_xception_cbam    -1 $s $B --arch legacy_xception.tf_in1k --head cbam
    run base_resnet50         -1 $s $B --arch resnet50.a1_in1k --head gap
    run base_vit_b16          -1 $s $B --arch vit_base_patch16_224.augreg2_in21k_ft_in1k --head gap --lr 5e-5
    run base_convnext_t       -1 $s $B --arch $ARCH --head gap
  done
fi

# ---------------------------------------------------------------------------
# Stage 4 - proposed model at full resolution: 5-fold CV + final model(s).
# Pick the variant (with/without metadata) from the ablation BEFORE running this.
if has final; then
  P="--split lesion --arch $ARCH --img_size $IMG_FINAL --epochs $EP_FINAL --head attn --loss la --lambda_align $LAMBDA --tta"
  for f in $FOLDS; do run dermalign $f 42 $P; done
  for s in $SEEDS_FINAL; do run dermalign -1 $s $P; done
fi

# ---------------------------------------------------------------------------
# Stage 4b - DermAlign-X: dual-stream CNN + ViT with cross-attention fusion (dermalign/hybrid.py).
# Same protocol, loss and alignment term as the proposed model; only the architecture differs.
if has hyb; then
  H="--split lesion --arch $ARCH --arch2 $ARCH2 --head xfuse --img_size $IMG_FINAL --epochs $EP_FINAL --loss la --lambda_align $LAMBDA --tta"
  for f in $FOLDS; do run dermalign_x $f 42 $H; done
  for s in $SEEDS_FINAL; do run dermalign_x -1 $s $H; done
fi

# ---------------------------------------------------------------------------
# Stage 4c - DermAlign-X component ablations (CV, lesion protocol, IMG_FAST so they are comparable
# with abl3_dermalign = CNN-only). hybabl0 is the full model at this resolution.
if has hybabl; then
  X="--split lesion --arch $ARCH --arch2 $ARCH2 --head xfuse --img_size $IMG_FAST --epochs $EP_FAST --loss la --lambda_align $LAMBDA --tta"
  for f in $FOLDS; do
    run hybabl0_full        $f 42 $X
    run hybabl1_concat      $f 42 $X --fuse_mode concat          # no cross-attention
    run hybabl2_nogate      $f 42 $X --fuse_mode xattn_nogate    # fixed 0.5 instead of the learned gate
    run hybabl3_depth2      $f 42 $X --fuse_depth 2
    run hybabl4_no_align    $f 42 $X --lambda_align 0            # last flag wins: no alignment loss
    run hybabl5_vit_only    $f 42 --split lesion --arch $ARCH2 --head attn --img_size $IMG_FAST \
                            --epochs $EP_FAST --loss la --lambda_align $LAMBDA --tta   # transformer stream alone
  done
fi

# ---------------------------------------------------------------------------
# Stage 5 - explanation quality (Table E): outer fold 0, same images for every model.
if has xai; then
  for e in abl1_gap_la abl2_attn_la abl3_dermalign dermalign dermalign_x; do
    d="$RUNS/$e/fold0_seed42"
    if [ -f "$d/model.pt" ] && [ ! -f "$d/xai_outer/summary.csv" ]; then
      python xai_eval.py --run_dir "$d" --csv $CSV --dataset outer $XAI_ARGS
    fi
  done
  # external masks (PH2), if prepared:  python xai_eval.py --run_dir $RUNS/dermalign/fold-1_seed42 --csv $CSV --dataset ph2
fi

# ---------------------------------------------------------------------------
if has eval; then
  python complexity.py --img_size $IMG_FINAL --arch $ARCH --arch2 $ARCH2 --out $OUT/complexity.csv
  python evaluate.py --runs_dir $RUNS --csv $CSV --ref dermalign --compare_prefix base_ --out $OUT
fi
