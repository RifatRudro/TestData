# DermAlign-Net

**Lesion-aligned attention for leakage-free, imbalance-robust and quantitatively explainable skin-lesion classification (HAM10000 → ISIC 2018 test → BCN20000).**

This repository implements the full experimental pipeline for the paper:

| Component | What it does | Where |
|---|---|---|
| Leakage-free protocol | 5-fold `StratifiedGroupKFold` on HAM10000 grouped by `lesion_id`, merged with the 20 duplicate pairs confirmed by Abhishek et al. (Sci Data 2025); inner 10 % grouped validation for model selection and temperature; ISIC 2018 Task 3 test used once | `prepare_data.py`, `dermalign/data.py` |
| Inflation experiment | Same model under three protocols: oversample-then-split (common mistake), image-level split, lesion-level split | `train.py --split {oversample_leak,image,lesion}` |
| DermAlign head | Spatial attention pooling; the attention map is the built-in explanation | `dermalign/models.py` |
| DermAlign-X (hybrid) | Dual-stream CNN (ConvNeXt) + ViT with bidirectional cross-attention, per-cell gated fusion, then the DermAlign attention head. `--head xfuse`, stage `hyb` | `dermalign/hybrid.py` |
| Leakage audit + repair | Independent check of the split (groups, duplicates, test-vs-HAM copies, finished runs) with automatic repair and quarantine of stale runs | `leakage_audit.py`, `dermalign/leakage.py` |
| Lesion-alignment loss | Penalises attention mass outside the dilated lesion mask (masks used **only during training**; inference is image-only) | `dermalign/losses.py` |
| Long-tail handling | Logit-adjusted cross-entropy (Menon et al., ICLR 2021); optional n^-q sampling with the matching prior; focal / weighted CE for ablation | `dermalign/losses.py`, `dermalign/data.py` |
| Reliability | Temperature scaling, ECE / Brier / NLL, reliability diagrams, abstain-and-refer (risk–coverage, AURC) | `dermalign/metrics.py`, `evaluate.py` |
| Baselines | DenseNet201, EfficientNet-B3, Xception+CBAM, ResNet50, ViT-B/16, ConvNeXt-T, re-trained under the same protocol | `run_all.sh` stage `base` |
| XAI evaluation | Attention, Grad-CAM, Grad-CAM++, Integrated Gradients **vs centre-prior and random controls**: relevance mass / rank accuracy, pointing game, border-shortcut share, deletion / insertion AUC, model-randomisation sanity check, inter-method agreement | `xai_eval.py`, `dermalign/xai.py` |
| Statistics | Mean ± SD over folds/seeds, bootstrap 95 % CIs (lesion-clustered when possible), paired bootstrap ΔBA, McNemar | `evaluate.py` |
| Paper tables | `results/report.md` (Markdown) and `results/tables.tex` (LaTeX) incl. literature comparison | `evaluate.py` |

> **Important.** The 96–99 % accuracies in the HAM10000 literature were obtained without a lesion-grouped split. Under this protocol, expect much lower numbers (the closest lesion-disjoint work, PLCRD, arXiv 2607.18773, reports BA 0.764 / lesion-level macro-F1 0.773 on HAM10000 and macro-F1 0.732 on ISIC 2018). Beat the *re-trained baselines and PLCRD under the same protocol*; use the inflation table to explain why the 98 % numbers are not comparable.

---

## 1. Install

```bash
pip install -r requirements.txt     # torch>=2.1, torchvision>=0.16, timm>=1.0
```

## 2. Data

### One command (recommended)

```bash
python download_data.py                    # ~3-4 GB into ./data/raw, then builds ./data/master.csv
python download_data.py --with_test_meta   # + age/sex/site of the ISIC 2018 test images (ISIC API)
pip install isic-cli && python download_data.py --with_bcn   # + BCN20000 external test set
```

What it fetches, all into this project's `data/` folder:

* ISIC 2018 Task 3 training (= the 10,015 HAM10000 images), lesion groupings (`lesion_id`), validation and test images with ground truth - from the official ISIC challenge bucket.
* HAM10000 metadata (age, sex, localization) and lesion masks - from the HAM10000 Harvard Dataverse API; if that fails, from the Kaggle mirrors (`kmader/skin-cancer-mnist-ham10000`, `tschandl/ham10000-lesion-segmentations`; needs `pip install kaggle` and `~/.kaggle/kaggle.json`). If neither works, metadata is rebuilt from the ISIC lesion groupings (no age/sex/site) and the script tells you where to put the masks.
* Then it runs `prepare_data.py` (folds, duplicate merging, resized cache).

Downloads resume after an interruption and finished steps are skipped, so just re-run it. PH2 needs a registration form, so it stays manual (`prepare_data.py --ph2_root`).

### Manual alternative

| Dataset | Use | Get it from |
|---|---|---|
| HAM10000 images + `HAM10000_metadata.csv` (has `lesion_id`) | training / CV | Harvard Dataverse doi:10.7910/DVN/DBW86T, or the Kaggle mirror `kmader/skin-cancer-mnist-ham10000`. The ISIC challenge copy of the same images is fine too: `ISIC2018_Task3_Training_LesionGroupings.csv` carries the lesion_id (this is what `download_data.py` uses). |
| HAM10000 lesion masks (`ISIC_xxxxxxx_segmentation.png`) | alignment loss + XAI scoring | HAM10000 Dataverse page (check the file list). If missing, generate masks with a segmentation model (e.g. MedSAM) and spot-check. Images without a mask simply get no alignment term. |
| ISIC 2018 Task 3 **test** (1,512 images) + ground truth | sealed internal test | https://challenge.isic-archive.com/data (Task 3 test input + test ground truth). If you have no GT CSV, `prepare_data.py` falls back on the bundled official labels from DermaMNIST-E. `ISIC_0035068` is dropped. |
| BCN20000 (Hospital Clínic Barcelona) | external, cross-site test | ISIC Archive (collection BCN20000) or the Sci Data 2024 release. Any CSV with an id column and a diagnosis column works; labels are mapped to the 7 classes (AK→akiec, SK/solar lentigo/LPLK→bkl, invasive SCC and "other" are excluded). Do **not** use ISIC 2019 as external – it contains HAM10000. |
| PH2 (optional) | explanation check on outside masks | https://www.fc.up.pt/addi/ph2%20database.html (registration) |

```bash
python prepare_data.py \
  --ham_dir /data/HAM10000 \
  --ham_masks_dir /data/HAM10000_segmentations \
  --isic18_test_dir /data/ISIC2018_Task3_Test_Input \
  --isic18_test_gt /data/ISIC2018_Task3_Test_GroundTruth.csv \
  --external bcn20000 /data/bcn/metadata.csv /data/bcn/images \
  --ph2_root /data/PH2Dataset \
  --dhash --check_test_overlap \
  --cache_dir /data/cache512 --cache_short 512 \
  --out data/master.csv
```

* `--cache_dir` resizes once (short side 512) and pre-applies Shades-of-Gray colour constancy – this makes training several times faster.
* `--dhash` merges exact / near-exact re-saved copies with the same diagnosis across lesion ids and writes `dhash_pairs_to_review.csv` – look at those pairs before the main runs.
* `--check_test_overlap` flags test images that are near-identical to a HAM10000 image (`test_dup_of_ham=1`; dropped by `evaluate.py`).
* The printout reports the share of image-level-split test images that have a sibling image of the same lesion in training – quote it in the paper.

## 3. Run the experiments

```bash
bash run_all.sh                                   # everything
STAGES="infl" bash run_all.sh                     # one stage
SEEDS_FINAL="42 7 2024" STAGES="base final" bash run_all.sh   # journal: 3 seeds
```

| Stage | Runs | Paper table | Rough GPU time (T4 / P100; ~4× faster on A100) |
|---|---|---|---|
| `audit` | leakage audit (+ repair, + quarantine of stale runs) over the master CSV and all finished runs | leakage report | minutes (CPU) |
| `infl` | ConvNeXt-T + GAP, 3 protocols × 5 folds, 224 px | Table A (top): inflation | ~5 h |
| `abl` | 5 variants × 5 folds, lesion protocol, 224 px | Table A (bottom): ablation | ~7 h |
| `base` | 6 literature architectures, all HAM10000, 384 px | Table B/C: test-set comparison | ~8 h per seed |
| `final` | DermAlign, 5 folds + final model, 384 px | Tables A/B/D | ~8 h |
| `hyb` | DermAlign-X, 5 folds + final model, 384 px | Tables A/B | ~12 h |
| `hybabl` | DermAlign-X ablations (concat / no gate / depth 2 / no alignment / ViT only) × 5 folds, 224 px | Table A (ablation of the hybrid) | ~10 h |
| `xai` | 4 models × 300 outer-fold images | Table E | ~1 h |
| `eval` | aggregation, CIs, tests, figures | all | minutes |

Finished runs are skipped, so a stopped Kaggle/Colab session can be resumed (keep the `runs/` folder). Choose the final variant (with or without metadata, λ) from the **cross-validation** ablation, never from the test sets.

Single run, for reference:

```bash
python train.py --csv data/master.csv --exp_name dermalign --split lesion --fold 0 \
  --arch convnext_tiny.fb_in22k_ft_in1k --head attn --loss la --lambda_align 0.5 \
  --img_size 384 --epochs 30 --tta
python xai_eval.py --run_dir runs/dermalign/fold0_seed42 --csv data/master.csv --dataset outer
python evaluate.py --runs_dir runs --csv data/master.csv --ref dermalign --compare_prefix base_ --out results
```

## 3b. Leakage audit and repair

The protocol is leakage-free **by construction** (grouped folds), and `leakage_audit.py` verifies it **from the files**, independently of `prepare_data.py`:

```bash
python leakage_audit.py --csv data/master.csv --runs_dir runs            # exit code 1 on any leakage
python leakage_audit.py --csv data/master.csv --runs_dir runs --fix --inplace --quarantine
```

| Check | Fails when |
|---|---|
| `group_single_fold`, `lesion_single_fold` | one lesion group sits in two folds (= the same lesion in train and test) |
| `unique_image_ids`, `no_lesion_id_overlap_with_test` | an id appears twice or is shared between HAM10000 and a test set |
| `test_not_copy_of_ham` | an ISIC 2018 / BCN20000 image is a near-copy (dHash ≤ 8/256 bits) or byte-copy of a HAM10000 image and is not flagged `test_dup_of_ham=1` |
| `no_cross_group_copies_across_folds` | two near-identical, same-diagnosis HAM10000 images from different groups sit in different folds |
| `largest_group_share` (warning) | one merged group holds > 5 % of HAM10000: `--max_dist` is chaining unrelated images |
| per-run replay | a finished `lesion` run trained on a lesion that is in its validation or outer set (judged with the *current* groups, from the run's exported predictions); `image` / `oversample_leak` runs are listed as *intentionally leaky* with the measured share |

Hashes are computed on the images the model actually reads (`path` in the master CSV, i.e. the resized, colour-corrected cache), so copies that only look identical after preprocessing are caught too.

`--fix` flags test copies, merges the groups of duplicate pairs, moves each merged group into the fold that already holds most of its images (minimal change; `--refold` regenerates every fold), re-audits, and lists `runs_to_retrain`. With `--quarantine` these runs are moved to `runs/_stale/`, so `run_all.sh` retrains them instead of skipping them. `run_all.sh` runs this as its first stage (`AUTOFIX=0` to only report and abort). Always read `results/leakage/leakage_report.md` and look at `dhash_pairs_to_review.csv`: an automatic repair is a safety net, not a substitute for looking at the pairs.

## 3c. DermAlign-X (hybrid CNN + Transformer)

`--head xfuse` (`dermalign/hybrid.py`) runs two streams on the same image: a CNN (default ConvNeXt-T, local texture) and a ViT (`--arch2`, default ViT-S/16, global context). Both are projected to `--fuse_dim` channels, then exchange information through **bidirectional cross-attention** (CNN tokens query the ViT tokens and vice versa, `--fuse_depth` blocks), are merged by a **per-cell, per-channel learned gate**, and pooled by the DermAlign attention head. The attention map lives on the CNN grid, so the lesion-alignment loss, Grad-CAM, Integrated Gradients and the randomisation sanity check apply unchanged. Metadata fusion, logit-adjusted loss and temperature scaling are shared with the other models.

Component ablations (`STAGES=hybabl`): `--fuse_mode concat` (no cross-attention), `--fuse_mode xattn_nogate` (fixed 0.5 instead of the gate), `--fuse_depth 2`, `--lambda_align 0`, and the ViT stream alone; the CNN stream alone is `abl3_dermalign`. `python complexity.py` reports parameters and GFLOPs of every model (Table F), so accuracy gains can be read against cost.

Practical note: ConvNeXt (ImageNet statistics) and the augreg ViT (0.5/0.5 statistics) are pretrained with different input normalisation while the data pipeline applies one; fine-tuning absorbs the mismatch, but say so in the methods.

## 3d. What numbers to expect (and what to distrust)

95–99 % accuracy on HAM10000 is what a split **without** lesion grouping gives; the `infl_*` runs reproduce that and quantify it. Under this protocol the closest comparable work (PLCRD) reports balanced accuracy 0.764, so a result far above that deserves suspicion: run the audit first. Report balanced accuracy, macro-F1, melanoma sensitivity and calibration on the sealed ISIC 2018 and BCN20000 sets against the re-trained baselines, not overall accuracy against the leaky literature. Nothing in this repository has been trained on the real data; the smoke test only checks that the code runs.

## 4. What ends up in `results/`

* **Table A** – CV on HAM10000: BA, macro-F1, accuracy, macro-AUC, melanoma sensitivity, lesion-level BA / macro-F1, ECE (calibrated / uncalibrated), mean ± SD.
* **Table B** – each test set: BA and macro-F1 with bootstrap 95 % CI, macro-AUC, melanoma sensitivity, single-model mean ± SD, ECE, AURC.
* **Table C** – DermAlign vs every baseline on the same images: ΔBA / Δmacro-F1 with paired-bootstrap CI and p, McNemar p (seed-matched).
* **Table D** – literature comparison with the protocol of every row (`dermalign/resources/literature.csv`; edit it if you update the review).
* **Table E** – explanation metrics for every evaluated model, with centre-prior and random controls.
* **Table F** – parameters and GFLOPs of every architecture (`complexity.csv`); `results/leakage/` – the leakage report.
* `per_class_*.csv`; selective-prediction columns in `test_summary.csv` (AURC, BA / melanoma sensitivity at 90 % and 80 % coverage, melanomas caught-or-referred); figures (`fig_confusion_*`, `fig_reliability_*`, `fig_risk_coverage_*`, `runs/*/*/xai_*/examples.png`).

## 5. Design decisions worth stating in the methods section

* `dx_type` is never used as a feature: only nevi were confirmed by follow-up, so it leaks the label.
* Masks are dilated by 5 % of the image height before the alignment loss so the border and immediate surrounding skin stay "allowed".
* The attention conv starts at zero (uniform attention = GAP), so λ = 0 is exactly an attention-pooling model and the ablation is clean.
* Logit adjustment is applied in the loss only; predictions use raw logits. If a sampler is used, the prior passed to the loss is the *effective* training prior.
* Every decision (checkpoint, temperature) uses the inner validation split; outer folds and test sets are only predicted.
* Localisation scores must be read against the centre prior: dermoscopy lesions are usually central, so a Gaussian blob already scores well.

## 6. Smoke test (CPU, synthetic data, no downloads)

```bash
python tests/make_synthetic.py --out /tmp/syn
python prepare_data.py --ham_dir /tmp/syn/HAM10000 --ham_masks_dir /tmp/syn/HAM10000_segmentations \
  --isic18_test_dir /tmp/syn/ISIC2018_Task3_Test_Input --isic18_test_gt /tmp/syn/ISIC2018_Task3_Test_GroundTruth.csv \
  --external bcn20000 /tmp/syn/BCN/metadata.csv /tmp/syn/BCN/images --cache_dir /tmp/syn/cache --cache_short 128 \
  --out /tmp/syn/master.csv
CSV=/tmp/syn/master.csv RUNS=/tmp/syn/runs OUT=/tmp/syn/results FOLDS="0" IMG_FAST=64 IMG_FINAL=64 EP_FAST=1 EP_FINAL=1 \
  WORKERS=0 BS=16 EXTRA="--no_pretrained --max_steps_per_epoch 2" XAI_ARGS="--n 20 --n_faith 3 --n_sanity 3 --ig_steps 4" \
  bash run_all.sh
```

The synthetic numbers are meaningless; the test only checks that every stage runs (`bash tests/smoke_test.sh` does the same in one command, after running the unit tests).

Mechanism check done during development (synthetic data, ResNet-18 from scratch, 128 px, 12 epochs – not a result): adding the alignment loss (λ = 1) raised the attention mass inside the lesion from 0.36 to 0.43 and cut the border-shortcut share from 0.085 to 0.051, while outer-fold BA fell from 0.78 to 0.72 because the synthetic set contains a planted ruler shortcut. Expect the same trade-off on real data: tune λ on the CV folds and report both accuracy and explanation metrics.

## 6b. Reproducibility checklist

* Seeds: `--seed` (default 42; journal runs `SEEDS_FINAL="42 7 2024"`) controls the inner split, sampler, augmentation and initialisation of new layers; `prepare_data.py --seed` controls the outer folds.
* Every run stores `config.json` (all arguments), `env.json` (python / torch / timm / numpy versions, CUDA, GPU, git commit), `log.csv` and the exported predictions.
* Selection and temperature use the inner validation split only; outer folds and test sets are predicted once. The ablation (not the test sets) decides the final variant.
* Tested with torch 2.14.1, timm 1.0.30, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1 (CPU smoke test). GPU runs are not bit-wise deterministic (cuDNN, AMP); report mean ± SD over folds/seeds.
* `pip install -r requirements-dev.txt && python -m pytest -q tests/test_units.py` runs the unit tests (model contracts, losses, leakage audit and repair on injected leaks).

## 7. References for the methods

Tschandl et al. 2018 (HAM10000); Codella et al. 2019 (ISIC 2018); Hernández-Pérez et al. 2024 (BCN20000); Mendonça et al. 2013 (PH2); Abhishek et al. 2025 (DermaMNIST/HAM10000 duplicates); Menon et al. 2021 (logit adjustment); Guo et al. 2017 (temperature scaling); Ross et al. 2017 and Rieger et al. 2020 (explanation-penalised training); Adebayo et al. 2018 (sanity checks); Petsiuk et al. 2018 (deletion/insertion); Arras et al. 2022 (relevance mass/rank accuracy); Finlayson & Trezzi 2004, Barata et al. 2015 (colour constancy); Woo et al. 2018 (CBAM); Liu et al. 2022 (ConvNeXt).
