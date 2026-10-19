"""CPU unit tests, no downloads, ~1 min:  python -m pytest -q tests/test_units.py"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dermalign.data import inner_split, outer_split  # noqa: E402
from dermalign.leakage import (audit_master, audit_runs, find_leak_pairs, fix_master,  # noqa: E402
                               hash_images, make_lesion_folds)
from dermalign.losses import LogitAdjustedLoss, alignment_loss  # noqa: E402
from dermalign.models import build_model  # noqa: E402
from dermalign.xai import randomise_top  # noqa: E402

VIT = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"


# --------------------------------------------------------------------------- models
@pytest.mark.parametrize("head,mode", [("gap", None), ("attn", None), ("cbam", None),
                                       ("xfuse", "xattn"), ("xfuse", "concat"), ("xfuse", "xattn_nogate")])
def test_model_contract(head, mode):
    cfg = dict(arch="resnet18", arch2=VIT, head=head, pretrained=False, img_size=64, use_meta=True)
    if mode:
        cfg["fuse_mode"] = mode
    m = build_model(cfg).train()
    x, meta = torch.randn(2, 3, 64, 64), torch.randn(2, m.meta_dim)
    out = m(x, meta)
    assert out["logits"].shape == (2, 7)
    if head == "gap":
        assert out["att"] is None
    else:
        assert out["att"].shape[0] == 2 and out["att"].shape[1] == 1
        torch.testing.assert_close(out["att"].flatten(1).sum(1), torch.ones(2), atol=1e-4, rtol=0)
    out["logits"].sum().backward()
    assert m.fc.weight.grad is not None and m.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in m.backbone.parameters() if p.requires_grad and p.ndim == 4)
    r = randomise_top(m)  # XAI sanity check must work for every head
    assert r is not m and r(x, meta)["logits"].shape == (2, 7)


def test_xfuse_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_model(dict(arch="resnet18", arch2=VIT, head="xfuse", pretrained=False, fuse_mode="nope"))


# --------------------------------------------------------------------------- losses
def test_alignment_loss_inside_vs_outside():
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 16:32, 16:32] = 1  # exactly grid cell (1, 1) of the 4x4 attention map
    inside = torch.zeros(1, 1, 4, 4)
    inside[0, 0, 1, 1] = 1
    outside = torch.zeros(1, 1, 4, 4)
    outside[0, 0, 0, 0] = 1
    valid = torch.ones(1)
    assert alignment_loss(inside, mask, valid) < 0.05
    assert alignment_loss(outside, mask, valid) > 0.95
    assert alignment_loss(outside, mask, torch.zeros(1)) == 0  # no usable mask -> no term


def test_logit_adjusted_loss_is_ce_at_tau0():
    logits, y = torch.randn(8, 7), torch.randint(0, 7, (8,))
    la = LogitAdjustedLoss(np.full(7, 1 / 7), tau=0.0)
    torch.testing.assert_close(la(logits, y), torch.nn.functional.cross_entropy(logits, y))


# --------------------------------------------------------------------------- leakage
@pytest.fixture()
def toy(tmp_path):
    """30 HAM lesions x 2 images, 12 test images; random-noise pictures (no accidental dHash matches)."""
    rng = np.random.RandomState(0)
    rows = []

    def add(image_id, dataset, lesion, dx):
        path = tmp_path / f"{image_id}.png"
        Image.fromarray(rng.randint(0, 256, (48, 48, 3), dtype=np.uint8)).save(path)
        rows.append(dict(image_id=image_id, path=str(path), dataset=dataset, dx=dx, label=0, lesion_id=lesion,
                         group_id=lesion, fold_lesion=-1, fold_image=-1, test_dup_of_ham=0))

    for i in range(30):
        for k in range(2):
            add(f"H{i}_{k}", "ham", f"L{i}", "nv" if i % 3 else "mel")
    for i in range(12):
        add(f"T{i}", "isic18_test", f"T{i}", "nv")
    df = pd.DataFrame(rows)
    ham = df.dataset == "ham"
    df.loc[ham, "fold_lesion"] = make_lesion_folds(df[ham], 5, 42)
    df.loc[~ham, "group_id"] = "isic18_test:" + df.lesion_id[~ham]
    return df


def _audit(df):
    return audit_master(df, find_leak_pairs(df, hash_images(df.path), max_dist=8))


def _failed(rep):
    return {c["name"] for c in rep["checks"] if not c["ok"] and c["severity"] == "error"}


def test_clean_master_passes(toy):
    rep = _audit(toy)
    assert rep["ok"], _failed(rep)
    assert rep["n_folds"] == 5


def test_lesion_folds_never_split_a_group(toy):
    ham = toy[toy.dataset == "ham"]
    assert (ham.groupby("group_id").fold_lesion.nunique() == 1).all()


def test_image_level_folds_are_flagged(toy):
    ham = toy.dataset == "ham"
    toy.loc[ham, "fold_lesion"] = np.arange(ham.sum()) % 5  # siblings land in different folds
    assert "group_single_fold" in _failed(_audit(toy))


def test_splits_replayed_from_master_are_disjoint(toy):
    ham = toy[toy.dataset == "ham"]
    for f in range(5):
        trainval, outer = outer_split(ham, "lesion", f, 42, 0)
        train, val = inner_split(trainval, "lesion", 42, 0.2)
        g = lambda d: set(d.group_id)  # noqa: E731
        assert not g(train) & g(val) and not g(trainval) & g(outer)


def test_test_copy_of_ham_is_found_flagged_and_stays_fixed(toy):
    shutil.copy(toy.path[toy.image_id == "H3_0"].iloc[0], toy.path[toy.image_id == "T2"].iloc[0])
    pairs = find_leak_pairs(toy, hash_images(toy.path))
    assert pairs["exact"] == 1
    assert "test_not_copy_of_ham" in _failed(audit_master(toy, pairs))
    fixed, info = fix_master(toy, pairs)
    assert info["newly_flagged_test_images"] == 1 and fixed.test_dup_of_ham.sum() == 1
    assert fixed.test_dup_of_ham[fixed.image_id == "T2"].iloc[0] == 1
    assert audit_master(fixed, pairs)["ok"]


def test_cross_lesion_duplicate_across_folds_is_merged(toy):
    a, b = toy[toy.image_id == "H1_0"].iloc[0], None
    # a different lesion with the same diagnosis in another fold, whose picture is a copy of `a`
    cand = toy[(toy.dataset == "ham") & (toy.dx == a.dx) & (toy.fold_lesion != a.fold_lesion)
               & (toy.group_id != a.group_id)]
    b = cand.iloc[0]
    shutil.copy(a.path, b.path)
    pairs = find_leak_pairs(toy, hash_images(toy.path))
    assert "no_cross_group_copies_across_folds" in _failed(audit_master(toy, pairs))
    fixed, info = fix_master(toy, pairs)
    assert info["merged_group_pairs"] == 1 and info["images_moved"] > 0 and info["folds_touched"]
    ga, gb = (fixed.group_id[fixed.image_id == i].iloc[0] for i in (a.image_id, b.image_id))
    assert ga == gb
    assert (fixed[fixed.group_id == ga].fold_lesion.nunique()) == 1
    assert audit_master(fixed, pairs)["ok"]
    again, info2 = fix_master(fixed, pairs)  # idempotent
    assert info2["merged_group_pairs"] == 0 and info2["images_moved"] == 0


def _write_run(root, name, df, outer_ids, val_ids, split="lesion", fold=0):
    run = root / name / f"fold{fold}_seed42"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"split": split, "fold": fold, "seed": 42}))
    pd.DataFrame({"image_id": sorted(val_ids)}).to_csv(run / "preds_innerval.csv", index=False)
    pd.DataFrame({"image_id": sorted(outer_ids)}).to_csv(run / "preds_outer.csv", index=False)
    return run


def test_audit_runs_detects_a_run_that_trained_on_an_outer_lesion(toy, tmp_path):
    ham = toy[toy.dataset == "ham"]
    outer = set(ham.image_id[ham.fold_lesion == 0])
    rest = sorted(set(ham.image_id) - outer)
    val = set(rest[:6])  # 3 whole lesions (2 images each, adjacent in the list)
    _write_run(tmp_path / "runs", "clean", toy, outer, val)
    sibling = next(i for i in outer if i.endswith("_0"))
    leaked_outer = (outer - {sibling.replace("_0", "_1")}) | set()  # drop one sibling from outer -> it trains
    _write_run(tmp_path / "runs", "leaky", toy, leaked_outer, val)
    res = audit_runs(toy, tmp_path / "runs").set_index("exp").status
    assert res["clean"] == "ok" and res["leaky"] == "LEAK"


def test_audit_runs_labels_image_level_runs_as_intentional(toy, tmp_path):
    ham = toy[toy.dataset == "ham"]
    outer = set(ham.image_id[::5])
    val = set(ham.image_id[1::7]) - outer
    _write_run(tmp_path / "runs", "infl_image", toy, outer, val, split="image")
    row = audit_runs(toy, tmp_path / "runs").iloc[0]
    assert row.status == "intentionally leaky" and row.outer_share_with_train_lesion > 0.5
