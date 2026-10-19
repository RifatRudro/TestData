"""Evaluation metrics: discrimination, per-class, calibration, selective prediction,
bootstrap confidence intervals and paired significance tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (cohen_kappa_score, f1_score, matthews_corrcoef, roc_auc_score,
                             roc_curve)

from .constants import CLASSES, MEL_IDX, NUM_CLASSES


def ece_score(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = prob.max(1)
    pred = prob.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(prob, y, n_bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "conf": float(conf[m].mean()) if m.any() else np.nan,
                     "acc": float((pred[m] == y[m]).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def balanced_accuracy(y, pred):
    present = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0))


def macro_auc(y, prob):
    aucs = []
    for c in np.unique(y):
        yc = (y == c).astype(int)
        if 0 < yc.sum() < len(yc):
            aucs.append(roc_auc_score(yc, prob[:, c]))
    return float(np.mean(aucs)) if aucs else np.nan


def sens_at_spec(y_bin, score, spec):
    fpr, tpr, _ = roc_curve(y_bin, score)
    ok = fpr <= (1 - spec) + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def classification_metrics(y, prob):
    """Returns (summary dict, per-class DataFrame)."""
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob.argmax(1)
    s = {
        "n": int(len(y)),
        "acc": float((pred == y).mean()),
        "bal_acc": balanced_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "weighted_f1": float(f1_score(y, pred, labels=np.unique(y), average="weighted", zero_division=0)),
        "macro_auc": macro_auc(y, prob),
        "mcc": float(matthews_corrcoef(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "ece": ece_score(prob, y),
        "brier": float(((prob - np.eye(NUM_CLASSES)[y]) ** 2).sum(1).mean()),
        "nll": float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean()),
    }
    ym = (y == MEL_IDX).astype(int)
    if 0 < ym.sum() < len(ym):
        s["mel_sens"] = float((pred[ym == 1] == MEL_IDX).mean())
        s["mel_spec"] = float((pred[ym == 0] != MEL_IDX).mean())
        s["mel_auc"] = float(roc_auc_score(ym, prob[:, MEL_IDX]))
        s["mel_sens@spec80"] = sens_at_spec(ym, prob[:, MEL_IDX], 0.80)
        s["mel_sens@spec90"] = sens_at_spec(ym, prob[:, MEL_IDX], 0.90)
    rows = []
    for c in range(NUM_CLASSES):
        yc, pc = (y == c), (pred == c)
        sup = int(yc.sum())
        tp = int((yc & pc).sum())
        prec = tp / pc.sum() if pc.sum() else np.nan
        sens = tp / sup if sup else np.nan
        spec = float((~pc[~yc]).mean()) if (~yc).any() else np.nan
        f1 = 2 * prec * sens / (prec + sens) if sup and pc.sum() and (prec + sens) > 0 else (0.0 if sup else np.nan)
        auc = roc_auc_score(yc.astype(int), prob[:, c]) if 0 < sup < len(y) else np.nan
        rows.append({"class": CLASSES[c], "support": sup, "sensitivity": sens, "specificity": spec,
                     "precision": prec, "f1": f1, "auc": auc})
    return s, pd.DataFrame(rows)


# --------------------------------------------------------------------------- selective prediction
def risk_coverage(y, prob):
    conf = prob.max(1)
    order = np.argsort(-conf, kind="stable")
    err = (prob.argmax(1)[order] != y[order]).astype(float)
    k = np.arange(1, len(y) + 1)
    risk = np.cumsum(err) / k
    coverage = k / len(y)
    return coverage, risk, float(risk.mean())


def selective_metrics(y, prob, coverages=(1.0, 0.9, 0.8)):
    y = np.asarray(y)
    conf = prob.max(1)
    order = np.argsort(-conf, kind="stable")
    out = {}
    _, _, aurc = risk_coverage(y, prob)
    out["aurc"] = aurc
    for c in coverages:
        keep = order[: max(1, int(round(c * len(y))))]
        yk, pk = y[keep], prob[keep].argmax(1)
        tag = int(round(c * 100))
        out[f"bal_acc@cov{tag}"] = balanced_accuracy(yk, pk)
        out[f"acc@cov{tag}"] = float((yk == pk).mean())
        mel = yk == MEL_IDX
        out[f"mel_sens@cov{tag}"] = float((pk[mel] == MEL_IDX).mean()) if mel.any() else np.nan
        # melanomas deferred to a dermatologist count as "caught" in a referral workflow
        all_mel = y == MEL_IDX
        if all_mel.any():
            deferred = np.setdiff1d(np.arange(len(y)), keep)
            caught = (pk[mel] == MEL_IDX).sum() + all_mel[deferred].sum()
            out[f"mel_caught_or_referred@cov{tag}"] = float(caught / all_mel.sum())
    return out


# --------------------------------------------------------------------------- resampling
def _confusion(y, pred):
    k = NUM_CLASSES
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def _fast_ba(y, p):
    cm = _confusion(y, p.argmax(1))
    sup = cm.sum(1)
    m = sup > 0
    return float((np.diag(cm)[m] / sup[m]).mean())


def _fast_f1(y, p):
    cm = _confusion(y, p.argmax(1))
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1)
    denom = 2 * tp + (cm.sum(0) - tp) + (sup - tp)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1[sup > 0].mean())


def _fast_auc(y, p):
    from scipy.stats import rankdata
    aucs = []
    for c in np.unique(y):
        pos = y == c
        n1, n0 = pos.sum(), (~pos).sum()
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(p[:, c])
        aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs)) if aucs else np.nan


METRIC_FNS = {  # fast numpy versions (identical values to the sklearn-based ones), used in resampling
    "bal_acc": _fast_ba,
    "macro_f1": _fast_f1,
    "macro_auc": _fast_auc,
    "acc": lambda y, p: float((p.argmax(1) == y).mean()),
    "mel_sens": lambda y, p: float((p[y == MEL_IDX].argmax(1) == MEL_IDX).mean()) if (y == MEL_IDX).any() else np.nan,
}


def bootstrap_ci(y, prob, metrics=("bal_acc", "macro_f1", "macro_auc", "mel_sens"), n_boot=2000,
                 groups=None, seed=0, alpha=0.05):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    vals = {m: [] for m in metrics}
    if groups is not None:  # precompute membership once
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    for _ in range(n_boot):
        if groups is None:
            idx = rng.randint(0, len(y), len(y))
        else:
            idx = np.concatenate([members[g] for g in rng.randint(0, len(members), len(members))])
        yb, pb = y[idx], prob[idx]
        for m in metrics:
            vals[m].append(METRIC_FNS[m](yb, pb))
    return {m: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for m, v in vals.items()}


def paired_bootstrap(y, prob_a, prob_b, metric="bal_acc", n_boot=2000, groups=None, seed=0):
    """Difference metric(a) - metric(b) on the same images: point estimate, 95% CI, two-sided p."""
    y = np.asarray(y)
    fn = METRIC_FNS[metric]
    rng = np.random.RandomState(seed)
    diffs = []
    if groups is not None:
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    for _ in range(n_boot):
        if groups is None:
            idx = rng.randint(0, len(y), len(y))
        else:
            idx = np.concatenate([members[g] for g in rng.randint(0, len(members), len(members))])
        diffs.append(fn(y[idx], prob_a[idx]) - fn(y[idx], prob_b[idx]))
    diffs = np.asarray(diffs)
    point = fn(y, prob_a) - fn(y, prob_b)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(point), "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)), "p": float(min(1.0, p))}


def mcnemar(y, pred_a, pred_b):
    ca, cb = pred_a == y, pred_b == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p": float(p)}


def aggregate_by_group(df: pd.DataFrame, prob_cols, group_col="group_id"):
    """Lesion-level predictions: mean probability over all images of a lesion."""
    g = df.groupby(group_col)
    out = g[prob_cols].mean()
    out["label"] = g["label"].agg(lambda s: s.mode().iat[0])
    return out.reset_index()
