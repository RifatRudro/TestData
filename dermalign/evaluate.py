#!/usr/bin/env python
"""Aggregate all runs into paper tables and figures.

python evaluate.py --runs_dir runs --csv data/master.csv --ref dermalign --out results

Produces in --out:
  cv_runs.csv / cv_summary.csv        outer-fold metrics per run and mean +- SD per experiment
  test_runs.csv / test_summary.csv    per-run and ensemble metrics (+ bootstrap 95% CI) per test set
  comparisons.csv                     ref vs every other experiment (paired bootstrap, McNemar)
  per_class_*.csv                     per-class table of the reference model (selective-prediction
                                      columns are in test_summary.csv)
  report.md, tables.tex               ready-to-paste tables incl. the literature comparison
  fig_*.png                           confusion matrices, reliability diagrams, risk-coverage curves
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dermalign.constants import CLASSES
from dermalign.metrics import (aggregate_by_group, bootstrap_ci, classification_metrics, mcnemar,
                               paired_bootstrap, reliability_bins, risk_coverage, selective_metrics)
from dermalign.utils import load_json

PROB = [f"prob_{c}" for c in CLASSES]
PROBU = [f"probu_{c}" for c in CLASSES]
RES = Path(__file__).parent / "dermalign" / "resources"
KEY = ["bal_acc", "macro_f1", "acc", "macro_auc", "mel_sens", "mel_auc", "ece", "brier"]


def discover(runs_dir):
    runs = []
    for cfgp in sorted(Path(runs_dir).glob("*/*/config.json")):
        rd = cfgp.parent
        if not (rd / "model.pt").exists():
            continue
        cfg = load_json(cfgp)
        preds = {p.stem.replace("preds_", ""): p for p in rd.glob("preds_*.csv")}
        runs.append({"exp": cfg["exp_name"], "split": cfg["split"], "fold": cfg["fold"], "seed": cfg["seed"],
                     "arch": cfg["arch"], "head": cfg["head"], "lambda_align": cfg.get("lambda_align", 0),
                     "loss": cfg.get("loss"), "use_meta": cfg.get("use_meta", False), "dir": rd, "preds": preds})
    return runs


def metrics_row(df, prob_cols=PROB):
    s, _ = classification_metrics(df.label.values, df[prob_cols].values)
    return s


def mean_sd(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        out[f"{k}_mean"] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        out[f"{k}_sd"] = np.nanstd(v, ddof=1) if np.isfinite(v).sum() > 1 else 0.0
    return out


def ensemble(frames):
    """Average calibrated and uncalibrated probabilities over runs (aligned on image_id)."""
    base = frames[0][["image_id", "group_id", "label"]].drop_duplicates("image_id").set_index("image_id")
    acc = None
    for f in frames:
        f = f.drop_duplicates("image_id").set_index("image_id").loc[base.index, PROB + PROBU]
        acc = f if acc is None else acc + f
    out = base.join(acc / len(frames)).reset_index()
    return out


def fmt(m, sd=None, ci=None, d=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "-"
    s = f"{m:.{d}f}"
    if sd is not None and not np.isnan(sd) and sd > 0:
        s += f" ± {sd:.{d}f}"
    if ci is not None:
        s += f" [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"
    return s


def fmt_p(p, n_boot=None):
    if n_boot and p == 0:
        return f"<{1 / n_boot:.4f}"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def tex_table(df, caption, label):
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("±", "$\\pm$").replace("&", "\\&")
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule",
             " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(r[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def collect_xai(runs_dir):
    rows = []
    for summ in sorted(Path(runs_dir).glob("*/*/xai_*/summary.csv")):
        xd = summ.parent
        exp, run = xd.parent.parent.name, xd.parent.name
        loc = pd.read_csv(summ, index_col=0)
        fa = pd.read_csv(xd / "faithfulness.csv", index_col=0) if (xd / "faithfulness.csv").exists() else None
        sa = pd.read_csv(xd / "sanity.csv", index_col=0) if (xd / "sanity.csv").exists() else None
        for m in loc.index:
            r = {"exp": exp, "run": run, "data": xd.name.replace("xai_", ""), "method": m,
                 "mass_in_mask": loc.at[m, "mass_in_mask_mean"], "rank_acc": loc.at[m, "rank_acc_mean"],
                 "pointing": loc.at[m, "pointing_mean"], "border_outside": loc.at[m, "border_outside_mean"]}
            if fa is not None and m in fa.index:
                r["deletion_auc"] = fa.at[m, "deletion_auc_mean"]
                r["insertion_auc"] = fa.at[m, "insertion_auc_mean"]
            if sa is not None and m in sa.index:
                r["sanity_rho"] = sa.at[m, "mean"]
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--csv", default=None, help="master CSV (to drop test images duplicated in HAM)")
    ap.add_argument("--ref", default=None, help="experiment treated as the proposed model")
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare_prefix", default=None,
                    help="only compare ref against experiments whose name starts with this (e.g. base_)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--no_figs", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = discover(args.runs_dir)
    if not runs:
        raise SystemExit(f"no finished runs under {args.runs_dir}")
    exps = sorted({r["exp"] for r in runs})
    ref = args.ref or next((e for e in exps if "dermalign" in e.lower()), exps[0])
    exclude = set()
    if args.csv:
        m = pd.read_csv(args.csv)
        if "test_dup_of_ham" in m.columns:
            exclude = set(m.loc[m.test_dup_of_ham == 1, "image_id"])
    print(f"{len(runs)} runs, {len(exps)} experiments, ref = {ref}")

    # ------------------------------------------------------------------ cross-validation
    cv_rows = []
    for r in runs:
        if "outer" not in r["preds"]:
            continue
        p = pd.read_csv(r["preds"]["outer"])
        s = metrics_row(p)
        su = metrics_row(p, PROBU)
        les = aggregate_by_group(p, PROB)
        sl = metrics_row(les)
        cv_rows.append({"exp": r["exp"], "split": r["split"], "fold": r["fold"], "seed": r["seed"], **s,
                        "ece_uncal": su["ece"], "les_bal_acc": sl["bal_acc"], "les_macro_f1": sl["macro_f1"],
                        "les_macro_auc": sl["macro_auc"]})
    cv_runs = pd.DataFrame(cv_rows)
    cv_sum = pd.DataFrame()
    if len(cv_runs):
        cv_runs.to_csv(out / "cv_runs.csv", index=False)
        recs = []
        for (e, sp), g in cv_runs.groupby(["exp", "split"]):
            recs.append({"exp": e, "split": sp, "n_runs": len(g),
                         **mean_sd(g.to_dict("records"), KEY + ["ece_uncal", "les_bal_acc", "les_macro_f1"])})
        cv_sum = pd.DataFrame(recs)
        cv_sum.to_csv(out / "cv_summary.csv", index=False)

    # ------------------------------------------------------------------ test sets
    test_rows, ens_rows, ens_preds, run_frames = [], [], {}, {}
    datasets = sorted({d for r in runs for d in r["preds"] if d not in ("outer", "innerval")})
    for ds in datasets:
        for e in exps:
            er = [r for r in runs if r["exp"] == e and ds in r["preds"]]
            for kind, sel in (("final", [r for r in er if r["fold"] < 0]), ("cv_ens", [r for r in er if r["fold"] >= 0])):
                if not sel:
                    continue
                frames = []
                for r in sel:
                    p = pd.read_csv(r["preds"][ds])
                    p = p[~p.image_id.isin(exclude)]
                    frames.append((r["seed"], p))
                    s = metrics_row(p)
                    test_rows.append({"dataset": ds, "exp": e, "kind": kind, "fold": r["fold"], "seed": r["seed"], **s})
                run_frames[(ds, e, kind)] = frames
                ens = ensemble([f for _, f in frames])
                ens_preds[(ds, e, kind)] = ens
                s = metrics_row(ens)
                su = metrics_row(ens, PROBU)
                ci = bootstrap_ci(ens.label.values, ens[PROB].values, n_boot=args.n_boot)
                single = [t for t in test_rows if t["dataset"] == ds and t["exp"] == e and t["kind"] == kind]
                ens_rows.append({"dataset": ds, "exp": e, "kind": kind, "n_models": len(sel),
                                 "seeds": sorted({sd for sd, _ in frames}), **s,
                                 "ece_uncal": su["ece"],
                                 **{f"{k}_ci": ci[k] for k in ci},
                                 **{f"single_{k}": v for k, v in mean_sd(single, ["bal_acc", "macro_f1"]).items()},
                                 **selective_metrics(ens.label.values, ens[PROB].values)})
                if e == ref:
                    _, pc = classification_metrics(ens.label.values, ens[PROB].values)
                    pc.to_csv(out / f"per_class_{ds}_{kind}.csv", index=False)
    test_runs = pd.DataFrame(test_rows)
    test_sum = pd.DataFrame(ens_rows)
    if len(test_runs):
        test_runs.to_csv(out / "test_runs.csv", index=False)
        test_sum.to_csv(out / "test_summary.csv", index=False)

    # ------------------------------------------------------------------ comparisons vs ref
    comp = []
    for (ds, e, kind), frames_b in run_frames.items():
        if e == ref or (ds, ref, kind) not in run_frames:
            continue
        if args.compare_prefix and not e.startswith(args.compare_prefix):
            continue
        frames_a = run_frames[(ds, ref, kind)]
        # same number of models on both sides: restrict to the seeds both experiments share
        common_seeds = {sd for sd, _ in frames_a} & {sd for sd, _ in frames_b}
        if common_seeds:
            frames_a = [(sd, f) for sd, f in frames_a if sd in common_seeds]
            frames_b = [(sd, f) for sd, f in frames_b if sd in common_seeds]
        a = ensemble([f for _, f in frames_a]).set_index("image_id")
        b = ensemble([f for _, f in frames_b]).set_index("image_id")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        y = a.label.values
        row = {"dataset": ds, "kind": kind, "ref": ref, "other": e, "n": len(common),
               "models_ref": len(frames_a), "models_other": len(frames_b),
               "seeds": sorted(common_seeds) if common_seeds else "all"}
        for met in ("bal_acc", "macro_f1", "mel_sens"):
            pb = paired_bootstrap(y, a[PROB].values, b[PROB].values, met, n_boot=args.n_boot)
            row.update({f"d_{met}": pb["diff"], f"d_{met}_ci": (pb["ci_lo"], pb["ci_hi"]), f"p_{met}": pb["p"]})
        mc = mcnemar(y, a[PROB].values.argmax(1), b[PROB].values.argmax(1))
        row.update({"mcnemar_p": mc["p"], "ref_only_correct": mc["a_only_correct"], "other_only_correct": mc["b_only_correct"]})
        comp.append(row)
    comp = pd.DataFrame(comp)
    if len(comp):
        comp.to_csv(out / "comparisons.csv", index=False)

    # ------------------------------------------------------------------ report
    md, tex = ["# DermAlign-Net results\n", f"Reference (proposed) experiment: **{ref}**\n"], []
    if len(cv_sum):
        t = cv_sum.sort_values(["split", "exp"])
        tab = pd.DataFrame({
            "Experiment": t.exp, "Protocol": t.split, "Runs": t.n_runs,
            "BA": [fmt(a, b) for a, b in zip(t.bal_acc_mean, t.bal_acc_sd)],
            "Macro-F1": [fmt(a, b) for a, b in zip(t.macro_f1_mean, t.macro_f1_sd)],
            "Acc": [fmt(a, b) for a, b in zip(t.acc_mean, t.acc_sd)],
            "Macro-AUC": [fmt(a, b) for a, b in zip(t.macro_auc_mean, t.macro_auc_sd)],
            "Mel sens": [fmt(a, b) for a, b in zip(t.mel_sens_mean, t.mel_sens_sd)],
            "Lesion BA": [fmt(a, b) for a, b in zip(t.les_bal_acc_mean, t.les_bal_acc_sd)],
            "Lesion macro-F1": [fmt(a, b) for a, b in zip(t.les_macro_f1_mean, t.les_macro_f1_sd)],
            "ECE (cal / uncal)": [f"{fmt(a)} / {fmt(b)}" for a, b in zip(t.ece_mean, t.ece_uncal_mean)],
        })
        md += ["## Table A. Internal 5-fold cross-validation on HAM10000 (outer folds, mean ± SD)\n",
               "Protocols other than `lesion` leak lesions between train and test; their numbers are "
               "shown only to quantify the inflation.\n", md_table(tab), ""]
        tex.append(tex_table(tab, "Five-fold cross-validation on HAM10000 (mean $\\pm$ SD over folds).", "tab:cv"))
    if len(test_sum):
        for ds, g in test_sum.groupby("dataset"):
            g = g.sort_values(["kind", "bal_acc"], ascending=[True, False])
            tab = pd.DataFrame({
                "Experiment": g.exp, "Models": [f"{k} ({n})" for k, n in zip(g.kind, g.n_models)],
                "BA [95% CI]": [fmt(a, ci=c) for a, c in zip(g.bal_acc, g.bal_acc_ci)],
                "Macro-F1 [95% CI]": [fmt(a, ci=c) for a, c in zip(g.macro_f1, g.macro_f1_ci)],
                "Macro-AUC": [fmt(a) for a in g.macro_auc],
                "Mel sens": [fmt(a) for a in g.mel_sens] if "mel_sens" in g else "-",
                "Acc": [fmt(a) for a in g.acc],
                "Single-model BA": [fmt(a, b) for a, b in zip(g.single_bal_acc_mean, g.single_bal_acc_sd)],
                "ECE cal/uncal": [f"{fmt(a)} / {fmt(b)}" for a, b in zip(g.ece, g.ece_uncal)],
                "AURC": [fmt(a) for a in g.aurc],
            })
            md += [f"## Table B. Held-out test: {ds} (ensemble of the listed models)\n", md_table(tab), ""]
            tex.append(tex_table(tab, f"Results on the held-out {ds.replace('_', ' ')} set.", f"tab:{ds}"))
    if len(comp):
        tab = pd.DataFrame({
            "Test set": comp.dataset, "Models": comp.kind, "vs": comp.other, "n": comp.n,
            "ΔBA [95% CI]": [fmt(a, ci=c) for a, c in zip(comp.d_bal_acc, comp.d_bal_acc_ci)],
            "Models (ref/other)": [f"{a}/{b}" for a, b in zip(comp.models_ref, comp.models_other)],
            "p (BA)": [fmt_p(p, args.n_boot) for p in comp.p_bal_acc],
            "ΔMacro-F1 [95% CI]": [fmt(a, ci=c) for a, c in zip(comp.d_macro_f1, comp.d_macro_f1_ci)],
            "McNemar p": [fmt_p(p) for p in comp.mcnemar_p],
        })
        md += [f"## Table C. {ref} vs other models (paired bootstrap, McNemar)\n", md_table(tab), ""]
    # literature comparison
    lit = pd.read_csv(RES / "literature.csv")
    ours = []
    if len(cv_sum) and ref in set(cv_sum.exp):
        r = cv_sum[cv_sum.exp == ref].iloc[0]
        ours.append({"study": f"Ours: {ref}", "protocol": f"{r.split}-grouped 5-fold CV", "eval_data": "HAM10000 (7 cls)",
                     "acc": r.acc_mean, "bal_acc": r.bal_acc_mean, "macro_f1": r.macro_f1_mean,
                     "lesion_macro_f1": r.les_macro_f1_mean, "auc": r.macro_auc_mean})
    if len(test_sum):
        for _, r in test_sum[test_sum.exp == ref].iterrows():
            ours.append({"study": f"Ours: {ref} ({r.kind})", "protocol": "sealed test, used once", "eval_data": r.dataset,
                         "acc": r.acc, "bal_acc": r.bal_acc, "macro_f1": r.macro_f1, "auc": r.macro_auc})
    lit_tab = pd.concat([lit, pd.DataFrame(ours)], ignore_index=True)
    show = lit_tab[["study", "protocol", "eval_data", "acc", "bal_acc", "macro_f1", "f1_reported", "auc", "note"]
                   if "f1_reported" in lit_tab else ["study", "protocol", "eval_data", "acc", "bal_acc", "macro_f1", "auc"]]
    show = show.copy()
    for c in ["acc", "bal_acc", "macro_f1", "f1_reported", "auc"]:
        if c in show:
            show[c] = show[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    show = show.fillna("")
    md += ["## Table D. Literature comparison\n",
           "Only rows with a lesion-disjoint protocol or an official held-out test set are directly comparable "
           "with ours; the others did not report a lesion-grouped split.\n", md_table(show), ""]
    tex.append(tex_table(show, "Comparison with the literature (protocol stated for each row).", "tab:lit"))
    xai = collect_xai(args.runs_dir)
    if len(xai):
        xai.to_csv(out / "xai_summary.csv", index=False)
        md += ["## Table E. Explanation quality (lesion masks; higher mass/rank/pointing/insertion = better, "
               "lower border/deletion = better; sanity: lower |rho| = passes)\n",
               "The centre prior and random map are controls: a method is only informative where it beats them.\n",
               md_table(xai.round(3).fillna("-")), ""]
        tex.append(tex_table(xai.round(3).fillna("-"), "Quantitative explanation evaluation.", "tab:xai"))
    cx = out / "complexity.csv"
    if cx.exists():
        cdf = pd.read_csv(cx)
        md += ["## Table F. Model size and forward cost (one image, no pretrained weights needed)\n",
               md_table(cdf.round(2)), ""]
        tex.append(tex_table(cdf.round(2), "Parameters and forward cost per image.", "tab:complexity"))
    (out / "report.md").write_text("\n".join(md))
    (out / "tables.tex").write_text("\n".join(tex))

    # ------------------------------------------------------------------ figures
    if not args.no_figs and ens_preds:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        for (ds, e, kind), ens in ens_preds.items():
            if e != ref:
                continue
            y, pr = ens.label.values, ens[PROB].values
            cm = confusion_matrix(y, pr.argmax(1), labels=range(len(CLASSES)), normalize="true")
            fig, ax = plt.subplots(figsize=(5, 4.4))
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
            ax.set_yticks(range(len(CLASSES)), CLASSES)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.5 else "black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{e} on {ds} ({kind})", fontsize=9)
            fig.colorbar(im, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out / f"fig_confusion_{ds}_{kind}.png", dpi=200)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(4, 4))
            for cols, lab in ((PROBU, "uncalibrated"), (PROB, "temperature-scaled")):
                rb = reliability_bins(ens[cols].values, y)
                rb = rb[rb.n > 0]
                ax.plot(rb.conf, rb.acc, marker="o", ms=3, label=lab)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=8)
            ax.set_title(f"Reliability: {ds}", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"fig_reliability_{ds}_{kind}.png", dpi=200)
            plt.close(fig)
        for ds in datasets:
            fig, ax = plt.subplots(figsize=(4.6, 3.6))
            for (d, e, kind), ens in ens_preds.items():
                if d != ds:
                    continue
                cov, risk, aurc = risk_coverage(ens.label.values, ens[PROB].values)
                ax.plot(cov, risk, lw=1.8 if e == ref else 1.0, label=f"{e} ({kind}) AURC={aurc:.3f}")
            ax.set_xlabel("Coverage (fraction not referred)")
            ax.set_ylabel("Error rate on accepted cases")
            ax.legend(fontsize=6)
            ax.set_title(f"Risk-coverage: {ds}", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"fig_risk_coverage_{ds}.png", dpi=200)
            plt.close(fig)
    print((out / "report.md").read_text()[:3000])
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
