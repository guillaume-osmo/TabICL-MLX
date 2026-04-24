#!/usr/bin/env python3
"""03 — Pat Walters-style Tukey-HSD boxplots for the ODT benchmark.

Reads results/02_cv5x5_results.csv (25 runs per method per split-kind)
and renders the comparison plots from his March 2025 post:

  https://practicalcheminformatics.blogspot.com/2025/03/
  even-more-thoughts-on-ml-method.html

Two panels per split kind (random / scaffold):
  • R² boxplot with Tukey-HSD significance stars between every pair
  • RMSE boxplot with Tukey-HSD significance stars

Also the "simultaneous CI" plot vs the best method (his other style).

Saves PNG + PDF in results/.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Optional: star annotations via statannotations (pip install statannotations)
try:
    from statannotations.Annotator import Annotator
    _HAS_ANNOTATIONS = True
except ImportError:
    _HAS_ANNOTATIONS = False


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def tukey_boxplot(df, x_col, y_col, xlab, ylab, ax, title="",
                  hide_ns=True, palette="Blues", rotate=30):
    """Pat Walters' tukey_boxplot, lightly modernized."""
    tukey = pairwise_tukeyhsd(endog=df[y_col], groups=df[x_col], alpha=0.05)
    t_data = tukey._results_table.data
    if hide_ns:
        pairs = [(x[0], x[1]) for x in t_data[1:] if x[3] < 0.05]
        p_vals = [x[3] for x in t_data[1:] if x[3] < 0.05]
    else:
        pairs = [(x[0], x[1]) for x in t_data[1:]]
        p_vals = [x[3] for x in t_data[1:]]

    # order by mean y
    order = df.groupby(x_col)[y_col].mean().sort_values(ascending=(y_col == "rmse")).index.tolist()

    sns.boxplot(x=x_col, y=y_col, data=df, ax=ax,
                order=order, palette=palette, linewidth=1.0, fliersize=2)
    sns.stripplot(x=x_col, y=y_col, data=df, ax=ax,
                  order=order, color="0.25", size=2.5, alpha=0.6, jitter=0.2)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")

    if _HAS_ANNOTATIONS and pairs:
        annot = Annotator(ax, pairs, data=df, x=x_col, y=y_col, order=order, verbose=False)
        annot.configure(text_format="star", loc="inside", line_height=0.02)
        annot.set_pvalues_and_annotate(p_vals)


def tukey_simultaneous(df, metric_col, ax, lower_better=False, title=""):
    """Tukey simultaneous CI against the best method — blue=best, red=worse."""
    grp = df.groupby("method")[metric_col].mean()
    best = grp.idxmin() if lower_better else grp.idxmax()
    tukey = pairwise_tukeyhsd(endog=df[metric_col], groups=df["method"], alpha=0.05)
    tukey.plot_simultaneous(comparison_name=best, ax=ax, figsize=(6, 4))
    ax.set_title(title)


def main():
    path = RESULTS / "02_cv5x5_results.csv"
    if not path.exists():
        print(f"  Missing {path}. Run `python 02_cv5x5_honest.py` first.")
        return
    df = pd.read_csv(path).dropna(subset=["r2", "rmse"])

    # 1-NN is a leakage probe only (script 01) — it crushes the y-axis on the
    # scaffold split (R²=-0.8) and hides real differences between ML models.
    EXCLUDE = {"1-NN ECFP4", "Predict-train-mean"}
    df = df[~df["method"].isin(EXCLUDE)].copy()
    print(f"  Loaded {len(df)} rows after dropping {EXCLUDE}, "
          f"methods: {sorted(df['method'].unique())}")

    for split_kind in ["random", "scaffold"]:
        sub = df[df["split"] == split_kind].copy()
        if sub.empty:
            continue

        # ── Boxplot panel (R² | RMSE) with Tukey stars ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        tukey_boxplot(sub, "method", "r2", "Method", r"$R^2$", axes[0],
                      title=f"R² — {split_kind} split (5 seeds × 5 folds)",
                      palette="Blues_r", hide_ns=True)
        tukey_boxplot(sub, "method", "rmse", "Method", "RMSE", axes[1],
                      title=f"RMSE — {split_kind} split (5 seeds × 5 folds)",
                      palette="Reds", hide_ns=True)
        plt.tight_layout()
        out_png = RESULTS / f"03_tukey_boxplot_{split_kind}.png"
        out_pdf = RESULTS / f"03_tukey_boxplot_{split_kind}.pdf"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_png.name} / {out_pdf.name}")

        # ── Simultaneous CI panel (horizontal CI against best) ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        tukey_simultaneous(sub, "r2",   axes[0], lower_better=False,
                           title=f"R² vs best — {split_kind}")
        tukey_simultaneous(sub, "rmse", axes[1], lower_better=True,
                           title=f"RMSE vs best — {split_kind}")
        plt.tight_layout()
        out_png = RESULTS / f"03_tukey_ci_{split_kind}.png"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_png.name}")

    # ── Summary CSV across both split kinds ──
    summary = df.groupby(["split", "method"]).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        n=("r2", "count"),
    ).reset_index().sort_values(["split", "r2_mean"], ascending=[True, False])
    summary.to_csv(RESULTS / "03_summary.csv", index=False)
    print(f"  Saved: 03_summary.csv")

    # ── Final summary table (console) ──
    print("\n" + "=" * 92)
    print(" FINAL RANKINGS")
    print("=" * 92)
    for split_kind in ["random", "scaffold"]:
        sub = summary[summary["split"] == split_kind]
        if sub.empty:
            continue
        print(f"\n  {split_kind.upper()} 5×5 CV:")
        print(f"  {'Method':<24}  {'R² (mean±std)':>16}  {'RMSE (mean±std)':>18}")
        print("  " + "─" * 62)
        for _, r in sub.iterrows():
            print(f"  {r['method']:<24}  {r['r2_mean']:+7.4f}±{r['r2_std']:.3f}  "
                  f"{r['rmse_mean']:7.4f}±{r['rmse_std']:.3f}")


if __name__ == "__main__":
    main()
