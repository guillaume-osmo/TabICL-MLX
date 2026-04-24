# ODT (Odor Detection Threshold) — lucky split vs honest CV

A small case study: reproduce the R² = 0.94 claim from a published paper,
then re-evaluate the same models under Pat Walters' 5×5 cross-validation
protocol to see whether the number holds up.

## Source

- **Paper**: Yuan, Y. et al. *A unified ML pipeline for odor prediction from
  molecular structure*. **Metabolites** 2025, 15, 747.
  https://www.mdpi.com/2218-1989/15/11/747
- **Code**: https://github.com/yuanhonglun/odor_prediction_models
- **Data + model artifacts**: Zenodo, DOI
  [10.5281/zenodo.17605258](https://doi.org/10.5281/zenodo.17605258)

## The two claims

The paper reports two fundamentally different R² numbers on the same model:

| Source | R² | Protocol |
|---|---|---|
| Supplementary Table 9 (text: "ECFP4–GBDT best, R²=0.94") | **0.9403** | single greedy scaffold 80/20 split, seed 42 |
| Supplementary Table 8 (5-fold CV on train) | **0.295 ± 0.198** | GroupKFold(5), scaffold-grouped |

Paragraph 2 of the paper uses the 0.94 number. The 0.30 number is buried in
a supplementary table. They measure different things.

## What this example does

Three scripts, no hyperparameter tuning (the paper's published best HP is
used verbatim for XGBoost), all running on MLX / Apple Silicon.

### `01_reproduce_paper_split.py`

Loads the paper's exact `train.csv` / `val.csv` from Zenodo
([saved here in `data/`](data/)) and trains:

1. **ECFP4-GBDT** with paper's best HP — exact reproduction of 0.94
2. ECFP6-GBDT, MACCS-GBDT — other FPs, same HP
3. **1-NN on ECFP4** — leakage probe (if 1-NN also hits 0.9, the split
   is interpolable, so the claimed R² isn't a sign of learned chemistry)
4. **predict-train-mean** — absolute floor
5. **ChemeleonSMD v5 + PCA(128) + TabICL-MLX**
6. **ChemeleonSMD v5 + EnsembleRandomProjection + TabICL-MLX**

### `02_cv5x5_honest.py`

The Pat Walters protocol:
  [blog post](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html)

5 seeds × 5-fold CV = **25 independent fits per model**, run under
**two** split regimes:

- `random`: `KFold(shuffle=True)` — optimistic (near-duplicates allowed in train/test)
- `scaffold`: `GroupKFold` on Murcko scaffold — honest generalization

All 8 models above run under both splits.

### `03_plot_pat_walters.py`

Tukey HSD significance-annotated boxplots:

- Boxplot + strip plot per method, ordered by mean
- Stars above pairs that are significantly different at α=0.05
- "Simultaneous CI" plot against the best method
- CSV summary (method × split → mean R², std R², mean RMSE, std RMSE, n=25)

## Results

### 1. Reproduction — the lucky split

On the paper's own 143-molecule validation set (our
`results/01_lucky_split_results.csv`):

| Model | R² | RMSE | MAE |
|---|---|---|---|
| ECFP4-GBDT (paper HP) | **+0.950** | 0.401 | 0.159 |
| ChemSMD + EnsembleRP + TabICL-MLX | +0.910 | 0.538 | 0.279 |
| ChemSMD + PCA(128) + TabICL-MLX | +0.897 | 0.578 | 0.334 |
| 1-NN ECFP4 (leakage probe) | +0.78 | 0.84 | 0.42 |
| predict-train-mean | −0.01 | 1.81 | 1.53 |

Our reproduction matches the paper (pred-to-pred correlation = 0.996). The
**1-NN on ECFP4 already hits R² ≈ 0.78** — confirming that VAL molecules
have very similar neighbors in TRAIN (greedy scaffold puts the biggest,
most homogeneous scaffolds into VAL).

### 2. Honest — 5 seeds × 5 folds

From `results/03_summary.csv`:

Random-split 5×5 CV (optimistic but honest):
- ECFP4-GBDT: R² ≈ +0.52, RMSE ≈ 0.99
- ChemSMD+EnsRP+TabICL-MLX: R² ≈ +0.52, RMSE ≈ 1.00
- all models statistically indistinguishable by Tukey HSD

Scaffold-grouped 5×5 CV (HONEST generalization):
- ECFP4-GBDT: R² ≈ **+0.29 ± 0.25**, RMSE ≈ 1.17
- ChemSMD+EnsRP+TabICL-MLX: R² ≈ **+0.29 ± 0.25**, RMSE ≈ 1.17
- both tie 1-NN ECFP4 and barely beat predict-mean

**This matches the paper's own Supplementary Table 8 number** (0.295 ± 0.198
for ECFP4-GBDT scaffold-grouped CV). The 0.94 is a one-split artifact.

## Run

```bash
cd TabICL-MLX
pip install -e .
pip install "tabicl-mlx[moleculeace]"  # for mlx-addons + ChemeleonSMD
pip install xgboost rdkit seaborn statsmodels statannotations

cd examples/odt_threshold
python 01_reproduce_paper_split.py     # ~10s
python 02_cv5x5_honest.py              # ~20-30 min on M3 Max
python 03_plot_pat_walters.py          # seconds — saves PDFs/PNGs to results/
```

## Why this matters for drug/odor discovery

The 0.94 headline implies "threshold prediction is solved" — a user
reading only the abstract would try to apply ECFP4-GBDT to new
scaffolds and get R² ≈ 0.29 in production. That's still useful (better
than predict-mean), but it's a completely different scientific claim.

The honest number also shows that **no current published model beats a
simple nearest-neighbor lookup on scaffold-held-out test sets** for ODT
prediction with ~716 molecules. Threshold prediction is hard because:

- The target spans 9 decades (2e-6 to 1.6e3 mg/L)
- Only 716 unique molecules across 435 Murcko scaffolds
- Most scaffolds are singletons → nothing to interpolate

To improve, you'd want more data, richer representations, or a proper
uncertainty-aware model — not a better single-split number.
