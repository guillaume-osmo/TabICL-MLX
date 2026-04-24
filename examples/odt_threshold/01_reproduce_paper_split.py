#!/usr/bin/env python3
"""01 — Reproduce the paper's R²=0.94 claim on its exact VAL split.

Paper: Yuan et al., Metabolites 2025, 15, 747 (MDPI)
       https://www.mdpi.com/2218-1989/15/11/747
Code:  https://github.com/yuanhonglun/odor_prediction_models
Data:  https://zenodo.org/records/17605258

Protocol (from the paper):
  1. Greedy Murcko-scaffold split (80/20, seed=42) → train.csv, val.csv
  2. RandomizedSearchCV(cv=GroupKFold(5)) on train to pick HP
  3. Refit best HP on full train, predict on val

The VAL split is the LUCKY one (largest scaffold families go to VAL,
homogeneous in y, easy to interpolate) — that's why R²=0.94 on VAL but
only R² ≈ 0.30 on the paper's own 5-fold CV on train.

We run IN PARALLEL:
  • ECFP4-GBDT with paper's best HP (exact reproduction)
  • ChemeleonSMD v5 + PCA(128) + TabICL-MLX
  • ChemeleonSMD v5 + EnsembleRandomProjection + TabICL-MLX
  • 1-NN baseline (scaffold leakage sanity check)
  • Predict-train-mean baseline (absolute floor)

Writes: results/01_lucky_split_results.csv
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.MACCSkeys import GenMACCSKeys
RDLogger.DisableLog("rdApp.*")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Github/ChemeleonSMD"))
from chemeleon_smd.inference import load_model, fingerprint as chemeleon_fingerprint, WEIGHTS_DIR
from mlx_addons.decomposition import PCA as MXPCA, EnsembleRandomProjection, ensemble_mean_predict
from tabicl_mlx import TabICLRegressorMLX

try:
    _clear_cache = mx.clear_cache
except AttributeError:
    _clear_cache = lambda: None

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

NBITS = 1024
SEED = 42


def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5


def metrics(y_true, y_pred):
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


# ── Features ──────────────────────────────────────────────────────

def build_ref_features(smiles_list, fp_kind="ECFP4", nbits=NBITS):
    radius = {"ECFP4": 2, "ECFP6": 3}
    size = nbits if fp_kind in radius else 167
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(np.zeros(size + 4, dtype=np.float32)); continue
        if fp_kind == "MACCS":
            bv = GenMACCSKeys(mol); bits = np.zeros(167, dtype=np.int8)
        else:
            bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius[fp_kind], nBits=nbits)
            bits = np.zeros(nbits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, bits)
        phys = np.array([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol),
            Descriptors.TPSA(mol), Descriptors.MolMR(mol),
        ], dtype=np.float32)
        out.append(np.concatenate([bits.astype(np.float32), phys]))
    return np.vstack(out)


def compute_chemeleon(smiles_list, weight_version="v5"):
    import chemeleon_smd.inference as inf
    inf._CACHED_MODEL = None
    gc.collect(); _clear_cache()
    model = load_model(WEIGHTS_DIR / f"score_dmpnn_distilled_{weight_version}.npz")
    fps_mx = chemeleon_fingerprint(smiles_list, model=model, batch_size=128)
    mx.eval(fps_mx); fps = np.array(fps_mx)
    del model, fps_mx; gc.collect(); _clear_cache()
    return fps


# ── Models ────────────────────────────────────────────────────────

def paper_best_gbdt():
    """ECFP4-GBDT with HP from Zenodo hpo_ECFP4_gbdt_best.json."""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600, max_depth=8, learning_rate=0.1,
        subsample=0.7, colsample_bytree=0.8, reg_lambda=0.0,
        random_state=SEED, n_jobs=-1, verbosity=0,
    )


def tabicl_fit_predict(Xtr, ytr, Xte, seed=SEED, n_est=8):
    f_mean = Xtr.mean(0, keepdims=True); f_std = Xtr.std(0, keepdims=True) + 1e-8
    Xtr_s = np.clip((Xtr - f_mean) / f_std, -6, 6).astype(np.float32)
    Xte_s = np.clip((Xte - f_mean) / f_std, -6, 6).astype(np.float32)
    ys = StandardScaler(); ytr_s = ys.fit_transform(ytr.reshape(-1, 1)).ravel().astype(np.float32)
    m = TabICLRegressorMLX(n_estimators=n_est, batch_size=1, random_state=seed)
    m.fit(Xtr_s, ytr_s)
    pred = ys.inverse_transform(m.predict(Xte_s).flatten().reshape(-1, 1)).ravel()
    del m, Xtr_s, Xte_s; _clear_cache(); gc.collect()
    return pred


def main():
    print("=" * 92)
    print(" 01 — Reproducing the paper's LUCKY validation split")
    print(" Dataset: examples/odt_threshold/data/paper_{train,val}.csv (from Zenodo)")
    print("=" * 92)

    train_df = pd.read_csv(DATA / "paper_train.csv")
    val_df   = pd.read_csv(DATA / "paper_val.csv")
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")
    print(f"Train y: [{train_df['y'].min():.2f}, {train_df['y'].max():.2f}] mean={train_df['y'].mean():+.2f}")
    print(f"Val y:   [{val_df['y'].min():.2f}, {val_df['y'].max():.2f}] mean={val_df['y'].mean():+.2f}")

    smi_tr, smi_va = train_df["SMILES"].tolist(), val_df["SMILES"].tolist()
    y_tr = train_df["y"].to_numpy(dtype=np.float32)
    y_va = val_df["y"].to_numpy(dtype=np.float32)

    rows = []

    # ── Baseline: predict train mean ──
    yhat = np.full_like(y_va, y_tr.mean())
    rows.append({"Model": "Baseline: predict train mean", **metrics(y_va, yhat)})

    # ── Baseline: 1-NN on ECFP4 (measures raw scaffold-leakage signal) ──
    t0 = time.time()
    X_tr_ref = build_ref_features(smi_tr, "ECFP4")
    X_va_ref = build_ref_features(smi_va, "ECFP4")
    knn = KNeighborsRegressor(n_neighbors=1).fit(X_tr_ref, y_tr)
    yhat = knn.predict(X_va_ref)
    rows.append({"Model": "1-NN ECFP4 (leakage probe)", **metrics(y_va, yhat), "time_s": time.time() - t0})

    # ── Paper's best: ECFP4-GBDT ──
    t0 = time.time()
    gbdt = paper_best_gbdt().fit(X_tr_ref, y_tr)
    yhat_gbdt = gbdt.predict(X_va_ref)
    rows.append({"Model": "ECFP4-GBDT (paper best HP)", **metrics(y_va, yhat_gbdt), "time_s": time.time() - t0})

    # ── ECFP6-GBDT & MACCS-GBDT (paper's other FPs, same HP) ──
    for fp in ["ECFP6", "MACCS"]:
        t0 = time.time()
        X_tr = build_ref_features(smi_tr, fp)
        X_va = build_ref_features(smi_va, fp)
        m = paper_best_gbdt().fit(X_tr, y_tr)
        yhat = m.predict(X_va)
        rows.append({"Model": f"{fp}-GBDT (paper HP)", **metrics(y_va, yhat), "time_s": time.time() - t0})

    # ── ChemeleonSMD + TabICL-MLX ──
    t0 = time.time()
    fps_tr = compute_chemeleon(smi_tr)
    fps_va = compute_chemeleon(smi_va)
    # Global (unsupervised) feature maps on train+val combined
    all_fps = np.vstack([fps_tr, fps_va])
    pca_g = MXPCA(n_components=128, random_state=SEED).fit(all_fps)
    ens_g = EnsembleRandomProjection(
        n_components=128, n_pca=1, n_sparse=2, n_gaussian=2, random_state=SEED,
    ).fit(all_fps)
    del all_fps
    fp_time = time.time() - t0

    t0 = time.time()
    yhat_pca = tabicl_fit_predict(pca_g.transform(fps_tr), y_tr, pca_g.transform(fps_va))
    rows.append({"Model": "ChemSMD + PCA(128) + TabICL-MLX",
                 **metrics(y_va, yhat_pca), "time_s": time.time() - t0})

    t0 = time.time()
    yhat_ens = ensemble_mean_predict(ens_g, tabicl_fit_predict, fps_tr, y_tr, fps_va)
    rows.append({"Model": "ChemSMD + EnsembleRP + TabICL-MLX",
                 **metrics(y_va, yhat_ens), "time_s": time.time() - t0})

    # ── Save + print ──
    df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    out = RESULTS / "01_lucky_split_results.csv"
    df.to_csv(out, index=False)

    print("\n" + "─" * 92)
    print(" LUCKY-SPLIT RESULTS (paper's exact validation split)")
    print("─" * 92)
    print(f"\n  {'Model':<40} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "─" * 67)
    for _, r in df.iterrows():
        print(f"  {r['Model']:<40} {r['R2']:>+8.4f} {r['RMSE']:>8.4f} {r['MAE']:>8.4f}")

    print(f"\n  Paper-reported (Table 9):                "
          f"+0.9403 0.4394 (ECFP4-GBDT)")
    print(f"  Fingerprint compute time: {fp_time:.1f}s")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
