#!/usr/bin/env python3
"""02 — Honest benchmark: 5-fold × 5-seed CV on the full deduped dataset.

Pat Walters' benchmark protocol (practicalcheminformatics.blogspot.com,
March 2025): 5 random seeds × 5-fold CV = 25 independent fits per model,
so each method has 25 R² / RMSE samples. We then compute Tukey's HSD for
pairwise significance.

Dataset: examples/odt_threshold/data/threshold_data.csv
Cleanup: same as the paper (canonicalize, dedup by geometric mean, drop NaN/≤0)
Target:  y = -log10(threshold)

Splits:
  • 'random'   — KFold(shuffle=True) — optimistic (no scaffold constraint)
  • 'scaffold' — GroupKFold on Murcko scaffold — honest generalization

Models:
  1. ECFP4-GBDT (paper's best HP)
  2. ECFP6-GBDT (paper HP)
  3. MACCS-GBDT (paper HP)
  4. RF on ECFP4
  5. 1-NN on ECFP4 (leakage probe)
  6. Predict train mean (floor)
  7. ChemSMD v5 + PCA(128) + TabICL-MLX
  8. ChemSMD v5 + EnsembleRP(1+2+2) + TabICL-MLX

Output: results/02_cv5x5_results.csv   (long format: seed, fold, split, method, r2, rmse)

~20-30 min runtime on M3 Max.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

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
N_FOLDS = 5
N_SEEDS = 5


def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5


def canonical_smiles(s):
    if not isinstance(s, str): return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def scaffold_of(s):
    mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if mol is None: return ""
    sc = MurckoScaffold.GetScaffoldForMol(mol)
    if sc is None or sc.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return Chem.MolToSmiles(sc, isomericSmiles=True)


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


def paper_gbdt(seed):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600, max_depth=8, learning_rate=0.1,
        subsample=0.7, colsample_bytree=0.8, reg_lambda=0.0,
        random_state=seed, n_jobs=-1, verbosity=0,
    )


def paper_rf(seed):
    return RandomForestRegressor(
        n_estimators=600, max_depth=30, max_features="sqrt",
        n_jobs=-1, random_state=seed,
    )


def tabicl_fit_predict(Xtr, ytr, Xte, seed=42, n_est=8):
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
    print(" 02 — Honest 5-fold × 5-seed CV (Pat Walters protocol)")
    print("=" * 92)

    df = pd.read_csv(DATA / "threshold_data.csv")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df = df.dropna(subset=["SMILES", "threshold"])
    df = df[df["threshold"] > 0].copy()
    df["SMILES"] = df["SMILES"].apply(canonical_smiles)
    df = df[~df["SMILES"].isna()]
    df = df.groupby("SMILES", as_index=False)["threshold"].apply(
        lambda s: 10 ** (-np.mean(-np.log10(s.values)))
    )
    df["y"] = -np.log10(df["threshold"])
    smiles = df["SMILES"].tolist()
    y = df["y"].to_numpy(dtype=np.float32)
    scaffolds = np.array([scaffold_of(s) for s in smiles])
    print(f"  N molecules: {len(smiles)},  unique scaffolds: {len(set(scaffolds))}")

    # ── Build feature sets once ──
    print("\n  Computing ECFP4, ECFP6, MACCS + physchem ...")
    X_ecfp4 = build_ref_features(smiles, "ECFP4")
    X_ecfp6 = build_ref_features(smiles, "ECFP6")
    X_maccs = build_ref_features(smiles, "MACCS")
    print(f"  ECFP4 {X_ecfp4.shape}, ECFP6 {X_ecfp6.shape}, MACCS {X_maccs.shape}")

    print("\n  Computing ChemeleonSMD v5 ...")
    fps = compute_chemeleon(smiles, "v5")
    print(f"  ChemSMD {fps.shape}")

    # Global feature maps (unsupervised, no label leakage)
    pca_g = MXPCA(n_components=128, random_state=0).fit(fps)
    ens_g = EnsembleRandomProjection(
        n_components=128, n_pca=1, n_sparse=2, n_gaussian=2, random_state=0,
    ).fit(fps)

    # ── Model wrappers ──
    def m_baseline_mean(Xtr, ytr, Xte, seed, tr_fp, te_fp):
        return np.full(len(te_fp), ytr.mean())

    def m_knn(Xtr, ytr, Xte, seed, *extra):
        return KNeighborsRegressor(n_neighbors=1).fit(Xtr, ytr).predict(Xte)

    def m_gbdt(Xtr, ytr, Xte, seed, *extra):
        return paper_gbdt(seed).fit(Xtr, ytr).predict(Xte)

    def m_rf(Xtr, ytr, Xte, seed, *extra):
        return paper_rf(seed).fit(Xtr, ytr).predict(Xte)

    def m_tabicl_pca(_Xtr, ytr, _Xte, seed, tr_fp, te_fp):
        return tabicl_fit_predict(pca_g.transform(tr_fp), ytr, pca_g.transform(te_fp), seed=seed)

    def m_tabicl_ens(_Xtr, ytr, _Xte, seed, tr_fp, te_fp):
        return ensemble_mean_predict(
            ens_g,
            lambda Ztr, ytr_, Zte: tabicl_fit_predict(Ztr, ytr_, Zte, seed=seed),
            tr_fp, ytr, te_fp,
        )

    # list: (label, X (or None for fp-only), needs_fp, model_fn)
    MODELS = [
        ("Predict-train-mean",            None,    False, m_baseline_mean),
        ("1-NN ECFP4",                    X_ecfp4, False, m_knn),
        ("ECFP4-GBDT",                    X_ecfp4, False, m_gbdt),
        ("ECFP6-GBDT",                    X_ecfp6, False, m_gbdt),
        ("MACCS-GBDT",                    X_maccs, False, m_gbdt),
        ("ECFP4-RF",                      X_ecfp4, False, m_rf),
        ("ChemSMD+PCA+TabICL",            None,    True,  m_tabicl_pca),
        ("ChemSMD+EnsRP+TabICL",          None,    True,  m_tabicl_ens),
    ]

    rows = []
    t_total = time.time()

    for split_kind in ["random", "scaffold"]:
        for seed in range(N_SEEDS):
            print(f"\n  — split={split_kind!s:<9}  seed={seed}")

            if split_kind == "random":
                splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(smiles))
            else:
                # GroupKFold doesn't accept seed; emulate by shuffling scaffold order
                rng = np.random.RandomState(seed)
                uniq = np.array(sorted(set(scaffolds)))
                rng.shuffle(uniq)
                # remap: use shuffled-scaffold-order as "fake" group labels
                remap = {s: i for i, s in enumerate(uniq)}
                groups = np.array([remap[s] for s in scaffolds])
                splits = list(GroupKFold(n_splits=N_FOLDS).split(smiles, y, groups=groups))

            for fold, (tr, te) in enumerate(splits):
                ytr, yte = y[tr], y[te]
                fp_tr, fp_te = fps[tr], fps[te]

                for label, X, needs_fp, fn in MODELS:
                    t0 = time.time()
                    if needs_fp:
                        Xtr = Xte = None  # model uses fp_tr/fp_te directly
                    else:
                        Xtr = X[tr] if X is not None else None
                        Xte = X[te] if X is not None else None
                    try:
                        yhat = fn(Xtr, ytr, Xte, seed, fp_tr, fp_te)
                        r2 = r2_score(yte, yhat)
                        rm = rmse(yte, yhat)
                    except Exception as e:
                        r2, rm = np.nan, np.nan
                        print(f"      [{label}] FAIL {e}")
                    dt = time.time() - t0
                    rows.append({
                        "split": split_kind, "seed": seed, "fold": fold,
                        "method": label, "r2": r2, "rmse": rm, "time_s": dt,
                    })
                print(f"    fold {fold+1}/{N_FOLDS} done")

    df_out = pd.DataFrame(rows)
    out = RESULTS / "02_cv5x5_results.csv"
    df_out.to_csv(out, index=False)

    # ── Print concise summary ──
    print("\n" + "=" * 92)
    print(" SUMMARY (mean ± std across 5 seeds × 5 folds = 25 runs)")
    print("=" * 92)
    for split_kind in ["random", "scaffold"]:
        sub = df_out[df_out["split"] == split_kind].groupby("method").agg(
            r2_mean=("r2", "mean"), r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        ).sort_values("r2_mean", ascending=False)
        print(f"\n  split = {split_kind}")
        print(f"  {'Method':<24} {'R² mean±std':>18} {'RMSE mean±std':>18}")
        print("  " + "─" * 62)
        for method, r in sub.iterrows():
            print(f"  {method:<24} {r['r2_mean']:+7.4f}±{r['r2_std']:.3f}  "
                  f"{r['rmse_mean']:7.4f}±{r['rmse_std']:.3f}")

    print(f"\n  Total time: {(time.time() - t_total)/60:.1f} min")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
