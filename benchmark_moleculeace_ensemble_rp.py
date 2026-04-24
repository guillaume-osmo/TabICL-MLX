#!/usr/bin/env python3
"""MoleculeACE sweep with EnsembleRandomProjection (PCA + SRP + GRP mix) + TabICL-MLX.

Reproduces the current best result on MoleculeACE with ChemeleonSMD
features: 1 PCA + 2 SparseRP + 2 GaussianRP, averaged across 5 feature
maps, TabICL-MLX as the regressor.

    Mean Overall RMSE  0.6313   (published leaderboard: 0.6337)
    Mean Cliff   RMSE  0.7458   (published leaderboard: 0.7508)
    Mean NonCliff RMSE 0.5675   (published leaderboard: 0.5683)

All three metrics beat the single-PCA baseline and the published number.

Requires ``pip install mlx-addons`` (see https://github.com/guillaume-osmo/mlx-addons).

Run: python benchmark_moleculeace_ensemble_rp.py
"""

import argparse
import gc
import os
import sys
import time
import warnings
from glob import glob

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlx.core as mx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.expanduser("~/Github/ChemeleonSMD"))
from chemeleon_smd.inference import load_model, fingerprint as chemeleon_fingerprint
from chemeleon_smd.inference import WEIGHTS_DIR

from mlx_addons.decomposition import (
    PCA,
    EnsembleRandomProjection,
    ensemble_mean_predict,
)
from tabicl_mlx import TabICLRegressorMLX

try:
    _clear_cache = mx.clear_cache
except AttributeError:
    try:
        import mlx.core.metal as _metal
        _clear_cache = _metal.clear_cache
    except Exception:
        _clear_cache = lambda: None

DATASETS_DIR = os.path.expanduser(
    "~/Github/tabpfn-molprop/datasets/MoleculeACE/Data/benchmark_data"
)
WEIGHT_DEFAULT = "v5"
PCA_DEFAULT = 128
N_EST_DEFAULT = 8
RANDOM_SEED = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def compute_all_fingerprints(all_smiles, weight_version, batch_size=128):
    import chemeleon_smd.inference as inf
    inf._CACHED_MODEL = None
    gc.collect(); _clear_cache()
    model = load_model(WEIGHTS_DIR / f"score_dmpnn_distilled_{weight_version}.npz")
    fps = chemeleon_fingerprint(all_smiles, model=model, batch_size=batch_size, as_numpy=True)
    del model; gc.collect(); _clear_cache()
    return fps


def tabicl_fit_predict(Xtr, ytr, Xte, *, n_estimators, random_state):
    f_mean = Xtr.mean(axis=0, keepdims=True)
    f_std = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr_s = np.clip((Xtr - f_mean) / f_std, -6, 6).astype(np.float32)
    Xte_s = np.clip((Xte - f_mean) / f_std, -6, 6).astype(np.float32)
    ys = StandardScaler(); ytr_s = ys.fit_transform(ytr.reshape(-1, 1)).ravel()
    m = TabICLRegressorMLX(n_estimators=n_estimators, batch_size=1, random_state=random_state)
    m.fit(Xtr_s, ytr_s); pred_s = m.predict(Xte_s).flatten()
    pred = ys.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    del m; _clear_cache(); gc.collect()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default=WEIGHT_DEFAULT, help="ChemeleonSMD version (v3-v6)")
    parser.add_argument("--pca-dim", type=int, default=PCA_DEFAULT)
    parser.add_argument("--n-estimators", type=int, default=N_EST_DEFAULT)
    parser.add_argument("--n-pca", type=int, default=1)
    parser.add_argument("--n-sparse", type=int, default=2)
    parser.add_argument("--n-gaussian", type=int, default=2)
    parser.add_argument("--out", default="moleculeace_results_ensemble_rp.csv")
    args = parser.parse_args()

    print("=" * 100)
    print(f"  MoleculeACE × EnsembleRandomProjection × TabICL-MLX")
    print(f"  Recipe: {args.n_pca} PCA + {args.n_sparse} SparseRP + {args.n_gaussian} GaussianRP")
    print(f"  ChemeleonSMD {args.weight}  |  k={args.pca_dim}  |  n_estimators={args.n_estimators}")
    print("=" * 100)

    csv_files = sorted(glob(f"{DATASETS_DIR}/*.csv"))
    bench_names = [os.path.basename(f).replace(".csv", "") for f in csv_files]
    print(f"\n  Phase 1: Loading {len(bench_names)} datasets...")
    datasets = {}
    all_smiles = []
    smiles_to_idx = {}
    for name in bench_names:
        df = pd.read_csv(f"{DATASETS_DIR}/{name}.csv")
        for smi in df["smiles"]:
            if smi not in smiles_to_idx:
                smiles_to_idx[smi] = len(all_smiles)
                all_smiles.append(smi)
        datasets[name] = df
    print(f"  Total unique molecules: {len(all_smiles)}")

    print(f"\n  Phase 2: Computing ChemeleonSMD {args.weight} fingerprints...")
    t0 = time.time()
    fps = compute_all_fingerprints(all_smiles, args.weight)
    print(f"    {fps.shape} in {time.time() - t0:.1f}s")

    print(f"\n  Phase 3: Fitting PCA + EnsembleRandomProjection globally on all molecules...")
    t0 = time.time()
    pca_global = PCA(n_components=args.pca_dim, random_state=RANDOM_SEED).fit(fps)
    ens_global = EnsembleRandomProjection(
        n_components=args.pca_dim,
        n_pca=args.n_pca, n_sparse=args.n_sparse, n_gaussian=args.n_gaussian,
        random_state=RANDOM_SEED,
    ).fit(fps)
    print(f"    PCA + EnsembleRP fitted in {time.time() - t0:.1f}s")

    print(f"\n  Phase 4: Running sweep ({len(bench_names)} targets)...")
    header = f"{'Dataset':<22} {'nTr':>5} {'nTe':>4}  {'PCA':>26}  {'Ensemble':>26}"
    print(header)
    print(f"{'':<22} {'':>5} {'':>4}  {'RMSE    Cliff   NonCl':>26}  {'RMSE    Cliff   NonCl':>26}")
    print("─" * len(header))

    rows = []
    t_sweep = time.time()
    for i, name in enumerate(bench_names):
        df = datasets[name]
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        y_tr = tr["y"].to_numpy().astype(np.float32)
        y_te = te["y"].to_numpy().astype(np.float32)
        cliff = te["cliff_mol"].to_numpy()
        tr_idx = [smiles_to_idx[s] for s in tr["smiles"]]
        te_idx = [smiles_to_idx[s] for s in te["smiles"]]
        X_tr_raw = fps[tr_idx]
        X_te_raw = fps[te_idx]

        # PCA baseline
        p_pca = tabicl_fit_predict(
            pca_global.transform(X_tr_raw), y_tr, pca_global.transform(X_te_raw),
            n_estimators=args.n_estimators, random_state=RANDOM_SEED,
        )

        # Ensemble RP
        def fp(Ztr, ytr, Zte):
            return tabicl_fit_predict(
                Ztr, ytr, Zte,
                n_estimators=args.n_estimators, random_state=RANDOM_SEED,
            )
        p_ens = ensemble_mean_predict(ens_global, fp, X_tr_raw, y_tr, X_te_raw)

        for lbl, p in [("PCA", p_pca), ("Ensemble", p_ens)]:
            o = rmse(y_te, p)
            c = rmse(y_te[cliff == 1], p[cliff == 1]) if cliff.sum() > 0 else np.nan
            n = rmse(y_te[cliff == 0], p[cliff == 0]) if (cliff == 0).sum() > 0 else np.nan
            rows.append({
                "benchmark": name, "n_train": len(tr), "n_test": len(te),
                "method": lbl, "overall_rmse": o, "cliff_rmse": c, "noncliff_rmse": n,
            })

        pca_r = next(r for r in rows[-2:] if r["method"] == "PCA")
        ens_r = next(r for r in rows[-2:] if r["method"] == "Ensemble")
        print(
            f"{name:<22} {len(tr):>5} {len(te):>4}  "
            f"{pca_r['overall_rmse']:.4f}  {pca_r['cliff_rmse']:.4f}  {pca_r['noncliff_rmse']:.4f}  "
            f"{ens_r['overall_rmse']:.4f}  {ens_r['cliff_rmse']:.4f}  {ens_r['noncliff_rmse']:.4f}"
        )

        # Save progress every 5 datasets so partial runs are recoverable
        if (i + 1) % 5 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    print("─" * len(header))
    print(f"\n  Sweep took {time.time() - t_sweep:.0f}s")

    df_all = pd.DataFrame(rows)
    df_all.to_csv(args.out, index=False)
    print(f"  Raw results saved to {args.out}")

    print(f"\n{'=' * 100}")
    print("  MEAN RMSE across 30 datasets")
    print(f"{'=' * 100}")
    print(f"\n  {'Method':<12} {'Overall':>9} {'Cliff':>9} {'NonCliff':>9}")
    print(f"  {'─' * 42}")
    for lbl in ["PCA", "Ensemble"]:
        sub = df_all[df_all["method"] == lbl]
        print(f"  {lbl:<12} {sub['overall_rmse'].mean():9.4f} {sub['cliff_rmse'].mean():9.4f} {sub['noncliff_rmse'].mean():9.4f}")

    d_o = df_all[df_all.method == "Ensemble"].overall_rmse.mean() - df_all[df_all.method == "PCA"].overall_rmse.mean()
    d_c = df_all[df_all.method == "Ensemble"].cliff_rmse.mean() - df_all[df_all.method == "PCA"].cliff_rmse.mean()
    d_n = df_all[df_all.method == "Ensemble"].noncliff_rmse.mean() - df_all[df_all.method == "PCA"].noncliff_rmse.mean()
    print(f"  {'Δ vs PCA':<12} {d_o:+9.4f} {d_c:+9.4f} {d_n:+9.4f}")

    print(f"\n  Published TabICL-MLX baseline: Overall=0.6337  Cliff=0.7508  NonCliff=0.5683")


if __name__ == "__main__":
    main()
