#!/usr/bin/env python3
"""Full MoleculeACE benchmark: ChemeleonSMD (v3-v6) + TabICL-MLX + PDL-SVR baseline.

Sweeps:
- ChemeleonSMD weight versions: v3, v4, v5, v6
- PCA dimensions: 96, 128, 160, 192
- Models: TabICL-MLX, PDL-SVR_lin (pairwise difference learning + linear SVR)

Global PCA fitted on ALL molecules (unsupervised — no leakage).

Run: python benchmark_moleculeace.py
"""

import argparse
import gc
import os
import sys
import time
import warnings
from glob import glob
from itertools import product

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlx.core as mx

try:
    _clear_cache = mx.clear_cache
except AttributeError:
    try:
        import mlx.core.metal as _metal
        _clear_cache = _metal.clear_cache
    except Exception:
        _clear_cache = lambda: None
from mlx_addons.decomposition import PCA  # Metal-accelerated: ~30x faster than sklearn.PCA on (35k, 2048)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.expanduser("~/Github/ChemeleonSMD"))
from chemeleon_smd.inference import load_model, fingerprint as chemeleon_fingerprint
from chemeleon_smd.inference import WEIGHTS_DIR

from tabicl_mlx import TabICLRegressorMLX

DATASETS_DIR = os.path.expanduser(
    "~/Github/tabpfn-molprop/datasets/MoleculeACE/Data/benchmark_data"
)

WEIGHT_VERSIONS = ["v3", "v4", "v5", "v6"]
PCA_DIMS = [96, 128, 160, 192]
RANDOM_SEED = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def compute_all_fingerprints(all_smiles, weight_version, batch_size=128):
    """Compute ChemeleonSMD fingerprints for a specific weight version.

    Streams per-batch results to host memory so Metal buffers don't
    accumulate across the ~50k-molecule corpus × 4 weight versions.
    """
    import chemeleon_smd.inference as inf
    inf._CACHED_MODEL = None  # force reload
    gc.collect(); _clear_cache()

    weights_path = WEIGHTS_DIR / f"score_dmpnn_distilled_{weight_version}.npz"
    model = load_model(weights_path)
    fps = chemeleon_fingerprint(all_smiles, model=model, batch_size=batch_size, as_numpy=True)
    del model
    gc.collect(); _clear_cache()
    return fps


def scale_and_clamp(train_feat, test_feat, train_y):
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1)).ravel()
    f_mean = train_feat.mean(axis=0, keepdims=True)
    f_std = train_feat.std(axis=0, keepdims=True) + 1e-8
    train_scaled = np.clip((train_feat - f_mean) / f_std, -6, 6)
    test_scaled = np.clip((test_feat - f_mean) / f_std, -6, 6)
    return train_scaled, test_scaled, y_scaled, y_scaler


# ── PDL-SVR_lin: Pairwise Difference Learning with Linear SVR ──────────

def pdl_svr_predict(train_feat, train_y, test_feat, n_anchors=50, seed=42):
    """Pairwise Difference Learning with Linear SVR.

    For each test sample, compute feature differences with anchor training
    samples, predict the label difference, and average.
    """
    rng = np.random.RandomState(seed)
    n_train = len(train_feat)
    n_anchors = min(n_anchors, n_train)

    # Select anchors
    anchor_idx = rng.choice(n_train, n_anchors, replace=False)
    X_anchor = train_feat[anchor_idx]
    y_anchor = train_y[anchor_idx]

    # Build pairwise training data: all pairs from train × anchors
    pairs_i, pairs_j = [], []
    for j in range(n_anchors):
        for i in range(n_train):
            if i != anchor_idx[j]:
                pairs_i.append(i)
                pairs_j.append(j)
    pairs_i = np.array(pairs_i)
    pairs_j = np.array(pairs_j)

    X_pairs = train_feat[pairs_i] - X_anchor[pairs_j]
    y_pairs = train_y[pairs_i] - y_anchor[pairs_j]

    # Train linear SVR on differences
    svr = LinearSVR(max_iter=5000, random_state=seed)
    svr.fit(X_pairs, y_pairs)

    # Predict: for each test sample, diff against all anchors, average
    predictions = np.zeros(len(test_feat))
    for k in range(len(test_feat)):
        diffs = test_feat[k:k+1] - X_anchor  # (n_anchors, d)
        deltas = svr.predict(diffs)  # (n_anchors,)
        predictions[k] = np.mean(y_anchor + deltas)

    return predictions


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoleculeACE sweep with TabICL-MLX (+ optional PDL-SVR)")
    parser.add_argument("--no-pdl", action="store_true",
                        help="Skip PDL-SVR baseline (runs only TabICL-MLX).")
    parser.add_argument("--out", type=str, default="moleculeace_results.csv")
    args = parser.parse_args()

    models_line = "TabICL-MLX" + ("" if args.no_pdl else " + PDL-SVR")
    print("=" * 100)
    print(f"  ChemeleonSMD (v3-v6) + {models_line} — Full MoleculeACE Sweep")
    print("  100% Apple Silicon MLX — zero PyTorch")
    print(f"  Weights: {WEIGHT_VERSIONS}  |  PCA: {PCA_DIMS}  |  Global PCA on all molecules")
    print("=" * 100)

    # ── Phase 1: Load all datasets ──
    csv_files = sorted(glob(f"{DATASETS_DIR}/*.csv"))
    benchmark_names = [os.path.basename(f).replace(".csv", "") for f in csv_files]
    print(f"\n  Phase 1: Loading {len(benchmark_names)} datasets...")

    datasets = {}
    all_smiles = []
    smiles_to_idx = {}

    for name in benchmark_names:
        df = pd.read_csv(f"{DATASETS_DIR}/{name}.csv")
        for smi in df["smiles"]:
            if smi not in smiles_to_idx:
                smiles_to_idx[smi] = len(all_smiles)
                all_smiles.append(smi)
        datasets[name] = df

    print(f"  Total unique molecules: {len(all_smiles)}")

    # ── Phase 2: Compute fingerprints for each weight version ──
    print(f"\n  Phase 2: Computing ChemeleonSMD fingerprints for each weight version...")
    all_fps = {}
    for wv in WEIGHT_VERSIONS:
        t0 = time.time()
        all_fps[wv] = compute_all_fingerprints(all_smiles, wv)
        dt = time.time() - t0
        print(f"    {wv}: {all_fps[wv].shape} in {dt:.1f}s")

    # ── Phase 3: Fit global PCA for each (version, dim) ──
    print(f"\n  Phase 3: Fitting global PCA models...")
    pca_models = {}
    for wv in WEIGHT_VERSIONS:
        for pdim in PCA_DIMS:
            pca = PCA(n_components=pdim, random_state=RANDOM_SEED)
            pca.fit(all_fps[wv])
            ev = pca.explained_variance_ratio_.sum()
            pca_models[(wv, pdim)] = pca
            print(f"    {wv} PCA {pdim:>4}: {ev:.2%} var", end="  ")
        print()

    # ── Phase 4: Run benchmarks ──
    print(f"\n  Phase 4: Running benchmarks ({len(benchmark_names)} datasets × "
          f"{len(WEIGHT_VERSIONS)} weights × {len(PCA_DIMS)} PCA × 2 models)...")

    all_results = []

    for i, benchmark_name in enumerate(benchmark_names):
        df = datasets[benchmark_name]
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]
        train_y = train_df["y"].to_numpy().astype(np.float32)
        test_y = test_df["y"].to_numpy().astype(np.float32)
        cliff_mol = test_df["cliff_mol"].to_numpy()
        n_train, n_test = len(train_df), len(test_df)

        train_idxs = [smiles_to_idx[s] for s in train_df["smiles"]]
        test_idxs = [smiles_to_idx[s] for s in test_df["smiles"]]

        print(f"\n[{i+1}/{len(benchmark_names)}] {benchmark_name} (n={n_train}+{n_test})")

        for wv in WEIGHT_VERSIONS:
            train_fps = all_fps[wv][train_idxs]
            test_fps = all_fps[wv][test_idxs]

            for pdim in PCA_DIMS:
                pca = pca_models[(wv, pdim)]
                train_feat = pca.transform(train_fps)
                test_feat = pca.transform(test_fps)

                train_scaled, test_scaled, y_scaled, y_scaler = scale_and_clamp(
                    train_feat, test_feat, train_y
                )

                # ── TabICL-MLX ──
                try:
                    model = TabICLRegressorMLX(
                        n_estimators=8, batch_size=1, random_state=RANDOM_SEED,
                        low_memory=(pdim >= 192),
                    )
                    t0 = time.time()
                    model.fit(train_scaled, y_scaled)
                    pred_scaled = model.predict(test_scaled).flatten()
                    dt = time.time() - t0
                    del model
                    _clear_cache(); gc.collect()
                    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

                    o = rmse(test_y, pred)
                    c = rmse(test_y[cliff_mol == 1], pred[cliff_mol == 1]) if cliff_mol.sum() > 0 else np.nan
                    n = rmse(test_y[cliff_mol == 0], pred[cliff_mol == 0]) if (cliff_mol == 0).sum() > 0 else np.nan
                    print(f"  {wv}/d{pdim}/ICL: {o:.3f} ({dt:.0f}s)", end="", flush=True)
                except Exception as e:
                    o = c = n = dt = np.nan
                    print(f"  {wv}/d{pdim}/ICL: ERR", end="", flush=True)

                all_results.append({
                    "benchmark": benchmark_name, "n_train": n_train, "n_test": n_test,
                    "weight": wv, "pca_dim": pdim, "model": "TabICL-MLX",
                    "overall_rmse": o, "cliff_rmse": c, "noncliff_rmse": n, "time_s": dt,
                })

                # ── PDL-SVR_lin (optional) ──
                if not args.no_pdl:
                    t0 = time.time()
                    pred_pdl = pdl_svr_predict(train_scaled, y_scaled, test_scaled, seed=RANDOM_SEED)
                    dt_pdl = time.time() - t0
                    pred_pdl_inv = y_scaler.inverse_transform(pred_pdl.reshape(-1, 1)).ravel()

                    o_pdl = rmse(test_y, pred_pdl_inv)
                    c_pdl = rmse(test_y[cliff_mol == 1], pred_pdl_inv[cliff_mol == 1]) if cliff_mol.sum() > 0 else np.nan
                    n_pdl = rmse(test_y[cliff_mol == 0], pred_pdl_inv[cliff_mol == 0]) if (cliff_mol == 0).sum() > 0 else np.nan

                    all_results.append({
                        "benchmark": benchmark_name, "n_train": n_train, "n_test": n_test,
                        "weight": wv, "pca_dim": pdim, "model": "PDL-SVR",
                        "overall_rmse": o_pdl, "cliff_rmse": c_pdl, "noncliff_rmse": n_pdl, "time_s": dt_pdl,
                    })

        print()

    # ── Summary ──
    df_all = pd.DataFrame(all_results)

    print(f"\n{'=' * 100}")
    print("  CROSS-COMPARISON: Mean RMSE across 30 datasets")
    print(f"{'=' * 100}")

    if args.no_pdl:
        # TabICL-only table
        for metric_name, metric_col in [("Overall RMSE", "overall_rmse"),
                                          ("Cliff RMSE", "cliff_rmse"),
                                          ("NonCliff RMSE", "noncliff_rmse")]:
            print(f"\n  {metric_name}:")
            print(f"  {'Weight':>6} {'PCA':>5}  {'TabICL-MLX':>11}")
            print(f"  {'─' * 30}")
            for wv in WEIGHT_VERSIONS:
                for pdim in PCA_DIMS:
                    sub_icl = df_all[(df_all["weight"] == wv) & (df_all["pca_dim"] == pdim) &
                                     (df_all["model"] == "TabICL-MLX")]
                    print(f"  {wv:>6} {pdim:>5}  {sub_icl[metric_col].mean():>11.4f}")
                print()
    else:
        for metric_name, metric_col in [("Overall RMSE", "overall_rmse"),
                                          ("Cliff RMSE", "cliff_rmse"),
                                          ("NonCliff RMSE", "noncliff_rmse")]:
            print(f"\n  {metric_name}:")
            print(f"  {'Weight':>6} {'PCA':>5}  {'TabICL-MLX':>11} {'PDL-SVR':>9} {'Winner':>8}")
            print(f"  {'─' * 50}")
            for wv in WEIGHT_VERSIONS:
                for pdim in PCA_DIMS:
                    sub_icl = df_all[(df_all["weight"] == wv) & (df_all["pca_dim"] == pdim) &
                                     (df_all["model"] == "TabICL-MLX")]
                    sub_svr = df_all[(df_all["weight"] == wv) & (df_all["pca_dim"] == pdim) &
                                     (df_all["model"] == "PDL-SVR")]
                    m_icl = sub_icl[metric_col].mean()
                    m_svr = sub_svr[metric_col].mean()
                    winner = "ICL" if m_icl < m_svr else "SVR"
                    print(f"  {wv:>6} {pdim:>5}  {m_icl:>11.4f} {m_svr:>9.4f} {'◀'+winner:>8}")
                print()

    # Best config
    print(f"\n{'=' * 100}")
    print("  BEST CONFIGS (lowest mean overall RMSE)")
    print(f"{'=' * 100}")
    summary = df_all.groupby(["weight", "pca_dim", "model"]).agg(
        mean_overall=("overall_rmse", "mean"),
        mean_cliff=("cliff_rmse", "mean"),
        mean_noncliff=("noncliff_rmse", "mean"),
        mean_time=("time_s", "mean"),
        n_ok=("overall_rmse", "count"),
    ).reset_index().sort_values("mean_overall")

    print(f"\n  {'Rank':>4} {'Weight':>6} {'PCA':>5} {'Model':<12} {'RMSE':>7} {'Cliff':>7} {'NonCl':>7} {'Time':>6}")
    print(f"  {'─' * 62}")
    for rank, (_, r) in enumerate(summary.head(20).iterrows(), 1):
        print(f"  {rank:>4} {r['weight']:>6} {r['pca_dim']:>5.0f} {r['model']:<12} "
              f"{r['mean_overall']:7.4f} {r['mean_cliff']:7.4f} {r['mean_noncliff']:7.4f} {r['mean_time']:5.1f}s")

    # Save
    df_all.to_csv(args.out, index=False)
    print(f"\n  Raw results saved to {args.out}")


if __name__ == "__main__":
    main()
