#!/usr/bin/env python3
"""MoleculeACE benchmark for TabPFN v2.6 (regressor).

Mirrors benchmark_moleculeace.py but uses TabPFNRegressor with a local
checkpoint. Runs on MPS when available, CPU otherwise.

Run:
    python benchmark_moleculeace_tabpfn.py \
        --ckpt /Users/tgg/Downloads/TabPFN2.6/tabpfn-v2.6-regressor-v2.6_default.ckpt
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.expanduser("~/Github/ChemeleonSMD"))
from chemeleon_smd.inference import load_model, fingerprint as chemeleon_fingerprint
from chemeleon_smd.inference import WEIGHTS_DIR

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

WEIGHT_VERSIONS = ["v3", "v4", "v5", "v6"]
PCA_DIMS = [96, 128, 160, 192]
RANDOM_SEED = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def compute_all_fingerprints(all_smiles, weight_version, batch_size=128):
    import chemeleon_smd.inference as inf
    inf._CACHED_MODEL = None
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to tabpfn-v2.6-regressor-*.ckpt")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | mps | cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--weights", nargs="+", default=WEIGHT_VERSIONS)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=PCA_DIMS)
    parser.add_argument("--out", type=str, default="moleculeace_results_tabpfn.csv")
    args = parser.parse_args()

    from tabpfn import TabPFNRegressor

    print("=" * 100)
    print(f"  ChemeleonSMD × TabPFN v2.6 — MoleculeACE")
    print(f"  Ckpt: {args.ckpt}")
    print(f"  Device: {args.device}  |  n_estimators={args.n_estimators}")
    print(f"  Chemeleon: {args.weights}  |  PCA: {args.pca_dims}")
    print("=" * 100)

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

    print(f"\n  Phase 2: Computing ChemeleonSMD fingerprints...")
    all_fps = {}
    for wv in args.weights:
        t0 = time.time()
        all_fps[wv] = compute_all_fingerprints(all_smiles, wv)
        print(f"    {wv}: {all_fps[wv].shape} in {time.time() - t0:.1f}s")

    print(f"\n  Phase 3: Fitting global PCA models...")
    pca_models = {}
    for wv in args.weights:
        for pdim in args.pca_dims:
            pca = PCA(n_components=pdim, random_state=RANDOM_SEED)
            pca.fit(all_fps[wv])
            pca_models[(wv, pdim)] = pca

    print(f"\n  Phase 4: Running TabPFN benchmark...")
    all_results = []

    for i, bench in enumerate(benchmark_names):
        df = datasets[bench]
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]
        train_y = train_df["y"].to_numpy().astype(np.float32)
        test_y = test_df["y"].to_numpy().astype(np.float32)
        cliff_mol = test_df["cliff_mol"].to_numpy()
        n_train, n_test = len(train_df), len(test_df)
        train_idxs = [smiles_to_idx[s] for s in train_df["smiles"]]
        test_idxs = [smiles_to_idx[s] for s in test_df["smiles"]]

        print(f"\n[{i+1}/{len(benchmark_names)}] {bench} (n={n_train}+{n_test})")

        for wv in args.weights:
            train_fps = all_fps[wv][train_idxs]
            test_fps = all_fps[wv][test_idxs]

            for pdim in args.pca_dims:
                pca = pca_models[(wv, pdim)]
                train_feat = pca.transform(train_fps)
                test_feat = pca.transform(test_fps)
                train_scaled, test_scaled, y_scaled, y_scaler = scale_and_clamp(
                    train_feat, test_feat, train_y
                )

                try:
                    model = TabPFNRegressor(
                        model_path=args.ckpt,
                        device=args.device,
                        n_estimators=args.n_estimators,
                        random_state=RANDOM_SEED,
                        ignore_pretraining_limits=True,
                    )
                    t0 = time.time()
                    model.fit(train_scaled, y_scaled)
                    pred_scaled = model.predict(test_scaled)
                    dt = time.time() - t0
                    del model
                    gc.collect()
                    pred = y_scaler.inverse_transform(
                        np.asarray(pred_scaled).reshape(-1, 1)
                    ).ravel()
                    o = rmse(test_y, pred)
                    c = rmse(test_y[cliff_mol == 1], pred[cliff_mol == 1]) if cliff_mol.sum() > 0 else np.nan
                    n = rmse(test_y[cliff_mol == 0], pred[cliff_mol == 0]) if (cliff_mol == 0).sum() > 0 else np.nan
                    print(f"  {wv}/d{pdim}/TabPFN: {o:.3f} ({dt:.0f}s)", end="", flush=True)
                except Exception as e:
                    o = c = n = dt = np.nan
                    print(f"  {wv}/d{pdim}/TabPFN: ERR({type(e).__name__}: {str(e)[:60]})", end="", flush=True)

                all_results.append({
                    "benchmark": bench, "n_train": n_train, "n_test": n_test,
                    "weight": wv, "pca_dim": pdim, "model": "TabPFN-v2.6",
                    "overall_rmse": o, "cliff_rmse": c, "noncliff_rmse": n, "time_s": dt,
                })
        print()

    df_all = pd.DataFrame(all_results)

    print(f"\n{'=' * 100}")
    print(f"  TabPFN v2.6 — Mean RMSE across {len(benchmark_names)} datasets")
    print(f"{'=' * 100}")
    summary = df_all.groupby(["weight", "pca_dim"]).agg(
        mean_overall=("overall_rmse", "mean"),
        mean_cliff=("cliff_rmse", "mean"),
        mean_noncliff=("noncliff_rmse", "mean"),
        mean_time=("time_s", "mean"),
    ).reset_index().sort_values("mean_overall")

    print(f"\n  {'Rank':>4} {'Weight':>6} {'PCA':>5} {'RMSE':>7} {'Cliff':>7} {'NonCl':>7} {'Time':>6}")
    print(f"  {'─' * 54}")
    for rank, (_, r) in enumerate(summary.head(20).iterrows(), 1):
        print(f"  {rank:>4} {r['weight']:>6} {r['pca_dim']:>5.0f} "
              f"{r['mean_overall']:7.4f} {r['mean_cliff']:7.4f} {r['mean_noncliff']:7.4f} {r['mean_time']:5.1f}s")

    df_all.to_csv(args.out, index=False)
    print(f"\n  Raw results saved to {args.out}")


if __name__ == "__main__":
    main()
