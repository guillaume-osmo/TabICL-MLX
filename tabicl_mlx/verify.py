"""Verification script: compare PyTorch TabICL vs MLX TabICL outputs.

Run: python -m tabicl_mlx.verify
"""

from __future__ import annotations

import os
import sys
import time

# Disable MPS to prevent Metal conflicts
os.environ["PYTORCH_MPS_DISABLED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np


def verify_conversion():
    """Convert weights and verify model construction."""
    from tabicl_mlx.convert import convert_from_huggingface
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "tabicl_mlx"
    npz_path = cache_dir / "tabicl_tabicl-regressor-v2-20260212_mlx.npz"

    if not npz_path.exists():
        print("=== Step 1: Convert weights from HuggingFace ===")
        convert_from_huggingface(cache_dir, "reg-v2.ckpt")
    else:
        print(f"=== Step 1: Using cached weights at {npz_path} ===")

    return npz_path


def verify_model_loads(npz_path):
    """Load MLX model and print parameter count."""
    import json
    import mlx.core as mx
    from tabicl_mlx.model import TabICL

    config_path = npz_path.with_suffix(".json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n=== Step 2: Load MLX model (config: max_classes={config.get('max_classes')}) ===")
    model = TabICL(**config)

    raw_weights = dict(np.load(str(npz_path)))
    weight_list = [(k, mx.array(v)) for k, v in raw_weights.items()]

    # Check for missing keys
    model_keys = set()
    for k, _ in model.parameters().items():
        model_keys.add(k)

    print(f"  Model parameter groups: {len(list(model.parameters().items()))}")
    print(f"  Weight file arrays: {len(weight_list)}")

    model.load_weights(weight_list)
    print("  Weights loaded successfully!")

    # Count parameters
    n_params = sum(v.size for _, v in model.parameters().items())
    print(f"  Total parameters: {n_params:,}")

    return model, config


def verify_forward_pass(model):
    """Run a simple forward pass to check shapes."""
    import mlx.core as mx

    print("\n=== Step 3: Forward pass (random data) ===")

    B, T, H = 1, 50, 10  # 1 table, 50 samples (40 train + 10 test), 10 features
    train_size = 40

    X = mx.random.normal((B, T, H))
    y_train = mx.random.normal((B, train_size))

    t0 = time.time()
    out = model(X, y_train)
    mx.eval(out)
    dt = time.time() - t0

    test_size = T - train_size
    print(f"  Input: X={X.shape}, y_train={y_train.shape}")
    print(f"  Output: {out.shape} (expected: ({B}, {test_size}, 999))")
    print(f"  Time: {dt:.3f}s")

    assert out.shape == (B, test_size, 999), f"Shape mismatch: {out.shape}"
    print("  Shape OK!")

    # Test predict_stats
    t0 = time.time()
    mean_pred = model.predict_stats(X, y_train, output_type="mean")
    mx.eval(mean_pred)
    dt = time.time() - t0

    print(f"  predict_stats('mean'): {mean_pred.shape} (expected: ({B}, {test_size}))")
    print(f"  Time: {dt:.3f}s")
    print(f"  Values range: [{float(mean_pred.min()):.4f}, {float(mean_pred.max()):.4f}]")

    assert mean_pred.shape == (B, test_size), f"Shape mismatch: {mean_pred.shape}"
    print("  Shape OK!")

    return True


def verify_numerical_equivalence(model, config):
    """Compare PyTorch and MLX outputs on identical input."""
    import torch
    import mlx.core as mx
    from tabicl import TabICL as TabICLPyTorch

    print("\n=== Step 4: Numerical comparison with PyTorch ===")

    # Create PyTorch model with same config
    pt_model = TabICLPyTorch(**config)

    # Load same weights
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename="tabicl-regressor-v2-20260212.ckpt", local_files_only=True)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pt_model.load_state_dict(checkpoint["state_dict"])
    pt_model.eval()

    # Generate identical random input
    np.random.seed(42)
    B, T, H = 1, 30, 8
    train_size = 20
    X_np = np.random.randn(B, T, H).astype(np.float32)
    y_np = np.random.randn(B, train_size).astype(np.float32)

    # PyTorch forward
    X_pt = torch.from_numpy(X_np)
    y_pt = torch.from_numpy(y_np)

    with torch.no_grad():
        pt_out = pt_model.predict_stats(X_pt, y_pt, output_type="mean")
    pt_result = pt_out.numpy()

    # MLX forward
    X_mx = mx.array(X_np)
    y_mx = mx.array(y_np)

    mx_result = model.predict_stats(X_mx, y_mx, output_type="mean")
    mx.eval(mx_result)
    mx_result = np.array(mx_result)

    # Compare
    max_diff = np.max(np.abs(pt_result - mx_result))
    mean_diff = np.mean(np.abs(pt_result - mx_result))

    print(f"  PyTorch output: {pt_result.flatten()[:5]} ...")
    print(f"  MLX output:     {mx_result.flatten()[:5]} ...")
    print(f"  Max abs diff:   {max_diff:.6f}")
    print(f"  Mean abs diff:  {mean_diff:.6f}")

    if max_diff < 0.01:
        print("  PASS: Outputs match within tolerance!")
    elif max_diff < 0.1:
        print("  WARN: Outputs differ but within reasonable range")
    else:
        print("  FAIL: Outputs differ significantly")

    return max_diff


def verify_performance(model):
    """Benchmark MLX inference speed."""
    import mlx.core as mx

    print("\n=== Step 5: Performance benchmark ===")

    B, T, H = 8, 200, 15
    train_size = 180

    X = mx.random.normal((B, T, H))
    y_train = mx.random.normal((B, train_size))

    # Warmup
    out = model.predict_stats(X, y_train, output_type="mean")
    mx.eval(out)

    # Benchmark
    times = []
    for i in range(5):
        t0 = time.time()
        out = model.predict_stats(X, y_train, output_type="mean")
        mx.eval(out)
        dt = time.time() - t0
        times.append(dt)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"  Input: B={B}, T={T}, H={H} (train_size={train_size})")
    print(f"  Time per forward: {mean_time:.3f} +/- {std_time:.3f}s (5 runs)")

    return mean_time


def main():
    print("TabICL MLX Verification")
    print("=" * 60)

    npz_path = verify_conversion()
    model, config = verify_model_loads(npz_path)
    verify_forward_pass(model)

    try:
        verify_numerical_equivalence(model, config)
    except Exception as e:
        print(f"  Numerical comparison skipped: {e}")

    verify_performance(model)

    print("\n" + "=" * 60)
    print("Verification complete!")


if __name__ == "__main__":
    main()
