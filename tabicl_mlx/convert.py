"""Weight conversion: PyTorch TabICL checkpoint -> MLX weights.

Usage:
    python -m tabicl_mlx.convert [--checkpoint-version reg-v2.ckpt] [--output weights.npz]

Or from Python:
    from tabicl_mlx.convert import convert_checkpoint
    convert_checkpoint("path/to/reg-v2.ckpt", "weights.npz")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


def _remap_sequential(key: str) -> str:
    """Remap nn.Sequential indexed keys to named layers.

    Examples:
        ssmax_layer.mlp.0.weight -> ssmax_layer.linear1.weight
        ssmax_layer.mlp.2.weight -> ssmax_layer.linear2.weight
        ssmax_layer.base_mlp.0.weight -> ssmax_layer.base_linear1.weight
        ssmax_layer.base_mlp.2.weight -> ssmax_layer.base_linear2.weight
        ssmax_layer.query_mlp.0.weight -> ssmax_layer.query_linear1.weight
        ssmax_layer.query_mlp.2.weight -> ssmax_layer.query_linear2.weight
        decoder.0.weight -> decoder_linear1.weight
        decoder.2.weight -> decoder_linear2.weight
    """
    # SSMax MLPs
    key = re.sub(r"\.mlp\.0\.", ".linear1.", key)
    key = re.sub(r"\.mlp\.2\.", ".linear2.", key)
    key = re.sub(r"\.base_mlp\.0\.", ".base_linear1.", key)
    key = re.sub(r"\.base_mlp\.2\.", ".base_linear2.", key)
    key = re.sub(r"\.query_mlp\.0\.", ".query_linear1.", key)
    key = re.sub(r"\.query_mlp\.2\.", ".query_linear2.", key)

    # ICL decoder
    key = re.sub(r"\.decoder\.0\.", ".decoder_linear1.", key)
    key = re.sub(r"\.decoder\.2\.", ".decoder_linear2.", key)

    return key


def _remap_layernorm(key: str) -> str:
    """Remap LayerNorm weight/bias names.

    PyTorch LayerNorm uses 'weight' and 'bias'.
    MLX LayerNorm uses 'weight' and 'bias' too, so no change needed.
    """
    return key


def _remap_blocks(key: str) -> str:
    """Remap nn.ModuleList 'blocks.N' to list-indexed 'blocks.N'.

    MLX uses the same pattern for lists, so no change needed.
    """
    return key


def _remap_skippable_linear(key: str) -> str:
    """Remap SkippableLinear keys.

    PyTorch SkippableLinear inherits nn.Linear, so weight/bias are direct attrs.
    MLX SkippableLinear wraps nn.Linear as self.linear, so:
        in_linear.weight -> in_linear.linear.weight
    """
    # in_linear (SkippableLinear) in ColEmbedding
    key = re.sub(
        r"col_embedder\.in_linear\.(weight|bias)",
        r"col_embedder.in_linear.linear.\1",
        key,
    )

    # out_w, out_b (SkippableLinear) in ColEmbedding
    key = re.sub(
        r"col_embedder\.out_w\.(weight|bias)",
        r"col_embedder.out_w.linear.\1",
        key,
    )
    key = re.sub(
        r"col_embedder\.out_b\.(weight|bias)",
        r"col_embedder.out_b.linear.\1",
        key,
    )

    return key


def _remap_one_hot_linear(key: str) -> str:
    """Remap OneHotAndLinear keys.

    PyTorch OneHotAndLinear inherits nn.Linear, weight/bias are direct attrs.
    MLX OneHotAndLinear wraps nn.Linear as self.linear:
        y_encoder.weight -> y_encoder.linear.weight
    """
    # y_encoder in ColEmbedding (when max_classes > 0)
    if re.match(r"col_embedder\.y_encoder\.(weight|bias)$", key):
        key = re.sub(
            r"col_embedder\.y_encoder\.(weight|bias)",
            r"col_embedder.y_encoder.linear.\1",
            key,
        )

    # y_encoder in ICLearning (when max_classes > 0)
    if re.match(r"icl_predictor\.y_encoder\.(weight|bias)$", key):
        # For regression (nn.Linear), no remap needed
        # For classification (OneHotAndLinear), need .linear. prefix
        # We'll handle this based on max_classes in the converter
        pass

    return key


def convert_checkpoint(
    pytorch_path: str | Path,
    output_path: str | Path,
    config_path: Optional[str | Path] = None,
) -> dict:
    """Convert PyTorch TabICL checkpoint to MLX weights.

    Parameters
    ----------
    pytorch_path : path to .ckpt file
    output_path : path for output .npz file
    config_path : optional path to save config JSON (defaults to output_path.with_suffix('.json'))

    Returns
    -------
    dict with 'config' and 'num_weights' keys
    """
    import torch

    pytorch_path = Path(pytorch_path)
    output_path = Path(output_path)
    if config_path is None:
        config_path = output_path.with_suffix(".json")
    else:
        config_path = Path(config_path)

    print(f"Loading PyTorch checkpoint from {pytorch_path}")
    checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=True)

    assert "config" in checkpoint, "Checkpoint missing 'config'"
    assert "state_dict" in checkpoint, "Checkpoint missing 'state_dict'"

    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]
    max_classes = config.get("max_classes", 10)

    # Keys to skip (non-persistent buffers, cached values, etc.)
    skip_keys = {
        "row_interactor.tf_row.rope.freqs",
        "row_interactor.tf_row.rope.dummy",
        "row_interactor.tf_row.rope.cached_freqs",
        "row_interactor.tf_row.rope.cached_scales",
        "row_interactor.tf_row.rope.scale",
    }

    mlx_weights = {}

    for key, tensor in state_dict.items():
        if key in skip_keys:
            continue
        np_array = tensor.numpy()

        # Split combined in_proj_weight into separate Q/K/V projections
        if "in_proj_weight" in key:
            prefix = key.replace("in_proj_weight", "")
            E = np_array.shape[0] // 3
            mlx_weights[prefix + "q_proj.weight"] = np_array[:E]
            mlx_weights[prefix + "k_proj.weight"] = np_array[E : 2 * E]
            mlx_weights[prefix + "v_proj.weight"] = np_array[2 * E : 3 * E]
            continue

        if "in_proj_bias" in key:
            prefix = key.replace("in_proj_bias", "")
            E = np_array.shape[0] // 3
            mlx_weights[prefix + "q_proj.bias"] = np_array[:E]
            mlx_weights[prefix + "k_proj.bias"] = np_array[E : 2 * E]
            mlx_weights[prefix + "v_proj.bias"] = np_array[2 * E : 3 * E]
            continue

        # Apply key remapping
        mlx_key = key
        mlx_key = _remap_sequential(mlx_key)
        mlx_key = _remap_skippable_linear(mlx_key)

        # OneHotAndLinear remap (only for classification y_encoder)
        if max_classes > 0:
            mlx_key = _remap_one_hot_linear(mlx_key)
            # Also handle icl_predictor y_encoder for classification
            mlx_key = re.sub(
                r"icl_predictor\.y_encoder\.(weight|bias)$",
                r"icl_predictor.y_encoder.linear.\1",
                mlx_key,
            )

        mlx_weights[mlx_key] = np_array

    # Save weights
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **mlx_weights)
    print(f"Saved {len(mlx_weights)} weight arrays to {output_path}")

    # Save config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    return {"config": config, "num_weights": len(mlx_weights)}


def convert_from_huggingface(
    output_dir: str | Path = ".",
    checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt",
) -> Path:
    """Download TabICL checkpoint from HuggingFace and convert to MLX.

    Parameters
    ----------
    output_dir : directory for output files
    checkpoint_version : HF filename (e.g., "reg-v2.ckpt", "clf-v2.ckpt")

    Returns
    -------
    Path to the output .npz file
    """
    from huggingface_hub import hf_hub_download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {checkpoint_version} from jingang/TabICL...")
    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=checkpoint_version)

    stem = checkpoint_version.replace(".ckpt", "")
    output_path = output_dir / f"tabicl_{stem}_mlx.npz"
    config_path = output_dir / f"tabicl_{stem}_mlx.json"

    convert_checkpoint(ckpt_path, output_path, config_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert TabICL PyTorch weights to MLX")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to local .ckpt file. If not given, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--checkpoint-version", type=str, default="tabicl-regressor-v2-20260212.ckpt",
        help="HuggingFace checkpoint filename (default: reg-v2.ckpt)",
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args()

    if args.checkpoint:
        stem = Path(args.checkpoint).stem
        out_path = Path(args.output) / f"tabicl_{stem}_mlx.npz"
        convert_checkpoint(args.checkpoint, out_path)
    else:
        convert_from_huggingface(args.output, args.checkpoint_version)
