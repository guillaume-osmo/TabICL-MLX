"""Convert a TabPFN v2.6 .ckpt into a flat MLX .npz + JSON config.

Key-name remapping:

  feature_group_embedder.0.weight       → feature_group_embedder_0.weight
  feature_group_embedder.2.weight       → feature_group_embedder_2.weight
  target_embedder.{weight,bias}         → target_embedder.{weight,bias}
  add_thinking_rows.row_token_values_TE → row_token_values_TE
  feature_positional_embedding_*        → feature_positional_embedding_embeddings.*
  blocks.{i}.per_sample_attention_between_features.* → blocks.{i}.per_sample_attention_between_features.*
  blocks.{i}.per_column_attention_between_cells.*    → blocks.{i}.per_column_attention_between_cells.*
  blocks.{i}.layernorm_{mha1,mha2,mlp}.weight        → blocks.{i}.layernorm_{...}.weight
  blocks.{i}.mlp.0.weight                             → blocks.{i}.mlp_lin1.weight
  blocks.{i}.mlp.2.weight                             → blocks.{i}.mlp_lin2.weight
  output_projection.0.{weight,bias}                   → output_projection_0.{weight,bias}
  output_projection.2.{weight,bias}                   → output_projection_2.{weight,bias}
  criterion.borders                                   → stored at top level under "borders"
  criterion.losses_per_bucket                         → dropped (training only)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def _remap_key(k: str) -> str | None:
    # Drop training-only tensors
    if k.startswith("criterion.losses_per_bucket"):
        return None

    if k == "criterion.borders":
        return "borders"

    if k == "add_thinking_rows.row_token_values_TE":
        return "row_token_values_TE"

    if k.startswith("feature_group_embedder."):
        # feature_group_embedder.0.weight, feature_group_embedder.2.weight
        m = re.match(r"feature_group_embedder\.(\d+)\.(.+)", k)
        if m:
            return f"feature_group_embedder_{m.group(1)}.{m.group(2)}"

    if k.startswith("output_projection."):
        m = re.match(r"output_projection\.(\d+)\.(.+)", k)
        if m:
            return f"output_projection_{m.group(1)}.{m.group(2)}"

    if k.startswith("blocks."):
        # blocks.{i}.mlp.{0|2}.weight → blocks.{i}.mlp_lin{1|2}.weight
        m = re.match(r"(blocks\.\d+)\.mlp\.(\d+)\.(.+)", k)
        if m:
            idx_map = {"0": "mlp_lin1", "2": "mlp_lin2"}
            sub = idx_map.get(m.group(2))
            if sub is not None:
                return f"{m.group(1)}.{sub}.{m.group(3)}"
        # All other block sub-modules pass through
        return k

    # target_embedder.*, feature_positional_embedding_embeddings.* : pass through
    return k


def convert_checkpoint(ckpt_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """Convert a TabPFN v2.6 ckpt to MLX npz + JSON.

    Returns (npz_path, json_path).
    """
    import torch  # local import so MLX-only environments can import this module

    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem
    npz_path = out_dir / f"{stem}_mlx.npz"
    json_path = out_dir / f"{stem}_mlx.json"

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    cfg = ckpt["config"]

    out: dict[str, np.ndarray] = {}
    dropped = 0
    for k, v in state.items():
        new_k = _remap_key(k)
        if new_k is None:
            dropped += 1
            continue
        arr = v.detach().cpu().numpy().astype(np.float32, copy=False)
        out[new_k] = arr

    np.savez(str(npz_path), **out)
    # Store the model config we need to build TabPFNV2p6 (only the forward-path fields)
    cfg_out = {
        "emsize": int(cfg["emsize"]),
        "nlayers": int(cfg["nlayers"]),
        "nhead": int(cfg["nhead"]),
        "features_per_group": int(cfg["features_per_group"]),
        "num_thinking_rows": int(cfg["num_thinking_rows"]),
        "encoder_type": str(cfg["encoder_type"]),
        "encoder_mlp_hidden_dim": int(cfg["encoder_mlp_hidden_dim"]),
        "num_buckets": int(cfg["num_buckets"]),
    }
    with open(json_path, "w") as f:
        json.dump(cfg_out, f, indent=2)

    print(f"Wrote {npz_path} ({len(out)} tensors, {dropped} training-only keys dropped)")
    print(f"Wrote {json_path}")
    return npz_path, json_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", default="tabpfn_mlx_cache")
    args = p.parse_args()
    convert_checkpoint(args.ckpt, args.out_dir)
