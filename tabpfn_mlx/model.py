"""TabPFN v2.6 (regressor) in MLX.

Faithful port of tabpfn.architectures.tabpfn_v2_6 covering the inference
forward path for regression. Classification / training / thinking-token
variants beyond v2.6 defaults are intentionally omitted.

Shape legend (consistent with the PyTorch source):
    Ri : number of input rows (train + test, before adding thinking rows)
    Rt : number of train-label rows
    R  : Ri + num_thinking_rows
    B  : batch size (one dataset per batch entry)
    C  : raw feature count (before grouping/padding)
    G  : number of feature groups after padding = ceil(C / features_per_group)
    F  : features_per_group
    E  : emsize (model embedding dim)
    H  : num heads
    D  : head_dim = E // H
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

NAN_INDICATOR = -2.0
INFINITY_INDICATOR = 2.0
NEG_INFINITY_INDICATOR = 4.0
ENCODING_SIZE_MULTIPLIER = 2


@dataclass
class TabPFNV2p6Config:
    emsize: int = 192
    nlayers: int = 24
    nhead: int = 3
    features_per_group: int = 3
    num_thinking_rows: int = 64
    encoder_type: str = "mlp"          # "linear" or "mlp"
    encoder_mlp_hidden_dim: int = 1024
    num_buckets: int = 5000


# ───────────────────────── Attention ─────────────────────────

class _Attention(nn.Module):
    """Shared QKV/out projections; no biases, matches TabPFN v2.6."""

    def __init__(self, emsize: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_projection = nn.Linear(emsize, head_dim * num_heads, bias=False)
        self.k_projection = nn.Linear(emsize, head_dim * num_heads, bias=False)
        self.v_projection = nn.Linear(emsize, head_dim * num_heads, bias=False)
        self.out_projection = nn.Linear(head_dim * num_heads, emsize, bias=False)


class AlongRowAttention(_Attention):
    """Multi-head self-attention over feature groups (C) for every row."""

    def __call__(self, x_BrCE: mx.array) -> mx.array:
        Br, C, _ = x_BrCE.shape
        q = self.q_projection(x_BrCE).reshape(Br, C, self.num_heads, self.head_dim)
        k = self.k_projection(x_BrCE).reshape(Br, C, self.num_heads, self.head_dim)
        v = self.v_projection(x_BrCE).reshape(Br, C, self.num_heads, self.head_dim)
        # MLX SDPA expects (..., H, S, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = 1.0 / (self.head_dim ** 0.5)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(Br, C, self.num_heads * self.head_dim)
        return self.out_projection(out)


class AlongColumnAttention(_Attention):
    """Multi-head self-attention over rows (R) for every column.

    Train rows attend to train rows only (keys/values from the first N rows).
    Test rows attend to train rows only, using only the first K/V head
    (multi-query attention for test) — matches the PyTorch implementation.
    """

    def __call__(self, x_BcRE: mx.array, single_eval_pos: Optional[int] = None) -> mx.array:
        Bc, R, _ = x_BcRE.shape
        N = R if single_eval_pos is None else single_eval_pos

        q = self.q_projection(x_BcRE).reshape(Bc, R, self.num_heads, self.head_dim)
        k_all = self.k_projection(x_BcRE[:, :N]).reshape(Bc, N, self.num_heads, self.head_dim)
        v_all = self.v_projection(x_BcRE[:, :N]).reshape(Bc, N, self.num_heads, self.head_dim)

        # Transpose to (Bc, H, S, D)
        q = q.transpose(0, 2, 1, 3)
        k_all = k_all.transpose(0, 2, 1, 3)
        v_all = v_all.transpose(0, 2, 1, 3)
        scale = 1.0 / (self.head_dim ** 0.5)

        if single_eval_pos is None or single_eval_pos == R:
            out = mx.fast.scaled_dot_product_attention(q, k_all, v_all, scale=scale)
        else:
            # Train queries attend to all train K/V heads
            q_tr = q[:, :, :N, :]
            q_te = q[:, :, N:, :]
            out_tr = mx.fast.scaled_dot_product_attention(q_tr, k_all, v_all, scale=scale)
            # Test queries use only the FIRST K/V head (multi-query), broadcast across H
            k_mq = mx.broadcast_to(k_all[:, :1, :, :], (Bc, self.num_heads, N, self.head_dim))
            v_mq = mx.broadcast_to(v_all[:, :1, :, :], (Bc, self.num_heads, N, self.head_dim))
            out_te = mx.fast.scaled_dot_product_attention(q_te, k_mq, v_mq, scale=scale)
            out = mx.concatenate([out_tr, out_te], axis=2)

        out = out.transpose(0, 2, 1, 3).reshape(Bc, R, self.num_heads * self.head_dim)
        return self.out_projection(out)


# ───────────────────────── Block ─────────────────────────

class TabPFNBlock(nn.Module):
    """Post-norm: (row-attn + residual) → RMSNorm → (col-attn + residual) → RMSNorm → (MLP + residual) → RMSNorm."""

    def __init__(self, emsize: int, nhead: int, dim_feedforward: int):
        super().__init__()
        head_dim = emsize // nhead
        self.per_sample_attention_between_features = AlongRowAttention(emsize, nhead, head_dim)
        self.per_column_attention_between_cells = AlongColumnAttention(emsize, nhead, head_dim)
        self.layernorm_mha1 = nn.RMSNorm(emsize)
        self.layernorm_mha2 = nn.RMSNorm(emsize)
        self.layernorm_mlp = nn.RMSNorm(emsize)
        self.mlp_lin1 = nn.Linear(emsize, dim_feedforward, bias=False)
        self.mlp_lin2 = nn.Linear(dim_feedforward, emsize, bias=False)

    def __call__(self, x_BRCE: mx.array, single_eval_pos: int) -> mx.array:
        B, R, C, E = x_BRCE.shape

        # Row attention (per row): fold R into batch
        x_flat = x_BRCE.reshape(B * R, C, E)
        attn_out = self.per_sample_attention_between_features(x_flat)
        x_flat = x_flat + attn_out
        x_BRCE = x_flat.reshape(B, R, C, E)
        x_BRCE = self.layernorm_mha1(x_BRCE)

        # Column attention (per column): transpose then fold C into batch
        x_BCRE = x_BRCE.transpose(0, 2, 1, 3).reshape(B * C, R, E)
        col_out = self.per_column_attention_between_cells(x_BCRE, single_eval_pos=single_eval_pos)
        x_BCRE = x_BCRE + col_out
        x_BCRE = x_BCRE.reshape(B, C, R, E).transpose(0, 2, 1, 3)
        x_BRCE = self.layernorm_mha2(x_BCRE)

        # MLP
        mlp_out = self.mlp_lin2(nn.gelu(self.mlp_lin1(x_BRCE)))
        x_BRCE = x_BRCE + mlp_out
        x_BRCE = self.layernorm_mlp(x_BRCE)
        mx.eval(x_BRCE)  # materialize per-block to reduce peak memory
        return x_BRCE


# ───────────────────────── Helpers ─────────────────────────

def _pad_and_reshape_feature_groups(x_RiBC: mx.array, F: int):
    """Pad columns to multiple of F and reshape to (Ri, B*G, F)."""
    Ri, B, C = x_RiBC.shape
    pad_cols = (-C) % F
    if pad_cols:
        x_RiBC = mx.concatenate(
            [x_RiBC, mx.zeros((Ri, B, pad_cols), dtype=x_RiBC.dtype)], axis=-1
        )
    G = (C + pad_cols) // F
    x_RiBgF = x_RiBC.reshape(Ri, B * G, F)
    return x_RiBgF, G


def _impute_nan_inf_with_train_mean(x: mx.array, num_train_rows: int) -> tuple[mx.array, mx.array]:
    """Replace NaN/±Inf in x with the mean of the same feature across train rows."""
    x_tr = x[:num_train_rows]
    mask_bad_tr = mx.isnan(x_tr) | mx.isinf(x_tr)
    x_tr_safe = mx.where(mask_bad_tr, mx.zeros_like(x_tr), x_tr)
    valid_count = mx.sum(1 - mask_bad_tr.astype(x_tr.dtype), axis=0, keepdims=True)
    valid_count = mx.maximum(valid_count, mx.array(1.0, dtype=x_tr.dtype))
    feature_means = mx.sum(x_tr_safe, axis=0, keepdims=True) / valid_count
    mask_bad = mx.isnan(x) | mx.isinf(x)
    # broadcast feature_means to x shape
    x_imp = mx.where(mask_bad, mx.broadcast_to(feature_means, x.shape), x)
    return x_imp, mask_bad


def _generate_nan_inf_indicator(x: mx.array) -> mx.array:
    is_nan = mx.isnan(x)
    is_pinf = mx.isinf(x) & (x > 0)
    is_ninf = mx.isinf(x) & (x < 0)
    ind = (
        is_nan.astype(x.dtype) * NAN_INDICATOR
        + is_pinf.astype(x.dtype) * INFINITY_INDICATOR
        + is_ninf.astype(x.dtype) * NEG_INFINITY_INDICATOR
    )
    return ind


def _standard_scaler(x: mx.array, num_train_rows: int) -> mx.array:
    """TorchStandardScaler (matches tabpfn.preprocessing.torch.torch_standard_scaler).

    - Uses sample variance (denominator N-1).
    - Replaces std=0 with 1 to avoid division by zero on constant features.
    - Adds fp32 eps to std before division.
    - Clips output to [-100, 100].
    """
    x_tr = x[:num_train_rows]
    N = float(num_train_rows)
    mean = mx.mean(x_tr, axis=0, keepdims=True)
    sq_diff = mx.sum((x_tr - mean) ** 2, axis=0, keepdims=True)
    denom = max(N - 1.0, 1.0)
    std = mx.sqrt(sq_diff / denom)
    # Replace std==0 with 1
    std = mx.where(std == 0, mx.ones_like(std), std)
    EPS_FP32 = 1.1920929e-7
    return mx.clip((x - mean) / (std + EPS_FP32), -100.0, 100.0)


def _normalize_feature_groups(x_RiBgF: mx.array, F: int) -> mx.array:
    """Scale by sqrt(F / num_used_features_in_group), zero out constant features."""
    Ri = x_RiBgF.shape[0]
    first = x_RiBgF[0:1]
    eq_first = mx.sum((x_RiBgF[1:] == first).astype(x_RiBgF.dtype), axis=0)
    non_const = (eq_first != (Ri - 1))
    num_used = mx.sum(non_const.astype(x_RiBgF.dtype), axis=-1, keepdims=True)
    num_used = mx.maximum(num_used, mx.array(1.0, dtype=x_RiBgF.dtype))
    scale = mx.sqrt(F / num_used)
    x_scaled = x_RiBgF * scale
    mask = mx.broadcast_to(non_const[None, :, :], x_RiBgF.shape)
    return mx.where(mask, x_scaled, mx.zeros_like(x_scaled))


# ───────────────────────── Top-level model ─────────────────────────

class TabPFNV2p6(nn.Module):
    """TabPFN v2.6 regressor inference path in MLX."""

    def __init__(self, config: TabPFNV2p6Config):
        super().__init__()
        self.config = config
        E = config.emsize
        H = config.nhead
        FFN = E * 2  # hidden_size = input_size * 2 per PyTorch source

        # Feature group embedder (MLP with 6 → 1024 → 192, no bias)
        encoding_size = config.features_per_group * ENCODING_SIZE_MULTIPLIER
        if config.encoder_type == "mlp":
            self.feature_group_embedder_0 = nn.Linear(encoding_size, config.encoder_mlp_hidden_dim, bias=False)
            self.feature_group_embedder_2 = nn.Linear(config.encoder_mlp_hidden_dim, E, bias=False)
            self._encoder_is_mlp = True
        else:
            self.feature_group_embedder_0 = nn.Linear(encoding_size, E, bias=False)
            self._encoder_is_mlp = False

        # Target embedder (with bias in ckpt)
        self.target_embedder = nn.Linear(ENCODING_SIZE_MULTIPLIER, E, bias=True)

        # Thinking rows (learnable, prepended)
        self.row_token_values_TE = mx.zeros((config.num_thinking_rows, E))

        # Column / positional embedding (pre-generated random → Linear(48, 192))
        self.feature_positional_embedding_embeddings = nn.Linear(E // 4, E, bias=True)

        # 24 blocks
        self.blocks = [TabPFNBlock(E, H, FFN) for _ in range(config.nlayers)]

        # Output projection: Linear(192, 384) → GELU → Linear(384, num_buckets)
        self.output_projection_0 = nn.Linear(E, FFN, bias=True)
        self.output_projection_2 = nn.Linear(FFN, config.num_buckets, bias=True)

    def _embed_features(self, x_RiBC: mx.array, num_train_labels: int) -> tuple[mx.array, int]:
        """Run the TabPFN feature preprocessor + embedder.

        Returns (embedded_BRiGX, G).
        """
        Ri, B, C = x_RiBC.shape
        F = self.config.features_per_group
        x_RiBgF, G = _pad_and_reshape_feature_groups(x_RiBC, F)
        nan_inf = _generate_nan_inf_indicator(x_RiBgF)
        x_RiBgF, _ = _impute_nan_inf_with_train_mean(x_RiBgF, num_train_labels)
        x_RiBgF = _standard_scaler(x_RiBgF, num_train_labels)
        x_RiBgF = _normalize_feature_groups(x_RiBgF, F)

        x_concat = mx.concatenate([x_RiBgF, nan_inf], axis=-1)  # (Ri, B*G, 2F)
        if self._encoder_is_mlp:
            emb = self.feature_group_embedder_2(nn.gelu(self.feature_group_embedder_0(x_concat)))
        else:
            emb = self.feature_group_embedder_0(x_concat)
        # (Ri, B*G, E) → (Ri, B, G, E) → (B, Ri, G, E)
        emb = emb.reshape(Ri, B, G, self.config.emsize).transpose(1, 0, 2, 3)
        return emb, G

    def _embed_targets(self, y: mx.array, num_train_rows: int, batch_size: int) -> mx.array:
        """Pad y with NaN, generate indicator, impute, concat, embed. Returns (B, Ri, E)."""
        Rt = y.shape[0]
        if y.ndim == 1:
            y_RtB1 = y.reshape(Rt, 1, 1)
        elif y.ndim == 2:
            y_RtB1 = y.reshape(Rt, batch_size, 1)
        else:
            y_RtB1 = y
        # Pad on row dim with NaN up to num_train_rows
        pad_rows = num_train_rows - Rt
        if pad_rows > 0:
            nan_pad = mx.full((pad_rows, y_RtB1.shape[1], 1), mx.nan, dtype=y_RtB1.dtype)
            y_RiB1 = mx.concatenate([y_RtB1, nan_pad], axis=0)
        else:
            y_RiB1 = y_RtB1

        nan_inf = _generate_nan_inf_indicator(y_RiB1)
        y_RiB1, _ = _impute_nan_inf_with_train_mean(y_RiB1, Rt)
        y_concat = mx.concatenate([y_RiB1, nan_inf], axis=-1)  # (Ri, B, 2)
        emb = self.target_embedder(y_concat)                    # (Ri, B, E)
        return emb.transpose(1, 0, 2)                           # (B, Ri, E)

    def _add_column_embedding(self, x_BRiGE: mx.array, column_seeds: mx.array) -> mx.array:
        """Add per-column learned positional embedding derived from fixed random seeds.

        column_seeds: (G, E//4). Produced externally (from the pre-generated column
        embeddings / a fixed-seed RNG) and broadcast here.
        """
        emb = self.feature_positional_embedding_embeddings(column_seeds)  # (G, E)
        return x_BRiGE + emb[None, None, :, :]

    def __call__(
        self,
        x_RiBC: mx.array,
        y_Rt: mx.array,
        column_seeds: mx.array,
    ) -> mx.array:
        """Forward pass. Returns logits of shape (M, B, num_buckets)."""
        Ri, B, C = x_RiBC.shape
        Rt = y_Rt.shape[0]

        emb_x, G = self._embed_features(x_RiBC, Rt)           # (B, Ri, G, E)
        emb_x = self._add_column_embedding(emb_x, column_seeds)
        emb_y = self._embed_targets(y_Rt, Ri, B)              # (B, Ri, E)
        # Append target as an extra "column": (B, Ri, G+1, E)
        x_BRiCD = mx.concatenate([emb_x, emb_y[:, :, None, :]], axis=2)

        # Prepend T thinking rows
        T = self.config.num_thinking_rows
        thinking = mx.broadcast_to(
            self.row_token_values_TE[None, :, None, :], (B, T, G + 1, self.config.emsize)
        )
        x_BRCE = mx.concatenate([thinking, x_BRiCD], axis=1)
        single_eval_pos = T + Rt

        for block in self.blocks:
            x_BRCE = block(x_BRCE, single_eval_pos)

        # Extract test embeddings from the y-column (last column), test rows only
        test_BMD = x_BRCE[:, single_eval_pos:, -1, :]          # (B, M, E)
        test_MBD = test_BMD.transpose(1, 0, 2)                 # (M, B, E)
        logits = self.output_projection_2(nn.gelu(self.output_projection_0(test_MBD)))
        return logits                                          # (M, B, num_buckets)
