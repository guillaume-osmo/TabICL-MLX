"""Core layers for TabICL MLX: attention blocks, custom linear layers.

Translated from tabicl/model/layers.py and tabicl/model/attention.py (PyTorch).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .ssmax import create_ssmax_layer
from .rope import RotaryEmbedding


class OneHotAndLinear(nn.Module):
    """One-hot encoding + linear projection for class embeddings."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_classes, embed_dim)

    def __call__(self, src: mx.array) -> mx.array:
        one_hot = mx.one_hot(src.astype(mx.int32), self.num_classes).astype(mx.float32)
        return self.linear(one_hot)


class SkippableLinear(nn.Module):
    """Linear layer that preserves skip_value sentinel for padded inputs."""

    def __init__(self, in_features: int, out_features: int, skip_value: float = -100.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.skip_value = skip_value

    def __call__(self, src: mx.array) -> mx.array:
        out = self.linear(src)
        skip_mask = mx.all(src == self.skip_value, axis=-1, keepdims=True)
        out = mx.where(skip_mask, self.skip_value, out)
        return out


class MultiheadAttention(nn.Module):
    """Multi-head attention with RoPE, SSMax support.

    Uses separate Q/K/V projections (PyTorch version uses combined in_proj_weight).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ssmax: str | bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"
        self.ssmax_layer = create_ssmax_layer(
            ssmax_type=ssmax, num_heads=num_heads, embed_dim=embed_dim
        )

    def __call__(
        self,
        query: mx.array,
        key: Optional[mx.array] = None,
        value: Optional[mx.array] = None,
        key_padding_mask: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> mx.array:
        """
        Parameters
        ----------
        query : shape (..., tgt_len, embed_dim)
        key : shape (..., src_len, embed_dim), defaults to query
        value : shape (..., src_len, embed_dim), defaults to key
        key_padding_mask : shape (..., src_len), True = ignore
        attn_mask : shape (tgt_len, src_len) or (..., num_heads, tgt_len, src_len), float additive
        rope : optional RotaryEmbedding

        Returns
        -------
        mx.array of shape (..., tgt_len, embed_dim)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        *batch_shape, tgt_len, _ = query.shape
        src_len = key.shape[-2]

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (..., num_heads, seq_len, head_dim)
        q = q.reshape(*batch_shape, tgt_len, self.num_heads, self.head_dim).transpose(
            *range(len(batch_shape)), -2, -3, -1
        )
        k = k.reshape(*batch_shape, src_len, self.num_heads, self.head_dim).transpose(
            *range(len(batch_shape)), -2, -3, -1
        )
        v = v.reshape(*batch_shape, src_len, self.num_heads, self.head_dim).transpose(
            *range(len(batch_shape)), -2, -3, -1
        )

        # Apply RoPE
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)

        # Apply SSMax scaling to queries
        if self.ssmax_layer is not None:
            q = self.ssmax_layer(q, src_len)

        # Build attention mask from key_padding_mask
        mask = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: (..., src_len), True = ignore
            # Convert to float additive mask: True -> -inf
            kpm = mx.where(
                key_padding_mask, mx.array(-1e9), mx.array(0.0)
            )
            # Expand: (..., 1, 1, src_len)
            kpm = mx.expand_dims(mx.expand_dims(kpm, axis=-2), axis=-2)
            # Broadcast to (..., num_heads, tgt_len, src_len)
            if mask is None:
                mask = kpm
            else:
                mask = mask + kpm

        # Flatten batch dims for SDPA (requires exactly 4D)
        flat_batch = 1
        for s in batch_shape:
            flat_batch *= s

        q_4d = q.reshape(flat_batch, self.num_heads, tgt_len, self.head_dim)
        k_4d = k.reshape(flat_batch, self.num_heads, src_len, self.head_dim)
        v_4d = v.reshape(flat_batch, self.num_heads, src_len, self.head_dim)

        if mask is not None:
            # Expand mask to (..., num_heads, tgt_len, src_len) then flatten
            if mask.ndim == 2:
                # (tgt_len, src_len) -> (1, 1, tgt_len, src_len)
                mask_4d = mx.expand_dims(mx.expand_dims(mask, axis=0), axis=0)
                mask_4d = mx.broadcast_to(
                    mask_4d, (flat_batch, self.num_heads, tgt_len, src_len)
                )
            else:
                mask_4d = mx.broadcast_to(
                    mask.reshape(flat_batch, -1, tgt_len, src_len),
                    (flat_batch, self.num_heads, tgt_len, src_len),
                )
        else:
            mask_4d = None

        scale = self.head_dim ** -0.5
        attn_out = mx.fast.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, scale=scale, mask=mask_4d
        )

        # Reshape back to (..., tgt_len, embed_dim)
        attn_out = attn_out.reshape(*batch_shape, self.num_heads, tgt_len, self.head_dim)
        # Transpose: (..., num_heads, tgt_len, head_dim) -> (..., tgt_len, num_heads, head_dim)
        attn_out = attn_out.transpose(
            *range(len(batch_shape)), -2, -3, -1
        )
        attn_out = attn_out.reshape(*batch_shape, tgt_len, self.embed_dim)

        return self.out_proj(attn_out)


class MultiheadAttentionBlock(nn.Module):
    """Attention block with pre-norm/post-norm, FFN, RoPE, SSMax.

    Replaces PyTorch's nn.TransformerEncoderLayer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: str | bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.attn = MultiheadAttention(d_model, nhead, ssmax=ssmax)
        self.norm1 = nn.LayerNorm(d_model, affine=True, bias=not bias_free_ln)
        self.norm2 = nn.LayerNorm(d_model, affine=True, bias=not bias_free_ln)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def __call__(
        self,
        q: mx.array,
        k: Optional[mx.array] = None,
        v: Optional[mx.array] = None,
        key_padding_mask: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
        train_size: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> mx.array:
        """
        Parameters
        ----------
        q : (..., tgt_len, d_model)
        k, v : optional, for cross-attention
        train_size : if set, k=v=q[..., :train_size, :]
        """
        if train_size is None:
            if k is None:
                k = q
            if v is None:
                v = k
        else:
            assert k is None and v is None
            k = q[..., :train_size, :]
            v = q[..., :train_size, :]

        if self.norm_first:
            q_normed = self.norm1(q)
            if train_size is None:
                # Check if k/v are the same as q (self-attention)
                k_normed = self.norm1(k)
                v_normed = self.norm1(v)
            else:
                k_normed = q_normed[..., :train_size, :]
                v_normed = k_normed

            attn = self.attn(
                q_normed, k_normed, v_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope=rope,
            )
            x = q + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            attn = self.attn(
                q, k, v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope=rope,
            )
            x = self.norm1(q + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _ff_block(self, x: mx.array) -> mx.array:
        return self.linear2(nn.gelu(self.linear1(x)))


class InducedSelfAttentionBlock(nn.Module):
    """Induced Self-Attention for efficient O(n) attention.

    Two-stage attention:
    1. Inducing points attend to input -> hidden
    2. Input attends to hidden -> output
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: str | bool = False,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value
        self.num_inds = num_inds
        self.d_model = d_model

        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"

        # Stage 1: inducing points -> input (with SSMax)
        self.multihead_attn1 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward,
            activation=activation, norm_first=norm_first,
            bias_free_ln=bias_free_ln, ssmax=ssmax,
        )
        # Stage 2: input -> hidden (no SSMax)
        self.multihead_attn2 = MultiheadAttentionBlock(
            d_model, nhead, dim_feedforward,
            activation=activation, norm_first=norm_first,
            bias_free_ln=bias_free_ln, ssmax=False,
        )

        # Learnable inducing points
        self.ind_vectors = mx.zeros((num_inds, d_model))

    def __call__(
        self,
        src: mx.array,
        train_size: Optional[int] = None,
    ) -> mx.array:
        """
        Parameters
        ----------
        src : (..., seq_len, d_model)
        train_size : if set, inducing points only attend to first train_size positions
        """
        # Check for skip values
        skip_mask = mx.all(
            mx.all(src == self.skip_value, axis=-1), axis=-1
        )  # batch shape

        if mx.all(skip_mask):
            return mx.full(src.shape, self.skip_value)

        # Expand inducing points to match batch shape
        *batch_shape, seq_len, d_model = src.shape
        ind_vectors = mx.broadcast_to(
            self.ind_vectors,
            (*batch_shape, self.num_inds, d_model),
        )

        # Stage 1: inducing points attend to (training) input
        if train_size is None:
            hidden = self.multihead_attn1(ind_vectors, src, src)
        else:
            src_train = src[..., :train_size, :]
            hidden = self.multihead_attn1(ind_vectors, src_train, src_train)

        # Stage 2: input attends to hidden
        out = self.multihead_attn2(src, hidden, hidden)

        # Restore skip values where needed
        if mx.any(skip_mask):
            skip_mask_expanded = mx.expand_dims(mx.expand_dims(skip_mask, axis=-1), axis=-1)
            out = mx.where(skip_mask_expanded, self.skip_value, out)

        return out
