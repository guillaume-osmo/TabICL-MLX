"""Rotary Position Embeddings for MLX.

Translated from tabicl/model/rope.py (PyTorch).
Manual implementation matching the exact PyTorch behavior.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _rotate_half_contiguous(x: mx.array) -> mx.array:
    """Rotate by splitting into contiguous halves: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _rotate_half_interleaved(x: mx.array) -> mx.array:
    """Interleaved rotation: pairs (0,1), (2,3), etc."""
    *shape, d = x.shape
    x = x.reshape(*shape, d // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    # Stack as (-x2, x1) interleaved
    result = mx.stack([-x2, x1], axis=-1)
    return result.reshape(*shape, d)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings matching PyTorch TabICL behavior.

    Parameters
    ----------
    dim : int
        Rotation dimension (head_dim).
    theta : float
        Base frequency.
    interleaved : bool
        If True, pairs are (0,1),(2,3),...
        If False, split into first/second halves.
    """

    def __init__(self, dim: int, theta: float = 10000.0, interleaved: bool = True):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.interleaved = interleaved

        # Precompute inverse frequencies
        # freqs = 1.0 / (theta ** (arange(0, dim, 2)[:dim//2] / dim))
        inv_freq = 1.0 / (
            theta ** (mx.arange(0, dim, 2).astype(mx.float32)[: dim // 2] / dim)
        )
        self._inv_freq = inv_freq

    def _get_freqs(self, seq_len: int) -> mx.array:
        """Compute frequency tensor for given sequence length.

        Returns
        -------
        For interleaved mode: (seq_len, dim)
        For contiguous mode: (seq_len, dim//2)
        """
        seq = mx.arange(seq_len).astype(mx.float32)
        # Outer product: (seq_len,) x (dim//2,) -> (seq_len, dim//2)
        freqs = mx.expand_dims(seq, axis=1) * mx.expand_dims(self._inv_freq, axis=0)

        if self.interleaved:
            # Repeat each freq for pairs: (seq_len, dim//2) -> (seq_len, dim)
            freqs = mx.repeat(freqs, repeats=2, axis=-1)

        return freqs

    def rotate_queries_or_keys(self, x: mx.array) -> mx.array:
        """Apply RoPE to queries or keys.

        Parameters
        ----------
        x : mx.array
            Shape (..., num_heads, seq_len, head_dim).

        Returns
        -------
        mx.array
            Rotated tensor, same shape.
        """
        seq_len = x.shape[-2]
        freqs = self._get_freqs(seq_len)  # (seq_len, dim or dim//2)

        if self.interleaved:
            rot_dim = freqs.shape[-1]  # = dim
            cos_f = mx.cos(freqs)
            sin_f = mx.sin(freqs)
            x_rot = x[..., :rot_dim]
            x_pass = x[..., rot_dim:]
            rotated = x_rot * cos_f + _rotate_half_interleaved(x_rot) * sin_f
            if x_pass.shape[-1] > 0:
                return mx.concatenate([rotated, x_pass], axis=-1)
            return rotated
        else:
            rot_dim = freqs.shape[-1] * 2  # freqs is (T, half), rot_dim = dim
            cos_f = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1)
            sin_f = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1)
            x_rot = x[..., :rot_dim]
            x_pass = x[..., rot_dim:]
            rotated = x_rot * cos_f + _rotate_half_contiguous(x_rot) * sin_f
            if x_pass.shape[-1] > 0:
                return mx.concatenate([rotated, x_pass], axis=-1)
            return rotated
