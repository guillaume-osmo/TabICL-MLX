"""Scalable Softmax (SSMax) variants for MLX.

Translated from tabicl/model/ssmax.py (PyTorch).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _logn(n: int) -> float:
    return math.log(max(n, 1))


class SSMax(nn.Module):
    """Scalable Softmax with learnable per-head scaling factors.

    q_scaled = q * (s * log(n)) where s is learnable per-head.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.scales = mx.ones((num_heads,))

    def __call__(self, q: mx.array, n: int) -> mx.array:
        logn = _logn(n)
        scales = self.scales.reshape(1, -1, 1, 1) * logn
        return q * scales


class SSMaxMLP(nn.Module):
    """Scalable Softmax using MLP to compute scaling factors.

    q_scaled = q * mlp(log(n))
    """

    def __init__(
        self,
        num_heads: int,
        n_hidden: int = 64,
        elementwise: bool = False,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.elementwise = elementwise
        self.num_heads = num_heads
        if elementwise:
            if head_dim is None:
                raise ValueError("head_dim must be provided when elementwise=True")
            out_dim = num_heads * head_dim
        else:
            out_dim = num_heads
        self.linear1 = nn.Linear(1, n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_dim)

    def __call__(self, q: mx.array, n: int) -> mx.array:
        logn = mx.array([[_logn(n)]])
        scales = self.linear2(nn.gelu(self.linear1(logn)))
        if self.elementwise:
            head_dim = q.shape[-1]
            scales = scales.reshape(1, self.num_heads, 1, head_dim)
        else:
            scales = scales.reshape(1, self.num_heads, 1, 1)
        return q * scales


class QASSMaxMLP(nn.Module):
    """Query-Aware Scalable Softmax using MLPs.

    q_scaled = q * base_mlp(log(n)) * (1 + tanh(query_mlp(q)))
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
        elementwise: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.elementwise = elementwise

        if elementwise:
            base_out_dim = num_heads * head_dim
            query_out_dim = head_dim
        else:
            base_out_dim = num_heads
            query_out_dim = 1

        self.base_linear1 = nn.Linear(1, n_hidden)
        self.base_linear2 = nn.Linear(n_hidden, base_out_dim)
        self.query_linear1 = nn.Linear(head_dim, n_hidden)
        self.query_linear2 = nn.Linear(n_hidden, query_out_dim)

    def __call__(self, q: mx.array, n: int) -> mx.array:
        logn = mx.array([[_logn(n)]])

        if self.elementwise:
            base_scales = self.base_linear2(nn.gelu(self.base_linear1(logn)))
            base_scales = base_scales.reshape(1, self.num_heads, 1, self.head_dim)
            modulation = 1 + mx.tanh(self.query_linear2(nn.gelu(self.query_linear1(q))))
        else:
            base_scales = self.base_linear2(nn.gelu(self.base_linear1(logn)))
            base_scales = base_scales.reshape(1, self.num_heads, 1, 1)
            modulation = 1 + mx.tanh(self.query_linear2(nn.gelu(self.query_linear1(q))))

        return q * (base_scales * modulation)


def create_ssmax_layer(ssmax_type: str, num_heads: int, embed_dim: int):
    """Factory function to create SSMax layer based on type."""
    if ssmax_type == "none":
        return None
    elif ssmax_type == "ssmax":
        return SSMax(num_heads)
    elif ssmax_type == "ssmax-mlp":
        return SSMaxMLP(num_heads)
    elif ssmax_type == "ssmax-mlp-elementwise":
        return SSMaxMLP(num_heads, head_dim=embed_dim // num_heads, elementwise=True)
    elif ssmax_type == "qassmax-mlp":
        return QASSMaxMLP(num_heads, embed_dim // num_heads)
    elif ssmax_type == "qassmax-mlp-elementwise":
        return QASSMaxMLP(num_heads, embed_dim // num_heads, elementwise=True)
    else:
        raise ValueError(f"Unknown {ssmax_type=}")
