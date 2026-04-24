"""Encoder stacks for TabICL MLX.

Translated from tabicl/model/encoders.py (PyTorch).
"""

from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .rope import RotaryEmbedding
from .layers import MultiheadAttentionBlock, InducedSelfAttentionBlock


class Encoder(nn.Module):
    """Stack of multihead attention blocks.

    Parameters
    ----------
    num_blocks : int
    d_model : int
    nhead : int
    dim_feedforward : int
    activation : str
    norm_first : bool
    bias_free_ln : bool
    use_rope : bool
    rope_base : float
    rope_interleaved : bool
    ssmax : str or bool
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        use_rope: bool = False,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        ssmax: Union[bool, str] = False,
        **kwargs,
    ):
        super().__init__()
        self.blocks = [
            MultiheadAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                norm_first=norm_first,
                bias_free_ln=bias_free_ln,
                ssmax=ssmax,
            )
            for _ in range(num_blocks)
        ]

        self.rope = (
            RotaryEmbedding(
                dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved
            )
            if use_rope
            else None
        )

    def __call__(
        self,
        src: mx.array,
        train_size: Optional[int] = None,
        eval_between_layers: bool = False,
    ) -> mx.array:
        """Process input through stacked blocks.

        Parameters
        ----------
        src : (..., seq_len, d_model)
        train_size : if set, keys/values restricted to first train_size positions
        eval_between_layers : if True, call mx.eval() after each block to
            materialize results and free the computation graph. Reduces peak
            memory at the cost of preventing cross-layer kernel fusion.
        """
        out = src
        for block in self.blocks:
            out = block(q=out, train_size=train_size, rope=self.rope)
            if eval_between_layers:
                mx.eval(out)
        return out


class SetTransformer(nn.Module):
    """Stack of induced self-attention blocks.

    Parameters
    ----------
    num_blocks : int
    d_model : int
    nhead : int
    dim_feedforward : int
    num_inds : int
    activation : str
    norm_first : bool
    bias_free_ln : bool
    ssmax : str or bool
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        **kwargs,
    ):
        super().__init__()
        self.blocks = [
            InducedSelfAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_inds=num_inds,
                activation=activation,
                norm_first=norm_first,
                bias_free_ln=bias_free_ln,
                ssmax=ssmax,
            )
            for _ in range(num_blocks)
        ]

    def __call__(
        self,
        src: mx.array,
        train_size: Optional[int] = None,
        eval_between_layers: bool = False,
    ) -> mx.array:
        """Process input through stacked ISAB blocks.

        Parameters
        ----------
        src : (..., seq_len, d_model)
        train_size : if set, inducing points only attend to training data
        eval_between_layers : if True, call mx.eval() after each block to
            reduce peak memory usage.
        """
        out = src
        for block in self.blocks:
            out = block(out, train_size)
            if eval_between_layers:
                mx.eval(out)
        return out
