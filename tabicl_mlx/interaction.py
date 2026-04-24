"""Row-wise interaction for TabICL MLX.

Translated from tabicl/model/interaction.py (PyTorch).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .encoders import Encoder


class RowInteraction(nn.Module):
    """Context-aware row-wise interaction using transformer with RoPE.

    Prepends learnable CLS tokens to feature embeddings, processes through
    attention blocks, then extracts and concatenates CLS outputs.

    Parameters
    ----------
    embed_dim : int
    num_blocks : int
    nhead : int
    dim_feedforward : int
    num_cls : int
    rope_base : float
    rope_interleaved : bool
    activation : str
    norm_first : bool
    bias_free_ln : bool
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.norm_first = norm_first

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=True,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
        )

        # Learnable CLS tokens
        self.cls_tokens = mx.zeros((num_cls, embed_dim))

        self.out_ln = nn.LayerNorm(embed_dim, affine=True, bias=not bias_free_ln) if norm_first else None

    def _aggregate_embeddings(self, embeddings: mx.array, eval_between_layers: bool = False) -> mx.array:
        """Process rows through transformer, extract CLS token outputs.

        Parameters
        ----------
        embeddings : (B, T, H+C, E)

        Returns
        -------
        (B, T, C*E) -- flattened CLS token outputs
        """
        rope = self.tf_row.rope

        # Process all blocks except the last
        for block in self.tf_row.blocks[:-1]:
            embeddings = block(embeddings, rope=rope)
            if eval_between_layers:
                mx.eval(embeddings)

        # Last block: cross-attention with CLS queries
        last_block = self.tf_row.blocks[-1]
        cls_outputs = last_block(
            q=embeddings[..., :self.num_cls, :],
            k=embeddings,
            v=embeddings,
            rope=rope,
        )

        if self.out_ln is not None:
            cls_outputs = self.out_ln(cls_outputs)

        # Flatten CLS tokens: (B, T, C, E) -> (B, T, C*E)
        *batch_shape, C, E = cls_outputs.shape
        return cls_outputs.reshape(*batch_shape[:-1], -1, C * E)

    def __call__(self, embeddings: mx.array, eval_between_layers: bool = False) -> mx.array:
        """Transform feature embeddings into row representations.

        Parameters
        ----------
        embeddings : (B, T, H+C, E)

        Returns
        -------
        (B, T, C*E)
        """
        B, T = embeddings.shape[:2]

        # Insert learnable CLS tokens at reserved positions
        cls_tokens = mx.broadcast_to(
            self.cls_tokens, (B, T, self.num_cls, self.embed_dim)
        )
        # Replace the reserved CLS positions
        embeddings_list = [cls_tokens, embeddings[:, :, self.num_cls:, :]]
        embeddings = mx.concatenate(embeddings_list, axis=2)

        return self._aggregate_embeddings(embeddings, eval_between_layers)
