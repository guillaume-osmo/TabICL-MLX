"""Dataset-wise in-context learning for TabICL MLX.

Translated from tabicl/model/learning.py (PyTorch).
Supports regression (max_classes=0) only in this initial version.
"""

from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .layers import OneHotAndLinear
from .encoders import Encoder


class ICLearning(nn.Module):
    """In-context learning via transformer encoder + decoder.

    For regression: encodes y_train via Linear(1, d_model), adds to train
    positions, runs through encoder with train_size masking, decodes via MLP.

    Parameters
    ----------
    out_dim : int
    max_classes : int
    d_model : int
    num_blocks : int
    nhead : int
    dim_feedforward : int
    activation : str
    norm_first : bool
    bias_free_ln : bool
    ssmax : str or bool
    """

    def __init__(
        self,
        out_dim: int,
        max_classes: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        **kwargs,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
        )

        if norm_first:
            self.ln = nn.LayerNorm(d_model, affine=True, bias=not bias_free_ln)

        if max_classes > 0:
            self.y_encoder = OneHotAndLinear(max_classes, d_model)
        else:
            self.y_encoder = nn.Linear(1, d_model)

        # Decoder MLP
        self.decoder_linear1 = nn.Linear(d_model, d_model * 2)
        self.decoder_linear2 = nn.Linear(d_model * 2, out_dim)

    def _decode(self, x: mx.array) -> mx.array:
        """Decoder MLP: Linear -> GELU -> Linear."""
        return self.decoder_linear2(nn.gelu(self.decoder_linear1(x)))

    def __call__(
        self,
        representations: mx.array,
        y_train: mx.array,
        eval_between_layers: bool = False,
    ) -> mx.array:
        """Run in-context learning.

        Parameters
        ----------
        representations : (B, T, D) where D = embed_dim * num_cls
        y_train : (B, train_size)

        Returns
        -------
        (B, test_size, out_dim) for regression
        """
        train_size = y_train.shape[1]

        # Encode targets
        if self.max_classes > 0:
            y_emb = self.y_encoder(y_train.astype(mx.float32))
        else:
            y_emb = self.y_encoder(mx.expand_dims(y_train, axis=-1))

        # Add target embeddings to training positions
        R = representations
        train_part = R[:, :train_size, :] + y_emb
        test_part = R[:, train_size:, :]
        R = mx.concatenate([train_part, test_part], axis=1)

        # Run through ICL encoder with train_size masking
        out = self.tf_icl(R, train_size=train_size, eval_between_layers=eval_between_layers)

        # Apply LayerNorm
        if self.norm_first:
            out = self.ln(out)

        # Decode
        out = self._decode(out)

        # Extract test portion
        return out[:, train_size:, :]
