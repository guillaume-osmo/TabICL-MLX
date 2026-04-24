"""Column-wise embedding for TabICL MLX.

Translated from tabicl/model/embedding.py (PyTorch).
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .layers import SkippableLinear, OneHotAndLinear
from .encoders import SetTransformer


class ColEmbedding(nn.Module):
    """Distribution-aware column-wise embedding.

    Processes each feature column through a shared set transformer to create
    distribution-aware embeddings.

    Parameters
    ----------
    embed_dim : int
    num_blocks : int
    nhead : int
    dim_feedforward : int
    num_inds : int
    activation : str
    norm_first : bool
    bias_free_ln : bool
    affine : bool
    feature_group : str or bool
    feature_group_size : int
    target_aware : bool
    max_classes : int
    reserve_cls_tokens : int
    ssmax : str or bool
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        affine: bool = True,
        feature_group: Union[bool, str] = False,
        feature_group_size: int = 3,
        target_aware: bool = False,
        max_classes: int = 10,
        reserve_cls_tokens: int = 4,
        ssmax: Union[bool, str] = False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.feature_group = feature_group
        self.feature_group_size = feature_group_size
        self.target_aware = target_aware
        self.max_classes = max_classes
        self.affine = affine

        self.in_linear = SkippableLinear(
            feature_group_size if feature_group else 1, embed_dim
        )

        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
        )

        if target_aware:
            if max_classes > 0:
                self.y_encoder = OneHotAndLinear(max_classes, embed_dim)
            else:
                self.y_encoder = nn.Linear(1, embed_dim)

        if affine:
            self.out_w = SkippableLinear(embed_dim, embed_dim)
            self.ln_w = nn.LayerNorm(embed_dim, affine=True, bias=not bias_free_ln) if norm_first else None
            self.out_b = SkippableLinear(embed_dim, embed_dim)
            self.ln_b = nn.LayerNorm(embed_dim, affine=True, bias=not bias_free_ln) if norm_first else None

    def feature_grouping(self, X: mx.array) -> mx.array:
        """Group features into fixed-size groups.

        Parameters
        ----------
        X : (B, T, H)

        Returns
        -------
        (B, T, G, feature_group_size)
        """
        if not self.feature_group:
            return mx.expand_dims(X, axis=-1)  # (B, T, H, 1)

        B, T, H = X.shape
        size = self.feature_group_size
        mode = "same" if self.feature_group is True else self.feature_group

        if mode == "same":
            idxs = mx.arange(H)
            groups = [X[:, :, (idxs + 2**i) % H] for i in range(size)]
            X = mx.stack(groups, axis=-1)
        else:
            x_pad_cols = (size - H % size) % size
            if x_pad_cols > 0:
                X = mx.pad(X, [(0, 0), (0, 0), (0, x_pad_cols)])
            X = X.reshape(B, T, -1, size)

        return X

    def _compute_embeddings(
        self,
        features: mx.array,
        train_size: int,
        y_train: Optional[mx.array] = None,
        embed_with_test: bool = False,
        eval_between_layers: bool = False,
    ) -> mx.array:
        """Core embedding computation.

        Parameters
        ----------
        features : (..., T, in_dim)
        train_size : int
        y_train : (..., train_size), optional
        embed_with_test : bool

        Returns
        -------
        (..., T, embed_dim)
        """
        src = self.in_linear(features)

        if not self.target_aware:
            src = self.tf_col(src, train_size=None if embed_with_test else train_size,
                              eval_between_layers=eval_between_layers)
        else:
            assert y_train is not None

            if self.max_classes > 0:
                y_emb = self.y_encoder(y_train.astype(mx.float32))
            else:
                y_emb = self.y_encoder(mx.expand_dims(y_train, axis=-1))

            # Add target embeddings to training positions
            train_part = src[..., :train_size, :] + y_emb
            test_part = src[..., train_size:, :]
            src = mx.concatenate([train_part, test_part], axis=-2)

            src = self.tf_col(src, train_size=None if embed_with_test else train_size,
                              eval_between_layers=eval_between_layers)

        if self.affine:
            w = self.out_w(src)
            if self.ln_w is not None:
                w = self.ln_w(w)
            b = self.out_b(src)
            if self.ln_b is not None:
                b = self.ln_b(b)
            embeddings = features * w + b
        else:
            embeddings = src

        return embeddings

    def __call__(
        self,
        X: mx.array,
        y_train: mx.array,
        embed_with_test: bool = False,
        eval_between_layers: bool = False,
    ) -> mx.array:
        """Transform input table into embeddings.

        Parameters
        ----------
        X : (B, T, H)
        y_train : (B, train_size)
        embed_with_test : bool

        Returns
        -------
        (B, T, G+C, E) where G=groups, C=cls tokens, E=embed_dim
        """
        train_size = y_train.shape[1]

        if self.feature_group:
            return self._forward_with_feature_group(X, y_train, train_size, embed_with_test, eval_between_layers)
        else:
            return self._forward_without_feature_group(X, y_train, train_size, embed_with_test, eval_between_layers)

    def _forward_with_feature_group(
        self,
        X: mx.array,
        y_train: mx.array,
        train_size: int,
        embed_with_test: bool,
        eval_between_layers: bool = False,
    ) -> mx.array:
        X = self.feature_grouping(X)  # (B, T, G, group_size)
        B, T, G, gs = X.shape

        if self.reserve_cls_tokens > 0:
            # Pad with skip value at the front of the group dimension
            cls_pad = mx.full((B, T, self.reserve_cls_tokens, gs), -100.0)
            X = mx.concatenate([cls_pad, X], axis=2)  # (B, T, C+G, gs)

        # Transpose: (B, T, G+C, gs) -> (B, G+C, T, gs)
        features = mx.transpose(X, axes=(0, 2, 1, 3))

        if self.target_aware:
            # Expand y_train: (B, train_size) -> (B, G+C, train_size)
            y_train = mx.broadcast_to(
                mx.expand_dims(y_train, axis=1),
                (y_train.shape[0], features.shape[1], y_train.shape[1]),
            )

        embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test, eval_between_layers)
        # Transpose back: (B, G+C, T, E) -> (B, T, G+C, E)
        return mx.transpose(embeddings, axes=(0, 2, 1, 3))

    def _forward_without_feature_group(
        self,
        X: mx.array,
        y_train: mx.array,
        train_size: int,
        embed_with_test: bool,
        eval_between_layers: bool = False,
    ) -> mx.array:
        B, T, H = X.shape

        if self.reserve_cls_tokens > 0:
            # Pad with skip value at the front of feature dimension
            cls_pad = mx.full((B, T, self.reserve_cls_tokens), -100.0)
            X = mx.concatenate([cls_pad, X], axis=2)  # (B, T, C+H)

        # (B, T, H+C) -> (B, H+C, T) -> (B, H+C, T, 1)
        features = mx.expand_dims(mx.transpose(X, axes=(0, 2, 1)), axis=-1)

        if self.target_aware:
            y_train = mx.broadcast_to(
                mx.expand_dims(y_train, axis=1),
                (y_train.shape[0], features.shape[1], y_train.shape[1]),
            )

        embeddings = self._compute_embeddings(features, train_size, y_train, embed_with_test, eval_between_layers)
        return mx.transpose(embeddings, axes=(0, 2, 1, 3))
