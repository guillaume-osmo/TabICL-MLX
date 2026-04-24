"""Top-level TabICL model for MLX.

Translated from tabicl/model/tabicl.py (PyTorch).
Inference-only, supports regression (max_classes=0).
"""

from __future__ import annotations

from typing import Union

import mlx.core as mx
import mlx.nn as nn

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning


class TabICL(nn.Module):
    """Tabular In-Context Learning Foundation Model (MLX).

    Three-stage pipeline:
    1. Column-wise embedding (distribution-aware)
    2. Row-wise interaction (feature interactions via RoPE transformer)
    3. Dataset-wise in-context learning (predictions from labeled examples)
    """

    def __init__(
        self,
        max_classes: int = 10,
        num_quantiles: int = 999,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 8,
        col_num_inds: int = 128,
        col_affine: bool = False,
        col_feature_group: Union[bool, str] = "same",
        col_feature_group_size: int = 3,
        col_target_aware: bool = True,
        col_ssmax: Union[bool, str] = "qassmax-mlp-elementwise",
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_rope_interleaved: bool = False,
        icl_num_blocks: int = 12,
        icl_nhead: int = 8,
        icl_ssmax: Union[bool, str] = "qassmax-mlp-elementwise",
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        **kwargs,
    ):
        super().__init__()
        icl_dim = embed_dim * row_num_cls

        if max_classes == 0:
            if num_quantiles <= 0:
                raise ValueError("For regression, num_quantiles must be > 0.")
            out_dim = num_quantiles
        else:
            out_dim = max_classes

        self.max_classes = max_classes
        self.num_quantiles = num_quantiles

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_inds=col_num_inds,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            affine=col_affine,
            feature_group=col_feature_group,
            feature_group_size=col_feature_group_size,
            target_aware=col_target_aware,
            max_classes=max_classes,
            reserve_cls_tokens=row_num_cls,
            ssmax=col_ssmax,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            rope_interleaved=row_rope_interleaved,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
        )

        self.icl_predictor = ICLearning(
            out_dim=out_dim,
            max_classes=max_classes,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=icl_ssmax,
        )

    def __call__(
        self,
        X: mx.array,
        y_train: mx.array,
        embed_with_test: bool = False,
        eval_between_layers: bool = False,
    ) -> mx.array:
        """Forward pass: ColEmbedding -> RowInteraction -> ICLearning.

        Parameters
        ----------
        X : (B, T, H) -- B tables, T samples, H features
        y_train : (B, train_size) -- training labels
        embed_with_test : bool
        eval_between_layers : if True, mx.eval() after each transformer block
            to reduce peak GPU memory (trades speed for memory).

        Returns
        -------
        (B, test_size, out_dim)
        """
        ebl = eval_between_layers
        embeddings = self.col_embedder(X, y_train=y_train, embed_with_test=embed_with_test, eval_between_layers=ebl)
        if ebl:
            mx.eval(embeddings)
        representations = self.row_interactor(embeddings, eval_between_layers=ebl)
        if ebl:
            mx.eval(representations)
        return self.icl_predictor(representations, y_train=y_train, eval_between_layers=ebl)

    def predict_stats(
        self,
        X: mx.array,
        y_train: mx.array,
        output_type: str = "mean",
        embed_with_test: bool = False,
        eval_between_layers: bool = False,
    ) -> mx.array:
        """Compute summary statistics from predicted quantiles.

        Parameters
        ----------
        X : (B, T, H)
        y_train : (B, train_size)
        output_type : "mean", "variance", or "median"
        embed_with_test : bool

        Returns
        -------
        (B, test_size) for mean/variance/median
        """
        assert self.max_classes == 0, "predict_stats is only for regression"

        raw_quantiles = self(X, y_train, embed_with_test=embed_with_test,
                             eval_between_layers=eval_between_layers)
        # Monotonize quantiles (equivalent to isotonic regression for CDF)
        quantiles = mx.sort(raw_quantiles, axis=-1)

        if output_type == "mean":
            return mx.mean(quantiles, axis=-1)
        elif output_type == "variance":
            return mx.var(quantiles, axis=-1)
        elif output_type == "median":
            mid = quantiles.shape[-1] // 2
            return quantiles[..., mid]
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
