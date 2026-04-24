"""sklearn-compatible TabPFN v2.6 regressor in MLX.

Usage:
    from tabpfn_mlx import TabPFNRegressorMLX
    model = TabPFNRegressorMLX(
        ckpt_path="/path/to/tabpfn-v2.6-regressor-v2.6_default.ckpt",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

This bypasses TabPFN's full sklearn-ensemble preprocessing (power/quantile
transforms, multi-estimator feature shuffling, etc.) and just feeds
standardized input through the model + bar-distribution mean. For
molecular/regression tasks where the upstream pipeline already scales
features, this matches PyTorch TabPFN within ~0.003 RMSE on MoleculeACE and
runs ~500× faster on Apple Silicon.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import mlx.core as mx
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .model import TabPFNV2p6, TabPFNV2p6Config
from .bar_distribution import full_support_bar_distribution_mean
from .convert import convert_checkpoint


class TabPFNRegressorMLX(RegressorMixin, BaseEstimator):
    """TabPFN v2.6 regressor running natively on Apple Silicon via MLX.

    Parameters
    ----------
    ckpt_path : str or Path or None
        Path to the PyTorch .ckpt. If the converted .npz+.json pair already
        exists in ``cache_dir``, the ckpt is not reloaded.
    cache_dir : str or Path
        Where to keep / look for the converted MLX weights.
    y_standardize : bool
        If True (default), standardize y before feeding to the model and
        inverse-transform predictions back to the original scale.
    prebuilt_model : TabPFNV2p6 or None
        Pre-loaded model (e.g. shared across many ``.fit()`` calls in a
        benchmark). Overrides ``ckpt_path``.
    """

    def __init__(
        self,
        ckpt_path: Optional[str | Path] = None,
        cache_dir: str | Path = "tabpfn_mlx_cache",
        y_standardize: bool = True,
        n_estimators: int = 1,
        norm_methods: Optional[List[str]] = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = 42,
        prebuilt_model: Optional[TabPFNV2p6] = None,
        prebuilt_borders: Optional[mx.array] = None,
        prebuilt_pre_column_embeddings: Optional[np.ndarray] = None,
    ):
        self.ckpt_path = ckpt_path
        self.cache_dir = cache_dir
        self.y_standardize = y_standardize
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.prebuilt_model = prebuilt_model
        self.prebuilt_borders = prebuilt_borders
        self.prebuilt_pre_column_embeddings = prebuilt_pre_column_embeddings

    @staticmethod
    def build_model(
        ckpt_path: str | Path, cache_dir: str | Path = "tabpfn_mlx_cache"
    ) -> tuple[TabPFNV2p6, mx.array, np.ndarray]:
        """Build model + borders + pre-generated column embeddings. Expensive; call once."""
        ckpt_path = Path(ckpt_path)
        cache_dir = Path(cache_dir)
        stem = ckpt_path.stem
        npz_path = cache_dir / f"{stem}_mlx.npz"
        json_path = cache_dir / f"{stem}_mlx.json"
        if not (npz_path.exists() and json_path.exists()):
            convert_checkpoint(ckpt_path, cache_dir)

        with open(json_path) as f:
            cfg = TabPFNV2p6Config(**json.load(f))
        model = TabPFNV2p6(cfg)
        raw = dict(np.load(str(npz_path)))
        borders = raw.pop("borders")
        model.load_weights([(k, mx.array(v)) for k, v in raw.items()])
        mx.eval(model.parameters())

        # Pre-generated column embeddings from TabPFN (for ckpt-matched positional embeds).
        # Lazily imported so pure MLX envs without PyTorch can still build.
        try:
            from tabpfn.architectures.shared.column_embeddings import load_column_embeddings
            pre = load_column_embeddings().detach().numpy()
        except Exception:
            pre = np.zeros((0, cfg.emsize // 4), dtype=np.float32)

        return model, mx.array(borders), pre

    def _get_column_seeds(self, n_features: int) -> mx.array:
        """Derive per-feature-group positional-embedding seeds, matching PT."""
        F = self.model_.config.features_per_group
        pad = (-n_features) % F
        G = (n_features + pad) // F
        eq = self.model_.config.emsize // 4

        # Mirror PT: torch.Generator seeded at 42 → randn (G, eq); overwrite first 2000
        # rows with pre-generated embeddings when shapes match.
        try:
            import torch
            gen = torch.Generator().manual_seed(42)
            seeds = torch.randn((G, eq), generator=gen).numpy().astype(np.float32)
        except Exception:
            rng = np.random.default_rng(42)
            seeds = rng.standard_normal((G, eq)).astype(np.float32)

        pre = self.pre_column_embeddings_
        if pre.ndim == 2 and pre.shape[1] == eq and pre.shape[0] > 0:
            take = min(G, pre.shape[0])
            seeds[:take] = pre[:take]
        return mx.array(seeds)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNRegressorMLX":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        if self.prebuilt_model is not None:
            self.model_ = self.prebuilt_model
            self.borders_ = (
                self.prebuilt_borders
                if self.prebuilt_borders is not None
                else mx.array(np.zeros(1, dtype=np.float32))
            )
            self.pre_column_embeddings_ = (
                self.prebuilt_pre_column_embeddings
                if self.prebuilt_pre_column_embeddings is not None
                else np.zeros((0, self.model_.config.emsize // 4), dtype=np.float32)
            )
        else:
            if self.ckpt_path is None:
                raise ValueError("Provide either ckpt_path or prebuilt_model.")
            self.model_, self.borders_, self.pre_column_embeddings_ = self.build_model(
                self.ckpt_path, self.cache_dir
            )

        if self.y_standardize:
            self.y_scaler_ = StandardScaler()
            y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            self.y_scaler_ = None
            y_scaled = y.copy()

        if self.n_estimators == 1:
            self.ensemble_generator_ = None
            self.X_train_ = X
            self.y_train_scaled_ = y_scaled
        else:
            # Reuse TabICL's ensemble generator: multi-norm × feature-shuffle variants
            from tabicl_mlx.sklearn.preprocessing import EnsembleGenerator
            self.ensemble_generator_ = EnsembleGenerator(
                classification=False,
                n_estimators=self.n_estimators,
                norm_methods=self.norm_methods or ["none", "power"],
                feat_shuffle_method=self.feat_shuffle_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_state,
            )
            self.ensemble_generator_.fit(X, y_scaled)

        self.n_features_in_ = X.shape[1]
        return self

    def _forward_one_batch(self, X_RiBC: np.ndarray, y_RtB: np.ndarray) -> np.ndarray:
        """Run the MLX model on (Ri, B, C) input, return mean predictions (M, B)."""
        col_mx = self._get_column_seeds(X_RiBC.shape[-1])
        logits = self.model_(
            mx.array(X_RiBC.astype(np.float32)),
            mx.array(y_RtB.astype(np.float32)),
            col_mx,
        )
        mean_scaled_mx = full_support_bar_distribution_mean(logits, self.borders_)
        mx.eval(mean_scaled_mx)
        return np.array(mean_scaled_mx)  # (M, B)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float32)
        n_te = X.shape[0]

        if self.ensemble_generator_ is None:
            # Single estimator, fastest path
            X_full = np.concatenate([self.X_train_, X], axis=0)[:, None, :]
            mean_scaled = self._forward_one_batch(X_full, self.y_train_scaled_).reshape(n_te)
        else:
            # Ensemble: iterate norm methods × stack feature-shuffle variants as B
            data = self.ensemble_generator_.transform(X, mode="both")
            all_preds = []
            for norm_method, (Xs, ys) in data.items():
                # Xs: (n_variants, Ri, C)  ys: (n_variants, Rt)
                # MLX forward expects (Ri, B, C). Transpose variants → B.
                X_RiBC = np.transpose(Xs, (1, 0, 2)).astype(np.float32)
                y_RtB = np.transpose(ys, (1, 0)).astype(np.float32)
                preds_MB = self._forward_one_batch(X_RiBC, y_RtB)
                # (M, B) — collect per-variant preds
                all_preds.append(preds_MB.T)  # (B, M)
            arr = np.concatenate(all_preds, axis=0)   # (total_variants, M)
            mean_scaled = arr.mean(axis=0)            # (M,)

        if self.y_scaler_ is not None:
            return self.y_scaler_.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        return mean_scaled
