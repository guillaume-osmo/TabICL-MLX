"""sklearn-compatible TabICL regressor using MLX.

Drop-in replacement for tabicl.TabICLRegressor that runs on MLX
instead of PyTorch, enabling Apple Silicon GPU acceleration without
the PyTorch MPS / MLX Metal conflict.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .model import TabICL
from .convert import convert_from_huggingface, convert_checkpoint


class TabICLRegressorMLX(RegressorMixin, BaseEstimator):
    """TabICL regressor using MLX for inference.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.
    norm_methods : list of str or None, default=None
        Normalization methods for the ensemble.
        If None, uses ["none", "power"].
    feat_shuffle_method : str, default="latin"
        Feature shuffle method for ensemble diversity.
    outlier_threshold : float, default=4.0
        Z-score threshold for outlier removal.
    batch_size : int or None, default=8
        Batch size for inference. If None, uses all estimators at once.
    model_path : str or Path or None, default=None
        Path to pre-converted MLX weights (.npz).
        If None, downloads and converts from HuggingFace.
    checkpoint_version : str, default="reg-v2.ckpt"
        HuggingFace checkpoint version (used if model_path is None).
    random_state : int or None, default=42
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[List[str]] = None,
        feat_shuffle_method: str = "latin",
        outlier_threshold: float = 4.0,
        batch_size: Optional[int] = 8,
        model_path: Optional[str | Path] = None,
        checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt",
        random_state: Optional[int] = 42,
        low_memory: bool = False,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.model_path = model_path
        self.checkpoint_version = checkpoint_version
        self.random_state = random_state
        self.low_memory = low_memory

    def _load_model(self) -> None:
        """Load the MLX TabICL model.

        If model_path points to a .npz file with a matching .json config,
        loads directly. If model_path is a .ckpt (PyTorch checkpoint),
        converts first. If model_path is None, downloads from HuggingFace.
        """
        if self.model_path is not None:
            model_path = Path(self.model_path)

            if model_path.suffix == ".npz":
                # Already converted MLX weights
                config_path = model_path.with_suffix(".json")
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Config file not found at {config_path}. "
                        "Run convert.py first or provide a .ckpt file."
                    )
                with open(config_path) as f:
                    config = json.load(f)
                weights_path = model_path

            elif model_path.suffix == ".ckpt":
                # PyTorch checkpoint -- convert on the fly
                npz_path = model_path.with_name(
                    model_path.stem + "_mlx.npz"
                )
                if not npz_path.exists():
                    print(f"Converting {model_path} to MLX format...")
                    convert_checkpoint(model_path, npz_path)
                config_path = npz_path.with_suffix(".json")
                with open(config_path) as f:
                    config = json.load(f)
                weights_path = npz_path
            else:
                raise ValueError(f"Unsupported model file format: {model_path.suffix}")
        else:
            # Download and convert from HuggingFace
            cache_dir = Path.home() / ".cache" / "tabicl_mlx"
            stem = self.checkpoint_version.replace(".ckpt", "")
            weights_path = cache_dir / f"tabicl_{stem}_mlx.npz"
            config_path = cache_dir / f"tabicl_{stem}_mlx.json"

            if not weights_path.exists():
                print("Converting TabICL checkpoint to MLX format...")
                convert_from_huggingface(cache_dir, self.checkpoint_version)

            with open(config_path) as f:
                config = json.load(f)

        # Build model
        self.model_ = TabICL(**config)
        self.model_config_ = config

        # Load weights
        raw_weights = dict(np.load(str(weights_path)))
        weight_list = [(k, mx.array(v)) for k, v in raw_weights.items()]
        self.model_.load_weights(weight_list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabICLRegressorMLX:
        """Fit the regressor.

        Prepares preprocessing (scaler, encoder, ensemble generator) and
        loads the pre-trained TabICL MLX model. The model itself is NOT
        trained -- it uses in-context learning at inference time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        from tabicl_mlx.sklearn.preprocessing import (
            TransformToNumerical,
            EnsembleGenerator,
        )
        from tabicl_mlx.sklearn.sklearn_utils import validate_data

        if y is None:
            raise ValueError("y is required but got None.")

        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        # Scale targets
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).flatten()

        # Transform features to numeric
        self.X_encoder_ = TransformToNumerical(verbose=False)
        X = self.X_encoder_.fit_transform(X)

        # Build ensemble generator
        self.ensemble_generator_ = EnsembleGenerator(
            classification=False,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none", "power"],
            feat_shuffle_method=self.feat_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        self.ensemble_generator_.fit(X, y_scaled)

        # Load MLX model
        self._load_model()

        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])

        return self

    def predict(
        self,
        X: np.ndarray,
        output_type: str = "mean",
    ) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        output_type : "mean", "variance", or "median"

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        from tabicl_mlx.sklearn.sklearn_utils import validate_data

        X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
        X = self.X_encoder_.transform(X)

        # Get ensemble data (train + test combined)
        data = self.ensemble_generator_.transform(X, mode="both")

        all_results = []
        for norm_method, (Xs, ys) in data.items():
            # Xs: (n_estimators, n_samples, n_features)
            # ys: (n_estimators, train_size)

            batch_size = self.batch_size or Xs.shape[0]
            n_batches = int(np.ceil(Xs.shape[0] / batch_size))
            Xs_split = np.array_split(Xs, n_batches)
            ys_split = np.array_split(ys, n_batches)

            for X_batch, y_batch in zip(Xs_split, ys_split):
                X_mx = mx.array(X_batch.astype(np.float32))
                y_mx = mx.array(y_batch.astype(np.float32))

                out = self.model_.predict_stats(
                    X_mx, y_mx, output_type=output_type,
                    eval_between_layers=self.low_memory,
                )
                mx.eval(out)
                all_results.append(np.array(out))

        # Stack and average across ensemble members
        arr = np.concatenate(all_results, axis=0)  # (total_estimators, n_test_samples)
        n_estimators = arr.shape[0]
        n_samples = arr.shape[1]

        # Inverse transform
        arr = self.y_scaler_.inverse_transform(
            arr.reshape(-1, 1)
        ).reshape(n_estimators, n_samples)

        return np.mean(arr, axis=0)
