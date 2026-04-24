"""TabPFNRegressorMLXNative — uses upstream TabPFN preprocessing end-to-end.

Reuses TabPFNRegressor.fit() for all CPU preprocessing (feature modality
detection, ordinal encoding, per-estimator Yeo-Johnson / quantile / robust
transforms, feature subsets, y z-score + per-estimator target transforms).

The only thing we replace is the PyTorch model forward pass — we run our
MLX port instead, then feed logits back into TabPFN's own border-translation
and ensemble-averaging path.

Expected outcome: identical RMSE (Overall / Cliff / NonCliff) to PyTorch
TabPFN, at Apple-Silicon speeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
import torch

from sklearn.base import BaseEstimator, TransformerMixin

from .model import TabPFNV2p6, TabPFNV2p6Config
from .convert import convert_checkpoint


class _MLXTruncatedSVD(TransformerMixin, BaseEstimator):
    """sklearn-compatible TruncatedSVD using randomized SVD on Metal.

    Halko-Martinsson-Tropp with ``n_iter`` power-iteration steps. The heavy
    matmul (X @ Ω, X.T @ Y) runs on Metal; the small QR + final (k+p, k+p)
    SVD run on the MLX CPU stream (Metal QR/SVD aren't landed yet).

    Benchmarks on Apple Silicon:
        shape=(633, 128)     :   4ms vs scipy ARPACK 406ms    (~100×)
        shape=(2000, 512)    :  21ms vs scipy ARPACK 17s      (~800×)
        shape=(10000, 2048)  :  39ms vs scipy ARPACK 40s      (~1000×)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_oversamples: int = 10,
        n_iter: int = 4,
        random_state: int | None = 42,
    ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X_np = np.ascontiguousarray(X, dtype=np.float32)
        n_samples, n_features = X_np.shape
        k = min(self.n_components, min(n_samples, n_features))
        p = self.n_oversamples
        kp = k + p

        rng = np.random.default_rng(self.random_state)
        Omega = rng.standard_normal((n_features, kp)).astype(np.float32)

        X_mx = mx.array(X_np)
        Omega_mx = mx.array(Omega)

        # Power iteration (Metal matmul)
        Y = X_mx @ Omega_mx
        for _ in range(self.n_iter):
            Y = X_mx @ (X_mx.T @ Y)
            mx.eval(Y)

        # QR (CPU), project (Metal), small SVD (CPU)
        with mx.stream(mx.cpu):
            Q, _ = mx.linalg.qr(Y)
            mx.eval(Q)
        B = Q.T @ X_mx
        mx.eval(B)
        with mx.stream(mx.cpu):
            _, S, Vt = mx.linalg.svd(B)
            mx.eval(S, Vt)

        Vt_np = np.array(Vt[:k, :], dtype=np.float32)
        # svd_flip (Vt-based): for each row of Vt, force the max-abs entry positive.
        max_abs_cols = np.argmax(np.abs(Vt_np), axis=1)
        signs = np.sign(Vt_np[np.arange(k), max_abs_cols])
        signs[signs == 0] = 1.0
        Vt_np = Vt_np * signs[:, None]

        self.components_ = Vt_np
        self.singular_values_ = np.array(S[:k], dtype=np.float32)
        return self

    def transform(self, X):
        X_np = np.ascontiguousarray(X, dtype=np.float32)
        return X_np @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class TabPFNRegressorMLXNative:
    """sklearn-style regressor using TabPFN's own preprocessing + MLX forward."""

    def __init__(
        self,
        ckpt_path: str | Path,
        cache_dir: str | Path = "tabpfn_mlx_cache",
        n_estimators: int = 8,
        random_state: int = 42,
        ignore_pretraining_limits: bool = True,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        fast_svd: bool = True,
        prebuilt_mlx_model: Optional[TabPFNV2p6] = None,
        prebuilt_pre_column_embeddings: Optional[np.ndarray] = None,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.cache_dir = Path(cache_dir)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.fast_svd = fast_svd
        self.prebuilt_mlx_model = prebuilt_mlx_model
        self.prebuilt_pre_column_embeddings = prebuilt_pre_column_embeddings

    @staticmethod
    def build_mlx_model(
        ckpt_path: str | Path, cache_dir: str | Path = "tabpfn_mlx_cache"
    ) -> tuple[TabPFNV2p6, np.ndarray]:
        """Build MLX model + fetch pre-generated column embeddings once.

        Pass the returned objects as ``prebuilt_mlx_model`` + ``prebuilt_pre_column_embeddings``
        to many regressor instances to avoid re-materializing 13M params per fit().
        """
        ckpt_path = Path(ckpt_path); cache_dir = Path(cache_dir)
        stem = ckpt_path.stem
        npz = cache_dir / f"{stem}_mlx.npz"
        jsn = cache_dir / f"{stem}_mlx.json"
        if not (npz.exists() and jsn.exists()):
            convert_checkpoint(ckpt_path, cache_dir)
        with open(jsn) as f:
            cfg = TabPFNV2p6Config(**json.load(f))
        model = TabPFNV2p6(cfg)
        raw = dict(np.load(str(npz)))
        _ = raw.pop("borders")
        model.load_weights([(k, mx.array(v)) for k, v in raw.items()])
        mx.eval(model.parameters())

        from tabpfn.architectures.shared.column_embeddings import load_column_embeddings
        pre = load_column_embeddings().detach().numpy()
        return model, pre

    @staticmethod
    def _patch_pt_model_cache():
        """Cache ``initialize_tabpfn_model`` output so running many fits doesn't
        reload the 50M-param PyTorch model each time.

        Fixes the 200+ GB blow-up when looping over many configs in a benchmark.
        """
        from tabpfn import base as _base
        if getattr(_base, "_mlx_model_cache_patched", False):
            return
        _orig = _base.initialize_tabpfn_model
        _cache: dict = {}

        def cached(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items(), key=lambda kv: kv[0])))
            try:
                if key not in _cache:
                    _cache[key] = _orig(*args, **kwargs)
                return _cache[key]
            except TypeError:
                # Un-hashable arg (e.g. Path object) — fall back to no-cache
                return _orig(*args, **kwargs)

        _base.initialize_tabpfn_model = cached
        _base._mlx_model_cache_patched = True

    @staticmethod
    def _patch_svd_to_mlx():
        """Replace ARPACK TruncatedSVD (scipy Fortran, ~99% of fit time) with MLX SVD.

        MLX's ``mx.linalg.svd`` runs on the CPU stream (Metal SVD isn't in MLX yet)
        but is still ~500× faster than scipy ARPACK's iterative eigensolver for our
        ~(1000, 128) matrices.
        """
        from tabpfn.preprocessing.steps import add_svd_features_step as mod
        if getattr(mod, "_mlx_svd_patched", False):
            return

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from tabpfn.preprocessing.steps.utils import make_scaler_safe

        def mlx_get_svd_features_transformer(
            global_transformer_name, n_samples, n_features, random_state=None
        ):
            divisor = 2 if global_transformer_name == "svd" else 4
            n_components = max(1, min(n_samples // 10 + 1, n_features // divisor))
            return Pipeline(steps=[
                ("save_standard", make_scaler_safe("standard", StandardScaler(with_mean=False))),
                ("svd", _MLXTruncatedSVD(n_components=n_components, random_state=random_state)),
            ])

        mod.get_svd_features_transformer = mlx_get_svd_features_transformer
        mod._mlx_svd_patched = True

    def _load_mlx_model(self) -> None:
        if self.prebuilt_mlx_model is not None:
            self.mlx_model_ = self.prebuilt_mlx_model
            self.mlx_cfg_ = self.prebuilt_mlx_model.config
            self.pre_column_embeddings_ = (
                self.prebuilt_pre_column_embeddings
                if self.prebuilt_pre_column_embeddings is not None
                else np.zeros((0, self.mlx_cfg_.emsize // 4), dtype=np.float32)
            )
            return
        self.mlx_model_, self.pre_column_embeddings_ = self.build_mlx_model(
            self.ckpt_path, self.cache_dir
        )
        self.mlx_cfg_ = self.mlx_model_.config

    def _column_seeds_for(self, n_features: int) -> mx.array:
        F = self.mlx_cfg_.features_per_group
        G = (n_features + ((-n_features) % F)) // F
        eq = self.mlx_cfg_.emsize // 4
        gen = torch.Generator().manual_seed(42)
        seeds = torch.randn((G, eq), generator=gen).numpy().astype(np.float32)
        pre = self.pre_column_embeddings_
        if pre.shape[1] == eq:
            take = min(G, pre.shape[0])
            seeds[:take] = pre[:take]
        return mx.array(seeds)

    def _mlx_forward(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> torch.Tensor:
        """Run MLX forward, return logits as torch.Tensor (M, 1, num_buckets)."""
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_full = np.concatenate([X_train, X_test], axis=0)[:, None, :]
        col_mx = self._column_seeds_for(X_full.shape[-1])

        logits_mx = self.mlx_model_(
            mx.array(X_full), mx.array(y_train), col_mx
        )
        mx.eval(logits_mx)
        return torch.from_numpy(np.array(logits_mx))  # (M, 1, num_buckets)

    def fit(self, X, y) -> "TabPFNRegressorMLXNative":
        """Fit via TabPFN's upstream preprocessing, then load MLX model."""
        # Patches are idempotent — safe to call every fit.
        self._patch_pt_model_cache()
        if self.fast_svd:
            self._patch_svd_to_mlx()
        from tabpfn import TabPFNRegressor
        # Use TabPFN's own fit to do all preprocessing. The PyTorch model is loaded
        # inside but only ever called by our overridden predict path.
        self.pt_regressor_ = TabPFNRegressor(
            model_path=str(self.ckpt_path),
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            device="cpu",  # keep tensors on CPU; we route forward through MLX anyway
            n_jobs=1,
            softmax_temperature=self.softmax_temperature,
            average_before_softmax=self.average_before_softmax,
        )
        self.pt_regressor_.fit(X, y)
        self._load_mlx_model()
        return self

    def predict(self, X) -> np.ndarray:
        """Run upstream pre/post-processing with MLX model in the middle."""
        from tabpfn.validation import ensure_compatible_predict_input_sklearn
        from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
        from tabpfn.utils import translate_probs_across_borders, transform_borders_one
        from tabpfn.preprocessing.configs import RegressorEnsembleConfig

        reg = self.pt_regressor_
        # Validate & clean test X the same way upstream does
        X = ensure_compatible_predict_input_sklearn(X, reg)
        from tabpfn.preprocessing.datamodel import FeatureModality
        cat_idx = reg.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)
        X = fix_dtypes(X, cat_indices=cat_idx)
        X = process_text_na_dataframe(X, ord_encoder=getattr(reg, "ordinal_encoder_", None))

        std_borders = reg.znorm_space_bardist_.borders.cpu().numpy()
        accumulated = None
        n_ok = 0

        for member in reg.executor_.ensemble_members:
            # Transform test input through this member's CPU preprocessor
            X_test_arr = member.transform_X_test(X)
            if hasattr(X_test_arr, "numpy"):
                X_test_arr = X_test_arr.numpy()

            X_train = member.X_train
            y_train = member.y_train
            if hasattr(X_train, "numpy"): X_train = X_train.numpy()
            if hasattr(y_train, "numpy"): y_train = y_train.numpy()

            # Run MLX forward
            logits = self._mlx_forward(X_train, y_train, X_test_arr).float()
            if self.softmax_temperature != 1:
                logits = logits / self.softmax_temperature

            config = member.config
            if isinstance(config, list) and len(config) == 1:
                config = config[0]

            if not isinstance(config, RegressorEnsembleConfig):
                raise ValueError("Non-regression config returned")

            # Border translation to common znorm space (same as upstream)
            if config.target_transform is None:
                borders_t = std_borders.copy()
                logit_cancel_mask = None
                descending = False
            else:
                logit_cancel_mask, descending, borders_t = transform_borders_one(
                    std_borders,
                    target_transform=config.target_transform,
                    repair_nan_borders_after_transform=reg.inference_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                )
                if descending:
                    borders_t = borders_t.flip(-1)

            if logit_cancel_mask is not None:
                logits = logits.clone()
                logits[..., logit_cancel_mask] = float("-inf")

            transformed = translate_probs_across_borders(
                logits,
                frm=torch.as_tensor(borders_t),
                to=reg.znorm_space_bardist_.borders,
            )
            if self.average_before_softmax:
                transformed = transformed.log()
            accumulated = transformed if accumulated is None else accumulated + transformed
            n_ok += 1

        if self.average_before_softmax:
            probs = (accumulated / n_ok).softmax(dim=-1)
        else:
            probs = accumulated / n_ok

        logits = probs.log()
        if logits.dtype == torch.float16:
            logits = logits.float()

        mean = reg.raw_space_bardist_.mean(logits).squeeze(-1).numpy()
        # mean shape: (1, n_test) → flatten
        return mean.reshape(-1)
