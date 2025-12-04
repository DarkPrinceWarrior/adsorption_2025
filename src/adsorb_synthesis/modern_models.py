"""Modern ensemble models for tabular data (2025)."""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .config import CATBOOST_MODEL_CONFIG
from .constants import HUBER_DELTA_DEFAULT


logger = logging.getLogger(__name__)


def _try_import_smote():
    """Try to import SMOTE/ADASYN from imbalanced-learn."""
    try:
        from imblearn.over_sampling import ADASYN, SMOTE
        return SMOTE, ADASYN
    except ImportError:
        return None, None


def _compute_focal_class_weights(y: np.ndarray, gamma: float = 2.0) -> Dict[int, float]:
    """Compute class weights for focal loss effect."""
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = {}
    for i, cls in enumerate(classes):
        # Scale weights exponentially for rare classes
        freq = np.sum(y == cls) / len(y)
        weight_dict[int(cls)] = float(class_weights[i] * (1.0 / (freq + 1e-6)) ** gamma)
    return weight_dict


def _apply_smote_if_available(X: np.ndarray, y: np.ndarray, random_state: int) -> tuple:
    """Apply SMOTE/ADASYN if available, otherwise return original data."""
    SMOTE, ADASYN = _try_import_smote()
    if SMOTE is None:
        warnings.warn(
            "imbalanced-learn not installed; SMOTE/ADASYN resampling will be skipped. "
            "Install with: pip install imbalanced-learn>=0.11.0",
            UserWarning,
        )
        return X, y
    
    try:
        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        
        # Only apply if we have enough samples and imbalance exists
        if min_samples >= 2 and len(unique) > 1 and counts.max() / min_samples > 1.5:
            # Try ADASYN first (adaptive synthetic sampling)
            try:
                sampler = ADASYN(random_state=random_state, n_neighbors=min(5, min_samples - 1))
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                return X_resampled, y_resampled
            except Exception:
                # Fall back to SMOTE if ADASYN fails
                try:
                    sampler = SMOTE(random_state=random_state, k_neighbors=min(5, min_samples - 1))
                    X_resampled, y_resampled = sampler.fit_resample(X, y)
                    return X_resampled, y_resampled
                except Exception:
                    return X, y
        return X, y
    except Exception:
        return X, y


def _import_catboost(problem_type: str):
    try:
        import catboost as cb
    except ImportError as exc:
        raise ImportError(
            "catboost is required for ModernTabularEnsemble. "
            "Install it with `pip install catboost`."
        ) from exc
    return cb.CatBoostClassifier if problem_type == "classification" else cb.CatBoostRegressor


def _default_catboost_params(problem_type: str, random_state: int) -> Dict:
    return CATBOOST_MODEL_CONFIG.to_params(problem_type, random_state)


class _ModernTabularEnsemble(BaseEstimator):
    def __init__(
        self,
        problem_type: str,
        catboost_params: Optional[Dict] = None,
        n_estimators: int = 5,
        validation_fraction: float = 0.2,
        optimize_weights: bool = True,
        weight_opt_params: Optional[Dict] = None,
        use_smote: bool = True,
        focal_gamma: float = 2.0,
        random_state: int = 42,
        physics_loss_fn: Optional[Callable] = None,
        physics_loss_weight: float = 0.0,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        self.problem_type = problem_type
        self.catboost_params = catboost_params
        self.n_estimators = n_estimators
        self.validation_fraction = validation_fraction
        self.optimize_weights = optimize_weights
        self.weight_opt_params = weight_opt_params
        self.use_smote = use_smote
        self.focal_gamma = focal_gamma
        self.random_state = random_state
        self.physics_loss_fn = physics_loss_fn
        self.physics_loss_weight = physics_loss_weight
        self.feature_names = feature_names

    def _make_catboost(self, seed: int):
        cls = _import_catboost(self.problem_type)
        params = _default_catboost_params(self.problem_type, seed)
        if self.catboost_params:
            params.update(dict(self.catboost_params))
        params['random_seed'] = seed
        return cls(**params)

    def _compute_physics_sample_weights(self, frames: Optional[pd.DataFrame], base_weights: np.ndarray) -> np.ndarray:
        """Compute sample weights incorporating physics violations."""
        if (
            self.physics_loss_fn is None
            or self.physics_loss_weight <= 0
            or frames is None
            or frames.empty
        ):
            return base_weights

        try:
            if isinstance(frames, pd.DataFrame):
                feature_names = list(frames.columns)
                frame_values = frames.to_numpy(dtype=np.float64, copy=False)
            else:
                frame_values = np.asarray(frames, dtype=np.float64)
                feature_names = self.feature_names

            if feature_names is None:
                raise ValueError("feature names are required for physics loss computation")

            violations = self.physics_loss_fn(frame_values, feature_names=feature_names)
            violations = np.asarray(violations, dtype=np.float64).reshape(-1)
            if violations.shape[0] != base_weights.shape[0]:
                warnings.warn(
                    "Physics loss vector length mismatch; falling back to base weights.",
                    RuntimeWarning,
                )
                return base_weights
            # Normalise violations to [0, 1] and construct weights
            violations = np.nan_to_num(violations, nan=0.0, posinf=0.0, neginf=0.0)
            finite = violations[np.isfinite(violations)]
            if not finite.size:
                return base_weights
            scale = np.nanpercentile(finite, 95)
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = np.nanmax(finite)
            if scale <= 0:
                return base_weights
            normalized = np.clip(violations / scale, 0.0, None)
            physics_weights = 1.0 + self.physics_loss_weight * normalized
            return base_weights * physics_weights
        except Exception as exc:
            warnings.warn(
                f"Physics loss computation failed: {exc}. Using base weights.",
                RuntimeWarning,
            )
            return base_weights

    def _normalise_proba(self, proba: np.ndarray, n_classes: int) -> np.ndarray:
        arr = np.asarray(proba, dtype=np.float64)
        if arr.ndim == 1:
            arr = np.vstack([1 - arr, arr]).T
        if arr.shape[1] == 1 and n_classes == 2:
            arr = np.hstack([1 - arr, arr])
        if arr.shape[1] < n_classes:
            pad = np.zeros((arr.shape[0], n_classes - arr.shape[1]), dtype=arr.dtype)
            arr = np.hstack([arr, pad])
        row_sums = arr.sum(axis=1, keepdims=True)
        zero_mask = row_sums == 0
        if np.any(zero_mask):
            arr[zero_mask, :] = 1.0 / max(arr.shape[1], 1)
            row_sums = arr.sum(axis=1, keepdims=True)
        return arr / row_sums

    def _optimize_weights(
        self,
        predictions: np.ndarray,
        y_val: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n_models = predictions.shape[0]
        if n_models == 1 or not self.optimize_weights:
            return np.ones(n_models, dtype=np.float64) / n_models

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim != 1 or sample_weight.shape[0] != y_val.shape[0]:
                raise ValueError("sample_weight must be 1D and match validation targets.")
            sw = sample_weight.copy()
            sw = sw / sw.sum() if sw.sum() > 0 else None
        else:
            sw = None

        try:
            from scipy.optimize import minimize
        except ImportError:
            warnings.warn("SciPy not available; using uniform ensemble weights.", RuntimeWarning)
            return np.ones(n_models, dtype=np.float64) / n_models

        initial = np.ones(n_models, dtype=np.float64) / n_models
        l2_penalty = 0.01  # L2 regularization strength

        if self.problem_type == "classification":
            n_classes = predictions.shape[-1]

            def objective(weights: np.ndarray) -> float:
                weights = np.clip(weights, 0.0, None)
                total = weights.sum()
                if total == 0:
                    return np.inf
                w_norm = weights / total
                ensemble = np.tensordot(w_norm, predictions, axes=(0, 0))
                ensemble = np.clip(ensemble, 1e-6, 1 - 1e-6)

                # Use Brier score for better calibration
                if n_classes == 2:
                    # Binary case: use brier_score_loss
                    if sw is not None:
                        brier = float(brier_score_loss(y_val, ensemble[:, 1], sample_weight=sw))
                    else:
                        brier = float(brier_score_loss(y_val, ensemble[:, 1]))
                else:
                    # Multi-class: average Brier score across all classes
                    y_one_hot = np.eye(n_classes)[y_val.astype(int)]
                    diff = (ensemble - y_one_hot) ** 2
                    if sw is not None:
                        brier = float(np.average(np.sum(diff, axis=1), weights=sw))
                    else:
                        brier = float(np.mean(np.sum(diff, axis=1)))

                # Add L2 regularization to prevent extreme weights
                reg = l2_penalty * float(np.sum((w_norm - initial) ** 2))
                return brier + reg
        else:
            def objective(weights: np.ndarray) -> float:
                weights = np.clip(weights, 0.0, None)
                total = weights.sum()
                if total == 0:
                    return np.inf
                w_norm = weights / total
                ensemble = np.tensordot(w_norm, predictions, axes=(0, 0))

                # Use MAE instead of MSE for robustness
                if sw is not None:
                    mae = float(mean_absolute_error(y_val, ensemble, sample_weight=sw))
                else:
                    mae = float(mean_absolute_error(y_val, ensemble))
                # Add L2 regularization
                reg = l2_penalty * float(np.sum((w_norm - initial) ** 2))
                return mae + reg

        constraints = {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}
        bounds = [(0.0, 1.0)] * n_models

        try:
            result = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                **(self.weight_opt_params or {}),
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Weight optimisation failed; using uniform weights. Reason: {exc}",
                RuntimeWarning,
            )
            return initial

        if not result.success:
            warnings.warn(
                f"Weight optimisation did not converge ({result.message}); using uniform weights.",
                RuntimeWarning,
            )
            return initial

        weights = np.clip(result.x, 0.0, None)
        return weights / weights.sum()

    def get_feature_importance(self) -> np.ndarray:
        check_is_fitted(self, "weights_")
        importances: List[np.ndarray] = []
        weights: List[float] = []

        for weight, model in zip(self.weights_, self.models_):
            if hasattr(model, "get_feature_importance"):
                importances.append(np.asarray(model.get_feature_importance(), dtype=np.float64))
                weights.append(weight)

        if not importances:
            raise ValueError("Feature importance is not available for any base model.")

        stacked = np.vstack(importances)
        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
        return np.average(stacked, axis=0, weights=weights_arr)


class ModernTabularEnsembleClassifier(_ModernTabularEnsemble, ClassifierMixin):
    """Ensemble of 5 CatBoost models for classification with calibration."""

    def __init__(
        self,
        catboost_params: Optional[Dict] = None,
        n_estimators: int = 5,
        validation_fraction: float = 0.2,
        optimize_weights: bool = True,
        weight_opt_params: Optional[Dict] = None,
        calibrate_predictions: bool = True,
        calibration_method: str = "isotonic",
        use_smote: bool = True,
        focal_gamma: float = 2.0,
        random_state: int = 42,
        physics_loss_fn: Optional[Callable] = None,
        physics_loss_weight: float = 0.0,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            problem_type="classification",
            catboost_params=catboost_params,
            n_estimators=n_estimators,
            validation_fraction=validation_fraction,
            optimize_weights=optimize_weights,
            weight_opt_params=weight_opt_params,
            use_smote=use_smote,
            focal_gamma=focal_gamma,
            random_state=random_state,
            physics_loss_fn=physics_loss_fn,
            physics_loss_weight=physics_loss_weight,
            feature_names=feature_names,
        )
        self.calibrate_predictions = calibrate_predictions
        self.calibration_method = calibration_method

    def fit(self, X, y, sample_weight=None, physics_frame: Optional[pd.DataFrame] = None):
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, ensure_min_samples=2)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_arr)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
            if sample_weight.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must match number of samples.")
        else:
            sample_weight = None

        counts = np.bincount(y_encoded)
        stratify = y_encoded if self.n_classes_ > 1 and np.all(counts >= 2) else None

        physics_train = physics_val = None
        if sample_weight is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val, train_idx, val_idx = train_test_split(
                X_arr,
                y_encoded,
                sample_weight,
                np.arange(X_arr.shape[0]),
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        else:
            X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
                X_arr,
                y_encoded,
                np.arange(X_arr.shape[0]),
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
            sw_train = sw_val = None

        if physics_frame is not None:
            physics_train = physics_frame.iloc[train_idx].reset_index(drop=True)
            physics_val = physics_frame.iloc[val_idx].reset_index(drop=True)

        if physics_train is not None and self.physics_loss_fn is not None and self.physics_loss_weight > 0:
            base_train = sw_train.copy() if sw_train is not None else np.ones(X_train.shape[0], dtype=np.float32)
            sw_train = self._compute_physics_sample_weights(physics_train, base_train)
            base_val = sw_val.copy() if sw_val is not None else np.ones(X_val.shape[0], dtype=np.float32)
            sw_val = self._compute_physics_sample_weights(physics_val, base_val)

        # Apply SMOTE/ADASYN to training data if enabled
        if self.use_smote and self.n_classes_ > 1 and sw_train is None:
            logger.info("Applying SMOTE/ADASYN for class balancing (n_classes=%s)", self.n_classes_)
            X_train_orig_shape = X_train.shape
            X_train, y_train = _apply_smote_if_available(X_train, y_train, self.random_state)
            if X_train.shape[0] != X_train_orig_shape[0]:
                logger.info("SMOTE applied: %s -> %s samples", X_train_orig_shape[0], X_train.shape[0])
            else:
                logger.info("SMOTE not applied (balanced or insufficient samples)")
        elif sw_train is not None and self.use_smote:
            logger.info("Skipping SMOTE due to provided sample weights.")

        # Compute focal class weights
        focal_weights = _compute_focal_class_weights(y_train, gamma=self.focal_gamma)
        logger.info("Focal weights computed (gamma=%s): %s", self.focal_gamma, focal_weights)

        models: List = []
        names: List[str] = []
        val_predictions: List[np.ndarray] = []

        for i in range(self.n_estimators):
            seed = self.random_state + i
            # CatBoost
            try:
                cat_model = self._make_catboost(seed=seed)
                if self.n_classes_ > 2:
                    cat_model.set_params(
                        auto_class_weights='None',
                        loss_function="MultiClass",
                        eval_metric="TotalF1",
                        class_weights=list(focal_weights.values()),
                    )
                else:
                    cat_model.set_params(
                        auto_class_weights='None',
                        loss_function="Logloss",
                        eval_metric="Logloss",
                        class_weights=list(focal_weights.values()),
                    )
                cat_model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=100,
                    verbose=False,
                    sample_weight=sw_train,
                )
                models.append(cat_model)
                names.append(f"catboost_{i}")
                val_predictions.append(self._normalise_proba(cat_model.predict_proba(X_val), self.n_classes_))
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"CatBoost training failed for seed {seed} and will be skipped: {exc}", RuntimeWarning)

        if not models:
            raise RuntimeError("All base models failed to train; cannot fit ensemble.")

        # Apply calibration to base models if requested
        if self.calibrate_predictions:
            calibrated_models: List = []
            calibrated_predictions: List[np.ndarray] = []

            for model, name in zip(models, names):
                try:
                    # Create a calibrated version of the model
                    cal_model = CalibratedClassifierCV(
                        model,
                        method=self.calibration_method,
                        cv="prefit",
                        ensemble=False,
                    )
                    # Use validation set for calibration
                    if sw_val is not None:
                        cal_model.fit(X_val, y_val, sample_weight=sw_val)
                    else:
                        cal_model.fit(X_val, y_val)
                    calibrated_models.append(cal_model)
                    calibrated_predictions.append(
                        self._normalise_proba(cal_model.predict_proba(X_val), self.n_classes_)
                    )
                except Exception as exc:  # pragma: no cover
                    warnings.warn(
                        f"Calibration failed for {name}, using uncalibrated model: {exc}",
                        RuntimeWarning,
                    )
                    calibrated_models.append(model)
                    calibrated_predictions.append(val_predictions[len(calibrated_models) - 1])
            
            self.models_ = calibrated_models
            val_predictions = calibrated_predictions
        else:
            self.models_ = models

        self.model_names_ = names
        val_stack = np.stack(val_predictions, axis=0)
        self.weights_ = self._optimize_weights(val_stack, y_val, sample_weight=sw_val)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "weights_")
        X_arr = check_array(X, accept_sparse=False)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        probs: List[np.ndarray] = []
        for model in self.models_:
            model_proba = model.predict_proba(X_arr)
            probs.append(self._normalise_proba(model_proba, self.n_classes_))
        prob_stack = np.stack(probs, axis=0)
        weights = self.weights_ / self.weights_.sum()
        ensemble = np.tensordot(weights, prob_stack, axes=(0, 0))
        return np.clip(ensemble, 1e-6, 1 - 1e-6)

    def predict(self, X):
        probs = self.predict_proba(X)
        labels = np.argmax(probs, axis=1)
        return self._label_encoder.inverse_transform(labels)


class ModernTabularEnsembleRegressor(_ModernTabularEnsemble, RegressorMixin):
    """Ensemble of 5 CatBoost models for regression with robust losses."""

    def __init__(
        self,
        catboost_params: Optional[Dict] = None,
        n_estimators: int = 5,
        validation_fraction: float = 0.2,
        optimize_weights: bool = True,
        weight_opt_params: Optional[Dict] = None,
        use_quantile: bool = False,
        quantile_alpha: float = 0.5,
        huber_delta: float = HUBER_DELTA_DEFAULT,
        random_state: int = 42,
        physics_loss_fn: Optional[Callable] = None,
        physics_loss_weight: float = 0.0,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            problem_type="regression",
            catboost_params=catboost_params,
            n_estimators=n_estimators,
            validation_fraction=validation_fraction,
            optimize_weights=optimize_weights,
            weight_opt_params=weight_opt_params,
            use_smote=False,  # Not applicable for regression
            random_state=random_state,
            physics_loss_fn=physics_loss_fn,
            physics_loss_weight=physics_loss_weight,
            feature_names=feature_names,
        )
        self.use_quantile = use_quantile
        self.quantile_alpha = quantile_alpha
        self.huber_delta = huber_delta

    def fit(self, X, y, sample_weight=None, physics_frame: Optional[pd.DataFrame] = None):
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        y_arr = np.asarray(y_arr, dtype=np.float32)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
            if sample_weight.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must match number of samples.")
            X_train, X_val, y_train, y_val, sw_train, sw_val, train_idx, val_idx = train_test_split(
                X_arr,
                y_arr,
                sample_weight,
                np.arange(X_arr.shape[0]),
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
        else:
            X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
                X_arr,
                y_arr,
                np.arange(X_arr.shape[0]),
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
            sw_train = sw_val = None

        physics_train = physics_val = None
        if physics_frame is not None:
            physics_train = physics_frame.iloc[train_idx].reset_index(drop=True)
            physics_val = physics_frame.iloc[val_idx].reset_index(drop=True)

        if physics_train is not None and self.physics_loss_fn is not None and self.physics_loss_weight > 0:
            base_train = sw_train.copy() if sw_train is not None else np.ones(X_train.shape[0], dtype=np.float32)
            sw_train = self._compute_physics_sample_weights(physics_train, base_train)
            base_val = sw_val.copy() if sw_val is not None else np.ones(X_val.shape[0], dtype=np.float32)
            sw_val = self._compute_physics_sample_weights(physics_val, base_val)

        models: List = []
        names: List[str] = []
        val_predictions: List[np.ndarray] = []

        for i in range(self.n_estimators):
            seed = self.random_state + i
            # CatBoost
            try:
                cat_model = self._make_catboost(seed=seed)
                if self.use_quantile:
                    logger.info("Using Quantile Regression (alpha=%s) for CatBoost", self.quantile_alpha)
                    cat_model.set_params(
                        loss_function=f"Quantile:alpha={self.quantile_alpha}",
                        eval_metric="MAE",
                        subsample=0.85,
                        rsm=0.85,
                    )
                else:
                    # Use MAE for robustness
                    logger.info("Using MAE loss for regression (CatBoost)")
                    cat_model.set_params(
                        loss_function="MAE",
                        eval_metric="MAE",
                        subsample=0.85,
                        rsm=0.85,
                    )
                cat_model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=100,
                    verbose=False,
                    sample_weight=sw_train,
                )
                models.append(cat_model)
                names.append(f"catboost_{i}")
                val_predictions.append(np.asarray(cat_model.predict(X_val), dtype=np.float64).ravel())
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"CatBoost training failed for seed {seed} and will be skipped: {exc}", RuntimeWarning)

        if not models:
            raise RuntimeError("All base models failed to train; cannot fit ensemble.")

        self.models_ = models
        self.model_names_ = names
        val_stack = np.stack(val_predictions, axis=0)
        self.weights_ = self._optimize_weights(val_stack, y_val, sample_weight=sw_val)
        return self

    def predict(self, X):
        check_is_fitted(self, "weights_")
        X_arr = check_array(X, accept_sparse=False)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        preds: List[np.ndarray] = []
        for model in self.models_:
            preds.append(np.asarray(model.predict(X_arr), dtype=np.float64).ravel())
        pred_stack = np.vstack(preds)
        weights = self.weights_ / self.weights_.sum()
        return np.dot(weights, pred_stack).astype(np.float32)

    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation (uncertainty).
        
        Uncertainty is estimated as the weighted standard deviation of the 
        predictions from the ensemble members.
        This captures epistemic uncertainty (model disagreement).
        """
        check_is_fitted(self, "weights_")
        X_arr = check_array(X, accept_sparse=False)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        
        # Get predictions from all base models
        preds: List[np.ndarray] = []
        for model in self.models_:
            preds.append(np.asarray(model.predict(X_arr), dtype=np.float64).ravel())
        
        # Shape: (n_models, n_samples)
        pred_stack = np.vstack(preds)
        
        # Weighted Mean
        weights = self.weights_ / self.weights_.sum()
        mean_pred = np.dot(weights, pred_stack)
        
        # Weighted Standard Deviation
        # sigma^2 = sum(w_i * (x_i - mu)^2)
        variance = np.dot(weights, (pred_stack - mean_pred) ** 2)
        std_pred = np.sqrt(variance)
        
        return mean_pred.astype(np.float32), std_pred.astype(np.float32)


__all__ = [
    "ModernTabularEnsembleClassifier",
    "ModernTabularEnsembleRegressor",
]
