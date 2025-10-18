"""Modern ensemble models for tabular data (2025)."""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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


def _import_tabnet(problem_type: str):
    try:
        if problem_type == "classification":
            from pytorch_tabnet.tab_model import TabNetClassifier as TabNetModel
        else:
            from pytorch_tabnet.tab_model import TabNetRegressor as TabNetModel
    except ImportError as exc:
        raise ImportError(
            "pytorch-tabnet is required for ModernTabularEnsemble. "
            "Install it with `pip install pytorch-tabnet`."
        ) from exc
    return TabNetModel


def _import_catboost(problem_type: str):
    try:
        import catboost as cb
    except ImportError as exc:
        raise ImportError(
            "catboost is required for ModernTabularEnsemble. "
            "Install it with `pip install catboost`."
        ) from exc
    return cb.CatBoostClassifier if problem_type == "classification" else cb.CatBoostRegressor


def _import_xgboost(problem_type: str):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for ModernTabularEnsemble. "
            "Install it with `pip install xgboost`."
        ) from exc
    return XGBClassifier if problem_type == "classification" else XGBRegressor


def _default_tabnet_params(problem_type: str, random_state: int) -> Dict:
    import torch

    params = dict(
        n_d=32,
        n_a=32,
        n_steps=3,
        gamma=1.25,
        lambda_sparse=1e-5,
        n_independent=1,
        n_shared=1,
        momentum=0.02,
        clip_value=2.0,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=8e-3, weight_decay=1e-4),
        scheduler_params=dict(T_max=200, eta_min=1e-4),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        seed=random_state,
    )
    if problem_type == "classification":
        params["mask_type"] = "sparsemax"
    return params


def _default_catboost_params(problem_type: str, random_state: int) -> Dict:
    base = dict(
        iterations=1600,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        min_data_in_leaf=5,
        bagging_temperature=0.2,
        random_strength=0.8,
        border_count=254,
        random_seed=random_state,
        verbose=False,
    )
    if problem_type == "classification":
        base.setdefault("loss_function", "Logloss")
    else:
        # Use MAE for robustness to outliers
        base.setdefault("loss_function", "MAE")
    return base


def _default_xgboost_params(problem_type: str, random_state: int) -> Dict:
    base = dict(
        n_estimators=1600,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    if problem_type == "classification":
        base["objective"] = "binary:logistic"
        base["use_label_encoder"] = False
        base["eval_metric"] = "logloss"
    else:
        # Use pseudo-Huber loss for robustness to outliers
        base["objective"] = "reg:pseudohubererror"
        base["huber_slope"] = 5.0  # Adjusted to match typical data scale
    return base


class _ModernTabularEnsemble(BaseEstimator):
    def __init__(
        self,
        problem_type: str,
        tabnet_params: Optional[Dict] = None,
        tabnet_fit_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
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
        self.tabnet_params = tabnet_params
        self.tabnet_fit_params = tabnet_fit_params
        self.catboost_params = catboost_params
        self.xgb_params = xgb_params
        self.validation_fraction = validation_fraction
        self.optimize_weights = optimize_weights
        self.weight_opt_params = weight_opt_params
        self.use_smote = use_smote
        self.focal_gamma = focal_gamma
        self.random_state = random_state
        self.physics_loss_fn = physics_loss_fn
        self.physics_loss_weight = physics_loss_weight
        self.feature_names = feature_names

    def _make_tabnet(self):
        TabNet = _import_tabnet(self.problem_type)
        params = _default_tabnet_params(self.problem_type, self.random_state)
        if self.tabnet_params:
            params.update(dict(self.tabnet_params))
        return TabNet(**params)

    def _make_catboost(self):
        cls = _import_catboost(self.problem_type)
        params = _default_catboost_params(self.problem_type, self.random_state)
        if self.catboost_params:
            params.update(dict(self.catboost_params))
        return cls(**params)

    def _make_xgboost(self):
        cls = _import_xgboost(self.problem_type)
        params = _default_xgboost_params(self.problem_type, self.random_state)
        if self.xgb_params:
            params.update(dict(self.xgb_params))
        return cls(**params)

    def _tabnet_fit_kwargs(self, n_samples: int) -> Dict:
        fit_params = self.tabnet_fit_params or {}
        max_epochs = int(fit_params.get("max_epochs", 400))
        patience = int(fit_params.get("patience", 60))
        batch_size = int(min(fit_params.get("batch_size", 128), max(n_samples, 16)))
        batch_size = max(batch_size, 16)
        virtual_batch_size = int(min(fit_params.get("virtual_batch_size", 64), batch_size))
        virtual_batch_size = max(virtual_batch_size, 16)
        return dict(
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
        )

    def _compute_physics_sample_weights(self, X: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
        """Compute sample weights incorporating physics violations."""
        if self.physics_loss_fn is None or self.physics_loss_weight <= 0:
            return base_weights
        
        if self.feature_names is None:
            warnings.warn(
                "feature_names not provided; physics loss cannot be computed. Using base weights.",
                RuntimeWarning,
            )
            return base_weights
        
        try:
            # Compute per-sample physics loss
            # Note: physics_loss_fn should return per-sample losses, not mean
            physics_violations = []
            for i in range(X.shape[0]):
                sample = X[i:i+1, :]
                loss = self.physics_loss_fn(sample, self.feature_names)
                physics_violations.append(loss)
            
            physics_violations = np.array(physics_violations)
            
            # Scale violations to [0, 1] range
            if physics_violations.max() > 0:
                physics_violations = physics_violations / (physics_violations.max() + 1e-6)
            
            # Increase weight for samples with high physics violations
            # Interpretation: train more on "physically wrong" samples to correct them
            physics_weights = 1.0 + self.physics_loss_weight * physics_violations
            
            # Combine with base weights
            combined_weights = base_weights * physics_weights
            
            # Normalize to preserve total weight
            combined_weights = combined_weights * (base_weights.sum() / combined_weights.sum())
            
            return combined_weights
            
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

        for weight, name, model in zip(self.weights_, self.model_names_, self.models_):
            if name == "catboost" and hasattr(model, "get_feature_importance"):
                importances.append(np.asarray(model.get_feature_importance(), dtype=np.float64))
                weights.append(weight)
            elif name == "xgboost" and hasattr(model, "feature_importances_"):
                importances.append(np.asarray(model.feature_importances_, dtype=np.float64))
                weights.append(weight)

        if not importances:
            raise ValueError("Feature importance is not available for any base model.")

        stacked = np.vstack(importances)
        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
        return np.average(stacked, axis=0, weights=weights_arr)


class ModernTabularEnsembleClassifier(_ModernTabularEnsemble, ClassifierMixin):
    """Ensemble of TabNet, CatBoost and XGBoost for classification with calibration."""

    def __init__(
        self,
        tabnet_params: Optional[Dict] = None,
        tabnet_fit_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
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
            tabnet_params=tabnet_params,
            tabnet_fit_params=tabnet_fit_params,
            catboost_params=catboost_params,
            xgb_params=xgb_params,
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

    def fit(self, X, y, sample_weight=None):
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

        if sample_weight is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                X_arr,
                y_encoded,
                sample_weight,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr,
                y_encoded,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
            sw_train = sw_val = None

        # Apply SMOTE/ADASYN to training data if enabled
        if self.use_smote and self.n_classes_ > 1 and sw_train is None:
            print(f"  [INFO] Applying SMOTE/ADASYN for class balancing (n_classes={self.n_classes_})")
            X_train_orig_shape = X_train.shape
            X_train, y_train = _apply_smote_if_available(X_train, y_train, self.random_state)
            if X_train.shape[0] != X_train_orig_shape[0]:
                print(f"  [INFO] SMOTE applied: {X_train_orig_shape[0]} → {X_train.shape[0]} samples")
            else:
                print(f"  [INFO] SMOTE not applied (balanced or insufficient samples)")
        elif sw_train is not None and self.use_smote:
            print("  [INFO] Skipping SMOTE due to provided sample weights.")

        # Compute focal class weights
        focal_weights = _compute_focal_class_weights(y_train, gamma=self.focal_gamma)
        print(f"  [INFO] Focal weights computed (gamma={self.focal_gamma}): {focal_weights}")

        models: List = []
        names: List[str] = []
        val_predictions: List[np.ndarray] = []

        # TabNet
        if sw_train is None:
            try:
                tabnet = self._make_tabnet()
                fit_kwargs = self._tabnet_fit_kwargs(X_train.shape[0])
                tabnet.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    **fit_kwargs,
                )
                models.append(tabnet)
                names.append("tabnet")
                val_predictions.append(self._normalise_proba(tabnet.predict_proba(X_val), self.n_classes_))
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"TabNet training failed and will be skipped: {exc}", RuntimeWarning)
        else:
            print("  [INFO] Skipping TabNet base learner (sample weights unsupported).")

        # CatBoost
        try:
            cat_model = self._make_catboost()
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
            names.append("catboost")
            val_predictions.append(self._normalise_proba(cat_model.predict_proba(X_val), self.n_classes_))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"CatBoost training failed and will be skipped: {exc}", RuntimeWarning)

        # XGBoost
        try:
            xgb_model = self._make_xgboost()
            # Compute sample weights from focal weights
            sample_weights = np.array([focal_weights[int(label)] for label in y_train], dtype=np.float32)
            if sw_train is not None:
                sample_weights = sample_weights * sw_train
            eval_sample_weights = None
            if sw_val is not None:
                eval_sample_weights = np.array(
                    [focal_weights[int(label)] for label in y_val],
                    dtype=np.float32,
                ) * sw_val
            
            if self.n_classes_ > 2:
                xgb_model.set_params(
                    objective="multi:softprob",
                    num_class=self.n_classes_,
                    eval_metric="mlogloss",
                    max_delta_step=1,
                )
            else:
                # For binary, use focal weights via sample_weight
                xgb_model.set_params(
                    objective="binary:logistic",
                    eval_metric=["logloss", "auc"],
                    max_delta_step=1,
                )
            xgb_model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
                sample_weight_eval_set=[eval_sample_weights] if eval_sample_weights is not None else None,
            )
            models.append(xgb_model)
            names.append("xgboost")
            val_predictions.append(self._normalise_proba(xgb_model.predict_proba(X_val), self.n_classes_))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"XGBoost training failed and will be skipped: {exc}", RuntimeWarning)

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
        for name, model in zip(self.model_names_, self.models_):
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
    """Ensemble of TabNet, CatBoost and XGBoost for regression with robust losses."""

    def __init__(
        self,
        tabnet_params: Optional[Dict] = None,
        tabnet_fit_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        validation_fraction: float = 0.2,
        optimize_weights: bool = True,
        weight_opt_params: Optional[Dict] = None,
        use_quantile: bool = False,
        quantile_alpha: float = 0.5,
        huber_delta: float = 1.0,
        random_state: int = 42,
        physics_loss_fn: Optional[Callable] = None,
        physics_loss_weight: float = 0.0,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            problem_type="regression",
            tabnet_params=tabnet_params,
            tabnet_fit_params=tabnet_fit_params,
            catboost_params=catboost_params,
            xgb_params=xgb_params,
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

    def fit(self, X, y, sample_weight=None):
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        y_arr = np.asarray(y_arr, dtype=np.float32)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
            if sample_weight.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must match number of samples.")
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                X_arr,
                y_arr,
                sample_weight,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr,
                y_arr,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
            sw_train = sw_val = None

        models: List = []
        names: List[str] = []
        val_predictions: List[np.ndarray] = []

        # TabNet
        if sw_train is None:
            try:
                tabnet = self._make_tabnet()
                fit_kwargs = self._tabnet_fit_kwargs(X_train.shape[0])
                tabnet.fit(
                    X_train,
                    y_train.reshape(-1, 1),
                    eval_set=[(X_val, y_val.reshape(-1, 1))],
                    **fit_kwargs,
                )
                models.append(tabnet)
                names.append("tabnet")
                val_predictions.append(np.asarray(tabnet.predict(X_val), dtype=np.float64).ravel())
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"TabNet training failed and will be skipped: {exc}", RuntimeWarning)
        else:
            print("  [INFO] Skipping TabNet base learner (sample weights unsupported).")

        # CatBoost
        try:
            cat_model = self._make_catboost()
            if self.use_quantile:
                print(f"  [INFO] Using Quantile Regression (alpha={self.quantile_alpha})")
                cat_model.set_params(
                    loss_function=f"Quantile:alpha={self.quantile_alpha}",
                    eval_metric="MAE",
                    subsample=0.85,
                    rsm=0.85,
                )
            else:
                # Use MAE for robustness (CatBoost Huber doesn't support delta parameter)
                print(f"  [INFO] Using MAE loss for regression (CatBoost)")
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
            names.append("catboost")
            val_predictions.append(np.asarray(cat_model.predict(X_val), dtype=np.float64).ravel())
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"CatBoost training failed and will be skipped: {exc}", RuntimeWarning)

        # XGBoost
        try:
            xgb_model = self._make_xgboost()
            if self.use_quantile:
                print(f"  [INFO] Using Quantile Regression (alpha={self.quantile_alpha}) for XGBoost")
                xgb_model.set_params(
                    objective=f"reg:quantileerror",
                    quantile_alpha=self.quantile_alpha,
                    eval_metric="mae",
                )
            else:
                # Use pseudo-Huber (reg:pseudohubererror) for robustness
                print(f"  [INFO] Using Pseudo-Huber loss (delta={self.huber_delta}) for XGBoost")
                xgb_model.set_params(
                    objective="reg:pseudohubererror",
                    huber_slope=self.huber_delta,
                    eval_metric="mae",
                )
            eval_sample_weights = [sw_val] if sw_val is not None else None
            xgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                sample_weight=sw_train,
                sample_weight_eval_set=eval_sample_weights,
            )
            models.append(xgb_model)
            names.append("xgboost")
            val_predictions.append(np.asarray(xgb_model.predict(X_val), dtype=np.float64).ravel())
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"XGBoost training failed and will be skipped: {exc}", RuntimeWarning)

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


__all__ = [
    "ModernTabularEnsembleClassifier",
    "ModernTabularEnsembleRegressor",
]
