"""Modern ensemble models for tabular data (2025)."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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
        base.setdefault("loss_function", "RMSE")
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
        base["objective"] = "reg:squarederror"
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
        random_state: int = 42,
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
        self.random_state = random_state

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

    def _optimize_weights(self, predictions: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        n_models = predictions.shape[0]
        if n_models == 1 or not self.optimize_weights:
            return np.ones(n_models, dtype=np.float64) / n_models

        try:
            from scipy.optimize import minimize
        except ImportError:
            warnings.warn("SciPy not available; using uniform ensemble weights.", RuntimeWarning)
            return np.ones(n_models, dtype=np.float64) / n_models

        initial = np.ones(n_models, dtype=np.float64) / n_models

        if self.problem_type == "classification":
            n_classes = predictions.shape[-1]
            labels = np.arange(n_classes)

            def objective(weights: np.ndarray) -> float:
                weights = np.clip(weights, 0.0, None)
                total = weights.sum()
                if total == 0:
                    return np.inf
                ensemble = np.tensordot(weights / total, predictions, axes=(0, 0))
                ensemble = np.clip(ensemble, 1e-6, 1 - 1e-6)
                return float(log_loss(y_val, ensemble, labels=labels))
        else:
            def objective(weights: np.ndarray) -> float:
                weights = np.clip(weights, 0.0, None)
                total = weights.sum()
                if total == 0:
                    return np.inf
                ensemble = np.tensordot(weights / total, predictions, axes=(0, 0))
                return float(mean_squared_error(y_val, ensemble))

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
    """Ensemble of TabNet, CatBoost and XGBoost for classification."""

    def __init__(
        self,
        tabnet_params: Optional[Dict] = None,
        tabnet_fit_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        validation_fraction: float = 0.2,
        optimize_weights: bool = True,
        weight_opt_params: Optional[Dict] = None,
        random_state: int = 42,
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
            random_state=random_state,
        )

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, ensure_min_samples=2)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_arr)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        counts = np.bincount(y_encoded)
        stratify = y_encoded if self.n_classes_ > 1 and np.all(counts >= 2) else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_arr,
            y_encoded,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )

        models: List = []
        names: List[str] = []
        val_predictions: List[np.ndarray] = []

        # TabNet
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

        # CatBoost
        try:
            cat_model = self._make_catboost()
            if self.n_classes_ > 2:
                cat_model.set_params(loss_function="MultiClass", eval_metric="TotalF1")
            else:
                cat_model.set_params(loss_function="Logloss", eval_metric="Logloss")
            if self.n_classes_ == 2:
                cat_model.set_params(auto_class_weights="Balanced")
            cat_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=False,
            )
            models.append(cat_model)
            names.append("catboost")
            val_predictions.append(self._normalise_proba(cat_model.predict_proba(X_val), self.n_classes_))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"CatBoost training failed and will be skipped: {exc}", RuntimeWarning)

        # XGBoost
        try:
            xgb_model = self._make_xgboost()
            if self.n_classes_ > 2:
                xgb_model.set_params(
                    objective="multi:softprob",
                    num_class=self.n_classes_,
                    eval_metric="mlogloss",
                    max_delta_step=1,
                )
            else:
                positives = np.count_nonzero(y_train == 1)
                negatives = y_train.shape[0] - positives
                if positives > 0:
                    xgb_model.set_params(scale_pos_weight=negatives / positives)
                xgb_model.set_params(
                    objective="binary:logistic",
                    eval_metric=["logloss", "auc"],
                    max_delta_step=1,
                )
            xgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            models.append(xgb_model)
            names.append("xgboost")
            val_predictions.append(self._normalise_proba(xgb_model.predict_proba(X_val), self.n_classes_))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"XGBoost training failed and will be skipped: {exc}", RuntimeWarning)

        if not models:
            raise RuntimeError("All base models failed to train; cannot fit ensemble.")

        self.models_ = models
        self.model_names_ = names
        val_stack = np.stack(val_predictions, axis=0)
        self.weights_ = self._optimize_weights(val_stack, y_val)
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
    """Ensemble of TabNet, CatBoost and XGBoost for regression."""

    def __init__(
        self,
        tabnet_params: Optional[Dict] = None,
        tabnet_fit_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        validation_fraction: float = 0.2,
        optimize_weights: bool = True,
        weight_opt_params: Optional[Dict] = None,
        random_state: int = 42,
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
            random_state=random_state,
        )

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        X_arr = np.asarray(X_arr, dtype=np.float32)
        y_arr = np.asarray(y_arr, dtype=np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X_arr,
            y_arr,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )

        models: List = []
        names: List[str] = []
        val_predictions: List[np.ndarray] = []

        # TabNet
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

        # CatBoost
        try:
            cat_model = self._make_catboost()
            cat_model.set_params(loss_function="RMSE", eval_metric="RMSE", subsample=0.85, rsm=0.85)
            cat_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=False,
            )
            models.append(cat_model)
            names.append("catboost")
            val_predictions.append(np.asarray(cat_model.predict(X_val), dtype=np.float64).ravel())
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"CatBoost training failed and will be skipped: {exc}", RuntimeWarning)

        # XGBoost
        try:
            xgb_model = self._make_xgboost()
            xgb_model.set_params(objective="reg:squarederror", eval_metric="rmse", max_delta_step=1)
            xgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
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
        self.weights_ = self._optimize_weights(val_stack, y_val)
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
