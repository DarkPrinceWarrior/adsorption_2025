"""Model configuration dataclasses for reusable defaults."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .constants import HUBER_DELTA_DEFAULT


@dataclass(frozen=True)
class TabNetConfig:
    """Default hyperparameters for TabNet models."""

    n_d: int = 32
    n_a: int = 32
    n_steps: int = 3
    gamma: float = 1.25
    lambda_sparse: float = 1e-5
    n_independent: int = 1
    n_shared: int = 1
    momentum: float = 0.02
    clip_value: float = 2.0
    optimizer_lr: float = 8e-3
    optimizer_weight_decay: float = 1e-4
    scheduler_T_max: int = 200
    scheduler_eta_min: float = 1e-4
    mask_type: str = "sparsemax"

    def to_params(self, problem_type: str, random_state: int) -> Dict:
        """Return a parameter dict compatible with TabNet constructors."""
        import torch

        params: Dict = dict(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            momentum=self.momentum,
            clip_value=self.clip_value,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay),
            scheduler_params=dict(T_max=self.scheduler_T_max, eta_min=self.scheduler_eta_min),
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            seed=random_state,
        )
        if problem_type == "classification":
            params["mask_type"] = self.mask_type
        return params


@dataclass(frozen=True)
class CatBoostConfig:
    """Default hyperparameters for CatBoost models."""

    iterations: int = 1600
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 6.0
    min_data_in_leaf: int = 5
    bagging_temperature: float = 0.2
    random_strength: float = 0.8
    border_count: int = 254
    verbose: bool = False

    def to_params(self, problem_type: str, random_state: int) -> Dict:
        params: Dict = dict(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=self.min_data_in_leaf,
            bagging_temperature=self.bagging_temperature,
            random_strength=self.random_strength,
            border_count=self.border_count,
            random_seed=random_state,
            verbose=self.verbose,
        )
        if problem_type == "classification":
            params.setdefault("loss_function", "Logloss")
        else:
            params.setdefault("loss_function", "MAE")
        return params


@dataclass(frozen=True)
class XGBoostConfig:
    """Default hyperparameters for XGBoost models."""

    n_estimators: int = 1600
    learning_rate: float = 0.03
    max_depth: int = 5
    min_child_weight: float = 4.0
    subsample: float = 0.85
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 0.7
    gamma: float = 0.2
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    tree_method: str = "hist"
    n_jobs: int = -1
    verbosity: int = 0

    def to_params(self, problem_type: str, random_state: int) -> Dict:
        params: Dict = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=random_state,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
        )
        if problem_type == "classification":
            params["objective"] = "binary:logistic"
            params["use_label_encoder"] = False
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "reg:pseudohubererror"
            params["huber_slope"] = HUBER_DELTA_DEFAULT
        return params


TABNET_MODEL_CONFIG = TabNetConfig()
CATBOOST_MODEL_CONFIG = CatBoostConfig()
XGBOOST_MODEL_CONFIG = XGBoostConfig()
