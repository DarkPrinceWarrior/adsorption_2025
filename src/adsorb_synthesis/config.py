"""Model configuration dataclasses for reusable defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class CatBoostConfig:
    """Default hyperparameters for CatBoost regressors (Forward Model)."""

    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 1.0
    min_data_in_leaf: int = 1
    subsample: float = 0.8
    colsample_bylevel: float = 0.8
    loss_function: str = "RMSE"
    verbose: bool = False
    allow_writing_files: bool = False

    def to_params(self, random_state: int) -> Dict:
        return dict(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=self.min_data_in_leaf,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            loss_function=self.loss_function,
            random_seed=random_state,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
        )


@dataclass(frozen=True)
class ForwardModelConfig:
    """Training-level settings for the forward model ensemble."""

    n_ensemble_splits: int = 5
    early_stopping_rounds: int = 100
    physics_penalty_weight: float = 1.0
    feature_selection_corr_threshold: float = 0.85
    feature_selection_vif_threshold: float = 10.0
    feature_selection_max_features: int = 15


CATBOOST_CONFIG = CatBoostConfig()
FORWARD_MODEL_CONFIG = ForwardModelConfig()
