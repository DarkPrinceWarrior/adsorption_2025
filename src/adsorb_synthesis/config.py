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

    # CV settings (used for OOF metrics only)
    n_ensemble_splits: int = 5
    early_stopping_rounds: int = 100
    physics_penalty_weight: float = 1.0
    minimum_sample_weight: float = 0.2
    feature_selection_corr_threshold: float = 0.85
    feature_selection_vif_threshold: float = 10.0
    feature_selection_max_features: int = 15

    # Production ensemble (true Deep Ensemble for BO inference)
    n_ensemble_members: int = 5
    ensemble_seed_step: int = 137  # seed spacing between members

    # Conformal prediction
    conformal_alpha: float = 0.10  # 90% prediction coverage
    mapie_method: str = "plus"


CATBOOST_CONFIG = CatBoostConfig()  # default fallback
FORWARD_MODEL_CONFIG = ForwardModelConfig()

# Per-target tuned hyperparameters (from tune_hyperparams.py, 80 trials each)
TUNED_CATBOOST_CONFIGS: Dict[str, CatBoostConfig] = {
    "E0, кДж/моль": CatBoostConfig(
        iterations=1700,
        learning_rate=0.0336,
        depth=8,
        l2_leaf_reg=2.2,
        min_data_in_leaf=10,
        subsample=0.921,
        colsample_bylevel=0.530,
    ),
    "х0, нм": CatBoostConfig(
        iterations=1600,
        learning_rate=0.0638,
        depth=7,
        l2_leaf_reg=1.846,
        min_data_in_leaf=4,
        subsample=0.950,
        colsample_bylevel=0.502,
    ),
    "Sme, м2/г": CatBoostConfig(
        iterations=2000,
        learning_rate=0.0469,
        depth=6,
        l2_leaf_reg=0.108,
        min_data_in_leaf=10,
        subsample=0.926,
        colsample_bylevel=0.623,
    ),
}


def get_catboost_config(target: str) -> CatBoostConfig:
    """Return tuned config for target, falling back to default."""
    return TUNED_CATBOOST_CONFIGS.get(target, CATBOOST_CONFIG)
