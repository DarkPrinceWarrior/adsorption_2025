"""Model configuration dataclasses for reusable defaults."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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


CATBOOST_MODEL_CONFIG = CatBoostConfig()
