"""Hyperparameter tuning with physics-aware objectives."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adsorb_synthesis.data_processing import (  # noqa: E402
    add_thermodynamic_features,
    build_lookup_tables,
    load_dataset,
)
from adsorb_synthesis.pipeline import (  # noqa: E402
    InverseDesignPipeline,
    StageConfig,
    _default_classifier,
    default_stage_configs,
)
from adsorb_synthesis.physics_losses import (  # noqa: E402
    DEFAULT_PHYSICS_EVALUATOR,
    physics_violation_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune physics-aware inverse design hyperparameters.")
    parser.add_argument("--data", default="data/SEC_SYN_with_features_DMFA_only_no_Y.csv")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials to run.")
    parser.add_argument("--storage", default=None, help="Optuna storage URL for persisting studies.")
    parser.add_argument("--study-name", default="adsorb_physics_tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=0, help="Sample size for quick experiments (0 = use full dataset).")
    parser.add_argument("--timeout", type=float, default=None, help="Study timeout in seconds.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel Optuna workers (>=1).")
    return parser.parse_args()


def build_stage_configs(
    base_configs: Sequence[StageConfig],
    physics_weight: float,
    w_thermo: float,
    w_energy: float,
) -> Sequence[StageConfig]:
    tuned: list[StageConfig] = []
    physics_kwargs = {"w_thermo": w_thermo, "w_energy": w_energy}
    for cfg in base_configs:
        if cfg.problem_type == "classification" and cfg.physics_weight > 0.0:
            kwargs_copy = dict(physics_kwargs)
            factory = partial(
                _default_classifier,
                enable_physics=True,
                physics_evaluator=cfg.physics_evaluator,
                physics_loss_kwargs=kwargs_copy,
                physics_weight=physics_weight,
            )
            tuned.append(
                replace(
                    cfg,
                    estimator_factory=factory,
                    physics_weight=physics_weight,
                    physics_loss_kwargs=kwargs_copy,
                )
            )
        else:
            tuned.append(cfg)
    return tuned


def evaluate_trial(
    configs: Sequence[StageConfig],
    df,
    lookups,
) -> tuple[float, float, float, float]:
    pipeline = InverseDesignPipeline(stage_configs=configs)
    pipeline.fit(df, lookup_tables=lookups)

    metal_metrics = pipeline.stage_results["metal"].metrics
    balanced_accuracy = float(metal_metrics.get("balanced_accuracy", metal_metrics.get("accuracy", 0.0)))

    preds = pipeline.predict(df, return_intermediate=True, enforce_physics=False)
    add_thermodynamic_features(preds)
    violation_scores = physics_violation_scores(preds, evaluator=DEFAULT_PHYSICS_EVALUATOR)
    physics_penalty = float(np.mean(violation_scores))
    if "Delta_G_residual" in preds:
        residual_mae = float(preds["Delta_G_residual"].abs().mean())
    else:
        residual_mae = float("nan")

    if "n_ratio_residual" in preds:
        ratio_mae = float(preds["n_ratio_residual"].abs().mean())
    else:
        ratio_mae = float("nan")

    return balanced_accuracy, physics_penalty, residual_mae, ratio_mae


def main() -> None:
    args = parse_args()

    df = load_dataset(args.data)
    if args.subset and args.subset > 0 and args.subset < len(df):
        df = df.sample(n=args.subset, random_state=args.seed).reset_index(drop=True)
    lookups = build_lookup_tables(df)

    base_stage_configs = default_stage_configs()

    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        physics_weight = trial.suggest_float("physics_weight", 0.0, 0.3)
        w_thermo = trial.suggest_float("w_thermo", 0.01, 0.2)
        w_energy = trial.suggest_float("w_energy", 0.001, 0.1)

        configs = build_stage_configs(base_stage_configs, physics_weight, w_thermo, w_energy)

        try:
            bal_acc, penalty, residual_mae, ratio_mae = evaluate_trial(configs, df, lookups)
        except Exception as exc:  # pragma: no cover - diagnostic output for failed trials
            trial.set_user_attr("exception", str(exc))
            raise

        trial.set_user_attr("residual_mae", residual_mae)
        trial.set_user_attr("ratio_residual_mae", ratio_mae)
        return bal_acc, penalty

    study_kwargs = dict(directions=["maximize", "minimize"], study_name=args.study_name)
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(sampler=sampler, **study_kwargs)
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    print("Pareto front:")
    for t in study.best_trials:
        print(
            f"Trial {t.number}: bal_acc={t.values[0]:.3f}, physics_penalty={t.values[1]:.4f}, "
            f"physics_weight={t.params['physics_weight']:.3f}, w_thermo={t.params['w_thermo']:.3f}, "
            f"w_energy={t.params['w_energy']:.3f}, "
            f"ΔG_res_mae={t.user_attrs.get('residual_mae'):.4f}, ratio_res_mae={t.user_attrs.get('ratio_residual_mae'):.4f}"
        )


if __name__ == "__main__":
    main()
