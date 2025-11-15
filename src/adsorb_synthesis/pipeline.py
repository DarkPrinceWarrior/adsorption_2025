"""Model training and inference pipeline for inverse design of adsorbents."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .constants import (
    ADSORPTION_FEATURES,
    LIGAND_DESCRIPTOR_FEATURES,
    METAL_DESCRIPTOR_FEATURES,
    DEFAULT_STOICHIOMETRY_BOUNDS,
    PROCESS_CONTEXT_FEATURES,
    RANDOM_SEED,
    SOLVENT_BOILING_POINTS_C,
    SOLVENT_DESCRIPTOR_FEATURES,
    STOICHIOMETRY_TARGETS,
    TEMPERATURE_CATEGORIES,
    TEST_SIZE,
    HUBER_DELTA_DEFAULT,
)
from .data_processing import (
    LookupTables,
    add_salt_mass_features,
    add_thermodynamic_features,
    build_lookup_tables,
)
from .modern_models import (
    ModernTabularEnsembleClassifier,
    ModernTabularEnsembleRegressor,
)
from .physics_losses import (
    DEFAULT_PHYSICS_EVALUATOR,
    PhysicsConstraintEvaluator,
    combined_physics_loss,
    physics_violation_scores,
    project_thermodynamics,
)

ProblemType = Literal["classification", "regression"]


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a single prediction stage in the pipeline."""

    name: str
    target: str
    problem_type: ProblemType
    feature_columns: Sequence[str]
    estimator_factory: Callable[[int], BaseEstimator]
    depends_on: Tuple[str, ...] = ()
    outlier_contamination: Optional[float] = None
    description: str = ""
    physics_weight: float = 0.0
    physics_evaluator: Optional[PhysicsConstraintEvaluator] = None
    physics_loss_kwargs: Optional[Dict[str, float]] = None
    physics_columns: Sequence[str] = ()
    target_transform: Optional[str] = None
    invert_target_to: Optional[str] = None


@dataclass
class StageResult:
    """Training artefacts and metrics for a single stage."""

    pipeline: Pipeline
    metrics: Dict[str, float]
    cv_mean: float
    cv_std: float
    feature_columns: List[str]


logger = logging.getLogger(__name__)


def _apply_target_transform(values: pd.Series, transform: Optional[str]) -> pd.Series:
    """Apply configured transform to a Series."""

    if transform is None:
        return pd.to_numeric(values, errors='coerce')
    numeric = pd.to_numeric(values, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    if transform == "log1p":
        with np.errstate(divide="ignore", invalid="ignore"):
            transformed = np.log1p(numeric)
    else:
        raise ValueError(f"Unsupported target transform '{transform}'")
    return pd.Series(transformed, index=values.index)


def _invert_target_transform(values: pd.Series, transform: Optional[str]) -> pd.Series:
    """Invert a transformed Series back to its physical scale."""

    numeric = pd.to_numeric(values, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    if transform is None:
        inverted = numeric
    elif transform == "log1p":
        with np.errstate(divide="ignore", invalid="ignore"):
            inverted = np.expm1(numeric)
    else:
        raise ValueError(f"Unsupported target transform '{transform}'")
    return pd.Series(inverted, index=values.index)


def _ensure_stage_target_column(df: pd.DataFrame, stage: StageConfig) -> None:
    """Guarantee that the transformed target column exists when needed."""

    if stage.target in df.columns:
        return
    source = stage.invert_target_to
    if source is None or source not in df.columns:
        return
    if stage.target_transform:
        df[stage.target] = _apply_target_transform(df[source], stage.target_transform)
    else:
        df[stage.target] = pd.to_numeric(df[source], errors='coerce')


def _default_classifier(
    random_state: int,
    *,
    enable_physics: bool = False,
    physics_evaluator: Optional[PhysicsConstraintEvaluator] = None,
    physics_loss_kwargs: Optional[Dict[str, float]] = None,
    physics_weight: float = 0.1,
) -> BaseEstimator:
    """Factory for classification models with optional physics-informed loss."""
    if enable_physics:
        loss_fn = partial(
            combined_physics_loss,
            evaluator=physics_evaluator or DEFAULT_PHYSICS_EVALUATOR,
            **(physics_loss_kwargs or {}),
        )
        return ModernTabularEnsembleClassifier(
            random_state=random_state,
            use_smote=True,
            focal_gamma=2.0,
            calibrate_predictions=True,
            calibration_method="isotonic",
            physics_loss_fn=loss_fn,
            physics_loss_weight=physics_weight,
        )
    return ModernTabularEnsembleClassifier(
        random_state=random_state,
        use_smote=True,
        focal_gamma=2.0,
        calibrate_predictions=True,
        calibration_method="isotonic",
    )


def _default_regressor(random_state: int) -> BaseEstimator:
    return ModernTabularEnsembleRegressor(
        random_state=random_state,
        use_quantile=False,
        huber_delta=HUBER_DELTA_DEFAULT,
    )


def _salt_mass_regressor(random_state: int) -> BaseEstimator:
    """Special regressor for salt_mass with extreme right-skew distribution."""
    return ModernTabularEnsembleRegressor(
        random_state=random_state,
        use_quantile=True,  # Predict median instead of mean
        quantile_alpha=0.5,  # Median is more robust for skewed data
    )


def default_stage_configs() -> List[StageConfig]:
    """Factory producing the recommended pipeline structure."""

    adsorption_features = list(ADSORPTION_FEATURES)
    ligand_features = sorted(set(adsorption_features + list(METAL_DESCRIPTOR_FEATURES) + ['Металл']))
    # Removed solvent_features - now using ligand_features directly with solvent descriptors
    process_context = list(PROCESS_CONTEXT_FEATURES)
    engineered_process_features = [
        'C_metal',
        'C_ligand',
        'log_C_metal',
        'log_C_ligand',
        'R_mass',
        'R_molar',
        'T_range',
        'T_activation',
        'T_dry_norm',
        'a0_calc',
        'E_calc',
        'Ws_W0_ratio',
        'delta_a0',
        'delta_E',
        'delta_Ws',
        'E_E0_ratio',
        'W0_per_SBET',
    ]

    # Salt features: ligand + ligand descriptors + solvent descriptors + engineered features
    salt_features = sorted(set(
        ligand_features + list(LIGAND_DESCRIPTOR_FEATURES) + 
        list(SOLVENT_DESCRIPTOR_FEATURES) + process_context +
        ['Лиганд', 'Metal_Ligand_Combo', 'Log_Metal_MW', 'Is_Cu', 'Is_Zn'] +
        engineered_process_features
    ))
    acid_features = sorted(set(salt_features + ['m (соли), г', 'n_соли', 'log_salt_mass']))
    volume_predictors = sorted(set(acid_features + ['m(кис-ты), г', 'n_кислоты', 'n_ratio']))
    tsyn_predictors = sorted(set(volume_predictors + ['Vсин. (р-ля), мл', 'Vsyn_m']))
    dry_predictors = sorted(set(tsyn_predictors + ['Tsyn_Category']))
    regen_predictors = sorted(set(dry_predictors + ['Tdry_Category']))
    thermodynamic_columns = ('Т.син., °С', 'Delta_G_equilibrium', 'K_equilibrium')
    physics_loss_defaults = {'w_thermo': 0.175, 'w_energy': 0.061}
    physics_weight_default = 0.017

    return [
        StageConfig(
            name="metal",
            target="Металл",
            problem_type="classification",
            feature_columns=adsorption_features,
            estimator_factory=partial(
                _default_classifier,
                enable_physics=True,
                physics_loss_kwargs=dict(physics_loss_defaults),
                physics_weight=physics_weight_default,
            ),
            physics_weight=physics_weight_default,
            physics_evaluator=DEFAULT_PHYSICS_EVALUATOR,
            physics_loss_kwargs=dict(physics_loss_defaults),
            physics_columns=thermodynamic_columns,
            description="Predict metal identity from adsorption performance with physics constraints.",
        ),
        StageConfig(
            name="ligand",
            target="Лиганд",
            problem_type="classification",
            feature_columns=ligand_features,
            estimator_factory=partial(
                _default_classifier,
                enable_physics=True,
                physics_loss_kwargs=dict(physics_loss_defaults),
                physics_weight=physics_weight_default,
            ),
            depends_on=("Металл",),
            physics_weight=physics_weight_default,
            physics_evaluator=DEFAULT_PHYSICS_EVALUATOR,
            physics_loss_kwargs=dict(physics_loss_defaults),
            physics_columns=thermodynamic_columns,
            description="Predict ligand conditioned on adsorption profile and metal with physics constraints.",
        ),
        # Solvent stage removed: dataset filtered to DMFA only (340/380 samples)
        # Solvent is now set as constant "ДМФА" in _ensure_process_defaults()
        StageConfig(
            name="salt_mass",
            target="log_salt_mass",  # Log-transformed target for better linearity
            problem_type="regression",
            feature_columns=salt_features,
            estimator_factory=_default_regressor,  # Back to Huber - log-space is linear-friendly
            depends_on=("Металл", "Лиганд"),
            outlier_contamination=0.02,  # Reduced from 0.05 - keep more data
            target_transform="log1p",
            invert_target_to="m (соли), г",
            description="Estimate required salt mass (log-transformed) for synthesis stage.",
        ),
        StageConfig(
            name="acid_mass",
            target="m(кис-ты), г",
            problem_type="regression",
            feature_columns=acid_features,
            estimator_factory=_default_regressor,
            depends_on=("Металл", "Лиганд", "m (соли), г", "log_salt_mass"),
            outlier_contamination=0.05,
            description="Estimate ligand acid mass conditioned on salt mass.",
        ),
        StageConfig(
            name="solvent_volume",
            target="Vсин. (р-ля), мл",
            problem_type="regression",
            feature_columns=volume_predictors,
            estimator_factory=_default_regressor,
            depends_on=(
                "Металл",
                "Лиганд",
                "m (соли), г",
                "m(кис-ты), г",
            ),
            outlier_contamination=0.05,
            description="Predict solvent volume required for synthesis.",
        ),
        StageConfig(
            name="tsyn_category",
            target="Tsyn_Category",
            problem_type="classification",
            feature_columns=tsyn_predictors,
            estimator_factory=_default_classifier,
            depends_on=(
                "Металл",
                "Лиганд",
                "m (соли), г",
                "m(кис-ты), г",
                "Vсин. (р-ля), мл",
            ),
            description="Classify synthesis temperature range.",
        ),
        StageConfig(
            name="tdry_category",
            target="Tdry_Category",
            problem_type="classification",
            feature_columns=dry_predictors,
            estimator_factory=_default_classifier,
            depends_on=(
                "Металл",
                "Лиганд",
                "m (соли), г",
                "m(кис-ты), г",
                "Vсин. (р-ля), мл",
                "Tsyn_Category",
            ),
            description="Classify drying temperature range.",
        ),
        StageConfig(
            name="treg_category",
            target="Treg_Category",
            problem_type="classification",
            feature_columns=regen_predictors,
            estimator_factory=_default_classifier,
            depends_on=(
                "Металл",
                "Лиганд",
                "m (соли), г",
                "m(кис-ты), г",
                "Vсин. (р-ля), мл",
                "Tsyn_Category",
                "Tdry_Category",
            ),
            description="Classify regeneration temperature range.",
        ),
    ]


class InverseDesignPipeline:
    """End-to-end orchestrator for training and inference."""

    def __init__(
        self,
        *,
        stage_configs: Optional[Sequence[StageConfig]] = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.stage_configs: List[StageConfig] = list(stage_configs or default_stage_configs())
        self.random_state = random_state
        self.stage_results: Dict[str, StageResult] = {}
        self.lookup_tables: Optional[LookupTables] = None
        self._trained = False

    def fit(
        self,
        dataset: pd.DataFrame,
        lookup_tables: Optional[LookupTables] = None,
        *,
        copy_dataset: bool = True,
    ) -> None:
        """Train models for every stage.

        Parameters
        ----------
        dataset : pd.DataFrame
            Training dataframe containing adsorption descriptors and synthesis context.
        lookup_tables : Optional[LookupTables]
            Pre-built lookup tables. When omitted they are reconstructed from ``dataset``.
        copy_dataset : bool, default True
            When False, the fit procedure mutates ``dataset`` in place instead of working
            on a defensive copy. Use with care to avoid side effects.
        """

        if lookup_tables is None:
            lookup_tables = build_lookup_tables(dataset)
        self.lookup_tables = lookup_tables

        data = dataset.copy() if copy_dataset else dataset
        _ensure_process_defaults(data)
        _augment_with_lookup_descriptors(data, lookup_tables)
        _update_stoichiometry_features(data)
        add_salt_mass_features(data)  # Add engineered features for salt_mass
        add_thermodynamic_features(data)

        rng_seed = self.random_state
        self.stage_results.clear()

        for stage in self.stage_configs:
            _ensure_stage_target_column(data, stage)
            stage_data = self._prepare_stage_dataframe(data, stage)
            if stage_data.empty:
                raise ValueError(f"Stage '{stage.name}' has no data after preprocessing")

            model = stage.estimator_factory(rng_seed)
            
            # Pass feature names to physics-informed models
            if hasattr(model, 'feature_names') and model.feature_names is None:
                feature_names = list(stage.feature_columns)
                for column in stage.physics_columns:
                    if column not in feature_names:
                        feature_names.append(column)
                model.feature_names = feature_names
            
            pipeline = self._build_pipeline(stage_data, stage.feature_columns, model)
            metrics, cv_mean, cv_std = self._train_and_evaluate(stage, pipeline, stage_data)

            self.stage_results[stage.name] = StageResult(
                pipeline=pipeline,
                metrics=metrics,
                cv_mean=cv_mean,
                cv_std=cv_std,
                feature_columns=list(stage.feature_columns),
            )

            # Persist transformed and inverted targets for downstream stages
            data.loc[stage_data.index, stage.target] = stage_data[stage.target]
            if stage.invert_target_to:
                inverted = _invert_target_transform(stage_data[stage.target], stage.target_transform)
                data.loc[stage_data.index, stage.invert_target_to] = inverted

        self._trained = True

    def predict(
        self,
        inputs: pd.DataFrame,
        *,
        return_intermediate: bool = True,
        enforce_physics: bool = True,
    ) -> pd.DataFrame:
        """Run sequential inference starting from adsorption descriptors."""

        if not self._trained:
            raise RuntimeError("Pipeline must be trained before calling predict().")
        if self.lookup_tables is None:
            raise RuntimeError("Lookup tables are unavailable. Train the pipeline first.")

        results = inputs.copy()
        _ensure_process_defaults(results)
        add_salt_mass_features(results)  # Add engineered features
        add_thermodynamic_features(results)

        for stage in self.stage_configs:
            _augment_with_lookup_descriptors(results, self.lookup_tables)
            _update_stoichiometry_features(results)

            features = list(stage.feature_columns)
            missing = [col for col in features if col not in results.columns]
            if missing:
                raise ValueError(
                    f"Missing required features for stage '{stage.name}': {missing}"
                )

            stage_result = self.stage_results[stage.name]
            predictions = stage_result.pipeline.predict(results[features])

            pred_series = pd.Series(predictions, index=results.index, name=stage.target)
            results[stage.target] = pred_series
            if stage.invert_target_to:
                inversed = _invert_target_transform(pred_series, stage.target_transform)
                results[stage.invert_target_to] = inversed

        if enforce_physics:
            results = project_thermodynamics(
                results,
                evaluator=DEFAULT_PHYSICS_EVALUATOR,
                overwrite=True,
                residual_column="K_equilibrium_residual",
            )
            add_thermodynamic_features(results)
            _project_stoichiometry(results)
            _enforce_temperature_limits(results)
            add_thermodynamic_features(results)

        if not return_intermediate:
            targets = [
                stage.invert_target_to or stage.target
                for stage in self.stage_configs
            ]
            return results[targets]
        return results

    def save(self, directory: str | Path) -> None:
        """Persist trained models and metadata to disk."""

        if not self._trained:
            raise RuntimeError("Cannot save an untrained pipeline.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        metadata = {
            'random_state': self.random_state,
            'stages': [],
        }

        for stage in self.stage_configs:
            stage_result = self.stage_results[stage.name]
            model_path = directory / f"{stage.name}_pipeline.joblib"
            joblib.dump(stage_result.pipeline, model_path)

            metadata['stages'].append({
                'name': stage.name,
                'target': stage.target,
                'problem_type': stage.problem_type,
                'feature_columns': stage_result.feature_columns,
                'metrics': stage_result.metrics,
                'cv_mean': stage_result.cv_mean,
                'cv_std': stage_result.cv_std,
                'model_path': model_path.name,
                'depends_on': stage.depends_on,
                'outlier_contamination': stage.outlier_contamination,
                'description': stage.description,
                'physics_weight': stage.physics_weight,
                'physics_columns': list(stage.physics_columns),
                'physics_loss_kwargs': dict(stage.physics_loss_kwargs) if stage.physics_loss_kwargs else None,
                'target_transform': stage.target_transform,
                'invert_target_to': stage.invert_target_to,
            })

        if self.lookup_tables is not None:
            for key, table in self.lookup_tables.to_dict().items():
                joblib.dump(table, directory / f"lookup_{key}.joblib")

        joblib.dump(metadata, directory / "pipeline_metadata.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "InverseDesignPipeline":
        """Restore a trained pipeline from disk."""

        directory = Path(directory)
        metadata_path = directory / "pipeline_metadata.joblib"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        metadata = joblib.load(metadata_path)

        stage_configs: List[StageConfig] = []
        for stage_meta in metadata.get('stages', []):
            problem_type = stage_meta['problem_type']
            physics_weight = stage_meta.get('physics_weight', 0.0)
            physics_columns = tuple(stage_meta.get('physics_columns', ()))
            physics_loss_kwargs_meta = stage_meta.get('physics_loss_kwargs')
            physics_loss_kwargs = dict(physics_loss_kwargs_meta) if physics_loss_kwargs_meta else None
            if problem_type == 'classification' and physics_weight > 0.0:
                estimator_factory = partial(
                    _default_classifier,
                    enable_physics=True,
                    physics_loss_kwargs=physics_loss_kwargs,
                    physics_weight=physics_weight,
                )
                physics_evaluator = DEFAULT_PHYSICS_EVALUATOR
            else:
                estimator_factory = _default_classifier if problem_type == 'classification' else _default_regressor
                physics_evaluator = DEFAULT_PHYSICS_EVALUATOR if problem_type == 'classification' and physics_weight > 0 else None
            stage_configs.append(StageConfig(
                name=stage_meta['name'],
                target=stage_meta['target'],
                problem_type=problem_type,
                feature_columns=tuple(stage_meta.get('feature_columns', ())),
                estimator_factory=estimator_factory,
                depends_on=tuple(stage_meta.get('depends_on', ())),
                outlier_contamination=stage_meta.get('outlier_contamination'),
                description=stage_meta.get('description', ''),
                physics_weight=physics_weight,
                physics_evaluator=physics_evaluator,
                physics_loss_kwargs=physics_loss_kwargs,
                physics_columns=physics_columns,
                target_transform=stage_meta.get('target_transform'),
                invert_target_to=stage_meta.get('invert_target_to'),
            ))

        pipeline = cls(stage_configs=stage_configs, random_state=metadata.get('random_state', RANDOM_SEED))

        lookup_tables = {}
        for key in ('metal', 'ligand', 'solvent'):
            lookup_path = directory / f"lookup_{key}.joblib"
            if lookup_path.exists():
                lookup_tables[key] = joblib.load(lookup_path)

        if len(lookup_tables) == 3:
            pipeline.lookup_tables = LookupTables(**lookup_tables)
        else:
            pipeline.lookup_tables = None

        pipeline.stage_results.clear()
        for stage_meta in metadata.get('stages', []):
            name = stage_meta['name']
            model_path = directory / stage_meta['model_path']
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file for stage '{name}': {model_path}")
            model_pipeline = joblib.load(model_path)

            stage_result = StageResult(
                pipeline=model_pipeline,
                metrics=stage_meta.get('metrics', {}),
                cv_mean=stage_meta.get('cv_mean', float('nan')),
                cv_std=stage_meta.get('cv_std', float('nan')),
                feature_columns=list(stage_meta.get('feature_columns', ())),
            )
            pipeline.stage_results[name] = stage_result

        pipeline._trained = True
        return pipeline

    def _prepare_stage_dataframe(self, data: pd.DataFrame, stage: StageConfig) -> pd.DataFrame:
        cols: List[str] = list(stage.feature_columns)
        for col in stage.physics_columns:
            if col not in cols:
                cols.append(col)
        if stage.target not in cols:
            cols.append(stage.target)
        df = data[cols].dropna()

        if stage.problem_type == "regression" and stage.outlier_contamination:
            iso = IsolationForest(
                contamination=stage.outlier_contamination,
                random_state=self.random_state,
            )
            mask = iso.fit_predict(df[[stage.target]]) == 1
            df = df.loc[mask]
        return df

    def _build_pipeline(self, data: pd.DataFrame, features: Sequence[str], estimator: BaseEstimator) -> Pipeline:
        cat_cols = [col for col in features if data[col].dtype == object or str(data[col].dtype) == 'category']
        num_cols = [col for col in features if col not in cat_cols]

        preprocess = ColumnTransformer(
            transformers=[
                (
                    'categorical',
                    Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ]),
                    cat_cols,
                ),
                (
                    'numeric',
                    SimpleImputer(strategy='median'),
                    num_cols,
                ),
            ],
            remainder='drop',
        )

        return Pipeline([
            ('preprocess', preprocess),
            ('model', estimator),
        ])

    def _train_and_evaluate(
        self,
        stage: StageConfig,
        pipeline: Pipeline,
        data: pd.DataFrame,
    ) -> Tuple[Dict[str, float], float, float]:
        features = list(stage.feature_columns)
        target = stage.target
        X = data[features]
        y = data[target]
        sample_weights: Optional[np.ndarray] = None
        evaluator = stage.physics_evaluator or DEFAULT_PHYSICS_EVALUATOR

        if stage.problem_type == "classification":
            class_counts = y.value_counts()
            stratify = None
            if y.nunique() > 1 and (class_counts < 2).sum() == 0:
                stratify = y
        else:
            stratify = None

        physics_frame = data if stage.physics_columns else data[features]
        if stage.physics_weight > 0.0:
            penalties = physics_violation_scores(physics_frame, evaluator=evaluator)
            if penalties.size:
                penalties = np.nan_to_num(penalties, nan=0.0, posinf=0.0, neginf=0.0)
                finite_mask = np.isfinite(penalties)
                finite_values = penalties[finite_mask]
                if finite_values.size:
                    scale = np.nanpercentile(finite_values, 95)
                    if not np.isfinite(scale) or scale <= 1e-6:
                        scale = np.nanmax(finite_values)
                    if scale > 0:
                        penalties = penalties / scale
                penalties = np.clip(penalties, 0.0, None)
                sample_weights = 1.0 + stage.physics_weight * penalties
                sample_weights = sample_weights.astype(np.float64, copy=False)

        indices = np.arange(len(X))
        split_arrays = [X, y, indices]
        if sample_weights is not None:
            split_arrays.append(sample_weights)

        split_result = train_test_split(
            *split_arrays,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=stratify,
        )

        if sample_weights is not None:
            X_train, X_test, y_train, y_test, train_idx, test_idx, w_train, w_test = split_result
        else:
            X_train, X_test, y_train, y_test, train_idx, test_idx = split_result
            w_train = w_test = None

        fit_kwargs = {}
        if w_train is not None:
            fit_kwargs['model__sample_weight'] = w_train
        model_step = pipeline.named_steps.get('model')
        if model_step is not None and hasattr(model_step, 'physics_loss_fn'):
            physics_subset = physics_frame.iloc[train_idx].reset_index(drop=True)
            fit_kwargs['model__physics_frame'] = physics_subset

        pipeline.fit(X_train, y_train, **fit_kwargs)
        y_pred = pipeline.predict(X_test)

        metrics: Dict[str, float]
        if stage.problem_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
            }
            cv_scores = self._cross_validate_stage(
                stage=stage,
                base_pipeline=pipeline,
                X=X,
                y=y,
                sample_weights=sample_weights,
                physics_frame=physics_frame,
            )
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': rmse,
                'mae': mean_absolute_error(y_test, y_pred),
            }
            cv_scores = self._cross_validate_stage(
                stage=stage,
                base_pipeline=pipeline,
                X=X,
                y=y,
                sample_weights=sample_weights,
                physics_frame=physics_frame,
            )

        return metrics, float(np.mean(cv_scores)), float(np.std(cv_scores))

    def _cross_validate_stage(
        self,
        *,
        stage: StageConfig,
        base_pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[np.ndarray],
        physics_frame: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """Manual cross-validation that respects sample weights."""
        if stage.problem_type == "classification":
            class_counts = y.value_counts()
            min_class = class_counts.min() if not class_counts.empty else 0
            if min_class >= 2:
                n_splits = min(5, int(min_class))
                splitter: Iterable = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
            else:
                n_splits = max(2, min(5, len(y)))
                splitter = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
        else:
            n_splits = max(2, min(5, len(y)))
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )

        scores: List[float] = []
        for train_idx, val_idx in splitter.split(X, y):
            X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
            y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]

            fit_kwargs = {}
            if sample_weights is not None:
                w_train = sample_weights[train_idx]
                fit_kwargs['model__sample_weight'] = w_train
            model_step = base_pipeline.named_steps.get('model')
            if physics_frame is not None and model_step is not None and hasattr(model_step, 'physics_loss_fn'):
                physics_subset = physics_frame.iloc[train_idx].reset_index(drop=True)
                fit_kwargs['model__physics_frame'] = physics_subset

            model = clone(base_pipeline)
            model.fit(X_train, y_train, **fit_kwargs)
            y_pred = model.predict(X_val)

            if stage.problem_type == "classification":
                score = balanced_accuracy_score(y_val, y_pred)
            else:
                mae = mean_absolute_error(y_val, y_pred)
                score = -float(mae)
            scores.append(float(score))

        return np.asarray(scores, dtype=np.float64)


def _augment_with_lookup_descriptors(df: pd.DataFrame, lookup: LookupTables) -> None:
    """Ensure descriptor columns are populated using lookup tables."""

    _apply_lookup(df, 'Металл', lookup.metal)
    _apply_lookup(df, 'Лиганд', lookup.ligand)
    _apply_lookup(df, 'Растворитель', lookup.solvent)


def _apply_lookup(df: pd.DataFrame, key_column: str, lookup_table: pd.DataFrame) -> None:
    if key_column not in df.columns:
        return

    df_columns = set(df.columns)
    lookup_cols = [col for col in lookup_table.columns if col not in df_columns]
    if lookup_cols:
        for col in lookup_cols:
            df[col] = np.nan

    # Align lookup table with dataframe columns and index
    aligned = lookup_table.reindex(columns=[col for col in df.columns if col in lookup_table.columns])
    aligned = aligned.add_suffix("_lookup")
    merged = df[[key_column]].join(aligned, on=key_column)

    for lookup_col in aligned.columns:
        target_col = lookup_col.removesuffix("_lookup")
        df[target_col] = merged[lookup_col].combine_first(df[target_col])


def _ensure_process_defaults(df: pd.DataFrame) -> None:
    """Provide default values for process context features when absent."""

    for feature in PROCESS_CONTEXT_FEATURES:
        if feature not in df.columns:
            df[feature] = 1.0
        df[feature] = df[feature].fillna(1.0)
    
    # Set solvent to ДМФА (constant after dataset filtering)
    if 'Растворитель' not in df.columns:
        df['Растворитель'] = 'ДМФА'
    df['Растворитель'] = df['Растворитель'].fillna('ДМФА')


def _update_stoichiometry_features(df: pd.DataFrame) -> None:
    """Derive stoichiometric helper features used by downstream stages."""

    if {'m (соли), г', 'Молярка_соли'}.issubset(df.columns):
        denom = df['Молярка_соли'].replace(0, np.nan)
        df['n_соли'] = df['m (соли), г'] / denom

    if {'m(кис-ты), г', 'Молярка_кислоты'}.issubset(df.columns):
        denom = df['Молярка_кислоты'].replace(0, np.nan)
        df['n_кислоты'] = df['m(кис-ты), г'] / denom

    if {'n_соли', 'n_кислоты'}.issubset(df.columns):
        denom = df['n_кислоты'].replace(0, np.nan)
        df['n_ratio'] = df['n_соли'] / denom
        (
            lower_bounds,
            upper_bounds,
            target_values,
            mode_values,
        ) = _compute_stoichiometry_bounds(df)

        ratio_values = pd.to_numeric(df['n_ratio'], errors='coerce').to_numpy(dtype=float)
        bounded = np.clip(ratio_values, lower_bounds, upper_bounds)
        target_or_bounded = np.where(np.isfinite(target_values), target_values, bounded)
        residual = ratio_values - target_or_bounded

        df['n_ratio_lower'] = lower_bounds
        df['n_ratio_upper'] = upper_bounds
        df['n_ratio_target'] = target_values
        df['n_ratio_mode'] = mode_values
        df['n_ratio_residual'] = residual

    if {'Vсин. (р-ля), мл', 'm (соли), г'}.issubset(df.columns):
        denom = df['m (соли), г'].replace(0, np.nan)
        df['Vsyn_m'] = df['Vсин. (р-ля), мл'] / denom


def _compute_stoichiometry_bounds(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper bounds and targets for each metal-ligand pair."""

    size = len(df)
    lower_default, upper_default = DEFAULT_STOICHIOMETRY_BOUNDS
    lower_arr = np.full(size, lower_default, dtype=float)
    upper_arr = np.full(size, upper_default, dtype=float)
    target_arr = np.full(size, np.nan, dtype=float)
    mode_arr = np.full(size, 'fallback', dtype=object)

    if {'Металл', 'Лиганд'}.issubset(df.columns):
        for idx, (metal, ligand) in enumerate(zip(df['Металл'], df['Лиганд'])):
            target_info = STOICHIOMETRY_TARGETS.get((metal, ligand))
            if not target_info:
                continue
            target = target_info['ratio']
            tol = target_info.get('tolerance', 0.1)
            lower_arr[idx] = target * (1 - tol)
            upper_arr[idx] = target * (1 + tol)
            target_arr[idx] = target
            mode_arr[idx] = 'target'

    return lower_arr, upper_arr, target_arr, mode_arr


def _project_stoichiometry(df: pd.DataFrame) -> None:
    """Adjust acid mass to keep molar ratio near material-specific targets."""

    required = {
        'Металл',
        'Лиганд',
        'm (соли), г',
        'Молярка_соли',
        'm(кис-ты), г',
        'Молярка_кислоты',
        'n_соли',
        'n_кислоты',
        'n_ratio',
    }
    if not required.issubset(df.columns):
        return

    ratio = pd.to_numeric(df['n_ratio'], errors='coerce').to_numpy(dtype=float)
    n_salt = pd.to_numeric(df['n_соли'], errors='coerce').to_numpy(dtype=float)
    molar_acid = pd.to_numeric(df['Молярка_кислоты'], errors='coerce').to_numpy(dtype=float)

    lower_bounds, upper_bounds, target_arr, _ = _compute_stoichiometry_bounds(df)

    desired_ratio = np.array(ratio, copy=True)
    fallback_mask = ~np.isfinite(target_arr)
    if np.any(fallback_mask):
        desired_ratio[fallback_mask] = np.clip(
            ratio[fallback_mask],
            lower_bounds[fallback_mask],
            upper_bounds[fallback_mask],
        )

    target_mask = np.isfinite(target_arr)
    if np.any(target_mask):
        within_tolerance = (
            (ratio >= lower_bounds) & (ratio <= upper_bounds)
        )
        target_adjust = target_mask & ~within_tolerance
        desired_ratio[target_adjust] = target_arr[target_adjust]

    adjust_mask = (
        np.isfinite(desired_ratio)
        & np.isfinite(n_salt)
        & np.isfinite(molar_acid)
        & (molar_acid > 0)
        & np.isfinite(ratio)
        & (np.abs(desired_ratio - ratio) > 1e-6)
    )
    if not np.any(adjust_mask):
        return

    target_n_acid = np.divide(
        n_salt,
        desired_ratio,
        out=np.full_like(n_salt, np.nan),
        where=desired_ratio > 0,
    )
    new_acid_mass = target_n_acid * molar_acid

    values = new_acid_mass[adjust_mask]
    column_dtype = df['m(кис-ты), г'].dtype
    if np.issubdtype(column_dtype, np.floating):
        values = values.astype(column_dtype, copy=False)
    df.loc[adjust_mask, 'm(кис-ты), г'] = values

    _update_stoichiometry_features(df)


_TEMPERATURE_SEQUENCE = (
    ('Tsyn_Category', 'Т.син., °С'),
    ('Tdry_Category', 'Т суш., °С'),
    ('Treg_Category', 'Tрег, ᵒС'),
)

_CATEGORY_MIDPOINT_CACHE: Dict[str, Dict[str, float]] = {}


def _temperature_midpoints(name: str) -> Dict[str, float]:
    if name in _CATEGORY_MIDPOINT_CACHE:
        return _CATEGORY_MIDPOINT_CACHE[name]
    spec = TEMPERATURE_CATEGORIES.get(name)
    if spec is None:
        _CATEGORY_MIDPOINT_CACHE[name] = {}
        return {}
    bins = spec['bins']
    labels = spec['labels']
    mapping = {
        label: float((bins[idx] + bins[idx + 1]) / 2.0)
        for idx, label in enumerate(labels)
    }
    _CATEGORY_MIDPOINT_CACHE[name] = mapping
    return mapping


def _numeric_to_temperature_category(values: np.ndarray, name: str, index: pd.Index) -> pd.Series:
    spec = TEMPERATURE_CATEGORIES.get(name)
    if spec is None:
        return pd.Series(np.nan, index=index, dtype=object)
    bins = np.asarray(spec['bins'], dtype=float)
    labels = spec['labels']
    series = pd.Series(values, index=index, dtype=float)
    upper = bins[-1] - 1e-6
    series = series.clip(lower=bins[0], upper=upper)
    categories = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False)
    return categories.astype(object)


def _max_temperature_below(limit: float, category_name: str) -> float:
    spec = TEMPERATURE_CATEGORIES.get(category_name)
    if spec is None or not np.isfinite(limit):
        return limit
    bins = np.asarray(spec['bins'], dtype=float)
    midpoints = np.array([(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)], dtype=float)
    candidates = midpoints[midpoints < limit]
    if candidates.size > 0:
        return float(np.max(candidates))
    return float(max(limit - 1.0, bins[0]))


def _enforce_temperature_limits(df: pd.DataFrame) -> None:
    """Enforce monotonic temperatures and solvent boiling constraints."""

    if df.empty:
        return

    numeric_values: Dict[str, np.ndarray] = {}

    for category, numeric_col in _TEMPERATURE_SEQUENCE:
        spec = TEMPERATURE_CATEGORIES.get(category)
        if spec is None:
            continue

        midpoint_map = _temperature_midpoints(category)

        numeric_series = pd.Series(np.nan, index=df.index, dtype=float)
        if numeric_col in df.columns:
            numeric_series = pd.to_numeric(df[numeric_col], errors='coerce')
        if category in df.columns:
            numeric_from_category = df[category].map(midpoint_map)
            numeric_series = numeric_series.fillna(numeric_from_category)
        else:
            df[category] = np.nan

        values = numeric_series.to_numpy(dtype=float, copy=True)
        numeric_values[category] = values

    # Enforce solvent boiling constraint for synthesis temperature
    syn_values = numeric_values.get('Tsyn_Category')
    if syn_values is not None and 'Растворитель' in df.columns:
        lookup = {str(k).strip().lower(): v for k, v in SOLVENT_BOILING_POINTS_C.items()}
        solvents = df['Растворитель'].astype(str).str.strip().str.lower()
        boiling = solvents.map(lookup).to_numpy(dtype=float, copy=False)
        violation_mask = (
            np.isfinite(syn_values)
            & np.isfinite(boiling)
            & (syn_values >= boiling)
        )
        if np.any(violation_mask):
            indices = np.where(violation_mask)[0]
            for idx in indices:
                row = df.index[idx]
                solvent = df.at[row, 'Растворитель']
                limit = boiling[idx]
                previous = syn_values[idx]
                new_value = min(_max_temperature_below(limit, 'Tsyn_Category'), limit - 1.0)
                new_value = max(new_value, 0.0)
                syn_values[idx] = new_value
                logger.warning(
                    "Adjusted Tsyn_Category at row %s: %.1f°C exceeds boiling point of %s (%.1f°C). "
                    "Projected to %.1f°C.",
                    row,
                    previous,
                    solvent,
                    limit,
                    new_value,
                )
            numeric_values['Tsyn_Category'] = syn_values

    prev_values: Optional[np.ndarray] = None
    for category, numeric_col in _TEMPERATURE_SEQUENCE:
        values = numeric_values.get(category)
        if values is None:
            prev_values = None
            continue

        spec = TEMPERATURE_CATEGORIES.get(category)
        bins = np.asarray(spec['bins'], dtype=float)
        finite_mask = np.isfinite(values)
        values[finite_mask] = np.clip(values[finite_mask], bins[0], bins[-1] - 1e-6)

        if prev_values is not None:
            mask = finite_mask & np.isfinite(prev_values)
            adjust_mask = mask & (values < prev_values)
            if np.any(adjust_mask):
                old_vals = values[adjust_mask].copy()
                new_vals = prev_values[adjust_mask]
                rows = df.index[np.where(adjust_mask)[0]]
                for row, previous, new_value in zip(rows, old_vals, new_vals):
                    logger.warning(
                        "Adjusted %s at row %s to maintain thermal order (%.1f°C -> %.1f°C).",
                        category,
                        row,
                        previous,
                        new_value,
                    )
                values[adjust_mask] = new_vals

        df[numeric_col] = values
        df[category] = _numeric_to_temperature_category(values, category, df.index)
        prev_values = values.copy()
