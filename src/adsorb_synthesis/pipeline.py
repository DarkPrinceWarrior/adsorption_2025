"""Model training and inference pipeline for inverse design of adsorbents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
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
    PROCESS_CONTEXT_FEATURES,
    RANDOM_SEED,
    SOLVENT_DESCRIPTOR_FEATURES,
    TEST_SIZE,
)
from .data_processing import LookupTables, add_salt_mass_features, build_lookup_tables
from .modern_models import (
    ModernTabularEnsembleClassifier,
    ModernTabularEnsembleRegressor,
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


@dataclass
class StageResult:
    """Training artefacts and metrics for a single stage."""

    pipeline: Pipeline
    metrics: Dict[str, float]
    cv_mean: float
    cv_std: float
    feature_columns: List[str]


def _default_classifier(random_state: int) -> BaseEstimator:
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
        huber_delta=5.0,  # Increased from 1.0 to match data scale (std~29)
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

    # Salt features: ligand + ligand descriptors + solvent descriptors + engineered features
    salt_features = sorted(set(
        ligand_features + list(LIGAND_DESCRIPTOR_FEATURES) + 
        list(SOLVENT_DESCRIPTOR_FEATURES) + process_context + 
        ['Лиганд', 'Metal_Ligand_Combo', 'Log_Metal_MW', 'Is_Cu', 'Is_Zn']
    ))
    acid_features = sorted(set(salt_features + ['m (соли), г', 'n_соли', 'log_salt_mass']))
    volume_predictors = sorted(set(acid_features + ['m(кис-ты), г', 'n_кислоты', 'n_ratio']))
    tsyn_predictors = sorted(set(volume_predictors + ['Vсин. (р-ля), мл', 'Vsyn_m']))
    dry_predictors = sorted(set(tsyn_predictors + ['Tsyn_Category']))
    regen_predictors = sorted(set(dry_predictors + ['Tdry_Category']))

    return [
        StageConfig(
            name="metal",
            target="Металл",
            problem_type="classification",
            feature_columns=adsorption_features,
            estimator_factory=_default_classifier,
            description="Predict metal identity from adsorption performance.",
        ),
        StageConfig(
            name="ligand",
            target="Лиганд",
            problem_type="classification",
            feature_columns=ligand_features,
            estimator_factory=_default_classifier,
            depends_on=("Металл",),
            description="Predict ligand conditioned on adsorption profile and metal.",
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

    def fit(self, dataset: pd.DataFrame, lookup_tables: Optional[LookupTables] = None) -> None:
        """Train models for every stage."""

        if lookup_tables is None:
            lookup_tables = build_lookup_tables(dataset)
        self.lookup_tables = lookup_tables

        data = dataset.copy()
        _ensure_process_defaults(data)
        _augment_with_lookup_descriptors(data, lookup_tables)
        _update_stoichiometry_features(data)
        add_salt_mass_features(data)  # Add engineered features for salt_mass

        rng_seed = self.random_state
        self.stage_results.clear()

        for stage in self.stage_configs:
            stage_data = self._prepare_stage_dataframe(data, stage)
            if stage_data.empty:
                raise ValueError(f"Stage '{stage.name}' has no data after preprocessing")

            model = stage.estimator_factory(rng_seed)
            pipeline = self._build_pipeline(stage_data, stage.feature_columns, model)
            metrics, cv_mean, cv_std = self._train_and_evaluate(stage, pipeline, stage_data)

            self.stage_results[stage.name] = StageResult(
                pipeline=pipeline,
                metrics=metrics,
                cv_mean=cv_mean,
                cv_std=cv_std,
                feature_columns=list(stage.feature_columns),
            )

            # Store predictions, inverse-transform log_salt_mass if needed
            if stage.target == "log_salt_mass":
                data["m (соли), г"] = np.expm1(stage_data[stage.target])  # Transform back to original
                data[stage.target] = stage_data[stage.target]  # Keep log version too
            else:
                data[stage.target] = stage_data[stage.target]

        self._trained = True

    def predict(
        self,
        inputs: pd.DataFrame,
        *,
        return_intermediate: bool = True,
    ) -> pd.DataFrame:
        """Run sequential inference starting from adsorption descriptors."""

        if not self._trained:
            raise RuntimeError("Pipeline must be trained before calling predict().")
        if self.lookup_tables is None:
            raise RuntimeError("Lookup tables are unavailable. Train the pipeline first.")

        results = inputs.copy()
        _ensure_process_defaults(results)
        add_salt_mass_features(results)  # Add engineered features

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
            
            # Inverse-transform log_salt_mass predictions
            if stage.target == "log_salt_mass":
                results["m (соли), г"] = np.expm1(predictions)  # Transform back
                results[stage.target] = predictions  # Keep log version
            else:
                results[stage.target] = predictions

        if not return_intermediate:
            targets = [stage.target if stage.target != "log_salt_mass" else "m (соли), г" 
                      for stage in self.stage_configs]
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
            estimator_factory = _default_classifier if problem_type == 'classification' else _default_regressor
            stage_configs.append(StageConfig(
                name=stage_meta['name'],
                target=stage_meta['target'],
                problem_type=problem_type,
                feature_columns=tuple(stage_meta.get('feature_columns', ())),
                estimator_factory=estimator_factory,
                depends_on=tuple(stage_meta.get('depends_on', ())),
                outlier_contamination=stage_meta.get('outlier_contamination'),
                description=stage_meta.get('description', ''),
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
        cols = list(stage.feature_columns) + [stage.target]
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

        if stage.problem_type == "classification":
            class_counts = y.value_counts()
            stratify = None
            if y.nunique() > 1 and (class_counts < 2).sum() == 0:
                stratify = y
        else:
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=stratify,
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics: Dict[str, float]
        if stage.problem_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
            }
            class_counts = y.value_counts()
            min_class = class_counts.min() if not class_counts.empty else 0
            if min_class >= 2:
                n_splits = min(5, int(min_class))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            else:
                n_splits = max(2, min(5, len(y)))
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy')
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': rmse,
                'mae': mean_absolute_error(y_test, y_pred),
            }
            n_splits = max(2, min(5, len(y)))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error')

        return metrics, float(np.mean(cv_scores)), float(np.std(cv_scores))


def _augment_with_lookup_descriptors(df: pd.DataFrame, lookup: LookupTables) -> None:
    """Ensure descriptor columns are populated using lookup tables."""

    _apply_lookup(df, 'Металл', lookup.metal)
    _apply_lookup(df, 'Лиганд', lookup.ligand)
    _apply_lookup(df, 'Растворитель', lookup.solvent)


def _apply_lookup(df: pd.DataFrame, key_column: str, lookup_table: pd.DataFrame) -> None:
    if key_column not in df.columns:
        return

    for key, row in lookup_table.iterrows():
        mask = df[key_column] == key
        if not mask.any():
            continue
        for column, value in row.items():
            if column not in df.columns:
                df[column] = np.nan
            df.loc[mask, column] = df.loc[mask, column].fillna(value)


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

    if {'Vсин. (р-ля), мл', 'm (соли), г'}.issubset(df.columns):
        denom = df['m (соли), г'].replace(0, np.nan)
        df['Vsyn_m'] = df['Vсин. (р-ля), мл'] / denom
