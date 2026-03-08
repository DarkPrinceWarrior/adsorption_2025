import pandas as pd

from scripts.benchmark_wave2 import build_inverse_benchmarks
from src.adsorb_synthesis.holdout_evaluation import build_chemistry_holdout_split


def test_build_chemistry_holdout_split_has_disjoint_groups():
    df = pd.DataFrame(
        {
            "Металл": ["Cu", "Cu", "Al", "Al", "Fe", "Fe"],
            "Лиганд": ["BTC", "BTC", "BTC", "BTC", "BDC", "BDC"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )

    split = build_chemistry_holdout_split(df, holdout_fraction=0.34, seed=42)

    train_groups = set(
        df.iloc[split.train_indices][["Металл", "Лиганд"]].astype(str).agg("|".join, axis=1)
    )
    holdout_groups = set(
        df.iloc[split.holdout_indices][["Металл", "Лиганд"]].astype(str).agg("|".join, axis=1)
    )

    assert train_groups
    assert holdout_groups
    assert train_groups.isdisjoint(holdout_groups)


def test_inverse_direct_is_separated_from_optimizer_benchmark():
    optimizer_shortlist = pd.DataFrame(
        {
            "backend": ["bofire"],
            "score": [0.2],
            "feasible": [True],
            "Металл": ["Cu"],
            "Лиганд": ["BTC"],
            "Растворитель": ["ДМФА"],
            "Т.син., °С": [130.0],
            "Т суш., °С": [130.0],
            "Tрег, ᵒС": [130.0],
        }
    )
    direct_predictions = pd.DataFrame(
        {
            "backend": ["inverse_direct"],
            "score": [0.9],
            "feasible": [True],
        }
    )

    optimizer_df, direct_df = build_inverse_benchmarks(
        {
            "/tmp/predictions_bofire.csv": optimizer_shortlist,
            "/tmp/inverse_direct/predictions.csv": direct_predictions,
        },
        {
            "/tmp/predictions_bofire_all.csv": optimizer_shortlist,
            "/tmp/inverse_direct/predictions.csv": direct_predictions,
        },
    )

    assert list(optimizer_df["backend"]) == ["bofire"]
    assert list(direct_df["backend"]) == ["inverse_direct"]
