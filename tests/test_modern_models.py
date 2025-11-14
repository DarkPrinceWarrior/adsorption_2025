import numpy as np
import pandas as pd

from src.adsorb_synthesis.modern_models import _ModernTabularEnsemble


def test_compute_physics_sample_weights_vectorized():
    ensemble = _ModernTabularEnsemble(problem_type='classification', physics_loss_weight=0.5)

    frames = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
    base = np.ones(3, dtype=np.float32)
    ensemble.physics_loss_fn = lambda df: np.array([0.0, 0.5, 1.0], dtype=np.float64)

    weights = ensemble._compute_physics_sample_weights(frames, base)
    np.testing.assert_allclose(weights.sum(), base.sum())
    assert weights[2] > weights[1] > weights[0]


def test_compute_physics_sample_weights_fallback():
    ensemble = _ModernTabularEnsemble(problem_type='classification', physics_loss_weight=0.5)
    base = np.ones(2, dtype=np.float32)
    weights = ensemble._compute_physics_sample_weights(None, base)
    np.testing.assert_array_equal(weights, base)
