from types import SimpleNamespace

import numpy as np
import pytest

from gsom_module import GSOM, train_gsom
from init_utils import make_init
from som_module import train_som


def data(seed=12):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, 50)) * 3
    X = np.vstack([c + rng.normal(size=(200, 50)) for c in centres])
    return (X - X.mean(0)) / X.std(0)


def test_initializer_shapes_and_errors():
    X = data()
    for method in ("pca", "kmeans++", "random"):
        assert make_init(method, X, grid_shape=(6, 6)).shape == (36, 50)
    with pytest.raises(ValueError, match="grid_shape"):
        make_init("pca", X, n_prototypes=36)
    with pytest.raises(ValueError, match="valid options"):
        make_init("not-a-method", X, n_prototypes=3)


def test_pca_is_rank_two_seed_independent_and_grid_oriented():
    X = data()
    first = make_init("pca", X, grid_shape=(6, 6), seed=1)
    second = make_init("pca", X, grid_shape=(6, 6), seed=99)
    assert np.array_equal(first, second)
    assert np.linalg.matrix_rank(first - first.mean(0)) == 2
    neighbour = np.mean([
        np.linalg.norm(first[r * 6 + c] - first[r * 6 + c + 1])
        for r in range(6) for c in range(5)
    ] + [
        np.linalg.norm(first[r * 6 + c] - first[(r + 1) * 6 + c])
        for r in range(5) for c in range(6)
    ])
    pairs = np.linalg.norm(
        first[:, None, :] - first[None, :, :], axis=2
    )
    assert neighbour < pairs[np.triu_indices(36, 1)].mean()
    rectangular = make_init("pca", X, grid_shape=(4, 7))
    row_span = np.linalg.norm(rectangular[21] - rectangular[0])
    col_span = np.linalg.norm(rectangular[6] - rectangular[0])
    assert col_span > row_span


def test_kmeans_and_random_initializers():
    X = data()
    first = make_init("kmeans++", X, n_prototypes=10, seed=1)
    second = make_init("kmeanspp", X, n_prototypes=10, seed=2)
    assert all(any(np.array_equal(row, source) for source in X) for row in first)
    assert not np.array_equal(first, second)
    with pytest.raises(ValueError):
        make_init("kmeans++", X, n_prototypes=len(X) + 1)
    random = make_init("random", X, n_prototypes=1000, seed=3)
    assert np.allclose(random.mean(0), X.mean(0), atol=0.15)
    assert np.allclose(random.std(0), X.std(0), atol=0.15)


def test_som_all_initializers():
    X = data()
    results = []
    for method in ("pca", "kmeans++", "random"):
        config = SimpleNamespace(
            init_method=method, init_seed=42, n_num=6, m_num=6,
            rough_learning_rate=0.2, rough_sigma=2.5,
            fine_learning_rate=0.05, fine_sigma=1.0, fine_lr_end=0.005,
            fine_sigma_end=1.0, rough_epochs_num=20, fine_epochs_num=40,
            train_in_blocks_flag=True, som_block_size=1000,
        )
        som, preds, qe_rough, qe_fine, hits = train_som(X, None, config)
        results.append(som.quantization_error(X))
        assert som.weights.shape == (36, 50)
        assert preds.shape == (len(X),)
        assert np.isfinite(results[-1])
        assert len(qe_rough) + len(qe_fine) == 2
        assert hits.shape == (6, 6)
    assert max(results) < 2 * min(results)


def test_gsom_all_initializers_and_validation(capsys):
    X = data()
    for method in ("pca", "kmeans++", "random"):
        config = SimpleNamespace(
            init_method=method, init_seed=42, gsom_spread_factor=0.6,
            gsom_learning_rate=0.2, gsom_iterations=30, gsom_max_nodes=16,
            gsom_growth_phase_ratio=1 / 3, gsom_init_grid_size=3,
        )
        result = train_gsom(X, None, config)
        assert len(result) == 3
        assert np.isfinite(result[2])
    with pytest.raises(ValueError, match="init_vectors"):
        GSOM(3, init_grid_size=3, init_vectors=np.zeros((2, 3)))
