import json
from types import SimpleNamespace

import numpy as np
import pytest

from gsom_module import GSOM, train_gsom


def synthetic_data(seed=7):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 50)) * 3
    X = np.vstack([center + rng.normal(size=(200, 50)) for center in centers])
    return (X - X.mean(axis=0)) / X.std(axis=0)


def test_learning_rate_schedule_and_validation():
    old = GSOM(2, init_lr=0.2)
    for progress in (0, 0.5, 1):
        assert old._lr_at(progress) == pytest.approx(0.2 * np.exp(-3 * progress))

    current = GSOM(2, init_lr=0.2, lr_end=0.005)
    assert current._lr_at(0) == pytest.approx(0.2, abs=1e-9)
    assert current._lr_at(1) == pytest.approx(0.005, abs=1e-9)
    assert current._lr_at(0.5) > current._lr_at(1)
    with pytest.raises(ValueError):
        GSOM(2, init_lr=0.2, lr_end=0.3)
    with pytest.raises(ValueError):
        GSOM(2, init_lr=0.2, lr_end=0)


def test_adapt_updates_only_bmu_and_direct_neighbours():
    gsom = GSOM(2, init_lr=0.2, init_grid_size=2, nb_lr_fraction=0.5,
                init_vectors=np.zeros((4, 2)))
    bmu = gsom.nodes[0]
    before = {id(node): node.weights.copy() for node in gsom.nodes}
    inp = np.ones(2)
    gsom._adapt(bmu, inp, 0.2)

    assert np.allclose(bmu.weights, before[id(bmu)] + 0.2 * (inp - before[id(bmu)]))
    neighbours = [nb for nb in (bmu.left, bmu.right, bmu.up, bmu.down) if nb is not None]
    for node in neighbours:
        assert np.allclose(
            node.weights,
            before[id(node)] + 0.1 * (inp - before[id(node)]),
        )
    untouched = [node for node in gsom.nodes if node is not bmu and node not in neighbours]
    assert all(np.array_equal(node.weights, before[id(node)]) for node in untouched)

    zero_nb = GSOM(2, init_grid_size=2, nb_lr_fraction=0,
                   init_vectors=np.zeros((4, 2)))
    before = [node.weights.copy() for node in zero_nb.nodes]
    zero_nb._adapt(zero_nb.nodes[0], inp, 0.2)
    assert all(np.array_equal(node.weights, old) for node, old in
               zip(zero_nb.nodes[1:], before[1:]))
    with pytest.raises(ValueError):
        GSOM(2, nb_lr_fraction=-0.1)
    with pytest.raises(ValueError):
        GSOM(2, nb_lr_fraction=1.1)


def test_seeded_runs_are_reproducible():
    X = synthetic_data()

    def run(seed):
        gsom = GSOM(50, init_lr=0.2, init_grid_size=3, max_nodes=16,
                    growth_phase_ratio=0.5, seed=seed)
        gsom.fit(X, iterations=120, verbose=False)
        return ([(n.x, n.y) for n in gsom.nodes],
                [n.weights.copy() for n in gsom.nodes])

    first = run(42)
    second = run(42)
    third = run(43)
    assert first[0] == second[0]
    assert all(np.array_equal(a, b) for a, b in zip(first[1], second[1]))
    assert first[0] != third[0] or any(
        not np.array_equal(a, b) for a, b in zip(first[1], third[1])
    )


def test_growth_diagnostics_warning_and_fail_fast(capsys):
    X = synthetic_data()
    gsom = GSOM(50, init_grid_size=3, max_nodes=16, seed=42)
    gsom.fit(X, iterations=5, verbose=False)
    summary = gsom.growth_summary()
    assert {"seed_nodes", "final_nodes", "max_nodes", "nodes_at_growth_end",
            "growth_events", "blocked_hits", "blocked_neighbours",
            "blocked_no_boundary", "blocked_cap", "blocked_error",
            "growth_cutoff", "iterations", "GT"} <= summary.keys()
    assert summary["final_nodes"] == len(gsom.nodes)
    assert summary["nodes_at_growth_end"] <= summary["final_nodes"]
    assert summary["final_nodes"] <= summary["max_nodes"]
    assert summary["growth_events"] == len(gsom.growth_log)

    config = SimpleNamespace(
        gsom_spread_factor=0.6, gsom_learning_rate=0.2, gsom_iterations=5,
        gsom_max_nodes=16, gsom_growth_phase_ratio=1 / 3,
        gsom_init_grid_size=3, gsom_seed=42, gsom_require_full_nodes=True,
    )
    with pytest.raises(RuntimeError):
        train_gsom(X, None, config)
    output = capsys.readouterr().out
    assert "GSOM growth summary:" in output
    assert "WARNING:" in output


def test_train_gsom_writes_summary(tmp_path):
    X = synthetic_data()
    config = SimpleNamespace(
        gsom_spread_factor=0.6, gsom_learning_rate=0.2, gsom_iterations=40,
        gsom_max_nodes=16, gsom_growth_phase_ratio=1 / 3,
        gsom_init_grid_size=3, gsom_seed=42, run_dir=str(tmp_path),
    )
    result = train_gsom(X, None, config)
    assert len(result) == 3
    summary_path = tmp_path / "gsom_growth_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["seed"] == 42
    assert "lr_end" in summary
    assert "nb_lr_fraction" in summary
