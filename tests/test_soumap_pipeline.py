from types import SimpleNamespace
import os

import csv
import numpy as np

import soumap_module


def _config(tmp_path):
    values = {
        "soumap_prototypes": 4,
        "soumap_random_state": 7,
        "soumap_som_rho_0": 2.0,
        "soumap_som_rho_f": 1.0,
        "soumap_som_nbr_topo_alpha_0": 0.5,
        "soumap_som_nbr_topo_alpha_f": 0.5,
        "soumap_som_coord_topo": "delaunay",
        "soumap_som_proto_topo": "CONN",
        "soumap_som_compute_dr_metrics": False,
        "soumap_umap_update_freq": 1,
        "soumap_umap_n_epochs": 1,
        "soumap_umap_min_dist_0": 0.01,
        "soumap_umap_min_dist_f": 0.01,
        "soumap_umap_spread_0": 2.0,
        "soumap_umap_spread_f": 2.0,
        "soumap_umap_lrate_0": 1.0,
        "soumap_umap_lrate_f": 0.1,
        "soumap_umap_negative_sample_rate": 5,
        "soumap_umap_gamma": 1.0,
        "soumap_umap_use_standard_umap": False,
        "soumap_ctrl_total_epochs": 2,
        "soumap_ctrl_min_nhbs": 3,
        "soumap_ctrl_fill_val": 0.5,
        "soumap_ctrl_coord_init": "pca",
        "soumap_ctrl_embedding_range": (-3.0, 3.0),
        "soumap_ctrl_n_jobs": 1,
        "soumap_ctrl_random_state": None,
        "soumap_ctrl_verbose": False,
        "soumap_ctrl_plot_every": 0,
        "run_dir": str(tmp_path),
        "premade_autoencoder_results_dir": "synthetic",
        "premade_encoded_already_scaled": True,
        "ae_bottleneck_dim": 16,
        "ae_use_bottleneck": True,
    }
    return SimpleNamespace(**values)


def test_soumap_adapter_outputs(tmp_path):
    X = np.random.default_rng(3).normal(size=(60, 16)).astype(np.float32)
    labels = [f"source_{i}_cont_.fits" for i in range(len(X))]
    config = _config(tmp_path)

    result = soumap_module.train_soumap(X, labels, config)
    assert result["bmu"].shape == (60,)
    assert np.issubdtype(result["bmu"].dtype, np.integer)
    assert np.all((result["bmu"] >= 0) & (result["bmu"] < 4))
    assert result["sample_coords"].shape == (60, 2)
    assert result["W"].shape == (4, 16)

    soumap_module.write_outputs(
        result, labels, labels, np.zeros(len(X), dtype=float), config
    )
    with open(tmp_path / "soumap_prototypes.csv", newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 4
    with open(tmp_path / "source_cluster_assignments.csv", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader)[:8] == soumap_module.SOURCE_COLUMNS
        assert sum(1 for _ in reader) == 60


def test_soumap_prototype_marker_sizes_are_bounded():
    sizes = soumap_module._prototype_marker_sizes([1, 500])
    assert np.all((sizes >= 30.0) & (sizes <= 600.0))
    assert sizes[0] == 30.0 + 570.0 / 500.0
    assert sizes[1] == 600.0


def test_mpec_round_trip_and_diagnostics(tmp_path):
    X = np.random.default_rng(8).normal(size=(60, 8)).astype(np.float32)
    labels = [f"source_{i}_cont_.fits" for i in range(len(X))]
    config = _config(tmp_path)
    config.mpec_output_dir = str(tmp_path / "mpec_plots")
    config.mpec_plot_which = "all"
    config.mpec_plot_format = "png"
    config.mpec_verbose = False
    result = soumap_module.train_soumap(X, labels, config)

    mpec = soumap_module.run_mpec(result, config)
    assert mpec.labels_.shape == (4,)
    assert np.sum(mpec.labels_ == -1) == np.sum(np.asarray(result["cadj"]).sum(axis=1) == 0)
    assert mpec.n_steps_ >= 1
    assert os.path.isfile(tmp_path / "mpec.h5")
    loaded = mpec.__class__.load(str(tmp_path / "mpec.h5"))
    np.testing.assert_array_equal(loaded.labels_, mpec.labels_)
    for name in (
        "step_tradeoff", "edge_distributions", "modularity",
        "cluster_sizes", "embedding", "view_agreement",
    ):
        path = tmp_path / "mpec_plots" / f"mpec_{name}.png"
        assert path.is_file() and path.stat().st_size > 0
