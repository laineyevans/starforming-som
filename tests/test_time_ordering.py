import csv
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embedding_diagnostics import _save_sim_colourings
from sim_time_ordering import parse_sim_metadata, plot_time_ordering


def _filenames(n, views=1):
    return [
        f"source_{i}_update_cnt_{i}_incl_{90 + view}_phi_{-90 + view}.fits"
        for view in range(views) for i in range(n)
    ]


def test_parse_sim_metadata_tolerates_real_data():
    filenames = [
        "source_update_cnt_12_incl_90_phi_-90.fits",
        "source_update_cnt_3_incl_45_phi_15.fits",
        "100132_cont_7MTM2TM1.image.fits",
    ]
    labels = ["original", "source [hflip]", "real"]
    meta = parse_sim_metadata(filenames, labels)
    np.testing.assert_array_equal(meta["frame"][:2], [12, 3])
    assert np.isnan(meta["frame"][2])
    assert meta["flip_type"].tolist() == ["original", "hflip", "original"]
    assert meta["view"][0] == "incl 90 phi -90"
    assert meta["track"][1] == "incl 45 phi 15 | hflip"


def test_time_ordering_grid_and_irregular_coordinates(tmp_path):
    n = 8
    filenames = _filenames(n)
    labels = ["original"] * n
    frames = np.arange(n, dtype=float)
    for coords in (
        np.column_stack((frames, frames % 3)),
        np.column_stack((frames * .7, np.sin(frames))),
    ):
        out = tmp_path / str(len(list(tmp_path.iterdir())))
        plot_time_ordering(np.arange(n), labels, filenames, SimpleNamespace(), coords, out)
        with open(out / "time_ordering_spearman.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert float(rows[0]["rho_x"]) > .99
        shuffled = np.random.default_rng(8).permutation(n)
        shuffled_out = tmp_path / f"shuffled_{len(list(tmp_path.iterdir()))}"
        plot_time_ordering(
            np.arange(n), labels, filenames, SimpleNamespace(),
            coords[shuffled], shuffled_out,
        )
        with open(shuffled_out / "time_ordering_spearman.csv", newline="") as handle:
            shuffled_rows = list(csv.DictReader(handle))
        assert abs(float(shuffled_rows[0]["rho_x"])) < .9


def test_save_sim_colourings(tmp_path):
    rng = np.random.default_rng(2)
    embedding = rng.normal(size=(12, 2))
    filenames = [
        f"source_{i}_update_cnt_{i}_incl_90_phi_-90.fits" for i in range(12)
    ]
    labels = ["original", "[hflip]", "[vflip]", "[hvflip]"] * 3
    meta = parse_sim_metadata(filenames, labels)
    _save_sim_colourings(str(tmp_path), "test", embedding, meta)
    expected = (
        "test_sim_frame.png", "test_sim_viewangle.png",
        "test_sim_flip.png", "test_sim_track.png",
    )
    assert all(os.path.isfile(tmp_path / name) for name in expected)
