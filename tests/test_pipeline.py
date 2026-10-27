"""Fast unit tests for the DESOM pipeline (run: pytest -q tests)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from desom.analysis.fp_normalize import normalize_fingerprints  # noqa: E402
from desom.analysis.metrics import original_space_errors, topographic_error  # noqa: E402
from desom.config import DEFAULT_CONFIG  # noqa: E402
from desom.data.consolidate import flux_outlier_masks, make_split_indices  # noqa: E402
from desom.models.som_layer import compute_map_distances, grid_distance_matrix, neighborhood_function  # noqa: E402


# ----------------------------------------------------------------------------- config
def test_paper_defaults():
    assert DEFAULT_CONFIG["preprocessing"]["min_baxel_sn"] == 10.0
    assert DEFAULT_CONFIG["preprocessing"]["rest_limits"] == [3650.0, 8950.0]
    assert DEFAULT_CONFIG["sample"]["split_mode"] == "galaxy"
    assert DEFAULT_CONFIG["training"]["map_distance"] == "manhattan"


# ----------------------------------------------------------------------------- distances
def _brute_manhattan(h, w):
    n = h * w
    D = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            D[a, b] = abs(a // w - b // w) + abs(a % w - b % w)
    return D


def test_manhattan_is_exact():
    np.testing.assert_array_equal(grid_distance_matrix((6, 4), "manhattan"), _brute_manhattan(6, 4))
    np.testing.assert_array_equal(grid_distance_matrix((15, 15), "manhattan"), _brute_manhattan(15, 15))


def test_paper_mode_replicates_original_formula():
    h = w = 15
    labels = np.arange(h * w)
    tmp = np.arange(h * w)[:, None]
    original = np.abs(tmp - labels) // w + np.abs(tmp % w - labels % w)  # Paper Original/DESOM.py
    np.testing.assert_array_equal(grid_distance_matrix((h, w), "paper"), original)


def test_modes_differ_only_where_original_is_approximate():
    exact = grid_distance_matrix((15, 15), "manhattan")
    paper = grid_distance_matrix((15, 15), "paper")
    assert (exact != paper).any()
    # Node 14 (row 0, col 14) and node 15 (row 1, col 0): exact = 1 + 14, original = 0 + 14
    assert exact[14, 15] == 15 and paper[14, 15] == 14


def test_unknown_distance_mode_raises():
    with pytest.raises(ValueError):
        grid_distance_matrix((3, 3), "euclid")


def test_neighborhood_weights():
    d = compute_map_distances(np.array([0, 4]), (3, 3))
    w = neighborhood_function(d, T=1.0)
    assert w[0, 0] == pytest.approx(1.0) and w[1, 4] == pytest.approx(1.0)
    assert w[0, 1] == pytest.approx(np.exp(-1.0))


# ----------------------------------------------------------------------------- fingerprints
def _reference_normalization(counts, pct):
    """Independent re-implementation of DESOM2_series.py (clip per node, minmax per node, per row)."""
    clipped = np.clip(counts, 0, np.percentile(counts, pct, axis=0))
    rng = clipped.max(0) - clipped.min(0)
    z2 = (clipped - clipped.min(0)) / np.where(rng == 0, 1, rng)
    zt = z2.T
    rng2 = zt.max(0) - zt.min(0)
    z3 = ((zt - zt.min(0)) / np.where(rng2 == 0, 1, rng2)).T
    return z3


def test_fingerprint_normalization_matches_paper_recipe():
    rng = np.random.RandomState(0)
    counts = rng.poisson(3, size=(200, 225)).astype(float)
    counts[:, 7] = 0  # never-used node (zero range)
    counts[5, 20] = 5000  # extreme outlier that must be clipped
    fp, stats = normalize_fingerprints(counts, (15, 15), percentile=99.9)
    assert fp.shape == (200, 15, 15, 1)
    np.testing.assert_allclose(fp.reshape(200, 225), _reference_normalization(counts, 99.9), atol=1e-6)
    assert fp.min() >= 0 and fp.max() <= 1
    assert stats["clip"][20] < 5000


def test_fingerprint_normalization_is_reusable_on_new_galaxies():
    rng = np.random.RandomState(1)
    counts = rng.poisson(2, size=(50, 25)).astype(float)
    full, stats = normalize_fingerprints(counts, (5, 5))
    again, _ = normalize_fingerprints(counts[:10], (5, 5), stats=stats)
    np.testing.assert_allclose(full[:10], again, atol=1e-6)


# ----------------------------------------------------------------------------- flux cuts / splits
def test_flux_outlier_masks():
    n_wave = 200
    wave = np.geomspace(3650, 8950, n_wave)
    X = np.ones((1000, n_wave), dtype=np.float32)
    X[3, 100] = -50.0           # strongly negative -> low-flux outlier
    X[10, 50] = 500.0           # very high away from emission lines -> removed
    k_line = int(np.argmin(np.abs(wave - 6563.0)))
    X[11, k_line] = 500.0       # very high but ON an emission line -> kept
    low, high, thr = flux_outlier_masks(X, wave, [6563.0], 0.5, 0.5, width_bins=3)
    assert low[3] and high[10] and not high[11]
    assert thr["max_threshold"] >= 1.0


@pytest.mark.parametrize("mode", ["galaxy", "spectrum"])
def test_split_partitions_everything_once(mode):
    labels = np.array(["train"] * 70 + ["validation"] * 15 + ["test"] * 15)
    tr, va, te = make_split_indices(labels, 100, mode, [0.7, 0.15, 0.15], seed=1)
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == 100 and len(set(allidx.tolist())) == 100


def test_galaxy_split_keeps_galaxies_together():
    labels = np.array(["train"] * 6 + ["test"] * 4)
    tr, va, te = make_split_indices(labels, 10, "galaxy", [0.7, 0.15, 0.15], seed=0)
    assert set(tr) == set(range(6)) and set(te) == set(range(6, 10)) and len(va) == 0


def test_spectrum_split_ignores_galaxy_labels_and_is_reproducible():
    labels = np.array(["train"] * 500 + ["test"] * 500)
    a = make_split_indices(labels, 1000, "spectrum", [0.7, 0.15, 0.15], seed=3)
    b = make_split_indices(labels, 1000, "spectrum", [0.7, 0.15, 0.15], seed=3)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert any(i >= 500 for i in a[0])  # train mixes both "galaxy" halves
    assert (len(a[0]), len(a[1]), len(a[2])) == (700, 150, 150)


def test_unknown_split_mode_raises():
    with pytest.raises(ValueError):
        make_split_indices(np.array(["train"]), 1, "bogus", [1, 0, 0], 0)


# ----------------------------------------------------------------------------- metrics
def test_topographic_error():
    d = np.array([[0.0, 1.0, 5.0, 5.0],    # best 0,1 -> adjacent on 2x2
                  [0.0, 5.0, 5.0, 1.0]])   # best 0,3 -> diagonal, not adjacent
    assert topographic_error(d, (2, 2)) == pytest.approx(0.5)


def test_original_space_errors_match_bruteforce():
    rng = np.random.RandomState(0)
    X, P = rng.rand(20, 30), rng.rand(9, 30)
    res = original_space_errors(X, P, (3, 3), chunk=7)
    d = ((X[:, None, :] - P[None]) ** 2).sum(2)
    assert res["quantization_err"] == pytest.approx(d.min(1).mean(), rel=1e-4)
    assert res["topographic_err"] == pytest.approx(topographic_error(d, (3, 3)))


# ----------------------------------------------------------------------------- models (need TensorFlow)
tf = pytest.importorskip("tensorflow")


def test_bmu_weights_graph_matches_numpy_path():
    from desom.models.desom1 import DESOM1

    m = DESOM1(input_dims=64, map_size=(4, 4), latent_dims=8)
    X = np.random.RandomState(0).rand(16, 64).astype("float32")
    m.init_som_weights(X)
    d, w = m.bmu_weights(X, T=2.0)
    bmu = m.predict_bmu(X)
    np.testing.assert_array_equal(np.argmin(d, axis=1), bmu)
    expected = neighborhood_function(compute_map_distances(bmu, (4, 4)), 2.0)
    np.testing.assert_allclose(w, expected, atol=1e-5)


def test_paper_distance_mode_changes_weights():
    from desom.models.desom1 import DESOM1

    m = DESOM1(input_dims=64, map_size=(4, 4), latent_dims=8, map_distance="paper")
    X = np.random.RandomState(0).rand(16, 64).astype("float32")
    d, w = m.bmu_weights(X, T=2.0)
    bmu = np.argmin(d, axis=1)
    expected = neighborhood_function(compute_map_distances(bmu, (4, 4), "paper"), 2.0)
    np.testing.assert_allclose(w, expected, atol=1e-5)


def test_training_engine_logs_metrics_and_resumes(tmp_path):
    from desom.models.desom1 import DESOM1
    from desom.training.engine import LOG_FIELDS, train_desom_model

    X = np.random.RandomState(0).rand(80, 64).astype("float32")
    m = DESOM1(input_dims=64, map_size=(3, 3), latent_dims=8)
    sec = dict(batch_size=16, iterations=6, som_iterations=4, Tmax=3, Tmin=0.5,
               checkpoint_interval=3, log_interval=2)
    out = train_desom_model(m, sec, X[:60], X[60:], tmp_path, "m.h5", "log.csv", "T", seed=0)
    assert out.name == "m.weights.h5" and out.exists()
    header = (tmp_path / "log.csv").read_text().splitlines()[0].split(",")
    assert header == LOG_FIELDS
    sec["iterations"] = 9
    train_desom_model(m, sec, X[:60], X[60:], tmp_path, "m.h5", "log.csv", "T", seed=0)
    assert (tmp_path / "log.csv").read_text().count("\n") > 4


def test_existing_checkpoints_still_load():
    """Weights trained with the previous version of the code must load into the refactored model."""
    w = ROOT / "workspace_data/experiments/run_500g/desom1/DESOM_model_final.weights.h5"
    cfg_file = ROOT / "workspace_data/experiments/run_500g/config.json"
    if not w.exists() or not cfg_file.exists():
        pytest.skip("no legacy checkpoint available")
    import json
    with open(cfg_file) as f:
        cfg = json.load(f)["desom1"]
    from desom.models.desom1 import DESOM1

    m = DESOM1(input_dims=4544, map_size=tuple(cfg["map_size"]), latent_dims=int(cfg["latent"]))
    m.load_weights(w)
    expected_prototypes = cfg["map_size"][0] * cfg["map_size"][1]
    assert m.prototypes.shape == (expected_prototypes, cfg["latent"])
