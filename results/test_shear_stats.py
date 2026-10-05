"""The selection, responses and shear bias, against a catalog with known answers.

Run with: python -m pytest results/test_shear_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shear_stats  # noqa: E402
from catalog import Catalog, CatalogError, find_run  # noqa: E402
from make_fixture import STEP, TRUTH, build  # noqa: E402
from shear_stats import (PAPER_CUT, SelectionCut, Sample, cut_from_name,  # noqa: E402
                         leakage_inputs, ngmix_passes, responses, shear_bias)

PAIR = ["zero", "g1_plus", "g1_minus"]


@pytest.fixture(scope="module")
def fixture_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("cat") / "fixture.fits"
    build(20000, seed=3).writeto(path)
    return path


@pytest.fixture(scope="module")
def cat(fixture_path):
    return Catalog(fixture_path)


def _value(result, key):
    return float(np.asarray(result[key][0]))


def _error(result, key):
    return float(np.asarray(result[key][1]))


# --------------------------------------------------------------------------
# the cut reads ngmix's own fit, of the right product, with strict inequalities
# --------------------------------------------------------------------------

def test_paper_cut_is_t_ratio_one_and_s2n_ten():
    assert PAPER_CUT == SelectionCut(min_t_ratio=1.0, min_s2n=10.0)
    assert cut_from_name("metacal") == PAPER_CUT
    assert cut_from_name("none") is None
    assert cut_from_name("metacal", min_s2n=15).min_s2n == 15
    with pytest.raises(ValueError):
        cut_from_name("superbit")


def test_each_product_is_cut_on_its_own_fit(tmp_path):
    hdul = build(200, seed=1, kappa=0.0)
    ngmix = hdul["NGMIX"].data
    # every object passes everywhere, then break exactly one thing per product
    for t in ("noshear", "1p", "1m", "2p", "2m"):
        ngmix[f"s2n_{t}"][:] = 50.0
        ngmix[f"T_{t}"][:] = 1.0
        ngmix[f"Tpsf_{t}"][:] = 0.1
    ngmix["s2n_1p"][0] = 10.0          # == threshold: fails (strict)
    ngmix["T_1m"][1] = 0.1             # T/Tpsf == 1: fails (strict)
    ngmix["flags_2p"][2] = 4           # ngmix failed on that product
    ngmix["s2n_noshear"][3] = 10.0001  # just passes
    path = tmp_path / "c.fits"
    hdul.writeto(path)
    cat = Catalog(path)
    scene = ["zero"]
    flat = {}
    for t in ("noshear", "1p", "1m", "2p", "2m"):
        ngmix_cols = cat.columns("NGMIX", shear_stats._cut_columns(t), scene)
        flat[t] = ngmix_passes(ngmix_cols, t, PAPER_CUT).reshape(-1)
    assert not flat["1p"][0] and flat["noshear"][0]
    assert not flat["1m"][1] and flat["noshear"][1]
    assert not flat["2p"][2] and flat["noshear"][2]
    assert flat["noshear"][3]
    assert all(flat[t][4:].all() for t in flat)


def test_selection_ignores_truth_and_stamp_quantities(tmp_path):
    """Scrambling every non-ngmix size or S/N must not change the sample."""
    hdul = build(500, seed=2)
    path_a, path_b = tmp_path / "a.fits", tmp_path / "b.fits"
    hdul.writeto(path_a)
    hdul["STAMP"].data["s2n_stamp"][:] = 0.0
    hdul["TRUTH"].data["e_prepsf"][:] = 0.0
    hdul.writeto(path_b)
    a = Sample(Catalog(path_a), PAIR).select["noshear"]
    b = Sample(Catalog(path_b), PAIR).select["noshear"]
    assert np.array_equal(a, b) and 0 < a.mean() < 1


# --------------------------------------------------------------------------
# responses
# --------------------------------------------------------------------------

def test_responses_recover_the_injected_matrices(cat):
    sample = Sample(cat, PAIR, cut=None)
    for est in ("ngmix", "shearnet"):
        r = responses(sample, est)
        assert np.allclose(r["Rg"][0], TRUTH[est]["R"] * np.eye(2), atol=1e-9)
        assert np.allclose(r["Rp"][0], TRUTH[est]["rho"] * np.eye(2), atol=1e-9)
        assert np.allclose(r["RS"][0], 0.0)          # no cut, no selection response


def test_the_cut_creates_a_selection_response_of_the_right_sign(cat):
    """S/N grows with measured g1, so a lower effective threshold under +g1
    admits galaxies with *lower* g1: R^S_11 < 0, R^S_22 ~ 0."""
    r = responses(Sample(cat, PAIR, cut=PAPER_CUT), "ngmix")
    rs, err = r["RS"]
    assert rs[0, 0] < -10 * err[0, 0]
    assert abs(rs[1, 1]) < 4 * err[1, 1] + 1e-12
    assert np.allclose(r["R"][0], r["Rg"][0] + r["RS"][0])


# --------------------------------------------------------------------------
# m and c
# --------------------------------------------------------------------------

def test_m_is_exact_without_a_cut(tmp_path):
    path = tmp_path / "m.fits"
    build(3000, seed=4, shearnet_m=0.02).writeto(path)
    sample = Sample(Catalog(path), PAIR, cut=None)
    assert abs(_value(shear_bias(sample, "ngmix"), "m")) < 1e-9
    assert _value(shear_bias(sample, "shearnet"), "m") == pytest.approx(0.02, abs=1e-9)


def test_selection_response_removes_the_selection_bias(cat):
    sample = Sample(cat, PAIR, cut=PAPER_CUT)
    for est in ("ngmix", "shearnet"):
        b = shear_bias(sample, est)
        # without R^S the cut biases m by ~ R^S / R, i.e. tens of percent
        assert _value(b, "m_without_RS") < -0.1
        # with it, m is the injected zero
        assert abs(_value(b, "m")) < 3.5 * _error(b, "m")
        assert _error(b, "m") < 0.01


def test_ngmix_c_is_psf_corrected_and_shearnet_c_is_not(cat, monkeypatch):
    sample = Sample(cat, PAIR, cut=None)
    corrected = _value(shear_bias(sample, "ngmix"), "c")
    monkeypatch.setitem(shear_stats.PSF_CORRECTED, "ngmix", False)
    raw = _value(shear_bias(sample, "ngmix"), "c")
    mean_psf_g2 = float(np.mean(sample.psf_g[..., 1]))
    expected_leak = TRUTH["ngmix"]["rho"] * mean_psf_g2 / TRUTH["ngmix"]["R"]
    assert raw - corrected == pytest.approx(expected_leak, rel=1e-6)
    assert abs(corrected) < 1e-3


def test_the_jackknife_drops_whole_objects_from_every_scene(cat):
    blocks = shear_stats._Blocks(cat.n_objects, 20)
    assert len(blocks.starts) == 20 and blocks.starts[0] == 0
    sample = Sample(cat, PAIR, cut=PAPER_CUT)
    blocks.count("n", sample.select["noshear"])
    # one block's sum is per scene, so deleting it removes those objects
    # from the +g and -g populations together
    assert blocks.sums["n"].shape == (len(PAIR), 20)


# --------------------------------------------------------------------------
# Figure 5 input
# --------------------------------------------------------------------------

def test_leakage_keeps_only_ring_complete_objects_and_the_same_ones(cat):
    sample = Sample(cat, PAIR, cut=PAPER_CUT)
    a = leakage_inputs(sample, "ngmix")
    b = leakage_inputs(sample, "shearnet")
    complete = sample.select["noshear"][0].all(axis=0)
    assert a["n_objects"] == b["n_objects"] == complete.sum() < cat.n_objects
    assert np.array_equal(a["e1_psf"], b["e1_psf"])


def test_leakage_slope_is_in_the_raw_original_shape(cat):
    sample = Sample(cat, PAIR, cut=None)
    for est in ("ngmix", "shearnet"):
        d = leakage_inputs(sample, est)
        for i, x in ((0, "e1"), (1, "e2")):
            slope = np.polyfit(d[f"{x}_psf"], d[f"{x}_gal"], 1)[0]
            assert slope == pytest.approx(TRUTH[est]["alpha"][i], abs=0.01)


# --------------------------------------------------------------------------
# the reader
# --------------------------------------------------------------------------

def test_old_layout_files_are_refused(tmp_path):
    path = tmp_path / "old.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(
        [fits.Column(name="x", array=np.zeros(3), format="D")], name="TAB_P")]).writeto(path)
    with pytest.raises(CatalogError, match="shearnet-eval"):
        Catalog(path)


def test_columns_reshape_to_scene_station_object(cat):
    rows = cat.column("TRUTH", "catalog_row", ["g1_minus"])
    assert rows.shape == (1, cat.n_rotations, cat.n_objects)
    assert np.array_equal(rows[0, 2], np.arange(cat.n_objects))
    g = cat.column("TRUTH", "g_applied", ["g1_plus", "g1_minus"])
    assert np.all(g[0, ..., 0] == 0.01) and np.all(g[1, ..., 0] == -0.01)
    assert cat.step == STEP


def test_find_run_reads_a_shearnet_runs_directory(tmp_path, fixture_path):
    run = tmp_path / "unit_tests" / "third" / "evaluations" / "default"
    run.mkdir(parents=True)
    target = run / "d4_unit_third_default.fits"
    target.symlink_to(fixture_path)
    assert find_run(tmp_path, "third") == target
    assert find_run(tmp_path / "unit_tests", "third") == target
    assert find_run(tmp_path, "first") is None
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "fourth.fits").symlink_to(fixture_path)
    assert find_run(flat, "fourth") == flat / "fourth.fits"
