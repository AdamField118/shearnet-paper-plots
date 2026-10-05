"""Science-facing checks for the new categorical figure (synthetic data only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from unit_test_bias import draw, main


def test_missing_is_not_zero_and_components_are_not_in_axis_labels():
    fig = draw({"runs": {"first": {"estimators": {
        "shearnet": {"m1": .012, "m1_err": .002, "c2": -.00003, "c2_err": .00001},
        "ngmix": {"m1": 0.0, "m1_err": .001, "c2": 0.0, "c2_err": .00001}}}}})
    assert [len(ax.patches) for ax in fig.axes] == [2, 2]
    np.testing.assert_allclose([p.get_height() for p in fig.axes[0].patches], [12, 0])
    np.testing.assert_allclose([p.get_height() for p in fig.axes[1].patches], [3, 0])  # |c|
    for ax in fig.axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == ["UT1", "UT2", "UT3", "UT4"]
        assert "_1" not in ax.get_ylabel() and "_2" not in ax.get_ylabel()
        assert len([t for t in ax.texts if t.get_text() == "…"]) == 6
    plt.close(fig)


def test_draft_replay_is_identical(tmp_path):
    a, b = tmp_path/"a.pdf", tmp_path/"b.pdf"
    main(["--draft", "--out", str(a)])
    main(["--values", str(a.with_suffix('.json')), "--out", str(b)])
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(a.with_suffix('.json').read_text())["components"] == {
        "m": 1, "c": 2, "applied_shear": 1}


def test_collect_uses_the_shear_stats_estimator(tmp_path):
    """The figure's numbers are shear_stats.shear_bias's, on the cut sample."""
    from argparse import Namespace
    from catalog import Catalog
    from make_fixture import build
    from shear_stats import G1_PAIR, PAPER_CUT, Sample, shear_bias
    from unit_test_bias import collect
    path = tmp_path/"first.fits"
    build(2000, seed=5).writeto(path)
    args = Namespace(cut="metacal", min_t_ratio=None, min_s2n=None, njack=20)
    data = collect(tmp_path, args)
    result = data["runs"]["first"]["estimators"]
    assert data["runs"]["second"]["estimators"] == {}
    assert "T_t/Tpsf_t > 1" in data["cut"] and "s2n_t > 10" in data["cut"]
    sample = Sample(Catalog(path), list(G1_PAIR), cut=PAPER_CUT)
    for est in ("shearnet", "ngmix"):
        expected = shear_bias(sample, est)
        np.testing.assert_allclose(result[est]["m1"], expected["m"][0])
        np.testing.assert_allclose(result[est]["m1_err"], expected["m"][1])
        np.testing.assert_allclose(result[est]["c2"], expected["c"][0])
        np.testing.assert_allclose(result[est]["c2_err"], expected["c"][1])
        assert result[est]["m1"] != result[est]["m1_without_RS"]
