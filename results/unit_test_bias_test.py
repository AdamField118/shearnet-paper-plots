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
    np.testing.assert_allclose([p.get_height() for p in fig.axes[1].patches], [-3, 0])
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


def test_collect_uses_the_table_statistics(tmp_path):
    from argparse import Namespace
    from make_fixture import write_fixture
    from evaluation_fits import Evaluation
    from paper_numbers import run_numbers
    from unit_test_bias import collect
    # The existing fixture and table estimator are used, not a second bias formula.
    path = tmp_path/"first.fits"
    write_fixture(path, n=300)
    args = Namespace(cut="none", correction=None, njack=20)
    result = collect(tmp_path, args)["runs"]["first"]["estimators"]
    for est in ("shearnet", "ngmix"):
        expected = run_numbers(Evaluation(path), est)
        for key in ("m1", "m1_err", "c2", "c2_err"):
            np.testing.assert_allclose(result[est][key], expected[key])
