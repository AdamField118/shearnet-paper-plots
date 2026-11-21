import numpy as np
import pytest

from catalog import Catalog
from flux_hlr_residuals import gaussian_hlr, residuals
from make_fixture import build
from shear_stats import SelectionCut


def test_radius_from_elliptical_gaussian():
    # For principal-axis widths a,b, the half-flux ellipse has equivalent
    # circular radius sqrt(2 ln(2) ab), independent of position angle.
    a, b = 0.6, 0.2
    mag = (a - b) / (a + b)
    g = np.array([[mag, 0], [0, mag], [0, 0], [1, 0]])
    got = gaussian_hlr(np.array([a*a + b*b, a*a + b*b, 2*b*b, 1.]), g)
    np.testing.assert_allclose(got[:3], np.sqrt(2*np.log(2) * np.array([a*b, a*b, b*b])))
    assert np.isnan(got[3])


def test_paired_residuals_from_original_stamps(tmp_path):
    n = 7
    hdus = build(n)
    ng, sn = (hdus[table].data for table in ("NGMIX", "SHEARNET"))
    ng["flags_original"] = 1
    ng["flags_original"][:5] = 0
    ng["flux_original"][:5] = [250, -10, 100, 200, 300]
    ng["g_original"][:5] = 0
    ng["T_original"][:5] = 1/np.log(2.)
    sn["flux_original"][:5] = [100, 80, 90, 110, 130]
    sn["hlr_original"][:5] = .5
    hdus["TRUTH"].data["flux_model"][:5] = 100.
    hdus["TRUTH"].data["hlr"][:5] = .75
    ng["T_noshear"][:5] = 2 * hdus["STAMP"].data["psf_T_admom"][:5]
    ng["s2n_noshear"][:5] = 20.
    ng["s2n_noshear"][2] = 10.  # strict paper cut
    sn["flags_original"][3] = 1
    ng["g_original"][4] = [1.1, 0]  # no physical Gaussian radius
    ng["flags_original"][4*n] = 0  # next scene; must not enter
    path = tmp_path / "fourth.fits"
    hdus.writeto(path)
    with Catalog(path) as cat:
        s, g, truth = residuals(cat, chunk=2)
        nogate = residuals(cat, cut=None, chunk=4)
        with pytest.raises(ValueError, match="no matched"):
            residuals(cat, SelectionCut(min_s2n=1000))
    np.testing.assert_allclose(s, [[0, -.25], [-20, -.25]])
    np.testing.assert_allclose(g, [[150, .25], [-110, .25]])
    np.testing.assert_allclose(truth, [[100, .75], [100, .75]])
    assert len(nogate[0]) == 3
