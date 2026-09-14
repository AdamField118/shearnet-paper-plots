"""Whether TeX can render on this machine. Nothing else.

This deliberately does NOT wrap ``superbit_lensing.plotter.pub_rc``. Figures
call ``pub_rc`` directly, as LITB-III-plots does
(``sec4/sec4.5/fig16_psf_leakage.ipynb``), so the styling that reaches the page
is theirs and not a local re-spelling of it. All that lives here is an
environment question, which is not a plotting concern.

The question has to be asked because ``pub_rc`` sets ``text.usetex: True``,
which is right for the paper and fatal on a TeX install missing the packages
matplotlib needs. On the cluster it fails inside ``tight_layout``::

    ! Package matplotlib-support Error: Missing cm-super package,
      required by Matplotlib.
    RuntimeError: latex was not able to process the following string: b'lp'

-- after the figure has been computed, with a traceback naming neither the
figure nor the fix.

Checking that the ``latex`` binary exists is not enough: it exists there and
still cannot render, because the failure is a missing font package rather than
a missing compiler. So this renders a label and looks.

Three ways to get the real fonts, least work first:

1. ``tlmgr`` ships with TinyTeX, which is what raised the error
   (``~/.TinyTeX/``). It is a TeX package manager, not a Python one::

       ~/.TinyTeX/bin/*/tlmgr install cm-super type1cm dvipng

2. conda, to leave the TeX install alone::

       conda install -c conda-forge texlive-core

3. Nothing: the caller drops ``text.usetex`` and mathtext draws the labels.

pip cannot help -- the missing piece is a TeX font package, not a Python one.

Set ``SHEARNET_NO_TEX=1`` to skip the probe and assume TeX is unavailable.
"""

from __future__ import annotations

import os
import warnings

_CACHED = None


def _probe() -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with plt.rc_context({"text.usetex": True}):
                figure = plt.figure(figsize=(1, 1))
                figure.text(0.5, 0.5, r"$R^{\gamma}$ lp")
                figure.canvas.draw()
                plt.close(figure)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def tex_available() -> bool:
    """Whether TeX rendering works here. Probed once per process."""
    global _CACHED
    if os.environ.get("SHEARNET_NO_TEX"):
        return False
    if _CACHED is None:
        _CACHED = _probe()
    return _CACHED


def warn_once() -> None:
    """Say what was lost and how to get it back, at most once per process."""
    if not getattr(warn_once, "_said", False):
        print("note: TeX cannot render here, so text.usetex is off and the\n"
              "      figure uses mathtext. For the paper's fonts:\n"
              "      ~/.TinyTeX/bin/*/tlmgr install cm-super type1cm dvipng")
        warn_once._said = True
