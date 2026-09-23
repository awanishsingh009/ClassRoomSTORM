"""Independent checks on the physical examples used in the revised figures."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/figures_src"))
from figure_data import airy, AIRY_ZERO, gaussian_pixels, two_emitter_example


def test_airy_peak_and_first_zero_have_physical_normalization():
    assert airy(0) == pytest.approx(1)
    assert AIRY_ZERO == pytest.approx(.6098349456)
    assert airy(AIRY_ZERO) < 1e-25
    # Evaluating a different radial range must not change the normalization.
    np.testing.assert_allclose(airy(np.array([.4,.5])), [airy(.4),airy(.5)])


def test_integrated_psf_conserves_photons_and_subpixel_position():
    expected = gaussian_pixels((61,61), 30.3, 29.7, 2.4, 5000)
    y,x = np.indices(expected.shape)
    assert expected.sum() == pytest.approx(5000, rel=1e-12)
    assert (expected*x).sum()/expected.sum() == pytest.approx(30.3, abs=1e-6)
    assert (expected*y).sum()/expected.sum() == pytest.approx(29.7, abs=1e-6)


def test_rayleigh_component_data_add_without_separate_renormalization():
    data = ROOT / "docs/figures_src/data"
    for name in ["above", "at", "below"]:
        table = np.loadtxt(data / f"rayleigh_{name}.csv", delimiter=",", skiprows=1)
        np.testing.assert_allclose(table[:,1]+table[:,2], table[:,3], atol=1e-14)


def test_synthetic_positions_validate_independently_of_display_width():
    ex = two_emitter_example(n_frames=80, seed=20260921)
    assert len(ex['result'].localizations) == 80
    assert abs(ex['parameters']['estimated_separation_px']-8) < .3
    assert ex['parameters']['rmse_position_px'] < .8
    assert ex['result'].superres.sum() == pytest.approx(80, rel=1e-6)
