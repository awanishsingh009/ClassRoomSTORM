"""Reproducible physical examples; no experimental results are synthesized."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import erf, j1

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from classroomstorm_core.reconstruction import reconstruct_frames

AIRY_ZERO = 3.8317059702075125 / (2 * np.pi)


def airy(r):
    """Peak-normalized circular-pupil intensity, r in lambda/NA."""
    z = 2 * np.pi * np.asarray(r, dtype=float)
    out = np.ones_like(z)
    np.divide(2 * j1(z), z, out=out, where=z != 0)
    return out ** 2


def gaussian_pixels(shape, x, y, sigma, photons):
    """Integrate a normalized Gaussian over camera pixels of unit pitch."""
    yy, xx = np.indices(shape)
    c = np.sqrt(2) * sigma
    px = 0.5 * (erf((xx + .5 - x) / c) - erf((xx - .5 - x) / c))
    py = 0.5 * (erf((yy + .5 - y) / c) - erf((yy - .5 - y) / c))
    return photons * px * py


def two_emitter_example(n_frames=600, seed=20260921):
    """One emitter active per frame; actual V1.2 reconstruction entry point."""
    rng = np.random.default_rng(seed)
    shape = (80, 80)
    centers = np.array([[35.5, 39.5], [43.5, 39.5]])
    sigma, photons, background = 6.0, 8000.0, 2.0
    ids = np.arange(n_frames) % 2
    rng.shuffle(ids)
    frames = np.stack([rng.poisson(background + gaussian_pixels(shape, *centers[k], sigma, photons)) for k in ids])
    result = reconstruct_frames(frames, blinker_mode="single", threshold_quantile=.995,
                                background_mode="frame_median", processing_backend="serial")
    truth = centers[ids]
    estimates = np.array([[p['x_px'], p['y_px']] for p in result.localizations])
    matched = truth[[p['frame'] for p in result.localizations]]
    errors = estimates - matched
    parameters = dict(seed=seed, frames=n_frames, shape=list(shape), sigma_psf_px=sigma,
                      separation_px=8., signal_photons_per_active_frame=photons,
                      mean_background_counts_per_pixel=background, active_emitters_per_frame=1,
                      threshold_quantile=.995, background_mode="frame_median",
                      data_kind="synthetic pixel-integrated Poisson counts; direct frame input",
                      render_sigma_px=1.5, measured_resolution=False,
                      mean_position_error_px=float(np.linalg.norm(errors, axis=1).mean()),
                      rmse_position_px=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
                      estimated_separation_px=float(abs(estimates[ids == 1, 0].mean()-estimates[ids == 0, 0].mean())))
    return dict(frames=frames, ids=ids, truth=truth, centers=centers, result=result,
                estimates=estimates, errors=errors, parameters=parameters)
