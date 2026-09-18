"""Shared acquisition-function math for the BO comparison.

All formulas are in MAXIMIZE convention (mean/std describe the quantity being maximized); each backend applies its own min/max sign flip before calling these.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import numpy as np

from parallel import worker_pool, slurm_cpu_budget  # noqa: F401 (re-exported)

N_CANDIDATES = 4096  # power of 2: exact for scrambled Sobol' candidate sets
UCB_BETA = 2.0
LENGTHSCALE_FRACTION = 0.15


def max_variance(std):
    """Pure space-filling / active-learning criterion: argmax sigma_n^2(x)."""
    return std**2


def gaussian_weight(x, x_best, bounds, fraction=LENGTHSCALE_FRACTION):
    """Pi-BO-style Gaussian prior weight centered on the incumbent x_best.

    x: (n, dim); x_best: (dim,); bounds: (2, dim) [lower, upper].
    """
    lengthscale = fraction * (bounds[1] - bounds[0])
    z = (x - x_best) / lengthscale
    return np.exp(-0.5 * np.sum(z**2, axis=-1))


def weighted_variance(std, weight):
    """w(x) * sigma_n^2(x)."""
    return weight * std**2


def regret_band(regret, lower_pct=10, upper_pct=90):
    """Median regret with an empirical [lower_pct, upper_pct] band across
    axis 0 (repeats)."""
    median = np.median(regret, axis=0)
    lower = np.percentile(regret, lower_pct, axis=0)
    upper = np.percentile(regret, upper_pct, axis=0)
    return median, lower, upper


def regret_of(history, optimum, direction):
    """Simple regret, clipped to >= 0."""
    regret = (optimum - history) if direction == "max" else (history - optimum)
    return np.maximum(regret, 0.0)
