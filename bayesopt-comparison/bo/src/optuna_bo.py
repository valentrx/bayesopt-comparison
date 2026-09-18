"""EI / MaxVar / WeightedVar / UCB Bayesian optimization loop, Optuna
backend. Drives optuna._gp (private API, pinned to optuna==4.2.1) directly, since optuna.samplers.GPSampler hardcodes logEI with no pluggable acquisition -- this is the same GP engine GPSampler calls internally, just without the Study/Sampler wrapper.

optuna._gp works in inputs normalized to [0, 1]^dim; train_x is kept in that normalized space throughout and only mapped back to problem.bounds to evaluate problem.fn and to report results.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import numpy as np
import torch
from scipy.stats import qmc

import optuna._gp.gp as gp
import optuna._gp.acqf as acqf
import optuna._gp.optim_mixed as optim_mixed
import optuna._gp.prior as gp_prior
import optuna._gp.search_space as sps

import utils

ACQUISITIONS = ("EI", "MaxVar", "WeightedVar", "UCB")
_ACQF_TYPE = {
    "EI": acqf.AcquisitionFunctionType.LOG_EI,
    "UCB": acqf.AcquisitionFunctionType.UCB,
}

# Plain MLE: fit_kernel_params minimizes -marginal_log_likelihood(params) - log_prior(params); identically 0 drops the regularization term that optuna's own default_log_prior (a Gamma-prior MAP) would add.
_zero_log_prior = lambda kernel_params: torch.tensor(0.0, dtype=torch.float64)


def _search_space(dim):
    return sps.SearchSpace(
        scale_types=np.full(dim, sps.ScaleType.LINEAR),
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        steps=np.zeros(dim),
    )


def _fit_kernel(X, Y, is_categorical, estimator="ML"):
    """estimator: "ML" (default, pure unregularized MLE) or "MAP" (Optuna's
    own default_log_prior -- Gamma(2,1) on kernel_scale, Gamma(1.1,30) on noise_var, mode away from 0). nu=2.5 is hardcoded in optuna._gp, not fit here."""
    y_std = (Y - Y.mean()) / max(Y.std(), 1e-6)
    log_prior = gp_prior.default_log_prior if estimator == "MAP" else _zero_log_prior
    kernel_params = gp.fit_kernel_params(
        X=X,
        Y=y_std,
        is_categorical=is_categorical,
        log_prior=log_prior,
        minimum_noise=gp_prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    return kernel_params, y_std


def _next_point(acquisition, X, Y, search_space, is_categorical, candidates, rng, estimator="ML"):
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition '{acquisition}'")
    kernel_params, y_std = _fit_kernel(X, Y, is_categorical, estimator=estimator)

    if acquisition in _ACQF_TYPE:
        params = acqf.create_acqf_params(
            acqf_type=_ACQF_TYPE[acquisition],
            kernel_params=kernel_params,
            search_space=search_space,
            X=X,
            Y=y_std,
            beta=utils.UCB_BETA if acquisition == "UCB" else None,
        )
        if candidates is not None:  # static grid: argmax only, no local refinement
            values = acqf.eval_acqf_no_grad(params, candidates)
            return candidates[np.argmax(values)]
        x_next, _ = optim_mixed.optimize_acqf_mixed(
            params, warmstart_normalized_params_array=X[np.argmax(y_std)][None, :], rng=rng
        )
        return x_next

    # MaxVar / WeightedVar: no native acqf_type in optuna._gp, so score a fixed Sobol' candidate set from the GP posterior directly.
    params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.UCB,
        kernel_params=kernel_params,
        search_space=search_space,
        X=X,
        Y=y_std,
        beta=utils.UCB_BETA,
    )
    mean, var = gp.posterior(
        kernel_params,
        torch.from_numpy(X),
        torch.from_numpy(is_categorical),
        torch.from_numpy(params.cov_Y_Y_inv),
        torch.from_numpy(params.cov_Y_Y_inv_Y),
        torch.from_numpy(candidates),
    )
    std = var.clamp_min(0.0).sqrt().numpy()
    if acquisition == "MaxVar":
        crit = utils.max_variance(std)
    else:  # WeightedVar
        x_best = X[np.argmax(y_std)]
        unit_box = np.array([np.zeros_like(x_best), np.ones_like(x_best)])
        weight = utils.gaussian_weight(candidates, x_best, unit_box)
        crit = utils.weighted_variance(std, weight)
    return candidates[np.argmax(crit)]


def run(acquisition, problem, seed, n_init, n_iter, direction="min", candidates=None, estimator="ML"):
    """Best-observed value (original f-scale) after each of n_iter
    iterations (length n_iter + 1, first entry = best of the initial design).

    candidates: externally-supplied static candidate set, normalized to [0, 1]^dim (e.g. a grid shared with another backend for a fair comparison), reused for every iteration -- every acquisition then picks argmax(acqf) over this fixed set instead of Optuna's native optimizer. When None (default): EI/UCB use Optuna's native optimizer (optim_mixed.optimize_acqf_mixed); MaxVar/WeightedVar have no native optimizer in optuna._gp, so they fall back to argmax over a fresh Sobol' redraw every iteration, from one persistent per-seed sequence.

    estimator: see _fit_kernel."""
    sign = 1 if direction == "max" else -1  # internal loop always maximizes Y = sign * f
    lo, hi = problem.bounds

    x_init_raw = problem.random_initial_design(n_init, seed)
    X = (x_init_raw - lo) / (hi - lo)
    Y = sign * problem.fn(x_init_raw)

    is_categorical = np.zeros(problem.dim, dtype=bool)
    search_space = _search_space(problem.dim)
    # Only acquisitions with no native optimizer (MaxVar/WeightedVar, see _next_point) need a scored fallback candidate set when none is supplied; EI/UCB must instead keep candidates=None all the way into _next_point so it reaches optim_mixed.optimize_acqf_mixed. Previously this Sobol' redraw fed *every* acquisition unconditionally, so EI/UCB always took the argmax-over-Sobol' branch and never used Optuna's native gradient-based optimizer.
    needs_sobol_fallback = candidates is None and acquisition not in _ACQF_TYPE
    sobol = qmc.Sobol(d=problem.dim, scramble=True, seed=seed) if needs_sobol_fallback else None
    rng = np.random.RandomState(seed)

    best = [Y.max()]
    for _ in range(n_iter):
        if candidates is not None:
            step_candidates = candidates
        elif sobol is not None:
            step_candidates = sobol.random(utils.N_CANDIDATES)
        else:
            step_candidates = None
        x_next = _next_point(acquisition, X, Y, search_space, is_categorical, step_candidates, rng, estimator=estimator)
        x_next_raw = lo + x_next * (hi - lo)
        y_next = sign * problem.fn(x_next_raw[None, :])[0]
        X = np.vstack([X, x_next])
        Y = np.append(Y, y_next)
        best.append(Y.max())

    return sign * np.array(best)
