"""EI / MaxVar / WeightedVar / UCB Bayesian optimization loop, BoTorch
backend.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import AnalyticAcquisitionFunction, LogExpectedImprovement, UpperConfidenceBound
from botorch.utils.transforms import t_batch_mode_transform
from botorch.optim import optimize_acqf

from utils import UCB_BETA, LENGTHSCALE_FRACTION

torch.set_default_dtype(torch.float64)

_ACQF_OPTIM = dict(num_restarts=10, raw_samples=256)
# Wide enough to never bind for a well-behaved fit -- inputs are normalized to the unit cube, so a lengthscale of 10 is already "flat".
_LENGTHSCALE_BOUNDS = Interval(1e-3, 10.0)
_OUTPUTSCALE_BOUNDS = Interval(1e-3, 1e3)
_NOISE_BOUNDS = Interval(1e-6, 1.0)


class PosteriorVariance(AnalyticAcquisitionFunction):
    """argmax sigma_n^2(x): pure space-filling / active-learning criterion."""

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        _, sigma = self._mean_and_sigma(X)
        return (sigma**2).squeeze(-1)


class WeightedPosteriorVariance(AnalyticAcquisitionFunction):
    """argmax w(x) sigma_n^2(x), w Gaussian around the incumbent x_best
    (Pi-BO-style, see utils.gaussian_weight)."""

    def __init__(self, model, x_best, bounds):
        super().__init__(model)
        self.register_buffer("x_best", x_best)
        self.register_buffer("lengthscale", LENGTHSCALE_FRACTION * (bounds[1] - bounds[0]))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        _, sigma = self._mean_and_sigma(X)
        z = (X.squeeze(-2) - self.x_best) / self.lengthscale
        weight = torch.exp(-0.5 * (z**2).sum(dim=-1))
        return weight * (sigma**2).squeeze(-1)


def _sign(direction):
    return 1 if direction == "max" else -1


def _fit_model(problem, train_x, train_y, estimator="ML"):
    """estimator: "ML" (default, pure unregularized MLE via explicit
    Interval-bounded covar_module/likelihood) or "MAP" (BoTorch's own default prior-attaching factories -- leaving covar_module/likelihood unset lets SingleTaskGP build them via get_covar_module_with_dim_scaled_prior / get_gaussian_likelihood_with_ lognormal_prior, a LogNormal prior on lengthscale/noise, Hvarfner et al. 2024)."""
    if estimator == "MAP":
        model = SingleTaskGP(
            train_x,
            train_y,
            mean_module=ConstantMean(),
            input_transform=Normalize(d=problem.dim, bounds=torch.tensor(problem.bounds)),
            outcome_transform=Standardize(m=1),
        )
    else:
        # nu is fixed, not fit by ML like lengthscale/outputscale below.
        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=problem.dim, lengthscale_constraint=_LENGTHSCALE_BOUNDS),
            outputscale_constraint=_OUTPUTSCALE_BOUNDS,
        )
        likelihood = GaussianLikelihood(noise_prior=None, noise_constraint=_NOISE_BOUNDS)
        model = SingleTaskGP(
            train_x,
            train_y,
            covar_module=covar_module,
            likelihood=likelihood,
            mean_module=ConstantMean(),
            input_transform=Normalize(d=problem.dim, bounds=torch.tensor(problem.bounds)),
            outcome_transform=Standardize(m=1),
        )
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    except ModelFittingError:
        pass  # keep the untrained default hyperparameters for this iteration
    return model


def _acquisition(name, model, train_x, train_y, bounds):
    if name == "EI":
        return LogExpectedImprovement(model, best_f=train_y.max())
    if name == "UCB":
        return UpperConfidenceBound(model, beta=UCB_BETA)
    if name == "MaxVar":
        return PosteriorVariance(model)
    if name == "WeightedVar":
        x_best = train_x[train_y.argmax()]
        return WeightedPosteriorVariance(model, x_best, bounds)
    raise ValueError(f"unknown acquisition '{name}'")


def _initial_data(problem, seed, n_init, sign):
    train_x = torch.from_numpy(problem.random_initial_design(n_init, seed))
    train_y = sign * problem.fn_torch(train_x).unsqueeze(-1)
    return train_x, train_y


def run(acquisition, problem, seed, n_init, n_iter, direction="min", candidates=None, estimator="ML"):
    """Best-observed value (original f-scale) after each of n_iter
    iterations (length n_iter + 1, first entry = best of the initial design).

    candidates: externally-supplied static candidate set (problem.bounds space, torch tensor), e.g. a grid shared with another backend for a fair comparison. When given, each iteration picks argmax(acqf) over this fixed set (no gradient refinement, no resampling) instead of BoTorch's native continuous multistart optimize_acqf.

    estimator: see _fit_model."""
    torch.manual_seed(seed)
    sign = _sign(direction)
    bounds = torch.tensor(problem.bounds)
    train_x, train_y = _initial_data(problem, seed, n_init, sign)

    best = [sign * train_y.max().item()]
    for _ in range(n_iter):
        model = _fit_model(problem, train_x, train_y, estimator=estimator)
        acqf = _acquisition(acquisition, model, train_x, train_y, bounds)
        if candidates is None:
            candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, **_ACQF_OPTIM)
        else:
            with torch.no_grad():
                values = acqf(candidates.unsqueeze(-2))
            candidate = candidates[values.argmax()].unsqueeze(0)
        y_new = sign * problem.fn_torch(candidate).unsqueeze(-1)
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, y_new])
        best.append(sign * train_y.max().item())

    return np.array(best)


def fit(acquisition, problem, seed, n_init, n_iter, direction="min", estimator="ML"):
    """Run n_iter iterations and return (model, train_x, train_y, sign),
    train_x/train_y including all n_init + n_iter points, model fit on all of them. f = sign * train_y."""
    torch.manual_seed(seed)
    sign = _sign(direction)
    bounds = torch.tensor(problem.bounds)
    train_x, train_y = _initial_data(problem, seed, n_init, sign)

    for _ in range(n_iter):
        model = _fit_model(problem, train_x, train_y, estimator=estimator)
        acqf = _acquisition(acquisition, model, train_x, train_y, bounds)
        candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, **_ACQF_OPTIM)
        y_new = sign * problem.fn_torch(candidate).unsqueeze(-1)
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, y_new])

    model = _fit_model(problem, train_x, train_y, estimator=estimator)  # fit on all n_init + n_iter points
    return model, train_x, train_y, sign
