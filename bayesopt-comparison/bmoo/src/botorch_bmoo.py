"""BoTorch fixed-grid Bayesian multi-objective optimization backend: EHVI
and PAL.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import time, warnings
import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import pal
from utils import nondominated_mask, reference_point

torch.set_default_dtype(torch.float64)

_ACQF_OPTIM = dict(num_restarts=10, raw_samples=256)  # mirrors bo_comparison's budget
# Wide enough to never bind for a well-behaved fit -- inputs are normalized to the unit cube, so a lengthscale of 10 is already "flat".
_LENGTHSCALE_BOUNDS = Interval(1e-3, 10.0)
_OUTPUTSCALE_BOUNDS = Interval(1e-3, 1e3)
_NOISE_BOUNDS = Interval(1e-6, 1.0)


def _fit(X, Y):
    """Independent per-objective SingleTaskGP, plain ML: explicit Interval-bounded covar_module/likelihood, no priors -- bypasses BoTorch's default prior-attaching factory (GammaPrior/LogNormal on lengthscale/outputscale/noise), which is MAP, not MLE. ``p=2`` in gpmp <=> ``nu=2.5`` here, fixed rather than fit, same as gpmp's p."""
    models = []
    for j in range(Y.shape[-1]):
        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=X.shape[-1], lengthscale_constraint=_LENGTHSCALE_BOUNDS),
            outputscale_constraint=_OUTPUTSCALE_BOUNDS,
        )
        likelihood = GaussianLikelihood(noise_prior=None, noise_constraint=_NOISE_BOUNDS)
        models.append(SingleTaskGP(
            X, Y[:, j:j+1], covar_module=covar_module, likelihood=likelihood,
            mean_module=ConstantMean(),
            input_transform=Normalize(X.shape[-1]), outcome_transform=Standardize(m=1),
        ))
    model = ModelListGP(*models)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        except ModelFittingError:
            pass  # keep the untrained default hyperparameters for this iteration
    return model


def _ehvi_acqf(model, Y_max, ref_max):
    """Y_max, ref_max: objectives and reference point in BoTorch's native
    maximization convention (i.e. already sign-flipped from this module's minimization inputs)."""
    partition = NondominatedPartitioning(ref_point=ref_max, Y=Y_max)
    return ExpectedHypervolumeImprovement(model, ref_point=ref_max.tolist(), partitioning=partition)


def optimize(problem, grid, initial_indices, n_iter=25, acquisition="EHVI",
             seed=0, ref_point=None, optimizer="grid",
             pal_epsilon=1e-3, pal_delta=0.1, pal_kappa=2.0):
    """optimizer: "grid" (default -- argmax over the fixed grid shared with
    GPmp/Optuna, the main fixed-grid comparison) or "gradient" (continuous L-BFGS-B multistart via optimize_acqf, EHVI only -- PAL's query rule is a discrete classification over a finite candidate pool, not a differentiable acquisition surface, see pal.py). In "gradient" mode, `grid` still seeds the initial design and reference point but evaluated points are no longer restricted to it: returned "indices" is None, since a continuous point generally isn't a grid row.
    """
    name = acquisition.lower()
    if name not in {"ehvi", "pal"}:
        raise ValueError("BoTorch acquisition must be EHVI or PAL")
    if optimizer not in ("grid", "gradient"):
        raise ValueError("optimizer must be 'grid' or 'gradient'")
    if optimizer == "gradient" and name == "pal":
        raise ValueError("optimizer='gradient' is not supported for PAL, see pal.py")

    torch.manual_seed(seed)
    selected = list(map(int, initial_indices)); Y = problem(grid[selected])
    ref = reference_point(Y) if ref_point is None else np.asarray(ref_point)
    bounds = torch.as_tensor(problem.bounds)
    X_extra = np.empty((0, problem.dim))  # gradient-mode points, off the grid

    pal_state = None
    if name == "pal":
        pal_state = pal.ParetoActiveLearning(len(grid), problem.n_obj, pal_epsilon, pal_delta, pal_kappa)
        for i, y in zip(selected, Y):
            pal_state.mark_evaluated(i, y)

    durations = []
    for _ in range(n_iter):
        start = time.perf_counter()
        available = np.setdiff1d(np.arange(len(grid)), np.asarray(selected))
        X = torch.as_tensor(np.vstack([grid[selected], X_extra]))
        Yt = torch.as_tensor(Y)

        if name == "ehvi":
            model = _fit(X, -Yt)
            acqf = _ehvi_acqf(model, -Yt, torch.as_tensor(-ref))
            if optimizer == "grid":
                C = torch.as_tensor(grid[available])
                score = acqf(C[:, None, :]).detach().numpy().reshape(-1)
                idx = int(available[np.nanargmax(score)])
                x_new = grid[idx:idx+1]
            else:  # gradient
                candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, **_ACQF_OPTIM)
                idx, x_new = None, candidate.detach().numpy()
        else:  # pal
            model = _fit(X, Yt)
            posterior = model.posterior(torch.as_tensor(grid[available]))
            mean = posterior.mean.detach().numpy()
            std = posterior.variance.clamp_min(1e-15).sqrt().detach().numpy()
            undecided = available[pal_state.status[available] == "U"]
            positions = np.searchsorted(available, undecided)
            queried = pal_state.step(mean[positions], std[positions], undecided)
            # Every undecided candidate got decided this round: fall back to the most uncertain point overall, a plain space-filling choice, rather than leave the fixed evaluation budget unspent.
            idx = queried if queried is not None else int(available[np.argmax((std**2).sum(axis=1))])
            x_new = grid[idx:idx+1]

        if idx is not None:
            selected.append(idx)
        else:
            X_extra = np.vstack([X_extra, x_new])
        y_new = problem(x_new)
        Y = np.vstack((Y, y_new))
        if pal_state is not None:
            pal_state.mark_evaluated(idx, y_new[0])
        durations.append(time.perf_counter()-start)

    indices = np.asarray(selected) if optimizer == "grid" else None
    X_all = grid[selected] if optimizer == "grid" else np.vstack([grid[selected], X_extra])
    return {"indices": indices, "X": X_all, "Y": Y, "times": np.asarray(durations),
            "ref_point": ref, "pareto_mask": nondominated_mask(Y)}
