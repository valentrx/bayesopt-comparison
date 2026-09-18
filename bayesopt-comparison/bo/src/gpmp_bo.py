"""EI / MaxVar / WeightedVar / UCB Bayesian optimization loop, gpmp-contrib
backend.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import numpy as np
import torch  # only for the autograd leaf tensor _neg_ei_value_and_grad needs
                # to hook gpmp's own torch backend into scipy's L-BFGS-B
from scipy.optimize import minimize
import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc
import gpmpcontrib.samplingcriteria as sampcrit

import utils

ACQUISITIONS = ("EI", "MaxVar", "WeightedVar", "UCB")
_ACQF_OPTIM = dict(num_restarts=10, raw_samples=256)  # mirrors botorch_bo's multistart budget


class BOGridSearch(gpc.SequentialStrategyGridSearch):
    """Sequential grid-search strategy driven by one of utils.py's criteria."""

    def __init__(self, problem, model, xt, acquisition, bounds):
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition '{acquisition}'")
        self.acquisition = acquisition
        self.bounds = bounds
        super().__init__(problem, model, xt)

    def update_current_estimate(self):
        self.current_estimate = gnp.min(self.zi)

    def sampling_criterion(self):
        if self.acquisition == "EI":
            return sampcrit.expected_improvement(-self.current_estimate, -self.zpm, self.zpv)
        if self.acquisition == "MaxVar":
            return self.zpv
        if self.acquisition == "WeightedVar":
            x_best = gnp.to_np(self.xi)[np.argmin(gnp.to_np(self.zi).reshape(-1))]
            weight = utils.gaussian_weight(gnp.to_np(self.xt), x_best, self.bounds)
            return gnp.asarray(weight).reshape(-1, 1) * self.zpv
        return -self.zpm + utils.UCB_BETA * gnp.sqrt(self.zpv)  # UCB


def _model(problem, estimator="ML"):
    """estimator: "ML" (default, pure unregularized MLE, fitting sigma^2 and rho by maximum likelihood, nu fixed) or "MAP" (gpmp-contrib's native REMAP: REML + Gaussian prior on log(sigma^2) + barrier-linear prior on logrho, prior hyperparameters resolved automatically from the data)."""
    # p (Matern smoothness) is fixed, not fit by ML.
    if estimator == "MAP":
        return gpc.Model_ConstantMean_Maternp_REMAP(
            problem.name,
            output_dim=1,
            mean_specification={"type": "constant"},
            covariance_specification={"p": 2},
        )
    return gpc.Model_ConstantMean_Maternp_ML(
        problem.name,
        output_dim=1,
        covariance_specification={"p": 2},
    )


def _build(problem, seed, n_init, acquisition, direction="min", xt=None, estimator="ML"):
    """xt: externally-supplied candidate grid (problem.bounds space), e.g. a
    grid shared with another backend for a fair comparison. Defaults to gpmp-contrib's own fixed low-discrepancy grid (ldrandunif) when None.estimator: see _model."""
    sign = -1 if direction == "max" else 1
    computer_experiment = gpc.ComputerExperiment(
        problem.dim, problem.bounds, single_objective=lambda x: sign * problem.fn(x)
    )
    if xt is None:
        gnp.set_seed(seed)
        xt = gp.misc.designs.ldrandunif(
            problem.dim, utils.N_CANDIDATES, computer_experiment.input_box
        )

    model = _model(problem, estimator)
    algo = BOGridSearch(computer_experiment, model, xt, acquisition, problem.bounds)
    algo.set_initial_design(problem.random_initial_design(n_init, seed))
    algo.sign = sign
    return algo


def _param_snapshot(algo):
    return [
        {
            "meanparam": gnp.copy(m["model"].meanparam) if m["model"].meanparam is not None else None,
            "covparam": gnp.copy(m["model"].covparam) if m["model"].covparam is not None else None,
        }
        for m in algo.model.models
    ]


def _restore_params(algo, snapshot):
    for m, snap in zip(algo.model.models, snapshot):
        if snap["meanparam"] is not None:
            m["model"].meanparam = snap["meanparam"]
        if snap["covparam"] is not None:
            m["model"].covparam = snap["covparam"]


def _safe_step(algo):
    # Plain ML has no REML/prior (no nugget/jitter) keeping the covariance well-conditioned; select_params can "succeed" (no exception) onto a near-singular covariance that then fails immediately afterwards, in update_predictions() -- retrying that same call with the same fresh (bad) hyperparameters fails identically, so first fall back to the last-known-good hyperparameters (snapshotted before step()). Even those aren't a guarantee: as more points accumulate the same hyperparameters can themselves become singular on the larger dataset (observed directly: order-51 params, fine one iteration earlier, failing again once evaluated against the order-52 data) -- exactly the "Consider using jitter" case gpmp's own warning names, with no jitter implemented here to add. So if the retry also fails, give up on refreshing zpm/zpv for just this one iteration (next iteration's sampling_criterion uses one-iteration-stale predictions) rather than crash; update_current_estimate is pure data reduction (gnp.min(self.zi)), never touches the covariance, so it's always safe to call. Either way the new observation itself is never lost: set_new_eval appends it before select_params runs.
    snapshot = _param_snapshot(algo)
    try:
        algo.step()
        return
    except RuntimeError:
        pass
    _restore_params(algo, snapshot)
    try:
        algo.update_predictions()
    except RuntimeError:
        pass
    algo.update_current_estimate()
    algo.n_iter += 1


def _neg_ei_value_and_grad(x_np, xi, zi, model, current_estimate):
    """Negative EI and its gradient at x, obtained by autodiff through
    gpmp's own (torch-backed) model.predict -- gradients w.r.t. the candidate point flow through it out of the box (verified against finite differences, see grid_ablation.py's module docstring), so this reuses gpmp's own prediction and gpmpcontrib.samplingcriteria.expected_improvement rather than reimplementing any GP/acquisition math. x_np: (dim,)."""
    x = torch.tensor(x_np.reshape(1, -1), dtype=torch.float64, requires_grad=True)
    zpm, zpv = model.predict(xi, zi, x, convert_out=False)
    ei = sampcrit.expected_improvement(-current_estimate, -zpm, zpv).sum()
    (-ei).backward()
    return float(-ei.item()), gnp.to_np(x.grad).reshape(-1)


def _gradient_argmax(model, xi, zi, current_estimate, lo, hi, num_restarts, raw_samples):
    """L-BFGS-B multistart argmax of EI over the continuous box [lo, hi] --
    gpmp-contrib has no native gradient-based acqf optimizer (only a fixed candidate grid or SMC), so this reimplements the standard "draw raw_samples, refine the best num_restarts with L-BFGS-B" recipe (mirrors botorch_bo's optimize_acqf budget) directly on top of gpmp's own autodiff-through-predict -- only optimizer plumbing is new, no duplicated GP/acquisition logic."""
    dim = lo.shape[0]
    raw = gnp.to_np(lo + gnp.rand(raw_samples, dim) * (hi - lo))

    zpm, zpv = model.predict(xi, zi, gnp.asarray(raw), convert_out=False)
    ei_raw = gnp.to_np(sampcrit.expected_improvement(-current_estimate, -zpm, zpv)).reshape(-1)
    starts = raw[np.argsort(-ei_raw)[:num_restarts]]

    scipy_bounds = list(zip(gnp.to_np(lo), gnp.to_np(hi)))
    best_x, best_neg_ei = starts[0], np.inf
    for x0 in starts:
        res = minimize(
            _neg_ei_value_and_grad, x0, args=(xi, zi, model, current_estimate),
            jac=True, method="L-BFGS-B", bounds=scipy_bounds,
        )
        if res.fun < best_neg_ei:
            best_neg_ei, best_x = res.fun, res.x
    return best_x


def _safe_new_eval(sp, xnew, znew):
    """Same fallback philosophy as _safe_step: set_new_eval appends the
    new point before select_params runs, so a failed refit never loses data -- just keep the last-known-good hyperparameters instead of a freshly (possibly degenerate) one."""
    snapshot = _param_snapshot(sp)
    try:
        sp.set_new_eval_with_model_selection(xnew, znew)
    except RuntimeError:
        _restore_params(sp, snapshot)


def _run_gradient(problem, seed, n_init, n_iter, direction, estimator, num_restarts, raw_samples):
    """EI only (see run's optimizer="gradient"). Reuses
    gpmpcontrib.SequentialPrediction for data/refit bookkeeping (the same set_new_eval_with_model_selection gpmpcontrib.SequentialStrategy.make_new_eval calls internally) instead of gpc.SequentialStrategyGridSearch, since there is no fixed candidate set here."""
    sign = -1 if direction == "max" else 1
    gnp.set_seed(seed)
    lo, hi = gnp.asarray(problem.bounds[0]), gnp.asarray(problem.bounds[1])

    model = _model(problem, estimator)
    sp = gpc.SequentialPrediction(model)
    x_init = problem.random_initial_design(n_init, seed)
    sp.set_data_with_model_selection(x_init, sign * problem.fn(x_init))

    best = [sign * gnp.to_scalar(gnp.min(sp.zi))]
    for _ in range(n_iter):
        current_estimate = gnp.to_scalar(gnp.min(sp.zi))
        x_new = _gradient_argmax(sp.model, sp.xi, sp.zi, current_estimate, lo, hi, num_restarts, raw_samples)
        x_new = x_new.reshape(1, -1)
        z_new = sign * problem.fn(x_new)
        _safe_new_eval(sp, x_new, z_new)
        best.append(sign * gnp.to_scalar(gnp.min(sp.zi)))
    return np.array(best)


def run(
    acquisition, problem, seed, n_init, n_iter, direction="min", xt=None, estimator="ML",
    optimizer="grid", num_restarts=_ACQF_OPTIM["num_restarts"], raw_samples=_ACQF_OPTIM["raw_samples"],
):
    """Best-observed value (original f-scale) after each of n_iter
    iterations (length n_iter + 1, first entry = best of the initial design). xt, estimator: see _build.

    optimizer: "grid" (default, gpc.SequentialStrategyGridSearch argmax over xt) or "gradient" (continuous L-BFGS-B multistart via autodiff, EI only -- see _run_gradient/_gradient_argmax; xt is ignored, there is no candidate grid in this mode)."""
    if optimizer == "gradient":
        if acquisition != "EI":
            raise ValueError("optimizer='gradient' currently only supports acquisition='EI'")
        return _run_gradient(problem, seed, n_init, n_iter, direction, estimator, num_restarts, raw_samples)

    algo = _build(problem, seed, n_init, acquisition, direction, xt=xt, estimator=estimator)
    best = [algo.sign * gnp.to_scalar(gnp.min(algo.zi))]
    for _ in range(n_iter):
        _safe_step(algo)
        best.append(algo.sign * gnp.to_scalar(gnp.min(algo.zi)))
    return np.array(best)


def fit(acquisition, problem, seed, n_init, n_iter, direction="min", estimator="ML"):
    """Run n_iter iterations and return the fitted algorithm, for posterior
    inspection (algo.predict, algo.xi, algo.zi, algo.sign)."""
    algo = _build(problem, seed, n_init, acquisition, direction, estimator=estimator)
    for _ in range(n_iter):
        _safe_step(algo)
    return algo
