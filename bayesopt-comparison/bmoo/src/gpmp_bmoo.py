"""GPmp BMOO: exact EHVI for 2/3 objectives, a from-scratch Feliot-style
Monte-Carlo estimate of it for any number of objectives, and PAL.

All objectives are minimized and modelled by independent GPmp GPs (plain maximum likelihood, gpmpcontrib.Model_ConstantMean_Maternp_ML -- not REML). Candidate optimization is an argmax (EHVI/MC) or PAL's own query rule over the single fixed grid supplied by compare.py.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import time
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
import gpmpcontrib as gpc
import pal
from utils import nondominated_mask, reference_point, hypervolume, hypervolume_contributions

# Matern, p=2 <=> BoTorch's nu=2.5. Fixed, not fit by the ML call below.
COVARIANCE_SPECIFICATION = {"p": 2}


def _build_model(n_obj):
    return gpc.Model_ConstantMean_Maternp_ML(
        "gpmp_bmoo", output_dim=n_obj, covariance_specification=COVARIANCE_SPECIFICATION
    )


def _fit_and_predict(model, X, Y, candidates):
    """Refit `model`'s ML hyperparameters on (X, Y) and predict at
    `candidates`. Plain ML has no REML/prior regularization keeping the optimizer off a near-singular fit; on a RuntimeError this keeps the model's last-known-good hyperparameters instead of crashing the run, the same fallback bo_comparison uses.
    """
    try:
        model.select_params(X, Y)
    except RuntimeError:
        pass
    mean, variance = model.predict(X, Y, candidates)
    return np.asarray(mean), np.maximum(np.asarray(variance), 1e-15)


def _normal_integral(lower, upper, mean, std):
    """Integral of Phi((z-mean)/std) from lower to upper."""
    def primitive(z):
        if np.isneginf(z):
            return 0.0
        a = (z - mean) / std
        return (z - mean) * ndtr(a) + std * np.exp(-0.5 * a * a) / np.sqrt(2*np.pi)
    return primitive(upper) - primitive(lower)


def _dominated(front, points):
    """Mask of the rows of points dominated by at least one front point."""
    return np.any(np.all(front[:, None, :] <= points[None, :, :], axis=2), axis=0)


def explicit_ehvi(mean, variance, observed, ref):
    """Exact EHVI for 2 or 3 independent Gaussian objectives.

    The non-dominated part of (-infinity, ref] is partitioned into axis-aligned cells induced by the observed Pareto-front coordinates. Since the expected improvement is the integral of P(Y <= z) over that region and the objectives are independent, every cell contributes a product of one-dimensional Gaussian CDF integrals available in closed form.
    """
    mean = np.atleast_2d(mean); variance = np.atleast_2d(variance)
    m = mean.shape[1]
    if m not in (2, 3):
        raise ValueError("Explicit EHVI is implemented only for 2 or 3 objectives")
    front = np.asarray(observed)[nondominated_mask(observed)]
    ref = np.asarray(ref)
    std = np.sqrt(variance)
    axes = []
    for j in range(m):
        interior = np.unique(front[front[:, j] < ref[j], j])
        axes.append(np.r_[-np.inf, interior, ref[j]])
    scores = np.zeros(len(mean))
    for index in np.ndindex(*(len(a)-1 for a in axes)):
        lower = np.array([axes[j][index[j]] for j in range(m)])
        upper = np.array([axes[j][index[j]+1] for j in range(m)])
        # Cell edges are front coordinates, so a front point dominates either the whole cell interior or none of it. The lower corner tells the two apart: probing the upper corner would also discard every cell that merely touches the front from below.
        if np.any(np.all(front <= lower, axis=1)):
            continue
        value = np.ones(len(mean))
        for j in range(m):
            value *= _normal_integral(lower[j], upper[j], mean[:, j], std[:, j])
        scores += value
    return scores


def _integration_box(mean, variance, observed, ref, n_sigma=6.0):
    """Finite box [lower, ref] holding all but a negligible part of the integral.

    The improvement region, the part of (-infinity, ref] not dominated by the front, is unbounded below. P(Y(x) <= z) is however negligible below every candidate posterior mean minus n_sigma standard deviations, so the region can be truncated there. The lower corner stays strictly below the observed ideal point, which leaves the non-dominated part of the box with a positive volume whatever the size of the front.
    """
    ref = np.asarray(ref, dtype=float)
    observed = np.asarray(observed, dtype=float)
    lower = np.minimum((mean - n_sigma*np.sqrt(variance)).min(axis=0),
                       observed.min(axis=0))
    return lower - 1e-6*np.maximum(ref-lower, 1e-9), ref


def _uniform_particles_non_dominated(observed, ref, lower, n_particles, rng,
                                     max_rounds=64):
    """Uniform particles on the non-dominated part of the box [lower, ref].

    The acceptance rate of the rejection loop also estimates the volume of that region, which normalizes the importance-sampling estimate downstream.
    """
    front = np.asarray(observed)[nondominated_mask(observed)]
    accepted, n_accepted, n_proposed = [], 0, 0
    batch = max(2*n_particles, 256)
    for _ in range(max_rounds):
        proposal = rng.uniform(lower, ref, size=(batch, len(ref)))
        n_proposed += batch
        keep = proposal[~_dominated(front, proposal)]
        if len(keep):
            accepted.append(keep); n_accepted += len(keep)
        if n_accepted >= n_particles:
            break
    if n_accepted == 0:
        # Pathological front shape: fall back on the corner strictly below the ideal point, always non-dominated, instead of looping forever.
        ideal = front.min(axis=0)
        particles = rng.uniform(lower, ideal, size=(n_particles, len(ref)))
        return particles, float(np.prod(ideal-lower)), lower
    particles = np.vstack(accepted)
    if len(particles) < n_particles:
        particles = particles[rng.integers(len(particles), size=n_particles)]
    else:
        particles = particles[:n_particles]
    return particles, float(np.prod(ref-lower))*n_accepted/n_proposed, lower


def _domination_probabilities(mean, std, particles):
    """P(Y(x) <= z) for every candidate x (rows) and particle z (columns)."""
    probabilities = np.ones((len(mean), len(particles)))
    for j in range(mean.shape[1]):
        probabilities *= ndtr((particles[None, :, j]-mean[:, None, j])/std[:, None, j])
    return probabilities


def _domination_envelope(mean, std, particles):
    """max_x P(Y(x) <= z), the SMC target shared by all candidates."""
    return np.maximum(
        _domination_probabilities(mean, std, particles).max(axis=0), 1e-300
    )


def smc_ehvi(mean, variance, observed, ref, rng, n_particles=2048, n_steps=6):
    """SMC estimate of the Feliot et al. expected hypervolume-improvement integral.

    Hand-rolled from scratch (not gpmp.mcmc.smc, which targets excursion sets / subset simulation, a different SMC problem). Particles live on the non-dominated part of the finite integration box. Tempering targets powers of the maximum candidate domination probability; multinomial resampling plus a Metropolis-Hastings random walk leave that target invariant. The final particles provide one common importance sample for every point of the fixed candidate grid:

    E[HVI(x)] = Z * E_q[P(Y(x) <= z) / envelope(z)], where Z = volume(region) * E_uniform[envelope]
    """
    lower, ref = _integration_box(mean, variance, observed, ref)
    particles, region_volume, lower = _uniform_particles_non_dominated(
        observed, ref, lower, n_particles, rng
    )
    front = np.asarray(observed)[nondominated_mask(observed)]
    std = np.sqrt(variance)
    envelope = _domination_envelope(mean, std, particles)
    # Particles are still uniform at this point, so this is the normalizing constant of the tempering target over the non-dominated region.
    normalizer = region_volume*float(envelope.mean())
    beta_previous = 0.0
    for beta in np.linspace(1/n_steps, 1.0, n_steps):
        incremental = envelope ** (beta-beta_previous)
        weights = incremental/incremental.sum()
        resampled = rng.choice(len(particles), len(particles), replace=True, p=weights)
        particles = particles[resampled]; envelope = envelope[resampled]
        scale = 0.15*(ref-lower)/np.sqrt(beta*n_steps)
        proposal = particles+rng.normal(size=particles.shape)*scale
        inside = (np.all((proposal >= lower) & (proposal <= ref), axis=1)
                  & ~_dominated(front, proposal))
        proposal_envelope = np.where(
            inside, _domination_envelope(mean, std, proposal), 0.0
        )
        # The random walk is symmetric, so the acceptance ratio is the ratio of tempered targets. Proposals leaving the region are rejected instead of clipped onto its boundary, which is what keeps the move reversible.
        accept = inside & (rng.random(len(particles))
                           < (proposal_envelope/envelope) ** beta)
        particles[accept] = proposal[accept]
        envelope[accept] = proposal_envelope[accept]
        beta_previous = beta
    probabilities = _domination_probabilities(mean, std, particles)
    return normalizer*np.mean(probabilities/envelope[None, :], axis=1)


def w_imse(model, X, Y, candidates, mean, ref, obj_scale=None):
    """One-step-lookahead w-IMSE: variance reduction on the currently
    estimated Pareto-optimal candidates from evaluating each candidate x.

    J_n(x) = sum_i w_n(x_i) * sum_j s_{j,n}^2 * sigma^2_{j,n+1}(x_i | x) w_n(x_i) = (V_{n,i} - V_n)/V_n  if x_i in N_n, else 0

    V_n is the hypervolume attained by Y; V_{n,i} is the hypervolume Y would reach if x_i's posterior mean were its true value (utils.hypervolume_contributions); N_n is implicitly the candidates with a positive contribution (a dominated-by-mean candidate always contributes 0, so no separate mask is needed). s_{j,n}^2 defaults to 1/range(Y_j)^2, the same per-objective normalization metrics.igd_plus uses.

    J_n(x) is minimized by the criterion, but every other acquisition in this module is chosen by argmax, so this returns

    sum_i w_n(x_i) * sum_j s_{j,n}^2 * [sigma^2_{j,n}(x_i) - sigma^2_{j,n+1}(x_i|x)]

    instead -- J_n(x) with the sign flipped and a candidate-independent additive constant (sum_i w_n(x_i) sum_j s_{j,n}^2 sigma^2_{j,n}(x_i)) dropped, so argmax here is exactly argmin J_n(x).

    The lookahead variance needs no refit: a GP variance update after a fictitious observation at x depends only on the covariance structure, not on the (unknown) value observed there, so it is read off the posterior covariance matrix already available at step n -- Model.kriging_predictor(..., return_type=1) -- once per objective instead of once per (reference point, candidate) pair.

    Cost note: that covariance matrix is (n_ref+n_cand) x (n_ref+n_cand), so this criterion is O(n_cand^2) per objective per iteration, unlike the O(n_cand) cost of explicit_ehvi/smc_ehvi/PAL -- expect it to be markedly slower at n_candidates=4096.
    """
    n_obj = Y.shape[1]
    if obj_scale is None:
        obj_scale = 1.0/np.maximum(np.ptp(Y, axis=0), 1e-12)**2

    contributions = hypervolume_contributions(Y, mean, ref)
    active = contributions > 0
    reduction = np.zeros(len(candidates))
    if not np.any(active):
        return reduction

    weight = contributions[active]/max(hypervolume(Y, ref), 1e-300)
    X_ref = candidates[active]
    n_ref = len(X_ref)

    for j in range(n_obj):
        gp = model.models[j]["model"]
        _, cov = gp.kriging_predictor(X, np.vstack([X_ref, candidates]), return_type=1)
        cov = np.asarray(cov)
        var_cand = np.maximum(np.diag(cov)[n_ref:], 1e-300)
        cross = cov[:n_ref, n_ref:]
        reduction += obj_scale[j]*(weight[:, None]*(cross**2/var_cand[None, :])).sum(axis=0)
    return reduction


def _fit_only(model, X, Y):
    """Refit `model`'s ML hyperparameters on (X, Y), no prediction -- used
    by the gradient-based candidate optimizer below, which then calls model.predict repeatedly (raw-sample screening, then L-BFGS-B refinement) against one fixed fit per outer iteration. Same RuntimeError fallback as _fit_and_predict."""
    try:
        model.select_params(X, Y)
    except RuntimeError:
        pass


def _neg_explicit_ehvi(x_flat, model, X, Y, ref):
    mean, variance = model.predict(X, Y, x_flat.reshape(1, -1))
    variance = np.maximum(np.asarray(variance), 1e-15)
    return -float(explicit_ehvi(np.asarray(mean), variance, Y, ref)[0])


def _gradient_argmax(model, X, Y, ref, lo, hi, num_restarts, raw_samples, rng):
    """L-BFGS-B multistart argmax of explicit_ehvi over the continuous box
    [lo, hi]. gpmp-contrib's torch backend makes model.predict differentiable w.r.t. the candidate point (bo_comparison/gpmp_bo.py exploits this for single-objective EI), but explicit_ehvi's own cell-decomposition integral is plain numpy/scipy.special.ndtr, not autodiff-traced -- so this reuses scipy's own finite-difference gradient (no `jac` passed to `minimize`) instead of reimplementing explicit_ehvi in torch. Same raw_samples + L-BFGS-B multistart recipe as bo_comparison/gpmp_bo._gradient_argmax and BoTorch's optimize_acqf.
    """
    dim = lo.shape[0]
    raw = lo + rng.random((raw_samples, dim)) * (hi - lo)
    mean, variance = model.predict(X, Y, raw)
    scores = explicit_ehvi(np.asarray(mean), np.maximum(np.asarray(variance), 1e-15), Y, ref)
    starts = raw[np.argsort(-scores)[:num_restarts]]

    scipy_bounds = list(zip(lo, hi))
    best_x, best_neg = starts[0], np.inf
    for x0 in starts:
        res = minimize(_neg_explicit_ehvi, x0, args=(model, X, Y, ref), method="L-BFGS-B", bounds=scipy_bounds)
        if res.fun < best_neg:
            best_neg, best_x = res.fun, res.x
    return best_x


def optimize(problem, grid, initial_indices, n_iter=25, acquisition="MC_sMC",
             seed=0, ref_point=None, n_particles=2048, smc_steps=6,
             pal_epsilon=1e-3, pal_delta=0.1, pal_kappa=2.0, optimizer="grid",
             num_restarts=10, raw_samples=256):
    """optimizer: "grid" (default -- argmax over `grid`, shared with
    BoTorch/PAL) or "gradient" (continuous L-BFGS-B multistart on explicit_ehvi, EHVI_explicit only -- MC_sMC's SMC particles, PAL's query rule and W_IMSE's lookahead variance are all tied to the fixed grid; mirrors botorch_bmoo.optimize's own "grid"/"gradient" split). In "gradient" mode `grid` still seeds the initial design and reference point, but evaluated points are no longer restricted to it: returned "indices" is None, exactly like botorch_bmoo.optimize's gradient mode.
    """
    rng = np.random.default_rng(seed)
    selected = list(map(int, initial_indices)); Y = problem(grid[selected])
    ref = reference_point(Y) if ref_point is None else np.asarray(ref_point)
    name = acquisition.lower()
    if name not in {"ehvi_explicit", "mc_smc", "mc", "smc", "pal", "w_imse"}:
        raise ValueError("GPmp acquisition must be EHVI_explicit, MC_sMC, PAL or W_IMSE")
    if optimizer not in ("grid", "gradient"):
        raise ValueError("optimizer must be 'grid' or 'gradient'")
    if optimizer == "gradient" and name != "ehvi_explicit":
        raise ValueError("optimizer='gradient' is only supported for EHVI_explicit")

    model = _build_model(problem.n_obj)
    pal_state = None
    if name == "pal":
        pal_state = pal.ParetoActiveLearning(len(grid), problem.n_obj, pal_epsilon, pal_delta, pal_kappa)
        for i, y in zip(selected, Y):
            pal_state.mark_evaluated(i, y)

    lo, hi = problem.bounds
    X_extra = np.empty((0, problem.dim))  # gradient-mode points, off the grid

    durations = []
    for _ in range(n_iter):
        start = time.perf_counter()
        available = np.setdiff1d(np.arange(len(grid)), np.asarray(selected))
        X = np.vstack([grid[selected], X_extra])

        if optimizer == "gradient":  # name == "ehvi_explicit", enforced above
            _fit_only(model, X, Y)
            x_new = _gradient_argmax(model, X, Y, ref, lo, hi, num_restarts, raw_samples, rng).reshape(1, -1)
            idx = None
        else:
            mean, variance = _fit_and_predict(model, X, Y, grid[available])
            if name == "ehvi_explicit":
                idx = int(available[np.nanargmax(explicit_ehvi(mean, variance, Y, ref))])
            elif name in {"mc_smc", "mc", "smc"}:
                score = smc_ehvi(mean, variance, Y, ref, rng, n_particles, smc_steps)
                idx = int(available[np.nanargmax(score)])
            elif name == "w_imse":
                score = w_imse(model, X, Y, grid[available], mean, ref)
                # No candidate improves any currently-estimated Pareto-optimal point's variance (e.g. a degenerate front): fall back to the most uncertain point overall, same rule PAL uses.
                idx = (int(available[np.nanargmax(score)]) if np.any(score > 0)
                       else int(available[np.argmax(variance.sum(axis=1))]))
            else:  # pal
                undecided = available[pal_state.status[available] == "U"]
                positions = np.searchsorted(available, undecided)
                queried = pal_state.step(mean[positions], np.sqrt(variance[positions]), undecided)
                # Every undecided candidate got decided this round: fall back to the most uncertain point overall, a plain space-filling choice, rather than leave the fixed evaluation budget unspent.
                idx = queried if queried is not None else int(available[np.argmax(variance.sum(axis=1))])

        if idx is not None:
            selected.append(idx); y_new = problem(grid[idx:idx+1])
        else:
            X_extra = np.vstack([X_extra, x_new]); y_new = problem(x_new)
        Y = np.vstack((Y, y_new))
        if pal_state is not None:
            pal_state.mark_evaluated(idx, y_new[0])
        durations.append(time.perf_counter()-start)

    indices = np.asarray(selected) if optimizer == "grid" else None
    X_all = grid[selected] if optimizer == "grid" else np.vstack([grid[selected], X_extra])
    return {"indices":indices, "X":X_all, "Y":Y,
            "times":np.asarray(durations), "ref_point":ref,
            "pareto_mask":nondominated_mask(Y)}
