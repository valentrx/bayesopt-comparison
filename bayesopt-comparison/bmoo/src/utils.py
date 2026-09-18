"""Common fixed-grid, Pareto and hypervolume utilities for BMOO.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import numpy as np
import gpmp.num as gnp
from gpmp.misc.designs import maximinlhs

from parallel import worker_pool, slurm_cpu_budget  # noqa: F401 (re-exported)


def fixed_lhs_grid(problem, n_candidates=4096, seed=0):
    """Generate the unique maximin LHS evaluation grid of a problem, via gpmp."""
    gnp.set_seed(seed)
    return maximinlhs(problem.dim, n_candidates, problem.bounds)


def initial_indices(grid, n_init, seed=0):
    """Select a repeat-specific maximin subset from an existing grid."""
    if n_init > len(grid):
        raise ValueError("n_init cannot exceed the grid size")

    rng = np.random.default_rng(seed)
    scale = np.maximum(np.ptp(grid, axis=0), 1e-15)
    normalized_grid = (grid - grid.min(axis=0)) / scale

    chosen = [int(rng.integers(len(grid)))]
    while len(chosen) < n_init:
        selected_points = normalized_grid[np.asarray(chosen)]
        squared_distances = (
            (normalized_grid[:, None, :] - selected_points[None, :, :]) ** 2
        ).sum(axis=2)
        score = squared_distances.min(axis=1)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))

    return np.asarray(chosen, dtype=int)


def nondominated_mask(Y):
    """Return nondominated rows for a minimization problem."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array")

    mask = np.ones(len(Y), dtype=bool)
    for i, point in enumerate(Y):
        dominated = np.any(np.all(Y <= point, axis=1) & np.any(Y < point, axis=1))
        mask[i] = not dominated
    return mask


def reference_point(Y, margin=0.1):
    """Construct a reference point worse than every observed objective."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array")

    objective_ranges = np.maximum(np.ptp(Y, axis=0), 1e-9)
    return np.max(Y, axis=0) + margin * objective_ranges


def hypervolume(Y, ref_point):
    """Calculate dominated hypervolume for minimization."""
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    Y = np.asarray(Y, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if len(Y) == 0:
        return 0.0

    pareto_front = Y[nondominated_mask(Y)]
    # BoTorch's Hypervolume uses a maximization convention; the signs are reversed because our objectives are minimized.
    calculator = Hypervolume(ref_point=torch.as_tensor(-ref_point, dtype=torch.double))
    return float(calculator.compute(torch.as_tensor(-pareto_front, dtype=torch.double)))


def hypervolume_history(Y, ref_point):
    """Return the attained hypervolume after each evaluation."""
    Y = np.asarray(Y, dtype=float)
    return np.asarray(
        [hypervolume(Y[:evaluation], ref_point) for evaluation in range(1, len(Y) + 1)]
    )


def hypervolume_contributions(observed_Y, candidate_Y, ref_point):
    """Compute the hypervolume gain produced by each candidate."""
    observed_Y = np.asarray(observed_Y, dtype=float)
    candidate_Y = np.atleast_2d(np.asarray(candidate_Y, dtype=float))
    baseline = hypervolume(observed_Y, ref_point)

    contributions = []
    for candidate in candidate_Y:
        augmented_Y = np.vstack([observed_Y, candidate])
        contribution = hypervolume(augmented_Y, ref_point) - baseline
        contributions.append(max(0.0, contribution))
    return np.asarray(contributions)
