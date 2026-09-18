"""Performance metrics for the fixed-grid BMOO benchmark.

Every metric compares a run against the same oracle: the Pareto front of the whole candidate grid. Since all algorithms are restricted to that grid, the oracle is the exact supremum of what any of them can attain, reached only by exhaustive evaluation. It is an oracle relative to the protocol, not to the problem: the grid being an LHS sample, it underestimates the hypervolume of the true continuous Pareto front by an amount that depends on the grid density.

The metrics fall in two families, and reporting one of each is deliberate:

- ``hypervolume`` and ``hypervolume_regret`` depend on the reference point r, in magnitude and -- between two mutually non-dominating fronts -- in ranking as well. Pushing r away from the front makes the absolute regret diverge and the relative one collapse to zero, so no choice of r is neutral.
- ``igd_plus`` and ``oracle_front_recall`` do not involve r at all.

When the two families agree on a ranking, that ranking is trustworthy. When they disagree, the fronts are mutually non-dominating and no total order over them is meaningful.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import numpy as np
from utils import nondominated_mask, hypervolume


def build_oracle(grid_values, ref_point):
    """Precompute everything the metrics need about a problem grid.

    Done once per problem: the Pareto-optimal subset of the grid costs O(|G|^2 m) and would otherwise be recomputed for every prefix of every run.
    """
    grid_values = np.asarray(grid_values, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if grid_values.ndim != 2:
        raise ValueError("grid_values must be a two-dimensional array")
    optimal_mask = nondominated_mask(grid_values)
    return {
        "grid_values": grid_values,
        "front": grid_values[optimal_mask],
        "optimal_mask": optimal_mask,
        "hypervolume": hypervolume(grid_values, ref_point),
        "scale": np.maximum(np.ptp(grid_values, axis=0), 1e-12),
        "ref_point": ref_point,
    }


def attained_hypervolume(Y, oracle):
    """Dominated hypervolume H(Y; r) of the evaluated objective vectors."""
    return hypervolume(Y, oracle["ref_point"])


def hypervolume_regret(Y, oracle, relative=True):
    """H* - H(Y; r), the multi-objective analogue of the single-objective regret.

    Relative by default. H* is a per-problem constant, independent of the number of evaluations and of the algorithm, so dividing by it only shifts the curve vertically on a log axis: convergence rates and rankings are unchanged. What it buys is comparability across problems and invariance under a per-objective affine rescaling of the objectives -- both H and H* pick up the same factor, since reference_point is equivariant. Pass relative=False for the raw volume in objective units.
    """
    regret = oracle["hypervolume"] - attained_hypervolume(Y, oracle)
    regret = max(regret, 0.0)  # Y is a subset of the grid, so H <= H*
    return regret/oracle["hypervolume"] if relative else regret


def igd_plus(Y, oracle, normalize=True):
    """Inverted generational distance IGD+ against the oracle front.

    Following Ishibuchi et al. (2015), for minimization:

    IGD+(A) = mean over z in P* of  min over a in A of  ||max(a - z, 0)||_2

    The modified distance charges only the coordinates in which a is worse than z. That is what makes IGD+ weakly compatible with Pareto dominance, where plain IGD is not: adding a dominating point can never increase IGD+.

    Coordinates are divided by the per-objective grid range by default, so the result is dimensionless and comparable across problems. Unlike the hypervolume family, no reference point enters.
    """
    Y = np.asarray(Y, dtype=float)
    front = Y[nondominated_mask(Y)]
    reference = oracle["front"]
    scale = oracle["scale"] if normalize else np.ones(reference.shape[1])
    # (len(reference), len(front), m): how much each observed point is worse than each oracle point, coordinate by coordinate.
    gap = np.maximum((front[None, :, :] - reference[:, None, :])/scale, 0.0)
    return float(np.mean(np.min(np.sqrt(np.sum(gap**2, axis=2)), axis=1)))


def oracle_front_recall(indices, oracle):
    """Fraction of the Pareto-optimal grid points that were actually evaluated.

    Purely combinatorial: no reference point, no distance, no normalization. Available only because the search space is a finite grid shared by every algorithm, which makes the set of attainable optima explicitly enumerable.
    """
    optimal_mask = oracle["optimal_mask"]
    n_optimal = int(optimal_mask.sum())
    if n_optimal == 0:
        return float("nan")
    evaluated = np.zeros(len(optimal_mask), dtype=bool)
    evaluated[np.asarray(indices, dtype=int)] = True
    return float((optimal_mask & evaluated).sum())/n_optimal


NAMES = ("hypervolume", "hypervolume_regret", "igd_plus", "oracle_front_recall")

# Axis labels stay short on purpose: the exact definitions live in the docstrings above, where they can be read, rather than on a crowded axis.
LABELS = {
    "hypervolume": "HV",
    "hypervolume_regret": "Relative HV regret",
    "igd_plus": "IGD+",
    "oracle_front_recall": "Oracle-front recall",
}

ABSOLUTE_REGRET_LABEL = "Absolute HV regret"

LOG_SCALE = {"hypervolume_regret", "igd_plus"}


def evaluation_histories(indices, oracle, relative_regret=True):
    """All four metrics, one value per evaluation, from the evaluated indices: indices (shape (n_evaluations,) or (n_repeats, n_evaluations)) are the grid indices in evaluation order, as returned by the optimizers; oracle is build_oracle's output for the matching grid; relative_regret is passed through to hypervolume_regret. Returns a dict of arrays, each shaped like indices. Indices alone are enough to rebuild everything: the grid is deterministic and saved alongside the results, so the evaluated objective vectors are exactly grid_values[indices] -- that is why compare.run stores them, letting any new metric be computed after the fact without re-running the optimizers."""
    indices = np.asarray(indices, dtype=int)
    squeeze = indices.ndim == 1
    indices = np.atleast_2d(indices)
    grid_values = oracle["grid_values"]
    out = {name: np.empty(indices.shape, dtype=float) for name in NAMES}
    for repeat, row in enumerate(indices):
        for n in range(1, len(row)+1):
            prefix = row[:n]
            Y = grid_values[prefix]
            out["hypervolume"][repeat, n-1] = attained_hypervolume(Y, oracle)
            out["hypervolume_regret"][repeat, n-1] = hypervolume_regret(
                Y, oracle, relative_regret
            )
            out["igd_plus"][repeat, n-1] = igd_plus(Y, oracle)
            out["oracle_front_recall"][repeat, n-1] = oracle_front_recall(
                prefix, oracle
            )
    return {k: (v[0] if squeeze else v) for k, v in out.items()}


def summarize(problem, histories, n_init=6, oracle=None):
    """Print the median final value of every metric, best method first."""
    print(f"\n{problem.name}: d={problem.dim}, m={problem.n_obj}", end="")
    if oracle is not None:
        print(f", H* = {oracle['hypervolume']:.6g}", end="")
    print(f"\n{'method':26s}{'H':>12s}{'regret':>10s}"
          f"{'IGD+':>10s}{'recall':>9s}")
    print("-"*67)
    rows = []
    for key, metric in histories.items():
        rows.append((
            float(np.median(np.atleast_2d(metric["hypervolume_regret"])[:, -1])),
            key, metric,
        ))
    for regret, key, metric in sorted(rows):
        final = {name: float(np.median(np.atleast_2d(metric[name])[:, -1]))
                 for name in NAMES}
        print(f"{key[0]+' / '+key[1]:26s}{final['hypervolume']:12.6g}"
              f"{100*regret:9.2f}%{final['igd_plus']:10.4f}"
              f"{100*final['oracle_front_recall']:8.1f}%")


def plot(problem, histories, n_init=6, path=None, relative_regret=True):
    """Draw the four metrics as a 2x2 panel, one curve per method.

    Median across repeats, with the 10th-90th percentile band. Regret and IGD+ use a log ordinate: both converge to zero, which is where the differences between methods live.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, name in zip(axes.ravel(), NAMES):
        for key, metric in histories.items():
            values = np.atleast_2d(metric[name])[:, n_init-1:]
            x = np.arange(n_init, n_init+values.shape[1])
            median = np.median(values, axis=0)
            low, high = np.percentile(values, [10, 90], axis=0)
            ax.plot(x, median, label=" / ".join(key))
            ax.fill_between(x, low, high, alpha=.12)
        label = LABELS[name]
        if name == "hypervolume_regret" and not relative_regret:
            label = ABSOLUTE_REGRET_LABEL
        ax.set(xlabel="Number of evaluations", ylabel=label)
        if name in LOG_SCALE:
            ax.set_yscale("log")
        ax.grid(alpha=.25)
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.suptitle(f"{problem.name}: $d={problem.dim}$, $m={problem.n_obj}$")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return fig, axes
