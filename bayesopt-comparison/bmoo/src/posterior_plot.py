"""Plots for BMOO observations, Pareto fronts and independent GP posteriors.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import nondominated_mask


def plot_objective_space(problem, Y, path=None):
    """Plot evaluated objective vectors and highlight nondominated points."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != problem.n_obj:
        raise ValueError(f"Y must have shape (n, {problem.n_obj})")
    pareto = nondominated_mask(Y)

    if problem.n_obj == 2:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(Y[:, 0], Y[:, 1], s=24, alpha=0.45, label="Evaluations")
        front = Y[pareto][np.argsort(Y[pareto, 0])]
        ax.plot(front[:, 0], front[:, 1], "o-", linewidth=1.5,
                label="Non-dominated front")
        ax.set_xlabel(r"Objective $f_1$")
        ax.set_ylabel(r"Objective $f_2$")
        ax.legend()
        axes = ax
    else:
        m = problem.n_obj
        fig, axes = plt.subplots(m, m, figsize=(2.35*m, 2.35*m), squeeze=False)
        for row in range(m):
            for col in range(m):
                ax = axes[row, col]
                if row == col:
                    ax.hist(Y[:, row], bins=min(20, max(5, len(Y)//2)), alpha=0.75)
                else:
                    ax.scatter(Y[:, col], Y[:, row], s=10, alpha=0.25)
                    ax.scatter(Y[pareto, col], Y[pareto, row], s=18,
                               label="Non-dominated" if (row, col) == (0, 1) else None)
                if row == m-1:
                    ax.set_xlabel(rf"$f_{{{col+1}}}$")
                if col == 0:
                    ax.set_ylabel(rf"$f_{{{row+1}}}$")
        handles, labels = axes[0, 1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{problem.name}: $m={problem.n_obj}$ objectives, minimization")
    fig.tight_layout()
    _save(fig, path)
    return fig, axes


def plot_pareto_comparison(problem, fronts, true_front=None, path=None):
    """Compare each algorithm's attained non-dominated front against the problem's true continuous Pareto front, in objective space. fronts is a dict {label: Y} of evaluated objective vectors per algorithm (one repeat, or pooled across repeats) -- only each Y's non-dominated subset is plotted, so callers do not need to filter beforehand. true_front (shape (n, n_obj)) defaults to problem.true_pareto_front(); pass it explicitly to avoid recomputing it (DTLZ7/WFG samples can have thousands of rows) when plotting several repeats or problems in a loop."""
    if true_front is None:
        true_front = problem.true_pareto_front()
    true_front = np.asarray(true_front, dtype=float)
    m = problem.n_obj
    attained = {label: np.asarray(Y, dtype=float)[nondominated_mask(np.asarray(Y, dtype=float))]
                for label, Y in fronts.items()}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if m == 2:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        order = true_front[:, 0].argsort()
        ax.plot(true_front[order, 0], true_front[order, 1], "-", color="0.6",
                linewidth=2, label="True Pareto front", zorder=1)
        for (label, front), color in zip(attained.items(), colors):
            ax.scatter(front[:, 0], front[:, 1], s=28, color=color, label=label, zorder=2)
        ax.set_xlabel(r"Objective $f_1$"); ax.set_ylabel(r"Objective $f_2$")
        ax.legend()
        axes = ax
    elif m == 3:
        # A real 3D scatter reads far better than the m x m pairwise grid below once there are only 3 axes to place -- that grid is kept for m >= 4, where no single 3D view could show all objectives anyway.
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2],
                   s=4, color="0.75", alpha=0.35, depthshade=False, label="True Pareto front")
        for (label, front), color in zip(attained.items(), colors):
            ax.scatter(front[:, 0], front[:, 1], front[:, 2],
                       s=40, color=color, depthshade=False, label=label)
        ax.set_xlabel(r"Objective $f_1$"); ax.set_ylabel(r"Objective $f_2$"); ax.set_zlabel(r"Objective $f_3$")
        ax.view_init(elev=22, azim=-60)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        axes = ax
    else:
        fig, axes = plt.subplots(m, m, figsize=(2.35*m, 2.35*m), squeeze=False)
        for row in range(m):
            for col in range(m):
                ax = axes[row, col]
                if row == col:
                    ax.hist(true_front[:, row], bins=30, color="0.75", alpha=0.6)
                else:
                    ax.scatter(true_front[:, col], true_front[:, row], s=6, color="0.75",
                               alpha=0.5, label="True front" if (row, col) == (0, 1) else None)
                    for (label, front), color in zip(attained.items(), colors):
                        ax.scatter(front[:, col], front[:, row], s=18, color=color,
                                   label=label if (row, col) == (0, 1) else None)
                if row == m-1:
                    ax.set_xlabel(rf"$f_{{{col+1}}}$")
                if col == 0:
                    ax.set_ylabel(rf"$f_{{{row+1}}}$")
        handles, labels = axes[0, 1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"{problem.name}: attained vs. true Pareto front, $m={m}$")
    fig.tight_layout()
    _save(fig, path)
    return fig, axes


def plot_decision_space(problem, X, Y, path=None):
    """Plot evaluated locations for one- or two-dimensional input spaces."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    pareto = nondominated_mask(Y)
    if problem.dim == 1:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.scatter(X[:, 0], np.zeros(len(X)), c="0.65", label="Evaluations")
        ax.scatter(X[pareto, 0], np.zeros(pareto.sum()),
                   label="Non-dominated solutions")
        ax.set_yticks([]); ax.set_xlabel(r"$x_1$"); ax.legend()
    elif problem.dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X[:, 0], X[:, 1], c="0.65", s=24, label="Evaluations")
        ax.scatter(X[pareto, 0], X[pareto, 1], s=36,
                   label="Non-dominated solutions")
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.legend()
    else:
        raise ValueError("Decision-space plotting is available only for input dimension 1 or 2")
    ax.set_title(f"{problem.name}: evaluated points in decision space")
    fig.tight_layout(); _save(fig, path)
    return fig, ax


def plot_gpmp_posterior(problem, X, Y, grid, path=None):
    """Plot GPmp marginal posterior means and standard deviations.

    For input dimension 1, each objective is shown as a curve. For input dimension 2, one contour panel is produced per objective. Higher input dimensions are intentionally rejected because a faithful surface plot is not available without slicing.
    """
    import gpmp_bmoo  # reuse the same ML model (COVARIANCE_SPECIFICATION)

    X, Y, grid = map(lambda z: np.asarray(z, dtype=float), (X, Y, grid))
    if Y.shape != (len(X), problem.n_obj):
        raise ValueError("Y must contain one column per objective")

    model = gpmp_bmoo._build_model(problem.n_obj)
    mean, variance = gpmp_bmoo._fit_and_predict(model, X, Y, grid)
    posterior = [(mean[:, j], np.sqrt(variance[:, j])) for j in range(problem.n_obj)]

    if problem.dim == 1:
        order = np.argsort(grid[:, 0])
        fig, axes = plt.subplots(problem.n_obj, 1,
                                 figsize=(8, 3*problem.n_obj), squeeze=False)
        for objective, (mean, std) in enumerate(posterior):
            ax = axes[objective, 0]
            ax.plot(grid[order, 0], mean[order], label="Posterior mean")
            ax.fill_between(grid[order, 0], mean[order]-1.96*std[order],
                            mean[order]+1.96*std[order], alpha=0.2,
                            label=r"95 % credible interval")
            ax.scatter(X[:, 0], Y[:, objective], s=20, label="Observations")
            ax.set_ylabel(rf"$f_{{{objective+1}}}$"); ax.legend()
        axes[-1, 0].set_xlabel(r"$x_1$")
    elif problem.dim == 2:
        fig, axes = plt.subplots(1, problem.n_obj,
                                 figsize=(5*problem.n_obj, 4.5), squeeze=False)
        for objective, (mean, _) in enumerate(posterior):
            ax = axes[0, objective]
            contour = ax.tricontourf(grid[:, 0], grid[:, 1], mean, levels=25)
            ax.scatter(X[:, 0], X[:, 1], c="white", edgecolors="black", s=20)
            ax.set_title(rf"Posterior mean of $f_{{{objective+1}}}$")
            ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
            fig.colorbar(contour, ax=ax)
    else:
        raise ValueError("Posterior surfaces are available only for input dimension 1 or 2")

    fig.suptitle(f"GPmp posteriors: {problem.name}, $m={problem.n_obj}$ objectives")
    fig.tight_layout(); _save(fig, path)
    return fig, axes


def _save(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
