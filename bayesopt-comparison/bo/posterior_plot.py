"""GP posterior snapshot after n evaluations, one column per method (2D/1D
problems only -- Branin/TwoBumps). Band shown is mean +/- BAND_BETA * std, an exact GP credible interval (BAND_BETA = utils.UCB_BETA, so for the "BoTorch UCB" column it doubles as the actual UCB acquisition surface, up to the min/max sign flip).

Usage: python3 posterior_plot.py <branin|twobumps> n [seed] [min|max] n          total number of evaluations shown (integer, > n_init) seed       initial-design seed, default 0 direction  "min" (default) or "max": which extremum is being sought

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

import gpmp.num as gnp
import gpmp_bo
import botorch_bo
import utils
from problem import BRANIN, TWOBUMPS

GRID_N_1D = 400
GRID_N_2D = 80
N_INIT = 4

PROBLEMS = {"branin": BRANIN, "twobumps": TWOBUMPS}

BAND_BETA = utils.UCB_BETA

# Branin's known global optimizers (for the 2D plot only): three minimizers, one maximizer at the box corner -- see problem.py.
OPTIMIZERS_2D = {
    "min": np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]),
    "max": np.array([[-5.0, 0.0]]),
}

ACQUISITION_PARAMS = {
    "gpmp-contrib EI": f"N candidates = {utils.N_CANDIDATES}",
    "BoTorch EI": "logEI, best_f = current best (no free parameter)",
    "BoTorch WeightedVar": f"Gaussian weight, lengthscale = {utils.LENGTHSCALE_FRACTION}x box range",
    "BoTorch UCB": f"β = {utils.UCB_BETA}",
}


def _grid(problem, grid_n):
    if problem.dim == 1:
        x1 = np.linspace(problem.bounds[0, 0], problem.bounds[1, 0], grid_n)
        return (x1,), x1.reshape(-1, 1)
    x1 = np.linspace(problem.bounds[0, 0], problem.bounds[1, 0], grid_n)
    x2 = np.linspace(problem.bounds[0, 1], problem.bounds[1, 1], grid_n)
    X1, X2 = np.meshgrid(x1, x2)
    return (X1, X2), np.column_stack([X1.ravel(), X2.ravel()])


def _gpmp_posterior(problem, seed, n_iter, Xg, direction):
    algo = gpmp_bo.fit("EI", problem, seed, N_INIT, n_iter, direction)
    zpm, zpv = algo.predict(Xg, convert_out=False)
    mean = algo.sign * gnp.to_np(zpm).reshape(-1)
    std = gnp.to_np(gnp.sqrt(zpv)).reshape(-1)
    return mean, std, gnp.to_np(algo.xi), algo.sign * gnp.to_np(algo.zi).reshape(-1)


def _botorch_posterior(acquisition, problem, seed, n_iter, Xg, direction):
    model, train_x, train_y, sign = botorch_bo.fit(acquisition, problem, seed, N_INIT, n_iter, direction)
    with torch.no_grad():
        post = model.posterior(torch.from_numpy(Xg))
    mean = sign * post.mean.squeeze(-1).numpy()
    std = post.variance.clamp_min(0.0).sqrt().squeeze(-1).numpy()
    return mean, std, train_x.numpy(), sign * train_y.squeeze(-1).numpy()


def snapshots(problem, n, seed, grid_n, direction):
    """n: total number of evaluations (initial design + BO iterations)."""
    if n <= N_INIT:
        raise ValueError(f"n must be > {N_INIT} (size of the initial design)")
    n_iter = n - N_INIT
    _, Xg = _grid(problem, grid_n)

    return {
        "gpmp-contrib EI": _gpmp_posterior(problem, seed, n_iter, Xg, direction),
        "BoTorch EI": _botorch_posterior("EI", problem, seed, n_iter, Xg, direction),
        "BoTorch WeightedVar": _botorch_posterior("WeightedVar", problem, seed, n_iter, Xg, direction),
        "BoTorch UCB": _botorch_posterior("UCB", problem, seed, n_iter, Xg, direction),
    }


def plot_1d(problem, data, n, seed, direction, path):
    (x1,), Xg = _grid(problem, GRID_N_1D)
    names = list(data.keys())
    truth = problem.fn(Xg)
    ucb_edge = "upper" if direction == "max" else "lower"

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2), sharey=True)
    for col, name in enumerate(names):
        mean, std, xi, zi = data[name]
        ax = axes[col]
        ax.plot(x1, truth, "k--", linewidth=1, label="truth" if col == 0 else None)
        ax.plot(x1, mean, color="tab:blue", label="predictor mean" if col == 0 else None)
        ax.fill_between(
            x1,
            mean - BAND_BETA * std,
            mean + BAND_BETA * std,
            color="tab:blue",
            alpha=0.2,
            label=f"mean ± {BAND_BETA:.0f}·std" if col == 0 else None,
        )
        ax.scatter(
            xi[:-1, 0], zi[:-1], s=20, c="black", zorder=5,
            label="evaluated point" if col == 0 else None,
        )
        ax.scatter(
            xi[-1:, 0], zi[-1:], s=90, c="red", marker="*", edgecolors="black", zorder=6,
            label="most recent evaluation" if col == 0 else None,
        )
        ax.set_title(f"{name}\n{ACQUISITION_PARAMS[name]}", fontsize=9)
        ax.set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[names.index("BoTorch UCB")].annotate(
        f"{ucb_edge} edge = UCB acquisition\n(up to sign flip)",
        xy=(0.5, 0.02), xycoords="axes fraction", fontsize=7, ha="center", color="tab:blue",
    )
    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        f"{problem.name} posterior after n = {n} evaluations "
        f"(seed = {seed}, searching for the {direction})"
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


def plot_2d(problem, data, n, seed, direction, path):
    (X1, X2), Xg = _grid(problem, GRID_N_2D)
    names = list(data.keys())
    truth = problem.fn(Xg).reshape(X1.shape)

    if direction == "max":
        rows = [
            ("predictor mean", lambda mean, std: mean),
            (f"upper bound  mean + {BAND_BETA:.0f}·std", lambda mean, std: mean + BAND_BETA * std),
        ]
    else:
        rows = [
            ("predictor mean", lambda mean, std: mean),
            (f"lower bound  mean − {BAND_BETA:.0f}·std", lambda mean, std: mean - BAND_BETA * std),
        ]

    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 7.7), layout="constrained")
    for row, (row_label, quantity) in enumerate(rows):
        surfaces = {name: quantity(*data[name][:2]).reshape(X1.shape) for name in names}
        vmin = min(s.min() for s in surfaces.values())
        vmax = max(s.max() for s in surfaces.values())
        for col, name in enumerate(names):
            ax = axes[row, col]
            cf = ax.contourf(X1, X2, surfaces[name], levels=30, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.contour(X1, X2, truth, levels=12, colors="white", linewidths=0.4, alpha=0.5)
            xi = data[name][2]
            label = row == 0 and col == 0
            ax.scatter(
                xi[:-1, 0], xi[:-1, 1], s=14, c="black", edgecolors="white", linewidths=0.4,
                label="evaluated point" if label else None,
            )
            ax.scatter(
                xi[-1:, 0], xi[-1:, 1], s=55, c="red", marker="*", edgecolors="black", linewidths=0.4, zorder=5,
                label="most recent evaluation" if label else None,
            )
            optimizers = OPTIMIZERS_2D[direction]
            ax.scatter(
                optimizers[:, 0], optimizers[:, 1], s=60, facecolors="none", edgecolors="cyan", linewidths=1.2,
                label=f"true global {'maximizer' if direction == 'max' else 'minimizer'}" if label else None,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{name}\n{ACQUISITION_PARAMS[name]}", fontsize=8.5)
            if row == 1 and name == "BoTorch UCB":
                ax.set_xlabel("= UCB acquisition surface (up to sign flip)", fontsize=7)
        fig.colorbar(cf, ax=axes[row, :].tolist(), shrink=0.85, pad=0.015)
        axes[row, 0].set_ylabel(row_label, fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle(
        f"{problem.name} posterior after n = {n} evaluations "
        f"(seed = {seed}, searching for the {direction})"
    )
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    try:
        problem_name = sys.argv[1]
        n = int(sys.argv[2])
    except (IndexError, ValueError):
        raise SystemExit(
            "usage: python3 posterior_plot.py <branin|twobumps> n [seed] [min|max]  (n must be an integer)"
        )
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    direction = sys.argv[4] if len(sys.argv) > 4 else "min"
    if direction not in ("min", "max"):
        raise SystemExit(f"direction must be 'min' or 'max', got '{direction}'")

    problem = PROBLEMS.get(problem_name)
    if problem is None:
        raise SystemExit(f"unknown problem '{problem_name}', choose from {list(PROBLEMS)}")

    print("acquisition parameters:")
    for name, params in ACQUISITION_PARAMS.items():
        print(f"  {name:<18s} {params}")

    grid_n = GRID_N_1D if problem.dim == 1 else GRID_N_2D
    data = snapshots(problem, n, seed, grid_n, direction)
    os.makedirs("results", exist_ok=True)
    path = f"results/posterior_{problem_name}_{direction}_n{n}.png"
    if problem.dim == 1:
        plot_1d(problem, data, n, seed, direction, path)
    else:
        plot_2d(problem, data, n, seed, direction, path)
