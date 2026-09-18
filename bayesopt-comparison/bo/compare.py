"""Compare gpmp-contrib, BoTorch and Optuna (EI, MaxVar, WeightedVar, UCB)
on Hartmann4, Hartmann6 and Ishigami.

One figure per problem: 4 panels (one per acquisition), each showing the 3 libraries' median simple regret (log-scale) with a [10th, 90th] percentile band.

Usage: python3 compare.py [n_repeats] [n_iter] [n_init] [min|max]

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import os
import sys
import time
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import gpmp as gp
import gpmp.num as gnp

import gpmp_bo
import botorch_bo
import optuna_bo
import utils
from problem import HARTMANN4, HARTMANN6, ISHIGAMI

PROBLEMS = {"Hartmann4": HARTMANN4, "Hartmann6": HARTMANN6, "Ishigami": ISHIGAMI}
ACQUISITIONS = ("EI", "MaxVar", "WeightedVar", "UCB")
LIBRARIES = {
    "gpmp-contrib": gpmp_bo.run,
    "BoTorch": botorch_bo.run,
    "Optuna": optuna_bo.run,
}
COLORS = {"gpmp-contrib": "tab:red", "BoTorch": "tab:blue", "Optuna": "tab:green"}
PANEL_BG = "#f4f4ec"  # physics.ai-style pale panel
DISPLAY_FLOOR = 1e-3  # exact 0 regret can't be placed on a log axis; display-only


def _lhs_grid_unit(problem, seed):
    """The one maximin-LHS candidate set for this problem, in [0, 1]^dim.

    All 3 libraries are handed the *same* points -- scaled to problem.bounds for gpmp-contrib/BoTorch, used as-is (already normalized) for Optuna -- and every acquisition optimizes by argmax over this one fixed set for the whole run: no gradient refinement (BoTorch), no local search (Optuna's optim_mixed), no resampling (Optuna's MaxVar/WeightedVar previously redrew a fresh Sobol' set every iteration). Each library still evaluates its own acquisition-function formula -- only the candidate optimizer is standardized, so any gap between libraries isolates the GP model, not the search/candidates. The grid is drawn once per problem and shared across every repeat, unlike the repeat-specific initial design (Problem.random_initial_design)."""
    gnp.set_seed(seed)
    unit_box = [[0.0] * problem.dim, [1.0] * problem.dim]
    return gnp.to_np(gp.misc.designs.maximinlhs(problem.dim, utils.N_CANDIDATES, unit_box))


def _gpmp_task(grid_raw, acquisition, problem, seed, n_init, n_iter, direction):
    return gpmp_bo.run(acquisition, problem, seed, n_init, n_iter, direction, xt=grid_raw)


def _botorch_task(grid_torch, acquisition, problem, seed, n_init, n_iter, direction):
    return botorch_bo.run(acquisition, problem, seed, n_init, n_iter, direction, candidates=grid_torch)


def _optuna_task(grid_unit, acquisition, problem, seed, n_init, n_iter, direction):
    return optuna_bo.run(acquisition, problem, seed, n_init, n_iter, direction, candidates=grid_unit)


def run_all(problem, n_repeats, n_init, n_iter, direction):
    """results[(library, acquisition)] has shape (n_repeats, n_iter + 1).

    Repeats run in a process pool sized to min(Slurm cpus-per-task, n_repeats) -- see utils.worker_pool -- since each repeat is an independent seed reading only the one shared grid built below."""
    grid_unit = _lhs_grid_unit(problem, seed=0)
    lo, hi = problem.bounds
    grid_raw = lo + grid_unit * (hi - lo)
    grid_torch = torch.from_numpy(grid_raw)

    libraries = {
        "gpmp-contrib": partial(_gpmp_task, grid_raw),
        "BoTorch": partial(_botorch_task, grid_torch),
        "Optuna": partial(_optuna_task, grid_unit),
    }

    results = {}
    with utils.worker_pool(n_repeats) as pool:
        for acquisition in ACQUISITIONS:
            for lib_name, run_fn in libraries.items():
                t0 = time.time()
                task = partial(run_fn, acquisition, problem, n_init=n_init, n_iter=n_iter, direction=direction)
                histories = list(pool.map(task, range(n_repeats)))
                results[(lib_name, acquisition)] = np.array(histories)
                print(f"  {problem.name} {acquisition:<12s} {lib_name:<13s} {time.time() - t0:.0f}s")
    return results


def plot(problem, results, n_init, n_iter, direction, path, ylim=None):
    """ylim: optional (ymin, ymax) applied to every panel's log-scale axis,
    e.g. to make figures comparable across problems/scripts."""
    n_evals = n_init + np.arange(n_iter + 1)
    n_repeats = next(iter(results.values())).shape[0]
    optimum = problem.global_optimum(direction)

    fig, axes = plt.subplots(1, len(ACQUISITIONS), figsize=(4.4 * len(ACQUISITIONS), 4.2))
    for ax, acquisition in zip(axes, ACQUISITIONS):
        for lib_name, color in COLORS.items():
            regret = utils.regret_of(results[(lib_name, acquisition)], optimum, direction)
            median, lower, upper = utils.regret_band(regret)
            ax.plot(n_evals, np.maximum(median, DISPLAY_FLOOR), label=lib_name, color=color, linewidth=2.0)
            ax.fill_between(
                n_evals,
                np.maximum(lower, DISPLAY_FLOOR),
                np.maximum(upper, DISPLAY_FLOOR),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.set_yscale("log")  # autoscaled per panel by default: acquisitions land in very different ranges
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("number of function evaluations")
        ax.set_title(acquisition, fontsize=11)
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, axis="y", which="major", color="white", linewidth=1.1, alpha=0.9)
        ax.grid(True, axis="y", which="minor", color="white", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    label = r"f(x^\star) - f(x^+)" if direction == "max" else r"f(x^+) - f(x^\star)"
    axes[0].set_ylabel(f"median simple regret ${label}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(
        f"{problem.name}, d={problem.dim}, searching for the {direction} "
        f"-- median over {n_repeats} repeats, shaded = [10th, 90th] pct."
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


def summarize(problem, results, direction):
    optimum = problem.global_optimum(direction)
    print(f"\n{problem.name}: final regret (median, [10th, 90th] percentile)")
    for acquisition in ACQUISITIONS:
        print(f"  {acquisition}:")
        for lib_name in LIBRARIES:
            regret = utils.regret_of(results[(lib_name, acquisition)], optimum, direction)[:, -1]
            median, lower, upper = utils.regret_band(regret[:, None])
            print(f"    {lib_name:<13s} {median[0]:.4f}  [{lower[0]:.4f}, {upper[0]:.4f}]")


if __name__ == "__main__":
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_init = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    direction = sys.argv[4] if len(sys.argv) > 4 else "min"
    if direction not in ("min", "max"):
        raise SystemExit(f"direction must be 'min' or 'max', got '{direction}'")

    os.makedirs("results/compare", exist_ok=True)
    for problem in PROBLEMS.values():
        results = run_all(problem, n_repeats, n_init, n_iter, direction)
        np.savez(
            f"results/compare/{problem.name.lower()}_histories_{direction}.npz",
            **{f"{lib}__{acq}": v for (lib, acq), v in results.items()},
        )
        plot(problem, results, n_init, n_iter, direction, f"results/compare/{problem.name.lower()}_comparison_{direction}.png")
        summarize(problem, results, direction)
