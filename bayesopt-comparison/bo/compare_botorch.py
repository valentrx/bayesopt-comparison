"""BoTorch only: compare 6 acquisitions -- EI, MaxVar, WeightedVar, UCB
(same as compare.py) plus Knowledge Gradient and Max-value Entropy Search, neither available in gpmp-contrib/Optuna. Ishigami only: KG's nested fantasy-model optimization runs ~135x EI's cost.

Usage: python3 compare_botorch.py [n_repeats] [n_iter] [n_init] [min|max]

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

from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.optim import optimize_acqf

import botorch_bo
import utils
import compare
from problem import ISHIGAMI

PROBLEMS = {"Ishigami": ISHIGAMI}
ACQUISITIONS = ("EI", "MaxVar", "WeightedVar", "UCB", "KG", "MES")
COLORS = {
    "EI": "tab:blue",
    "MaxVar": "tab:orange",
    "WeightedVar": "tab:green",
    "UCB": "tab:red",
    "KG": "tab:purple",
    "MES": "tab:brown",
}
_KG_NUM_FANTASIES = 64  # qKnowledgeGradient default
_MES_CANDIDATE_SET_SIZE = 1000  # BoTorch tutorial convention for Gumbel-sampling the max-value distribution


_BOTORCH_BO_ACQUISITIONS = ("EI", "MaxVar", "WeightedVar", "UCB")


def _acquisition(name, model, train_x, train_y, bounds):
    if name in _BOTORCH_BO_ACQUISITIONS:
        return botorch_bo._acquisition(name, model, train_x, train_y, bounds)
    if name == "KG":
        return qKnowledgeGradient(model, num_fantasies=_KG_NUM_FANTASIES)
    if name == "MES":
        lo, hi = bounds
        candidate_set = lo + torch.rand(_MES_CANDIDATE_SET_SIZE, lo.shape[0]) * (hi - lo)
        return qLowerBoundMaxValueEntropy(model, candidate_set=candidate_set)
    raise ValueError(f"unknown acquisition '{name}'")


def run(acquisition, problem, seed, n_init, n_iter, direction="min"):
    """Mirrors botorch_bo.run, extended with KG/MES. Best-observed value
    (original f-scale) after each of n_iter iterations."""
    torch.manual_seed(seed)
    sign = botorch_bo._sign(direction)
    bounds = torch.tensor(problem.bounds)
    train_x, train_y = botorch_bo._initial_data(problem, seed, n_init, sign)

    best = [sign * train_y.max().item()]
    for _ in range(n_iter):
        model = botorch_bo._fit_model(problem, train_x, train_y)
        acqf = _acquisition(acquisition, model, train_x, train_y, bounds)
        candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, **botorch_bo._ACQF_OPTIM)
        y_new = sign * problem.fn_torch(candidate).unsqueeze(-1)
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, y_new])
        best.append(sign * train_y.max().item())
    return np.array(best)


def run_all(n_repeats, n_init, n_iter, direction):
    """results[(problem_name, acquisition)] has shape (n_repeats, n_iter + 1).

    Repeats run in a process pool sized to min(Slurm cpus-per-task, n_repeats) -- see utils.worker_pool. This is where it matters most: KG alone runs ~200-230s/repeat (see module docstring), so n_repeats=100 serial is ~5.7h; spread over the job's cpus-per-task it drops proportionally."""
    results = {}
    with utils.worker_pool(n_repeats) as pool:
        for problem_name, problem in PROBLEMS.items():
            for acquisition in ACQUISITIONS:
                t0 = time.time()
                task = partial(run, acquisition, problem, n_init=n_init, n_iter=n_iter, direction=direction)
                histories = list(pool.map(task, range(n_repeats)))
                results[(problem_name, acquisition)] = np.array(histories)
                print(f"  {problem_name:<10s} {acquisition:<12s} {time.time() - t0:.0f}s")
    return results


def plot(results, n_init, n_iter, direction, path):
    n_evals = n_init + np.arange(n_iter + 1)
    n_repeats = next(iter(results.values())).shape[0]

    panel_width = 4.6 * len(PROBLEMS) if len(PROBLEMS) > 1 else 6.5  # a single panel needs more room for the legend
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(panel_width, 4.4), squeeze=False)
    for ax, (problem_name, problem) in zip(axes[0], PROBLEMS.items()):
        optimum = problem.global_optimum(direction)
        for acquisition, color in COLORS.items():
            regret = utils.regret_of(results[(problem_name, acquisition)], optimum, direction)
            median, lower, upper = utils.regret_band(regret)
            ax.plot(n_evals, np.maximum(median, compare.DISPLAY_FLOOR), label=acquisition, color=color, linewidth=2.0)
            ax.fill_between(
                n_evals,
                np.maximum(lower, compare.DISPLAY_FLOOR),
                np.maximum(upper, compare.DISPLAY_FLOOR),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("number of function evaluations")
        ax.set_title(problem_name, fontsize=11)
        ax.set_facecolor(compare.PANEL_BG)
        ax.grid(True, axis="y", which="major", color="white", linewidth=1.1, alpha=0.9)
        ax.grid(True, axis="y", which="minor", color="white", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    label = r"f(x^\star) - f(x^+)" if direction == "max" else r"f(x^+) - f(x^\star)"
    axes[0][0].set_ylabel(f"median simple regret ${label}$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"Ishigami, BoTorch, 6-acquisition comparison -- median over {n_repeats} repeats, shaded = [10th, 90th] pct.")
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_init = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    direction = sys.argv[4] if len(sys.argv) > 4 else "min"
    if direction not in ("min", "max"):
        raise SystemExit(f"direction must be 'min' or 'max', got '{direction}'")

    os.makedirs("results/compare", exist_ok=True)
    results = run_all(n_repeats, n_init, n_iter, direction)
    np.savez(
        f"results/compare/ishigami_botorch_6acq_histories_{direction}.npz",
        **{f"{p}__{a}": v for (p, a), v in results.items()},
    )
    plot(results, n_init, n_iter, direction, f"results/compare/ishigami_botorch_6acq_comparison_{direction}.png")
