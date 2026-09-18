"""EI candidate-strategy ablation across gpmp, BoTorch and Optuna: "LHS,
fixed grid" vs "Sobol, resampled" vs "Gradient (BFGS)", isolating the candidate optimizer from the GP model compare.py compares. run_all/plot also take estimator="MAP" -- see grid_ablation_map.py, which reuses them as-is.

Usage: python3 grid_ablation.py [n_repeats] [n_iter] [n_init] [problem] problem in {Hartmann4, Hartmann6, Ishigami}, default Ishigami

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
from scipy.stats import qmc

import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc
from botorch.acquisition.analytic import LogExpectedImprovement

import gpmp_bo
import botorch_bo
import optuna_bo
import utils
import compare
from problem import HARTMANN4, HARTMANN6, ISHIGAMI

PROBLEMS = {"Hartmann4": HARTMANN4, "Hartmann6": HARTMANN6, "Ishigami": ISHIGAMI}
N_GRID = utils.N_CANDIDATES
_GRID_SEED = 0  # fixed, single draw for "LHS, fixed grid" -- independent of the per-repeat seed j


# --- Protocol 1: "LHS, fixed grid" ---


def _lhs_grid_unit(problem):
    """One maximin-LHS candidate set in [0, 1]^dim, drawn once for the
    whole ablation (not per repeat -- confirmed dense enough), seeded independently of the per-repeat seed j."""
    gnp.set_seed(_GRID_SEED)
    unit_box = [[0.0] * problem.dim, [1.0] * problem.dim]
    return gnp.to_np(gp.misc.designs.maximinlhs(problem.dim, N_GRID, unit_box))


# --- Protocol 2: "Sobol, resampled" ---


def _sobol_candidates(problem, seed):
    """Infinite generator of fresh unit-box candidate batches, drawn from
    one persistent qmc.Sobol(seed=seed) sequence that advances across calls -- the shared resampling mechanism handed to all 3 libraries."""
    sobol = qmc.Sobol(d=problem.dim, scramble=True, seed=seed)
    while True:
        yield sobol.random(utils.N_CANDIDATES)


def _gpmp_resample_step(algo, xt):
    """Swap in a freshly-drawn xt and refresh zpm/zpv there before
    stepping, so both the criterion argmax and the resulting self.xt[best_idx] refer to the new candidate set -- step() alone would use zpm/zpv left over from the previous xt."""
    algo.xt = xt
    algo.nt = xt.shape[0]
    try:
        algo.update_predictions()
    except RuntimeError:
        pass  # keep stale zpm/zpv for this one iteration -- same fallback
              # philosophy as gpmp_bo._safe_step
    gpmp_bo._safe_step(algo)


def gpmp_sobol_resampled(problem, seed, n_init, n_iter, direction, estimator="ML"):
    sign = -1 if direction == "max" else 1
    computer_experiment = gpc.ComputerExperiment(
        problem.dim, problem.bounds, single_objective=lambda x: sign * problem.fn(x)
    )
    lo, hi = problem.bounds
    gen = _sobol_candidates(problem, seed)

    model = gpmp_bo._model(problem, estimator)
    algo = gpmp_bo.BOGridSearch(computer_experiment, model, lo + next(gen) * (hi - lo), "EI", problem.bounds)
    algo.set_initial_design(problem.random_initial_design(n_init, seed))
    algo.sign = sign

    best = [algo.sign * gnp.to_scalar(gnp.min(algo.zi))]
    for _ in range(n_iter):
        _gpmp_resample_step(algo, lo + next(gen) * (hi - lo))
        best.append(algo.sign * gnp.to_scalar(gnp.min(algo.zi)))
    return np.array(best)


def botorch_sobol_resampled(problem, seed, n_init, n_iter, direction, estimator="ML"):
    torch.manual_seed(seed)
    sign = botorch_bo._sign(direction)
    lo, hi = problem.bounds
    gen = _sobol_candidates(problem, seed)

    train_x, train_y = botorch_bo._initial_data(problem, seed, n_init, sign)
    best = [sign * train_y.max().item()]
    for _ in range(n_iter):
        model = botorch_bo._fit_model(problem, train_x, train_y, estimator=estimator)
        ei = LogExpectedImprovement(model, best_f=train_y.max())
        candidates = torch.from_numpy(lo + next(gen) * (hi - lo))
        with torch.no_grad():
            values = ei(candidates.unsqueeze(-2))
        x_new = candidates[values.argmax()].unsqueeze(0)
        y_new = sign * problem.fn_torch(x_new).unsqueeze(-1)
        train_x = torch.cat([train_x, x_new])
        train_y = torch.cat([train_y, y_new])
        best.append(sign * train_y.max().item())
    return np.array(best)


def optuna_sobol_resampled(problem, seed, n_init, n_iter, direction, estimator="ML"):
    sign = 1 if direction == "max" else -1
    lo, hi = problem.bounds
    x_init_raw = problem.random_initial_design(n_init, seed)
    X = (x_init_raw - lo) / (hi - lo)
    Y = sign * problem.fn(x_init_raw)
    is_categorical = np.zeros(problem.dim, dtype=bool)
    search_space = optuna_bo._search_space(problem.dim)
    gen = _sobol_candidates(problem, seed)
    rng = np.random.RandomState(seed)

    best = [Y.max()]
    for _ in range(n_iter):
        x_next = optuna_bo._next_point("EI", X, Y, search_space, is_categorical, next(gen), rng, estimator=estimator)
        x_next_raw = lo + x_next * (hi - lo)
        y_next = sign * problem.fn(x_next_raw[None, :])[0]
        X = np.vstack([X, x_next])
        Y = np.append(Y, y_next)
        best.append(Y.max())
    return sign * np.array(best)


# --- Protocol 3: "Gradient (BFGS)" ---


def gpmp_gradient(problem, seed, n_init, n_iter, direction, estimator="ML"):
    return gpmp_bo.run("EI", problem, seed, n_init, n_iter, direction, estimator=estimator, optimizer="gradient")  # continuous L-BFGS-B multistart via autodiff through gpmp's own predict, see gpmp_bo._run_gradient


def botorch_gradient(problem, seed, n_init, n_iter, direction, estimator="ML"):
    return botorch_bo.run("EI", problem, seed, n_init, n_iter, direction, estimator=estimator)  # candidates=None -> optimize_acqf, L-BFGS-B multistart


def optuna_gradient(problem, seed, n_init, n_iter, direction, estimator="ML"):
    return optuna_bo.run("EI", problem, seed, n_init, n_iter, direction, estimator=estimator)  # candidates=None -> optim_mixed.optimize_acqf_mixed (bug-fixed path)


PROTOCOLS = ["LHS, fixed grid", "Sobol, resampled", "Gradient (BFGS)"]
_PROTOCOL_SLUG = {"LHS, fixed grid": "lhs_fixed", "Sobol, resampled": "sobol_resampled", "Gradient (BFGS)": "gradient_bfgs"}
LIBRARY_COLORS = {"gpmp-contrib": "tab:red", "BoTorch": "tab:blue", "Optuna": "tab:green"}


def _gpmp_lhs_fixed_task(grid_raw, problem, seed, n_init, n_iter, direction, estimator="ML"):
    return gpmp_bo.run("EI", problem, seed, n_init, n_iter, direction, xt=grid_raw, estimator=estimator)


def _botorch_lhs_fixed_task(grid_torch, problem, seed, n_init, n_iter, direction, estimator="ML"):
    return botorch_bo.run("EI", problem, seed, n_init, n_iter, direction, candidates=grid_torch, estimator=estimator)


def _optuna_lhs_fixed_task(grid_unit, problem, seed, n_init, n_iter, direction, estimator="ML"):
    return optuna_bo.run("EI", problem, seed, n_init, n_iter, direction, candidates=grid_unit, estimator=estimator)


def run_all(problem, n_repeats, n_init, n_iter, direction, estimator="ML"):
    """estimator: "ML" (default, see gpmp_bo/botorch_bo/optuna_bo) or "MAP"
    (each library's own native MAP prior, see grid_ablation_map.py).

    Repeats run in a process pool sized to min(Slurm cpus-per-task, n_repeats) -- see utils.worker_pool -- since each repeat is an independent seed reading only the one shared grid built below."""
    lo, hi = problem.bounds
    grid_unit = _lhs_grid_unit(problem)
    grid_raw = lo + grid_unit * (hi - lo)
    grid_torch = torch.from_numpy(grid_raw)

    variants = {
        ("LHS, fixed grid", "gpmp-contrib"): partial(_gpmp_lhs_fixed_task, grid_raw),
        ("LHS, fixed grid", "BoTorch"): partial(_botorch_lhs_fixed_task, grid_torch),
        ("LHS, fixed grid", "Optuna"): partial(_optuna_lhs_fixed_task, grid_unit),
        ("Sobol, resampled", "gpmp-contrib"): gpmp_sobol_resampled,
        ("Sobol, resampled", "BoTorch"): botorch_sobol_resampled,
        ("Sobol, resampled", "Optuna"): optuna_sobol_resampled,
        ("Gradient (BFGS)", "gpmp-contrib"): gpmp_gradient,
        ("Gradient (BFGS)", "BoTorch"): botorch_gradient,
        ("Gradient (BFGS)", "Optuna"): optuna_gradient,
    }

    results = {}
    with utils.worker_pool(n_repeats) as pool:
        for (protocol, lib_name), run_fn in variants.items():
            t0 = time.time()
            task = partial(run_fn, problem, n_init=n_init, n_iter=n_iter, direction=direction, estimator=estimator)
            histories = list(pool.map(task, range(n_repeats)))
            results[(protocol, lib_name)] = np.array(histories)
            print(f"  {protocol:<18s} {lib_name:<13s} {time.time() - t0:.0f}s")
    return results


def plot(problem, results, n_init, n_iter, direction, path, estimator="ML", ylim=None):
    """ylim: optional (ymin, ymax) applied to the shared log-scale axis
    (panels already share y via sharey=True), e.g. to make figures comparable across problems/scripts."""
    n_evals = n_init + np.arange(n_iter + 1)
    n_repeats = next(iter(results.values())).shape[0]
    optimum = problem.global_optimum(direction)

    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(4.6 * len(PROTOCOLS), 4.4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        for lib_name, color in LIBRARY_COLORS.items():
            key = (protocol, lib_name)
            if key not in results:
                continue
            regret = utils.regret_of(results[key], optimum, direction)
            median, lower, upper = utils.regret_band(regret)
            ax.plot(n_evals, np.maximum(median, compare.DISPLAY_FLOOR), label=lib_name, color=color, linewidth=2.0)
            ax.fill_between(
                n_evals,
                np.maximum(lower, compare.DISPLAY_FLOOR),
                np.maximum(upper, compare.DISPLAY_FLOOR),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("number of function evaluations")
        ax.set_title(protocol, fontsize=11)
        ax.set_facecolor(compare.PANEL_BG)
        ax.grid(True, axis="y", which="major", color="white", linewidth=1.1, alpha=0.9)
        ax.grid(True, axis="y", which="minor", color="white", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    label = r"f(x^\star) - f(x^+)" if direction == "max" else r"f(x^+) - f(x^\star)"
    axes[0].set_ylabel(f"median simple regret ${label}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    est_suffix = "" if estimator == "ML" else f", {estimator} fit"
    fig.suptitle(f"{problem.name}, EI, candidate-strategy ablation{est_suffix} -- median over {n_repeats} repeats")
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_init = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    problem_name = sys.argv[4] if len(sys.argv) > 4 else "Ishigami"
    if problem_name not in PROBLEMS:
        raise SystemExit(f"problem must be one of {list(PROBLEMS)}, got '{problem_name}'")
    problem = PROBLEMS[problem_name]
    direction = "min"

    os.makedirs("results/grid/ml", exist_ok=True)
    results = run_all(problem, n_repeats, n_init, n_iter, direction)
    np.savez(
        f"results/grid/ml/{problem.name.lower()}_grid_ablation_histories.npz",
        **{f"{_PROTOCOL_SLUG[p]}__{l}": v for (p, l), v in results.items()},
    )
    plot(problem, results, n_init, n_iter, direction, f"results/grid/ml/{problem.name.lower()}_grid_ablation.png")
