"""EHVI candidate-strategy ablation across GPmp and BoTorch: "LHS, fixed
grid" vs "Sobol, resampled" vs "Gradient (BFGS)", scored by hypervolume and IGD+ (front coverage). Restricted to problem.n_obj in (2, 3), the range of GPmp's analytic explicit_ehvi. No Optuna: TPE has no analytic EHVI. Not to be confused with grid_ablation.py, which ablates the grid's *size*.

Usage: python3 candidate_ablation.py [n_repeats] [n_iter] [n_init] [problem] problem: any key of problem.PROBLEMS_BY_NAME with n_obj in (2, 3) (e.g. zdt1), default zdt1

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import qmc

import gpmp_bmoo
import botorch_bmoo
import metrics
import posterior_plot
from problem import PROBLEMS_BY_NAME
from utils import fixed_lhs_grid, initial_indices, reference_point, worker_pool

N_CANDIDATES = 4096  # power of 2: exact for scrambled Sobol' candidate sets, same as compare.py's default grid size
PROTOCOLS = ["LHS, fixed grid", "Sobol, resampled", "Gradient (BFGS)"]
_PROTOCOL_SLUG = {"LHS, fixed grid": "lhs_fixed", "Sobol, resampled": "sobol_resampled", "Gradient (BFGS)": "gradient_bfgs"}
LIBRARY_COLORS = {"GPmp": "tab:red", "BoTorch": "tab:blue"}

# metrics.NAMES minus oracle_front_recall: that one needs exact grid indices (is *this* grid point evaluated?), which has no meaning for the continuous Sobol/gradient protocols -- see the module docstring.
_METRIC_NAMES = ("hypervolume", "hypervolume_regret", "igd_plus")
_METRICS_TO_PLOT = ("hypervolume", "igd_plus")  # regret is a rescaling of hypervolume, redundant on one figure
_METRIC_LABELS = {"hypervolume": "HV", "igd_plus": "IGD+ (front coverage)"}
_METRIC_LOG_SCALE = {"igd_plus"}  # converges to 0, where the differences between protocols live


def _metric_history(Y, oracle):
    """Per-evaluation hypervolume, relative HV regret and IGD+ computed
    directly from the evaluated objective vectors Y -- see metrics.py. Mirrors metrics.evaluation_histories, but keyed off Y instead of grid indices so it works uniformly for grid-restricted and continuous candidates alike."""
    out = {name: np.empty(len(Y)) for name in _METRIC_NAMES}
    for n in range(1, len(Y) + 1):
        prefix = Y[:n]
        out["hypervolume"][n - 1] = metrics.attained_hypervolume(prefix, oracle)
        out["hypervolume_regret"][n - 1] = metrics.hypervolume_regret(prefix, oracle)
        out["igd_plus"][n - 1] = metrics.igd_plus(prefix, oracle)
    return out


# --- Protocol 1: "LHS, fixed grid" -- both backends already support this via their own optimizer="grid" (the default), nothing new needed. ---


def _gpmp_lhs_fixed(problem, grid, n_iter, ref, init, seed):
    return gpmp_bmoo.optimize(problem, grid, init, n_iter, "EHVI_explicit", seed, ref, optimizer="grid")


def _botorch_lhs_fixed(problem, grid, n_iter, ref, init, seed):
    return botorch_bmoo.optimize(problem, grid, init, n_iter, "EHVI", seed, ref, optimizer="grid")


# --- Protocol 2: "Sobol, resampled" ---


def _sobol_candidates(problem, seed, n_candidates):
    """Infinite generator of fresh unit-box candidate batches, drawn from
    one persistent qmc.Sobol(seed=seed) sequence that advances across calls -- the shared resampling mechanism handed to both libraries, same pattern as bo_comparison/grid_ablation.py's _sobol_candidates."""
    sobol = qmc.Sobol(d=problem.dim, scramble=True, seed=seed)
    while True:
        yield sobol.random(n_candidates)


def gpmp_sobol_resampled(problem, grid, n_iter, ref, init, seed, n_candidates=N_CANDIDATES):
    lo, hi = problem.bounds
    X = grid[np.asarray(init, dtype=int)]
    Y = problem(X)
    model = gpmp_bmoo._build_model(problem.n_obj)
    gen = _sobol_candidates(problem, seed, n_candidates)

    durations = []
    for _ in range(n_iter):
        start = time.perf_counter()
        candidates = lo + next(gen) * (hi - lo)
        mean, variance = gpmp_bmoo._fit_and_predict(model, X, Y, candidates)
        x_new = candidates[np.nanargmax(gpmp_bmoo.explicit_ehvi(mean, variance, Y, ref))][None, :]
        y_new = problem(x_new)
        X = np.vstack([X, x_new]); Y = np.vstack([Y, y_new])
        durations.append(time.perf_counter() - start)
    return {"X": X, "Y": Y, "times": np.asarray(durations), "ref_point": ref}


def botorch_sobol_resampled(problem, grid, n_iter, ref, init, seed, n_candidates=N_CANDIDATES):
    torch.manual_seed(seed)
    lo, hi = problem.bounds
    X = grid[np.asarray(init, dtype=int)]
    Y = problem(X)
    ref_max = torch.as_tensor(-ref)
    gen = _sobol_candidates(problem, seed, n_candidates)

    durations = []
    for _ in range(n_iter):
        start = time.perf_counter()
        Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
        model = botorch_bmoo._fit(Xt, -Yt)
        acqf = botorch_bmoo._ehvi_acqf(model, -Yt, ref_max)
        candidates = torch.as_tensor(lo + next(gen) * (hi - lo))
        with torch.no_grad():
            scores = acqf(candidates[:, None, :]).detach().numpy().reshape(-1)
        x_new = candidates[np.nanargmax(scores)].unsqueeze(0).numpy()
        y_new = problem(x_new)
        X = np.vstack([X, x_new]); Y = np.vstack([Y, y_new])
        durations.append(time.perf_counter() - start)
    return {"X": X, "Y": Y, "times": np.asarray(durations), "ref_point": ref}


# --- Protocol 3: "Gradient (BFGS)" -- both backends already support this via their own optimizer="gradient", nothing new needed here. ---


def _gpmp_gradient(problem, grid, n_iter, ref, init, seed):
    return gpmp_bmoo.optimize(problem, grid, init, n_iter, "EHVI_explicit", seed, ref, optimizer="gradient")


def _botorch_gradient(problem, grid, n_iter, ref, init, seed):
    return botorch_bmoo.optimize(problem, grid, init, n_iter, "EHVI", seed, ref, optimizer="gradient")


VARIANTS = {
    ("LHS, fixed grid", "GPmp"): _gpmp_lhs_fixed,
    ("LHS, fixed grid", "BoTorch"): _botorch_lhs_fixed,
    ("Sobol, resampled", "GPmp"): gpmp_sobol_resampled,
    ("Sobol, resampled", "BoTorch"): botorch_sobol_resampled,
    ("Gradient (BFGS)", "GPmp"): _gpmp_gradient,
    ("Gradient (BFGS)", "BoTorch"): _botorch_gradient,
}


def run_all(problem, n_repeats=10, n_init=6, n_iter=25, seed=0, n_candidates=N_CANDIDATES):
    """Every variant function shares the (problem, grid, n_iter, ref, init,
    seed) signature -- init and seed last, so a partial binding the first four args (as below) can be called by ProcessPoolExecutor.map with two positional iterables, exactly like compare.py's _run_repeat. Repeats run in a process pool sized to min(Slurm cpus-per-task, n_repeats) -- see utils.worker_pool -- one pool opened per problem, reused across every (protocol, library) pair.

    Returns (results, metadata). results[(protocol, lib)] is a dict of per-evaluation metric arrays, each shape (n_repeats, n_init+n_iter) -- see _METRIC_NAMES/_metric_history. metadata["fronts"][(protocol, lib)] holds the raw evaluated Y per repeat (shape (n_repeats, n_init+n_iter, n_obj)), for plot_pareto_fronts.
    """
    if problem.n_obj not in (2, 3):
        raise ValueError(
            "candidate_ablation is EHVI-only: requires problem.n_obj in (2, 3), see gpmp_bmoo.explicit_ehvi"
        )

    grid = fixed_lhs_grid(problem, n_candidates, seed)
    grid_values = problem(grid)
    ref = reference_point(grid_values)
    oracle = metrics.build_oracle(grid_values, ref)
    inits = [initial_indices(grid, n_init, seed + 10_000 + repeat) for repeat in range(n_repeats)]

    results, fronts = {}, {}
    with worker_pool(n_repeats) as pool:
        for (protocol, lib_name), run_fn in VARIANTS.items():
            t0 = time.time()
            task = partial(run_fn, problem, grid, n_iter, ref)
            outputs = list(pool.map(task, inits, range(seed, seed + n_repeats)))
            histories = [_metric_history(out["Y"], oracle) for out in outputs]
            results[(protocol, lib_name)] = {name: np.array([h[name] for h in histories]) for name in _METRIC_NAMES}
            fronts[(protocol, lib_name)] = np.array([out["Y"] for out in outputs])
            print(f"  {protocol:<18s} {lib_name:<8s} {time.time() - t0:.0f}s")
    return results, {
        "grid": grid, "grid_values": grid_values, "ref_point": ref,
        "initial_indices": np.array(inits), "fronts": fronts,
    }


def plot(problem, results, n_init, path=None):
    """One column per protocol, one row per metric in _METRICS_TO_PLOT,
    colored by library -- mirrors bo_comparison/grid_ablation.py's protocol-column layout, extended with a second row (see the module docstring for why HV alone isn't enough). Drops the partial initial design like compare.py.plot / metrics.plot: history starts at n_init evaluations, not at the first individual observation."""
    n_repeats = next(iter(results.values()))["hypervolume"].shape[0]
    fig, axes = plt.subplots(
        len(_METRICS_TO_PLOT), len(PROTOCOLS),
        figsize=(4.6 * len(PROTOCOLS), 3.6 * len(_METRICS_TO_PLOT)),
        sharex=True, squeeze=False,
    )
    for row, metric_name in enumerate(_METRICS_TO_PLOT):
        for col, protocol in enumerate(PROTOCOLS):
            ax = axes[row, col]
            for lib_name, color in LIBRARY_COLORS.items():
                key = (protocol, lib_name)
                if key not in results:
                    continue
                history = results[key][metric_name][:, n_init - 1:]
                x = np.arange(n_init, n_init + history.shape[1])
                med = np.median(history, axis=0)
                lo, hi = np.percentile(history, [10, 90], axis=0)
                ax.plot(x, med, label=lib_name, color=color, linewidth=2.0)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            if metric_name in _METRIC_LOG_SCALE:
                ax.set_yscale("log")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(protocol, fontsize=11)
            if row == len(_METRICS_TO_PLOT) - 1:
                ax.set_xlabel("number of evaluations")
            if col == 0:
                ax.set_ylabel(_METRIC_LABELS[metric_name])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{problem.name}, EHVI, candidate-strategy ablation -- median over {n_repeats} repeats")
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_pareto_fronts(problem, metadata, repeat=0, path=None):
    """Attained front per (protocol, library), one repeat, against the
    problem's true continuous Pareto front -- the direct visual read of the failure mode HV/IGD+ only summarize numerically: points clustering in one corner of objective space instead of spreading across the front.

    Needs metadata["fronts"] (run_all's raw per-repeat Y, index-free) -- unlike compare.plot_pareto_fronts, which needs grid indices and so cannot be reused for the continuous Sobol/gradient protocols.
    """
    fronts = {" / ".join(key): Y[repeat] for key, Y in metadata["fronts"].items()}
    return posterior_plot.plot_pareto_comparison(problem, fronts, path=path)


if __name__ == "__main__":
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_init = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    problem_name = sys.argv[4] if len(sys.argv) > 4 else "zdt1"
    if problem_name not in PROBLEMS_BY_NAME:
        raise SystemExit(f"problem must be one of {sorted(PROBLEMS_BY_NAME)}, got '{problem_name}'")
    problem = PROBLEMS_BY_NAME[problem_name]

    results, metadata = run_all(problem, n_repeats, n_init, n_iter)
    directory = Path("results/candidate_ablation")
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{problem.name.lower()}_{problem.n_obj}obj_candidate_ablation"
    np.savez_compressed(
        directory / f"{stem}_histories.npz",
        **{f"{_PROTOCOL_SLUG[p]}__{l}__{m}": v for (p, l), hist in results.items() for m, v in hist.items()},
        **{f"{_PROTOCOL_SLUG[p]}__{l}__Y": v for (p, l), v in metadata["fronts"].items()},
        grid=metadata["grid"], grid_values=metadata["grid_values"],
        ref_point=metadata["ref_point"], initial_indices=metadata["initial_indices"],
    )
    plot(problem, results, n_init, path=directory / f"{stem}.png")
    try:
        plot_pareto_fronts(problem, metadata, path=directory / f"{stem}_pareto.png")
    except NotImplementedError:
        pass  # no known Pareto front for this problem in pymoo
