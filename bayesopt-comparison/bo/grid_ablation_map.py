"""Same candidate-strategy ablation as grid_ablation.py, each library fit
with its own native MAP prior instead of pure MLE -- does that regularize away the near-singular-covariance failures MLE hits? Reuses grid_ablation.py's run_all/plot as-is; results land in results/grid/map/, next to results/grid/ml/.

Usage: python3 grid_ablation_map.py [n_repeats] [n_iter] [n_init] [problem] problem in {Hartmann4, Hartmann6, Ishigami}, default Ishigami

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import os
import sys

import numpy as np

import grid_ablation as ga

if __name__ == "__main__":
    n_repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_init = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    problem_name = sys.argv[4] if len(sys.argv) > 4 else "Ishigami"
    if problem_name not in ga.PROBLEMS:
        raise SystemExit(f"problem must be one of {list(ga.PROBLEMS)}, got '{problem_name}'")
    problem = ga.PROBLEMS[problem_name]
    direction = "min"

    os.makedirs("results/grid/map", exist_ok=True)
    results = ga.run_all(problem, n_repeats, n_init, n_iter, direction, estimator="MAP")
    np.savez(
        f"results/grid/map/{problem.name.lower()}_grid_ablation_map_histories.npz",
        **{f"{ga._PROTOCOL_SLUG[p]}__{l}": v for (p, l), v in results.items()},
    )
    ga.plot(
        problem, results, n_init, n_iter, direction,
        f"results/grid/map/{problem.name.lower()}_grid_ablation_map.png",
        estimator="MAP",
    )
