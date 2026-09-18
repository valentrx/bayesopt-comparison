"""Ablation of the unique fixed-grid size using the GPmp BMOO backend.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
import gpmp_bmoo
from utils import fixed_lhs_grid, initial_indices, reference_point, hypervolume_history


def run(problem, grid_sizes=(256, 512, 1024, 2048, 4096), n_repeats=10,
        n_init=6, n_iter=25, acquisition="MC_sMC", seed=0):
    """Create one grid per (problem, grid size), not one grid per repeat."""
    histories, times, grids, initial_sets = {}, {}, {}, {}
    for size in grid_sizes:
        grid = fixed_lhs_grid(problem, size, seed)
        ref = reference_point(problem(grid))
        h = np.empty((n_repeats, n_init+n_iter)); t = np.empty((n_repeats,n_iter))
        starts = np.empty((n_repeats,n_init), dtype=int)
        for repeat in range(n_repeats):
            init = initial_indices(grid,n_init,seed+10_000+repeat); starts[repeat]=init
            out = gpmp_bmoo.optimize(problem,grid,init,n_iter,acquisition,seed+repeat,ref)
            h[repeat]=hypervolume_history(out["Y"],ref); t[repeat]=out["times"]
        histories[size]=h; times[size]=t; grids[size]=grid; initial_sets[size]=starts
    return histories,times,{"grids":grids,"initial_indices":initial_sets}


def plot(problem, histories, n_init=6, path=None):
    fig,ax=plt.subplots(figsize=(8,5))
    for size,h in histories.items():
        # One hypervolume per evaluation: drop the partial initial design.
        history=h[:,n_init-1:]; x=np.arange(n_init,n_init+history.shape[1])
        med=np.median(history,0); low,high=np.percentile(history,[10,90],axis=0)
        ax.plot(x,med,label=rf"$|\mathcal{{G}}|={size}$")
        ax.fill_between(x,low,high,alpha=.12)
    ax.set(xlabel="Number of evaluations", ylabel="HV",
           title=f"Grid-size ablation (GPmp): {problem.name}, $m={problem.n_obj}$")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=180)
    return fig,ax
