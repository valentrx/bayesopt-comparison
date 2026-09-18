"""Fixed-grid benchmark for GPmp and BoTorch BMOO methods: EHVI and PAL.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
from pathlib import Path
from functools import partial
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
import gpmp_bmoo, botorch_bmoo, metrics, posterior_plot
from problem import ZDT, DTLZ, WFG, PROBLEMS_BY_NAME
from utils import fixed_lhs_grid, initial_indices, hypervolume_history, reference_point, worker_pool

# Both backends optimize by argmax (EHVI) or PAL's own query rule over the same fixed grid (optimizer="grid" is botorch_bmoo's default). Optuna/TPE and BoTorch's ParEGO/MaxVar were dropped from this comparison: the point now is EHVI-formula vs. EHVI-Monte-Carlo vs. PAL, not a broader acquisition sweep -- optuna_bmoo.py and the ParEGO/MaxVar acquisitions it replaced still exist for anyone who wants that broader comparison back.
BASE_ALGORITHMS = {
    ("GPmp", "MC_sMC"): gpmp_bmoo.optimize,
    ("GPmp", "PAL"): gpmp_bmoo.optimize,
    ("GPmp", "W_IMSE"): gpmp_bmoo.optimize,
    ("BoTorch", "EHVI"): botorch_bmoo.optimize,
    ("BoTorch", "PAL"): botorch_bmoo.optimize,
}

def algorithms_for(problem):
    methods = dict(BASE_ALGORITHMS)
    if problem.n_obj in (2, 3):
        methods[("GPmp", "EHVI_explicit")] = gpmp_bmoo.optimize
    return methods

def _run_repeat(optimizer,problem,grid,n_iter,acq,ref,init,seed):
    return optimizer(problem,grid,init,n_iter,acq,seed,ref)

def run(problem,n_repeats=10,n_iter=25,n_init=6,n_candidates=4096,seed=0,algorithms=None):
    """Repeats run in a process pool sized to min(Slurm cpus-per-task,
    n_repeats) -- see utils.worker_pool -- since each repeat is an independent seed reading only the one shared grid built below. One pool is opened for the whole problem and reused across every (library, acquisition) pair, so worker processes are spawned once, not once per algorithm."""
    algorithms=algorithms_for(problem) if algorithms is None else algorithms
    grid=fixed_lhs_grid(problem,n_candidates,seed); grid_values=problem(grid)
    ref=reference_point(grid_values)
    # The evaluated grid indices are what make any metric recomputable after the fact: the grid is deterministic and saved, so the objective vectors of a run are exactly grid_values[selected]. See metrics.py.
    starts=np.stack([initial_indices(grid,n_init,seed+10_000+repeat) for repeat in range(n_repeats)])
    results={}; times={}; selected={}
    with worker_pool(n_repeats) as pool:
        for (lib,acq),optimizer in algorithms.items():
            task=partial(_run_repeat,optimizer,problem,grid,n_iter,acq,ref)
            outputs=list(pool.map(task,starts,range(seed,seed+n_repeats)))
            results[(lib,acq)]=np.array([hypervolume_history(out["Y"],ref) for out in outputs])
            times[(lib,acq)]=np.array([out["times"] for out in outputs])
            selected[(lib,acq)]=np.array([out["indices"] for out in outputs])
    return results,times,{"grid":grid,"grid_values":grid_values,"initial_indices":starts,
                          "ref_point":ref,"selected":selected}

def plot(problem,results,n_init=6,n_iter=None,path=None):
    fig,ax=plt.subplots(figsize=(9,5.5))
    for key,v in results.items():
        # v carries one hypervolume per evaluation, starting at the first one. Keep the value attained by the initial design plus one point per BO iteration, so the curve spans n_init..n_init+n_iter evaluations.
        stop=v.shape[1] if n_iter is None else n_init+n_iter
        history=v[:,n_init-1:stop]; x=np.arange(n_init,n_init+history.shape[1])
        med=np.median(history,0); lo,hi=np.percentile(history,[10,90],axis=0)
        ax.plot(x,med,label=" / ".join(key)); ax.fill_between(x,lo,hi,alpha=.12)
    ax.set(xlabel="Number of evaluations", ylabel="HV",
           title=f"{problem.name}: $d={problem.dim}$, $m={problem.n_obj}$")
    ax.grid(alpha=.25); ax.legend(ncol=2); fig.tight_layout()
    if path: Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=180)
    return fig,ax

def plot_pareto_fronts(problem,metadata,repeat=0,path=None):
    """Attained front per algorithm (one repeat) against the problem's true
    continuous Pareto front -- see posterior_plot.plot_pareto_comparison. Needs metadata["selected"] (compare.run's grid indices per algorithm).
    """
    selected=metadata["selected"]
    grid_values=metadata["grid_values"]
    fronts={" / ".join(key):grid_values[indices[repeat]] for key,indices in selected.items()}
    return posterior_plot.plot_pareto_comparison(problem,fronts,path=path)

def summarize(problem,results,times):
    print(f"\n{problem.name}: d={problem.dim}, m={problem.n_obj}")
    for key,v in results.items():
        final=v[:,-1]; elapsed=times[key].sum(1)
        print(f"{key[0]:8s} {key[1]:14s} | HV {np.median(final):.6g} [{np.percentile(final,10):.6g}, {np.percentile(final,90):.6g}] | time {np.median(elapsed):.3f}s")

def save(problem,results,times,metadata,directory="results"):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); stem=f"{problem.name.lower()}_{problem.n_obj}obj"
    metadata=dict(metadata); selected=metadata.pop("selected",{})
    arrays={f"result__{a}__{b}":v for (a,b),v in results.items()}; arrays.update({f"time__{a}__{b}":v for (a,b),v in times.items()})
    # selected is keyed by algorithm, so it is flattened the same way as results and times instead of being handed to savez as a dict.
    arrays.update({f"selected__{a}__{b}":v for (a,b),v in selected.items()})
    arrays.update(metadata)
    np.savez_compressed(directory/f"{stem}_histories.npz",**arrays)
    n_init=int(np.asarray(metadata["initial_indices"]).shape[1])
    if selected:
        # The metrics panel already carries HV on one of its four axes, so the older single-metric figure would only duplicate it.
        oracle=metrics.build_oracle(metadata["grid_values"],metadata["ref_point"])
        histories={k:metrics.evaluation_histories(i,oracle) for k,i in selected.items()}
        fig,_=metrics.plot(problem,histories,n_init,path=directory/f"{stem}_metrics.png")
        plt.close(fig)
        try:
            pf_fig,_=plot_pareto_fronts(problem,{**metadata,"selected":selected},
                                        path=directory/f"{stem}_pareto.png")
            plt.close(pf_fig)
        except NotImplementedError:
            pass  # no known Pareto front for this problem in pymoo
    else:
        fig,_=plot(problem,results,n_init,path=directory/f"{stem}_comparison.png")
        plt.close(fig)

def load(path):
    data=np.load(path); results={}; times={}; metadata={k:data[k] for k in ("grid","grid_values","initial_indices","ref_point")}
    # Always present, empty for archives written before indices were stored.
    metadata["selected"]={}
    for key in data.files:
        p=key.split("__")
        if p[0]=="result": results[(p[1],p[2])]=data[key]
        elif p[0]=="time": times[(p[1],p[2])]=data[key]
        elif p[0]=="selected": metadata["selected"][(p[1],p[2])]=data[key]
    return results,times,metadata

if __name__=="__main__":
    nr=int(sys.argv[1]) if len(sys.argv)>1 else 10; ni=int(sys.argv[2]) if len(sys.argv)>2 else 25; n0=int(sys.argv[3]) if len(sys.argv)>3 else 6; m=int(sys.argv[4]) if len(sys.argv)>4 else 2
    # families: which problems to run -- "zdt" (5 problems, m forced to 2, m argument ignored), "dtlz_wfg" (16 problems at the given m), "wfg" (just the 9 WFG problems at the given m, e.g. for a quick test), "all" (zdt + dtlz_wfg, the default), or a comma-separated list of specific PROBLEMS_BY_NAME keys (e.g. "dtlz2,dtlz7,wfg9") to run only those. Family shortcuts split this way -- not by looping every m value in one job -- because ZDT doesn't depend on m at all: running m=2..5 in one job would otherwise recompute the same 5 ZDT problems 4 times.
    families=sys.argv[5] if len(sys.argv)>5 else "all"
    if families in ("all","zdt","dtlz_wfg","wfg"):
        bases=[]
        if families in ("all","zdt"): bases+=[(base,2) for base in ZDT]
        if families in ("all","dtlz_wfg"): bases+=[(base,m) for base in DTLZ+WFG]
        if families=="wfg": bases+=[(base,m) for base in WFG]
    else:
        # Comma-separated explicit problem names (PROBLEMS_BY_NAME keys), e.g. "dtlz2,dtlz7,wfg9" -- run only those, at the given m (ignored for zdt* names, which are always 2, same as the family shortcuts above).
        names=families.split(",")
        unknown=[n for n in names if n not in PROBLEMS_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown problem(s) {unknown}; families must be 'all', 'zdt', "
                              f"'dtlz_wfg', 'wfg', or a comma-separated list of {sorted(PROBLEMS_BY_NAME)}")
        bases=[(PROBLEMS_BY_NAME[n], 2 if n.startswith("zdt") else m) for n in names]
    for base,n_obj in bases:
        problem=base.with_n_obj(n_obj); r,t,meta=run(problem,nr,ni,n0); save(problem,r,t,meta); summarize(problem,r,t)
