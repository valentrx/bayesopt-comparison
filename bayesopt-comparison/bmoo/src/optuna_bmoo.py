"""Optuna multi-objective TPE restricted to the common fixed grid."""
import time
import numpy as np
import optuna
from utils import nondominated_mask, reference_point

def optimize(problem,grid,initial_indices,n_iter=25,acquisition="TPE",seed=0,ref_point=None):
    if acquisition.lower()!="tpe": raise ValueError("Optuna acquisition must be TPE")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    selected=list(map(int,initial_indices)); Y=problem(grid[selected])
    ref=reference_point(Y) if ref_point is None else np.asarray(ref_point); durations=[]
    dist=optuna.distributions.IntDistribution(0,len(grid)-1)
    sampler=optuna.samplers.TPESampler(seed=seed,multivariate=True,group=True)
    study=optuna.create_study(directions=["minimize"]*problem.n_obj,sampler=sampler)
    for idx,y in zip(selected,Y):
        study.add_trial(optuna.trial.create_trial(params={"candidate":idx},distributions={"candidate":dist},values=y.tolist()))
    for _ in range(n_iter):
        start=time.perf_counter(); selected_set=set(selected)
        while True:
            trial=study.ask({"candidate":dist}); idx=int(trial.params["candidate"])
            if idx not in selected_set: break
            study.tell(trial,Y[selected.index(idx)].tolist())
        y=problem(grid[idx:idx+1])[0]; study.tell(trial,y.tolist())
        selected.append(idx); Y=np.vstack((Y,y)); durations.append(time.perf_counter()-start)
    return {"indices":np.asarray(selected),"X":grid[selected],"Y":Y,"times":np.asarray(durations),"ref_point":ref,"pareto_mask":nondominated_mask(Y)}
