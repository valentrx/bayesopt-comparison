# BO library comparison: gpmp-contrib vs BoTorch vs Optuna

Compares three Bayesian-optimization libraries on the same 4 acquisition
functions and the same 3 test problems. `compare.py` standardizes candidate
search across all 3 backends (one shared, fixed maximin-LHS grid per
problem, drawn once from gpmp and reused by every repeat and every
library) so any gap between libraries isolates GP-model quality rather
than search machinery; `grid_ablation.py` is where the candidate-strategy
question itself (fixed grid vs. resampled Sobol' vs. each library's own
gradient/local-search optimizer) is studied directly.

GP hyperparameters are fit by plain maximum likelihood, pinned to the same
estimator across all 3 libraries: β (constant mean), σ² (variance) and ρ
(lengthscale, one per dimension via ARD) are fit this way; ν (Matérn
smoothness, `nu=2.5` everywhere) is fixed, not fit -- no library exposes a
gradient for it, and it isn't separately identifiable from ρ by MLE anyway.

## Layout

| path | what |
|---|---|
| `src/utils.py` | MaxVar / Pi-BO-style WeightedVar formulas, shared constants (`UCB_BETA`, `N_CANDIDATES`, ...), regret-band helpers; re-exports `worker_pool` from the top-level `../src/parallel.py`. EI/UCB have no shared formula here -- each backend uses its own library's native one where it exists |
| `src/problem.py` | Test problems: `HARTMANN4`, `HARTMANN6`, `ISHIGAMI` (the 3 used below), plus `BRANIN`/`TWOBUMPS` (2D/1D, used only by `posterior_plot.py`) |
| `src/gpmp_bo.py` | gpmp-contrib backend (`SequentialStrategyGridSearch` subclass) |
| `src/botorch_bo.py` | BoTorch backend (`AnalyticAcquisitionFunction` subclasses for MaxVar/WeightedVar) |
| `src/optuna_bo.py` | Optuna backend (drives `optuna._gp` directly, see its docstring) |
| `compare.py` | Runs all 3 libraries x 4 acquisitions on a problem over the one shared fixed candidate grid, plots median regret + [10th, 90th] pct. band |
| `compare_botorch.py` | BoTorch only, 6 acquisitions (the 4 above + Knowledge Gradient + Max-value Entropy Search, neither available natively in gpmp-contrib/Optuna), Ishigami only -- KG's nested fantasy-model optimization is ~135x EI's per-repeat cost, see its module docstring |
| `posterior_plot.py` | 2D/1D posterior snapshots (Branin/TwoBumps only — the 3 comparison problems are too high-dimensional to plot) |
| `grid_ablation.py` | EI on one problem (default Ishigami), 3 candidate-search protocols x 3 libraries ("LHS, fixed grid" / "Sobol, resampled" / "Gradient (BFGS)") instead of libraries alone -- isolates candidate strategy from GP-model quality |
| `grid_ablation_map.py` | Same 3 protocols x 3 libraries, MAP instead of MLE (each library's own native prior) -- reuses `grid_ablation.py`'s `run_all`/`plot` as-is, results land in `results/grid/map/` next to `grid_ablation.py`'s `results/grid/ml/` |
| `bo_comparison.ipynb` | Walkthrough notebook: math (LaTeX), what's reused from each library vs. new code, `grid_ablation` then `compare` (quick live demo) |

## Running it

```
pip install -r ../requirements.txt   # see that file for the gpmp/gpmp-contrib editable-install step

python3 compare.py [n_repeats] [n_iter] [n_init] [min|max]                 # defaults: 10 25 6 min
python3 compare_botorch.py [n_repeats] [n_iter] [n_init] [min|max]         # defaults: 10 25 6 min, Ishigami only
python3 posterior_plot.py {branin|twobumps} <n_evals> <seed> [min|max]
python3 grid_ablation.py [n_repeats] [n_iter] [n_init] [problem]           # defaults: 10 25 6 Ishigami; problem in {Hartmann4, Hartmann6, Ishigami}
python3 grid_ablation_map.py [n_repeats] [n_iter] [n_init] [problem]       # same defaults, MAP estimator
jupyter notebook bo_comparison.ipynb                                       # walkthrough, see above
```

All of `compare.py`, `compare_botorch.py` and `grid_ablation*.py` run their
`n_repeats` independent seeds in parallel via a process pool sized to
`min(SLURM_CPUS_PER_TASK, n_repeats)` (`src/utils.worker_pool`, from the
shared `../src/parallel.py`) -- processes rather than threads, since
`gnp.set_seed` mutates gpmp's global RNG state, which concurrent threads in
one process would race on; each worker is pinned to 1 BLAS/torch thread so
`n_workers` processes don't oversubscribe the allocation. Outside Slurm this
falls back to the local core count, so it runs the same way on a laptop.

Both `compare.py`/`compare_botorch.py` and `grid_ablation*.py` write into
`results/`: `compare.py` produces `<problem>_comparison_<direction>.png`
(the plot) and `<problem>_histories_<direction>.npz` (raw per-repeat,
per-(library, acquisition) histories, so the plot can be regenerated
without rerunning the optimization); `posterior_plot.py` produces
`posterior_<problem>_<direction>_n<n_evals>.png`.

## Acquisitions

`EI`, `MaxVar` (pure posterior-variance maximization), `WeightedVar`
(Pi-BO-style variance x Gaussian-weight-on-incumbent), `UCB` — the same
formula applied to each backend's own GP posterior, searched over the one
shared fixed candidate grid (see above). MaxVar/WeightedVar/the UCB constant
are shared code (`src/utils.py`); EI/UCB otherwise use each library's own
native acquisition class where one exists. `compare_botorch.py` adds
Knowledge Gradient and Max-value Entropy Search, BoTorch-only.
