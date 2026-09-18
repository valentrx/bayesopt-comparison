# Fixed-grid BMOO benchmark

Multi-objective comparison on the ZDT (2 objectives), DTLZ1-7 and WFG1-9
(2 to 5 objectives, via pymoo) suites, minimizing all objectives. GPmp
(`MC_sMC`, `EHVI_explicit`, `PAL`, `W_IMSE`) is compared against BoTorch
(`EHVI`, `PAL`), on the same fixed candidate grid.

GP hyperparameters are fit by plain maximum likelihood, identical to `bo/`:
β (constant mean), σ² (variance) and ρ (lengthscale, one per dimension) are
fit this way; ν (Matérn smoothness, `nu=2.5` everywhere) is fixed, not fit.

## Layout

| path | what |
|---|---|
| `src/utils.py` | Fixed-grid, Pareto and hypervolume utilities (`fixed_lhs_grid`, `initial_indices`, `nondominated_mask`, `hypervolume*`); re-exports `worker_pool` from the top-level `../src/parallel.py` |
| `src/problem.py` | `ZDT`, `DTLZ`, `WFG` problem families, wrapping `pymoo.problems.get_problem` |
| `src/metrics.py` | Hypervolume, IGD+, oracle front recall -- scored against the grid-restricted Pareto front |
| `src/pal.py` | Pareto Active Learning classification state, shared by both backends -- see "PAL" below |
| `src/gpmp_bmoo.py` | gpmp-contrib backend: exact EHVI, the Monte-Carlo `smc_ehvi` estimator, `W_IMSE`, and PAL glue -- see "SMC" below |
| `src/botorch_bmoo.py` | BoTorch backend: `ExpectedHypervolumeImprovement` and PAL glue |
| `src/optuna_bmoo.py` | Optuna TPE backend, kept importable but not in the default comparison (TPE has no posterior variance, which PAL needs) |
| `src/posterior_plot.py` | Plotting library: objective-space scatter, Pareto-front comparison, GPmp posterior slices -- used by `compare.py`/`candidate_ablation.py`, no CLI of its own |
| `compare.py` | Runs the benchmark: one fixed maximin-LHS grid per problem, `n_repeats` repeats varying only the initial points, plots hypervolume/IGD+ |
| `candidate_ablation.py` | EHVI only: "LHS, fixed grid" vs "Sobol, resampled" vs "Gradient (BFGS)" candidate strategies, scored by hypervolume and IGD+ |
| `grid_ablation.py` | Ablates the fixed grid's *size* (not the candidate strategy), GPmp only |
| `bmoo_comparison.ipynb` | Walkthrough notebook: full benchmark, grid-size ablation, candidate-strategy ablation |

## What SMC and PAL actually do

Two pieces of this benchmark are hand-built, not reused from a library --
worth knowing what they do before reading the rest of the code:

- **SMC (`gpmp_bmoo.smc_ehvi`)** -- a from-scratch sequential Monte Carlo
  estimator of the expected hypervolume improvement (Feliot, Bect & Vazquez),
  for any number of objectives (the closed-form `explicit_ehvi` only covers
  2-3). Particles start uniform on the non-dominated part of a bounded
  integration box, then get tempered towards the region where a candidate is
  likely to dominate, via multinomial resampling and a Metropolis-Hastings
  random walk. The final particle set gives one shared importance sample
  that scores every grid candidate at once. It is not `gpmp.mcmc.smc`
  (that one targets a different problem, excursion-set subset simulation) --
  this is a purpose-built sampler, validated numerically against
  `explicit_ehvi` (correlation > 0.998 on a real posterior).
- **PAL (`src/pal.py`)** -- Pareto Active Learning (Zuluaga, Krause, Sergent
  & Puy, ICML 2013): each grid candidate gets a shrinking confidence
  hyperrectangle from independent-GP predictions, and is classified as
  confidently Pareto-optimal, confidently dominated, or undecided; the most
  uncertain undecided candidate is queried each round. Shared, backend-agnostic
  state -- both `gpmp_bmoo.py` and `botorch_bmoo.py` fit their own GP and
  hand it `(mean, std)`. This is a documented simplification of the paper's
  bookkeeping (see the module docstring for the exact deviations), not a
  claimed exact reproduction.

## Running it

```bash
pip install -r ../requirements.txt
python3 compare.py [n_repeats] [n_iter] [n_init] [n_obj] [families]   # defaults: 10 25 6 2 all
python3 candidate_ablation.py [n_repeats] [n_iter] [n_init] [problem] # defaults: 10 25 6 zdt1
jupyter notebook bmoo_comparison.ipynb
```

`compare.py`'s `families` argument selects which problems to run: `zdt`
(5 problems, `n_obj` forced to 2), `dtlz_wfg` (16 problems at the given
`n_obj`), `wfg`, `all` (the default), or a comma-separated list of
`problem.PROBLEMS_BY_NAME` keys (e.g. `dtlz2,dtlz7,wfg9`).

Like `bo/`, repeats run in a process pool sized to
`min(SLURM_CPUS_PER_TASK, n_repeats)` (`src/utils.worker_pool`), falling
back to the local core count outside Slurm.
