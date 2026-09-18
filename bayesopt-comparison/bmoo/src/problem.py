"""Multi-objective benchmark problems used by the fixed-grid BMOO code: the
real ZDT, DTLZ and WFG test suites (via pymoo), not objectives synthesized from a single scalar function.

An earlier version built "conflicting objectives" by rolling, reflecting and rescaling the input of one scalar test function (Branin, Hartmann4/6, Ishigami) before evaluating it. That construction was dropped: roll has period `dim` and reflect has period 2, so objectives past the lcm(dim, 2)-th reused an already-taken (roll, reflect) pair and relied on a sub-box rescaling alone to look distinct -- which it mostly didn't (measured pairwise correlation up to 0.93 between two of Hartmann4's five "objectives"). ZDT/DTLZ/WFG are the literature suites this benchmark is actually expected to run on, with genuinely independent objectives and known Pareto-front geometry.

DTLZ and WFG are scalable in the number of objectives; the input dimension is grown with n_obj to keep a fixed number of "distance" parameters (DTLZ's k, WFG's l), following each family's own convention -- but with smaller k/l than the canonical papers (which target dim up in the tens) so dimensions stay in the same range as the rest of this benchmark (Hartmann6 tops out at dim=6). ZDT is only defined for 2 objectives; ZDT5 (an 80-bit binary-coded problem) is excluded, since it isn't a continuous box and doesn't fit the LHS/continuous-GP protocol used here.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
from dataclasses import dataclass, field
from functools import partial
from typing import Callable
import numpy as np
from pymoo.problems import get_problem

Array = np.ndarray


def _as_2d(X: Array, dim: int) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != dim:
        raise ValueError(f"X must have shape (n, {dim}), got {X.shape}")
    return X


@dataclass(frozen=True)
class MultiObjectiveProblem:
    """Wraps one pymoo test-problem instance behind the interface the rest of
    bmoo_comparison expects (.dim, .bounds, .n_obj, __call__).

    _factory: n_obj -> a freshly built pymoo Problem. Called once here and again by with_n_obj, so it must not depend on external mutable state.
    """
    name: str
    n_obj: int
    _factory: Callable[[int], object] = field(repr=False)
    dim: int = field(init=False)
    bounds: Array = field(init=False, repr=False)

    def __post_init__(self):
        problem = self._factory(self.n_obj)
        if problem.n_obj != self.n_obj:
            raise ValueError(
                f"{self.name}: pymoo built n_obj={problem.n_obj}, expected {self.n_obj}"
            )
        object.__setattr__(self, "_pymoo", problem)
        object.__setattr__(self, "dim", int(problem.n_var))
        object.__setattr__(
            self, "bounds", np.stack([problem.xl, problem.xu]).astype(float)
        )

    def with_n_obj(self, n_obj: int) -> "MultiObjectiveProblem":
        """Return the same problem family rebuilt for a different objective count."""
        return MultiObjectiveProblem(self.name, int(n_obj), self._factory)

    def evaluate(self, X: Array) -> Array:
        """Evaluate all objectives; returns an array of shape (n, n_obj)."""
        X = _as_2d(X, self.dim)
        if np.any((X < self.bounds[0] - 1e-9) | (X > self.bounds[1] + 1e-9)):
            raise ValueError("At least one point lies outside the problem bounds")
        Y = np.asarray(self._pymoo.evaluate(X), dtype=float)
        if Y.shape != (len(X), self.n_obj):
            raise ValueError(f"pymoo returned shape {Y.shape}, expected {(len(X), self.n_obj)}")
        return Y

    __call__ = evaluate

    def evaluate_objective(self, X: Array, objective: int) -> Array:
        if not 0 <= objective < self.n_obj:
            raise IndexError("objective index out of range")
        return self.evaluate(X)[:, objective]

    def true_pareto_front(self) -> Array:
        """The problem's known continuous Pareto front (objective space),
        exact for ZDT/most DTLZ, a dense sample for DTLZ7 and WFG -- see pymoo.problems.Problem.pareto_front. Independent of the grid: this is ground truth, not the grid-restricted oracle metrics.py scores runs against."""
        front = self._pymoo.pareto_front()
        if front is None:
            raise NotImplementedError(f"{self.name} has no known Pareto front in pymoo")
        return np.asarray(front, dtype=float)


Problem = MultiObjectiveProblem


# Factories are module-level functions bound with functools.partial, not closures returned by a builder function: a closure (a function object defined inside another function) cannot be pickled, and MultiObjectiveProblem instances must be -- they cross a process boundary every time compare.py parallelizes repeats across worker processes (see utils.worker_pool). partial(func, ...) of a module-level func pickles by reference, correctly.

def _zdt_build(n_obj, name, n_var):
    """ZDT is not scalable in n_obj: always 2, per the original definition."""
    if n_obj != 2:
        raise ValueError(f"{name.upper()} is only defined for n_obj=2")
    return get_problem(name, n_var=n_var)


# DTLZ's "distance" parameter k: n_var = n_obj + k - 1. Canonical papers use k=5 (DTLZ1), k=10 (DTLZ2-6) or k=20 (DTLZ7); scaled down here to keep dim comparable to the rest of this benchmark (dim <= 6 for n_obj <= 5).
_DTLZ_K = {"dtlz1": 2, "dtlz2": 3, "dtlz3": 3, "dtlz4": 3, "dtlz5": 3, "dtlz6": 3, "dtlz7": 5}


def _dtlz_build(n_obj, name):
    k = _DTLZ_K[name]
    return get_problem(name, n_var=n_obj + k - 1, n_obj=n_obj)


def _wfg_build(n_obj, name, l=4):
    """WFG's position parameter k must be >= 4 and a multiple of (n_obj-1);
    l is the distance parameter, n_var = k + l."""
    k = max(4, 2 * (n_obj - 1))
    return get_problem(name, n_var=k + l, n_obj=n_obj, k=k)


ZDT1 = MultiObjectiveProblem("ZDT1", 2, partial(_zdt_build, name="zdt1", n_var=6))
ZDT2 = MultiObjectiveProblem("ZDT2", 2, partial(_zdt_build, name="zdt2", n_var=6))
ZDT3 = MultiObjectiveProblem("ZDT3", 2, partial(_zdt_build, name="zdt3", n_var=6))
ZDT4 = MultiObjectiveProblem("ZDT4", 2, partial(_zdt_build, name="zdt4", n_var=6))
ZDT6 = MultiObjectiveProblem("ZDT6", 2, partial(_zdt_build, name="zdt6", n_var=6))
# ZDT5 intentionally omitted: 80 binary variables, not a continuous box.

DTLZ1 = MultiObjectiveProblem("DTLZ1", 3, partial(_dtlz_build, name="dtlz1"))
DTLZ2 = MultiObjectiveProblem("DTLZ2", 3, partial(_dtlz_build, name="dtlz2"))
DTLZ3 = MultiObjectiveProblem("DTLZ3", 3, partial(_dtlz_build, name="dtlz3"))
DTLZ4 = MultiObjectiveProblem("DTLZ4", 3, partial(_dtlz_build, name="dtlz4"))
DTLZ5 = MultiObjectiveProblem("DTLZ5", 3, partial(_dtlz_build, name="dtlz5"))
DTLZ6 = MultiObjectiveProblem("DTLZ6", 3, partial(_dtlz_build, name="dtlz6"))
DTLZ7 = MultiObjectiveProblem("DTLZ7", 3, partial(_dtlz_build, name="dtlz7"))

WFG1 = MultiObjectiveProblem("WFG1", 3, partial(_wfg_build, name="wfg1"))
WFG2 = MultiObjectiveProblem("WFG2", 3, partial(_wfg_build, name="wfg2"))
WFG3 = MultiObjectiveProblem("WFG3", 3, partial(_wfg_build, name="wfg3"))
WFG4 = MultiObjectiveProblem("WFG4", 3, partial(_wfg_build, name="wfg4"))
WFG5 = MultiObjectiveProblem("WFG5", 3, partial(_wfg_build, name="wfg5"))
WFG6 = MultiObjectiveProblem("WFG6", 3, partial(_wfg_build, name="wfg6"))
WFG7 = MultiObjectiveProblem("WFG7", 3, partial(_wfg_build, name="wfg7"))
WFG8 = MultiObjectiveProblem("WFG8", 3, partial(_wfg_build, name="wfg8"))
WFG9 = MultiObjectiveProblem("WFG9", 3, partial(_wfg_build, name="wfg9"))

ZDT = (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)
DTLZ = (DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7)
WFG = (WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9)
PROBLEMS = ZDT + DTLZ + WFG
PROBLEMS_BY_NAME = {problem.name.lower(): problem for problem in PROBLEMS}
