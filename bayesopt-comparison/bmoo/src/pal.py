"""Pareto Active Learning (PAL-style, Zuluaga, Krause, Sergent & Puy, ICML
2013): classify fixed-grid candidates into confidently Pareto-optimal / confidently dominated / undecided from independent-GP confidence hyperrectangles, and query the most uncertain undecided candidate each round. Shared, backend-agnostic state: gpmp_bmoo.py and botorch_bmoo.py each fit their own GP and hand this module (mean, std) at the candidates it asks for.

This is a *documented simplification* of the original paper's bookkeeping, not a claimed byte-for-byte reproduction -- check against the source paper before citing exact fidelity in a writeup:

- Discard rule matches the paper's definition directly: an active candidate x (status P, U or E -- evaluated) is moved to "dominated" (R) once some other active candidate y epsilon-dominates it, i.e. y's best case is, with an epsilon margin, still at least as good as x's worst case in every objective. Confidence regions only ever shrink (cumulative intersection across rounds), so a discard decision is never revisited.
- Accept rule is a deliberate simplification. The paper's own pairwise promotion test was not reproduced from memory with confidence; the "must dominate every other undecided point" rule that a literal symmetric reading of "discard" would suggest is actively wrong -- it would reject every point whenever two candidates are mutually non-dominated, which happens for any real Pareto front with more than one point. Instead, a surviving candidate graduates to the accepted Pareto set once its own confidence-region diagonal has shrunk below `epsilon`: further sampling could not change its classification by more than the same epsilon that already governs discarding, so it is a self-consistent (if not paper-identical) stopping rule.
- Confidence width defaults to a fixed `kappa=2.0` multiplier on `std` (ParetoActiveLearning's `kappa` parameter) rather than the paper's theoretical, delta-controlled `sqrt(beta_t)` schedule (confidence_width below) -- a further, deliberate deviation from the paper, traded for a width that stays flat across rounds instead of (very slowly) widening; pass `kappa=None` to use the theoretical schedule instead.

All formulas use a minimization convention: lower is better.
"""
import numpy as np


def confidence_width(t, n_candidates, n_obj, delta=0.1):
    """sqrt(beta_t): GP-UCB confidence multiplier (Srinivas et al. 2010),
    union-bounded over every objective and every grid candidate so that |mu_t(x) - f(x)| <= sqrt(beta_t) sigma_t(x) holds simultaneously for all of them with probability >= 1 - delta, for every round t >= 1.
    """
    beta_t = 2.0*np.log(n_obj*max(n_candidates, 1)*(np.pi**2)*(t**2)/(6.0*delta))
    return float(np.sqrt(max(beta_t, 0.0)))


class ParetoActiveLearning:
    """PAL-style classification state for one fixed candidate grid. n_candidates/n_obj size the beta_t schedule (confidence_width, unused when `kappa` is set); epsilon is the accuracy target in objective units, candidates accept once their confidence-region diagonal drops below it; delta is confidence_width's failure-probability budget, ignored when `kappa` is set. kappa (default 2.0) is a fixed confidence-width multiplier -- the hyperrectangle half-width at every round is `kappa * std`, a constant, instead of the theoretical delta-controlled `sqrt(beta_t)` schedule, a common GP-UCB simplification that trades away the schedule's `1 - delta` coverage guarantee; pass `kappa=None` to use the theoretical schedule instead."""

    def __init__(self, n_candidates, n_obj, epsilon=1e-3, delta=0.1, kappa=2.0):
        self.n_obj = n_obj
        self.epsilon = epsilon
        self.delta = delta
        self.kappa = kappa
        self.n_candidates = n_candidates
        self.lower = np.full((n_candidates, n_obj), -np.inf)
        self.upper = np.full((n_candidates, n_obj), np.inf)
        # U: undecided: P: accepted Pareto-optimal; R: discarded (dominated); E: evaluated (exact lower=upper, retired from querying but still usable to discard/accept its neighbours).
        self.status = np.full(n_candidates, "U", dtype="<U1")
        self.t = 0

    def mark_evaluated(self, index, y):
        """Record an exact observation: retires `index` from querying while
        keeping it in the active pool with a zero-width confidence region."""
        y = np.asarray(y, dtype=float).reshape(-1)
        self.lower[index] = y
        self.upper[index] = y
        self.status[index] = "E"

    def step(self, mean, std, indices):
        """Refine the confidence regions of `indices` (currently "U") with a
        fresh posterior, reclassify them, and return the grid index of the undecided candidate to sample next, or None if every candidate at `indices` is now decided.

        mean, std : arrays, shape (len(indices), n_obj) Posterior mean/std at grid[indices] from the current GP fit.indices : array of int Grid indices these predictions correspond to; every one of them must currently have status "U".
        """
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return None
        if np.any(self.status[indices] != "U"):
            raise ValueError("step() expects indices currently marked \"U\"")

        self.t += 1
        w = self.kappa if self.kappa is not None \
            else confidence_width(self.t, self.n_candidates, self.n_obj, self.delta)
        self.lower[indices] = np.maximum(self.lower[indices], mean - w*std)
        self.upper[indices] = np.minimum(self.upper[indices], mean + w*std)

        for idx in indices:
            if self.status[idx] == "U" and self._epsilon_dominated(idx):
                self.status[idx] = "R"

        undecided = indices[self.status[indices] == "U"]
        if len(undecided):
            diagonal = np.linalg.norm(self.upper[undecided] - self.lower[undecided], axis=1)
            self.status[undecided[diagonal < self.epsilon]] = "P"

        undecided = indices[self.status[indices] == "U"]
        if len(undecided) == 0:
            return None
        diagonal = np.linalg.norm(self.upper[undecided] - self.lower[undecided], axis=1)
        return int(undecided[np.argmax(diagonal)])

    def _epsilon_dominated(self, index):
        """True if some other active (non-discarded) candidate y
        epsilon-dominates `index`: y's best case is, with an epsilon margin, still at least as good as index's worst case in every objective."""
        active = np.flatnonzero(self.status != "R")
        others = active[active != index]
        if len(others) == 0:
            return False
        dominates = np.all(self.lower[others] + self.epsilon <= self.upper[index], axis=1)
        return bool(np.any(dominates))

    def pareto_indices(self):
        """Grid indices currently classified as confidently Pareto-optimal."""
        return np.flatnonzero(self.status == "P")
