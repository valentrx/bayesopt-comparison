"""Test functions for the BO library comparison.

Each Problem bundles a numpy function with its bounds and known optima; fn_torch calls fn through numpy so every backend optimizes the same function. BRANIN/TWOBUMPS are used only by posterior_plot.py (2D/1D).

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import gpmp as gp


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    bounds: np.ndarray  # (2, dim): lower, upper
    fn: Callable[[np.ndarray], np.ndarray]  # (n, dim) -> (n,)
    global_minimum: float
    global_maximum: float

    def fn_torch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) -> (...,)."""
        batch_shape = x.shape[:-1]
        y = self.fn(x.reshape(-1, self.dim).detach().cpu().numpy())
        return torch.from_numpy(y).to(dtype=x.dtype, device=x.device).reshape(batch_shape)

    def random_initial_design(self, n, seed):
        rng = np.random.default_rng(seed)
        lo, hi = self.bounds[0], self.bounds[1]
        return lo + rng.random((n, self.dim)) * (hi - lo)

    def global_optimum(self, direction):
        return self.global_maximum if direction == "max" else self.global_minimum


def _branin(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    a, r, s = 1.0, 6.0, 10.0
    b, c, t = 5.1 / (4 * np.pi**2), 5.0 / np.pi, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


# Branin-Hoo, d=2, https://www.sfu.ca/~ssurjano/branin.html. Maximum sits exactly at the box corner (-5, 0): evaluate _branin() there rather than hardcode a rounded literal.
_branin_argmax = np.array([[-5.0, 0.0]])

BRANIN = Problem(
    name="Branin",
    dim=2,
    bounds=np.array([[-5.0, 0.0], [10.0, 15.0]]),
    fn=_branin,
    global_minimum=0.397887,
    global_maximum=float(_branin(_branin_argmax)[0]),
)

TWOBUMPS = Problem(
    name="TwoBumps",
    dim=1,
    bounds=np.array([[-1.0], [1.0]]),
    fn=gp.misc.testfunctions.twobumps,
    global_minimum=-1.1889453,  # dense grid scan + Brent polish, x* ~= 0.14652
    global_maximum=1.3058651,  # dense grid scan + L-BFGS-B polish, x* ~= -0.57837
)

# Hartmann4/6 and Ishigami: gp.misc.testfunctions.{hartmann4,hartmann6,ishigami}. global_minimum/global_maximum: differential evolution (12 restarts) + L-BFGS-B polish; matches the published Hartmann6 optimum (-3.32237 at z=(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573), see https://www.sfu.ca/~ssurjano/hart6.html). Hartmann4/6 maxima sit exactly at a box corner: evaluate the formula there rather than hardcode a rounded literal.
_h4_argmax = np.array([[1.0, 1.0, 0.0, 1.0]])
_h6_argmax = np.array([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]])

HARTMANN4 = Problem(
    name="Hartmann4",
    dim=4,
    bounds=np.array([[0.0] * 4, [1.0] * 4]),
    fn=gp.misc.testfunctions.hartmann4,
    global_minimum=-3.1344941412224,
    global_maximum=float(gp.misc.testfunctions.hartmann4(_h4_argmax)[0]),
)

HARTMANN6 = Problem(
    name="Hartmann6",
    dim=6,
    bounds=np.array([[0.0] * 6, [1.0] * 6]),
    fn=gp.misc.testfunctions.hartmann6,
    global_minimum=-3.3223680114155147,
    global_maximum=float(gp.misc.testfunctions.hartmann6(_h6_argmax)[0]),
)

ISHIGAMI = Problem(
    name="Ishigami",
    dim=3,
    bounds=np.array([[-np.pi] * 3, [np.pi] * 3]),
    fn=gp.misc.testfunctions.ishigami,  # a=5, b=0.1
    global_minimum=-1.0 - 0.1 * np.pi**4,
    global_maximum=6.0 + 0.1 * np.pi**4,  # 1 + a + 0.1*pi**4
)
