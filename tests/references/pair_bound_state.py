#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""pair_bound_state.py -- exact bound state of two particles in 2-D with a
radial pair potential V(r).

Exact reference for paired systems: the energy and size of an isolated pair
(e.g. the electron-hole bilayer exciton) anchor the QMC runs, and its orbital
seeds the pairing function of a `pairing` trial.

Two particles of masses m_a, m_b in the plane, interacting through V(|r_a - r_b|).
The centre of mass separates; the relative coordinate r has reduced mass
mu = m_a m_b / (m_a + m_b) and, for angular momentum l (psi = R(r) e^{i l theta}),

    [ -(1/2mu) (1/r) d/dr (r d/dr) + l^2/(2 mu r^2) + V(r) ] R = E R,

normalized as  int_0^inf R^2 2 pi r dr = 1.  The pair size is the r.m.s.
separation r_rms = sqrt(<r^2>), <r^2> = int r^2 R^2 2 pi r dr.

Electron-hole bilayer (charges -1, +1, layers a distance d apart, dielectric
eps):  V(r) = -1 / (eps sqrt(r^2 + d^2)).  At d = 0, mu = 1/2 this is the 2-D
hydrogen atom, E_n = -mu Z^2 / (2 (n - 1/2)^2): E = -1 for the ground state
(Yang et al., PRA 43, 1186 (1991)).

The radial equation is solved by the symmetric finite-volume eigenproblem of
tests/references/dot_two_electron_exact.py on a uniform grid of cell centres, with the
energy and <r^2> Richardson-extrapolated from spacings h and h/2.  The box
edge r_max must sit where R has decayed (R ~ e^{-kappa r}, kappa =
sqrt(2 mu |E|) for a potential that vanishes at infinity); --check doubles
the grid and enlarges the box to report the remaining error.

    py -3 tests/references/pair_bound_state.py --d 0.4
    py -3 tests/references/pair_bound_state.py --d 0.4 --masses 1 1 --eps 1 --check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.linalg import eigh_tridiagonal

Potential = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PairState:
    """Lowest radial state of one angular momentum.

    energy and r2 are Richardson-extrapolated; r and R are the orbital on the
    finer grid, normalized so that sum(R^2 2 pi r) h = 1.
    """
    energy: float
    r2: float               # <r^2>
    r: np.ndarray           # [n] cell centres of the finer grid
    R: np.ndarray           # [n] R(r), sign fixed so that R > 0 near the origin

    @property
    def r_rms(self) -> float:
        return math.sqrt(self.r2)


def reduced_mass(m_a: float, m_b: float) -> float:
    if not (math.isfinite(m_a) and math.isfinite(m_b) and m_a > 0 and m_b > 0):
        raise ValueError(f"masses must be positive and finite, got {m_a}, {m_b}")
    return m_a * m_b / (m_a + m_b)


def bilayer_coulomb(d: float, eps: float = 1.0, charge_product: float = -1.0) -> Potential:
    """q_a q_b / (eps sqrt(r^2 + d^2)): Coulomb between charges in two planes."""
    if not (math.isfinite(d) and d >= 0 and math.isfinite(eps) and eps > 0):
        raise ValueError(f"bilayer Coulomb needs d >= 0 and eps > 0, got d={d}, eps={eps}")
    return lambda r: charge_product / (eps * np.sqrt(r * r + d * d))


def radial_state_h(potential: Potential, mu: float, ell: int, n_grid: int,
                   r_max: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Lowest level of angular momentum ell at one spacing h = r_max / n_grid.

    Finite volumes on the cell centres r_i = (i - 1/2) h, faces r_{i+-1/2}:
      -(1/2mu r) d/dr (r dR/dr) ~ -[r_{i+1/2}(R_{i+1}-R_i) - r_{i-1/2}(R_i-R_{i-1})]/(2mu r_i h^2).
    The face at r = 0 has zero area, the regular-origin condition for every
    ell, and R = 0 beyond the last cell.  With R_i = y_i / sqrt(r_i) the
    matrix is symmetric tridiagonal and sum(y^2) 2 pi h is the norm.
    Returns (energy, r, R) with R normalized.
    """
    if not (isinstance(ell, int) and isinstance(n_grid, int) and n_grid >= 2):
        raise ValueError("ell and n_grid must be integers, n_grid >= 2")
    if not (math.isfinite(mu) and mu > 0 and math.isfinite(r_max) and r_max > 0):
        raise ValueError(f"need mu > 0 and r_max > 0, got mu={mu}, r_max={r_max}")
    h = r_max / n_grid
    r = h * (np.arange(1, n_grid + 1) - 0.5)
    rp = r + 0.5 * h                         # r_{i+1/2}
    rm = r - 0.5 * h                         # r_{i-1/2}  (0 at the origin)
    v = np.asarray(potential(r), dtype=float)
    if v.shape != r.shape or not np.all(np.isfinite(v)):
        raise ValueError("the potential must return a finite value at every grid point")
    k = 1.0 / (2.0 * mu * h * h)
    diag = k * (rp + rm) / r + ell * ell / (2.0 * mu * r * r) + v
    off = -k * rp[:-1] / np.sqrt(r[:-1] * r[1:])
    w, y = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
    energy = float(w[0])
    # A level above the potential at the box edge is a box state, not a bound
    # one: its value would depend on r_max.
    if energy >= v[-1]:
        raise ValueError(f"no bound state: lowest level {energy:.6g} is not below "
                         f"V(r_max) = {v[-1]:.6g}")
    y = y[:, 0]
    y /= math.sqrt(2.0 * math.pi * h * float(np.dot(y, y)))
    if y[0] < 0:
        y = -y
    return energy, r, y / np.sqrt(r)


def mean_r2(r: np.ndarray, R: np.ndarray) -> float:
    """<r^2> = int r^2 R^2 2 pi r dr by the midpoint rule of the grid."""
    h = r[1] - r[0]
    return float(np.sum(r ** 3 * R * R)) * 2.0 * math.pi * h


def bound_state(potential: Potential, mu: float, ell: int = 0,
                n_grid: int = 20000, r_max: float = 60.0) -> PairState:
    """Lowest bound state of angular momentum ell, Richardson-extrapolated.

    The midpoint discretization errs as O(h^2) in both the energy and <r^2>,
    so (4 x(h/2) - x(h)) / 3 removes the leading term (error O(h^4)).
    """
    e1, r1, R1 = radial_state_h(potential, mu, ell, n_grid, r_max)
    e2, r2, R2 = radial_state_h(potential, mu, ell, 2 * n_grid, r_max)
    energy = (4.0 * e2 - e1) / 3.0
    r2_mean = (4.0 * mean_r2(r2, R2) - mean_r2(r1, R1)) / 3.0
    return PairState(energy=energy, r2=r2_mean, r=r2, R=R2)


def cutoff_polynomial_fit(r: np.ndarray, R: np.ndarray, L: float, order: int) -> list[float]:
    """Seed of the polynomial part of a pairing orbital (`cosine_polynomial`),

        phi(r) = (1 - r/L)^3 Theta(L - r) sum_{m=0}^{M} c_m r^m,   c_1 = 3 c_0 / L,

    fitted to R(r) by least squares over r < L with the area weight 2 pi r.
    c_1 is not free: it makes phi cuspless (phi'(0) = 0, as R'(0) = 0 for a
    potential finite at the origin).  Returned as [c_0, c_2, ..., c_M] with
    c_0 = 1, the order of Alea's `params`.  The fit runs in x = r/L, where the
    basis (1 - x)^3 x^m is well conditioned, and c_m = a_m / L^m.
    """
    if not (isinstance(order, int) and order >= 1 and math.isfinite(L) and L > 0):
        raise ValueError(f"need order >= 1 and L > 0, got order={order}, L={L}")
    inside = r < L
    x, y, w = r[inside] / L, R[inside], np.sqrt(r[inside])
    envelope = (1.0 - x) ** 3
    basis = [envelope * (1.0 + 3.0 * x)] + [envelope * x ** m for m in range(2, order + 1)]
    a = np.linalg.lstsq(np.stack(basis, axis=1) * w[:, None], y * w, rcond=None)[0]
    if a[0] == 0.0:
        raise ValueError("the fitted c_0 vanishes; phi(0) = 0 cannot be normalized")
    return [1.0] + [float(a[i] / a[0]) / L ** m for i, m in enumerate(range(2, order + 1), 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d", type=float, required=True, help="layer separation")
    ap.add_argument("--eps", type=float, default=1.0, help="dielectric constant")
    ap.add_argument("--masses", type=float, nargs=2, default=[1.0, 1.0],
                    metavar=("M_A", "M_B"))
    ap.add_argument("--ell", type=int, default=0, help="relative angular momentum")
    ap.add_argument("--n-grid", type=int, default=20000)
    ap.add_argument("--r-max", type=float, default=60.0)
    ap.add_argument("--check", action="store_true",
                    help="also solve with a doubled grid and a 1.5x box, print the differences")
    a = ap.parse_args()

    mu = reduced_mass(*a.masses)
    potential = bilayer_coulomb(a.d, a.eps)
    s = bound_state(potential, mu, a.ell, a.n_grid, a.r_max)
    print(f"# V(r) = -1/(eps sqrt(r^2 + d^2)), d = {a.d}, eps = {a.eps}, "
          f"mu = {mu}, l = {a.ell}")
    print(f"E      = {s.energy:.10f}")
    print(f"<r^2>  = {s.r2:.10f}")
    print(f"r_rms  = {s.r_rms:.10f}")
    if a.check:
        fine = bound_state(potential, mu, a.ell, 2 * a.n_grid, a.r_max)
        wide = bound_state(potential, mu, a.ell, int(1.5 * a.n_grid), 1.5 * a.r_max)
        print(f"# grid x2:  dE = {fine.energy - s.energy:.1e}  d<r^2> = {fine.r2 - s.r2:.1e}")
        print(f"# box x1.5: dE = {wide.energy - s.energy:.1e}  d<r^2> = {wide.r2 - s.r2:.1e}")


if __name__ == "__main__":
    main()
