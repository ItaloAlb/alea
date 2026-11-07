#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""dot_two_electron_exact.py -- exact energies of two electrons in a 2-D
parabolic quantum dot in a perpendicular magnetic field.

Reference for alea's fixed-phase DMC (tests/configs/dot2e_*.yaml).

Hartree atomic units, effective mass m = 1, charge q = -1, dielectric eps:

    H = sum_i [ (p_i - q A(r_i))^2 / 2 + 1/2 w0^2 r_i^2 ] + 1/(eps |r_1 - r_2|),
    A = B/2 (-y, x)  (symmetric gauge),  w_c = q B / m = -B.

Kohn's theorem: the centre of mass (mass 2, charge 2q) separates exactly and
feels only the dot, so its ground level is W = sqrt(w0^2 + w_c^2/4).  The
relative coordinate r = r_1 - r_2 (mu = 1/2, same w_c) carries the Coulomb
repulsion; for relative angular momentum m,

    E_rel(m) = eps_|m| - (w_c/2) m,
    [-(1/2mu)(d2/dr2 - (m^2 - 1/4)/r^2) + 1/2 mu W^2 r^2 + 1/(eps r)] u = eps u,

with R = u / sqrt(r).  Pauli: even m is the spin singlet, odd m the triplet.
E(total) = W (CM ground) + E_rel(m).  (Wagner, Merkt & Chaplik, PRB 45, 1951
(1992): the ground state switches singlet -> triplet as B grows.)

The radial equation is solved by a symmetric finite-volume eigenproblem on
a uniform grid, Richardson-extrapolated (h, h/2); --check doubles the grid
to report the remaining discretisation error.

    py -3 tests/references/dot_two_electron_exact.py --w0 0.25 --B 0 0.2 0.4 --m 0 -1
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.linalg import eigh_tridiagonal




def radial_spectrum_h(m: int, W: float, eps: float, n_grid: int,
                      r_max: float, n_levels: int) -> np.ndarray:
    """Lowest n_levels radial energies at one grid spacing.

    Finite volumes on the cell centres r_i = (i - 1/2) h, faces r_{i+-1/2}:
      -(1/2mu r) d/dr (r dR/dr) ~ -[r_{i+1/2}(R_{i+1}-R_i) - r_{i-1/2}(R_i-R_{i-1})]/(2mu r_i h^2)
    The face at r = 0 has zero area, which is the regular-origin condition for
    every m (no 1/r^2 singularity in the stencil).  Scaling by sqrt(r_i) makes
    the matrix symmetric tridiagonal.
    """
    if (not isinstance(m, int) or not isinstance(n_grid, int)
            or not isinstance(n_levels, int) or not 1 <= n_levels <= n_grid
            or n_grid < 2 or not math.isfinite(W) or W <= 0
            or math.isnan(eps) or eps <= 0
            or not math.isfinite(r_max) or r_max <= 0):
        raise ValueError("radial spectrum needs integer sizes, W > 0, eps > 0 and r_max > 0")
    mu = 0.5
    h = r_max / n_grid
    r = h * (np.arange(1, n_grid + 1) - 0.5)
    rp = r + 0.5 * h                       # r_{i+1/2}
    rm = r - 0.5 * h                       # r_{i-1/2}  (0 at the origin)
    k = 1.0 / (2.0 * mu * h * h)
    V = m * m / (2.0 * mu * r * r) + 0.5 * mu * W * W * r * r + 1.0 / (eps * r)
    diag = k * (rp + rm) / r + V
    off = -k * rp[:-1] / np.sqrt(r[:-1] * r[1:])
    return eigh_tridiagonal(diag, off, select="i", select_range=(0, n_levels - 1),
                            eigvals_only=True)


def radial_level_h(m: int, W: float, eps: float, n_grid: int, r_max: float) -> float:
    """Lowest eps_|m| at one grid spacing; preserve the ground-state API."""
    return float(radial_spectrum_h(m, W, eps, n_grid, r_max, 1)[0])


def radial_level(m: int, W: float, eps: float, n_grid: int, r_max: float) -> float:
    """eps_|m| Richardson-extrapolated from h and h/2 (error O(h^4))."""
    e1 = radial_level_h(m, W, eps, n_grid, r_max)
    e2 = radial_level_h(m, W, eps, 2 * n_grid, r_max)
    return (4.0 * e2 - e1) / 3.0


def total_energy(m: int, B: float, w0: float, eps: float, n_grid: int, r_max: float):
    wc = -B                                   # q B / m with q = -1, m = 1
    W = math.sqrt(w0 * w0 + 0.25 * wc * wc)
    return W + radial_level(m, W, eps, n_grid, r_max) - 0.5 * wc * m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--w0", type=float, required=True)
    ap.add_argument("--B", type=float, nargs="+", required=True)
    ap.add_argument("--m", type=int, nargs="+", default=[0, -1],
                    help="relative angular momenta (even: singlet, odd: triplet)")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--n-grid", type=int, default=4000)
    ap.add_argument("--r-max", type=float, default=0.0,
                    help="radial box (default: 12 oscillator lengths)")
    ap.add_argument("--check", action="store_true",
                    help="also solve at half the spacing and print the difference")
    a = ap.parse_args()

    print(f"# w0 = {a.w0}  eps = {a.eps}   E = W + eps_|m| + (B/2) m")
    print("#     B          m      E_total           (grid error)")
    for B in a.B:
        W = math.sqrt(a.w0 ** 2 + 0.25 * B * B)
        r_max = a.r_max or 12.0 / math.sqrt(0.5 * W)
        for m in a.m:
            e = total_energy(m, B, a.w0, a.eps, a.n_grid, r_max)
            extra = ""
            if a.check:
                e2 = total_energy(m, B, a.w0, a.eps, 2 * a.n_grid + 1, r_max)
                extra = f"   ({abs(e2 - e):.1e})"
            print(f"  {B:9.5f}  {m:4d}   {e:.10f}{extra}")


if __name__ == "__main__":
    main()
