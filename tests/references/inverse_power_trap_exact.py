#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""inverse_power_trap_exact.py -- exact ground energy of two identical particles
in a 2-D isotropic harmonic trap interacting through V(r) = C / r^p.

Reference for alea's VMC/DMC with the `inverse_power` term (p = 3: dipoles
polarized perpendicular to the plane, e.g. Astrakharchik et al., PRL 98,
060405 (2007), in units of r0 = M C_dd/(4 pi hbar^2) and hbar^2/(M r0^2)).

    H = sum_i [ p_i^2/(2M) + 1/2 M w^2 r_i^2 ] + C / |r_1 - r_2|^p

The centre of mass (mass 2M) separates exactly and feels only the trap, so its
ground level is w.  The relative coordinate r (mu = M/2) with angular
momentum m obeys

    [-(1/2mu)(d2/dr2 - (m^2 - 1/4)/r^2) + 1/2 mu w^2 r^2 + C/r^p] u = eps u,

with R = u / sqrt(r).  E(total) = w + eps_|m|.  The bosonic (and the
distinguishable) ground state is m = 0.  For p > 2 the pair wavefunction
vanishes at contact as exp(-(2 sqrt(2 mu C)/(p-2)) r^{-(p-2)/2}); no boundary
condition is imposed for it, the potential itself enforces it.

The radial equation is solved by the symmetric finite-volume eigenproblem of
tests/references/dot_two_electron_exact.py (uniform cell centres, zero-area face at the
origin), Richardson-extrapolated over h and h/2; --check also solves at half
the spacing and prints the remaining discretization error.

Checks with closed forms (tests/test_inverse_power_trap_exact.py): C = 0 gives
E = 2w; p = 2 only shifts the centrifugal term, m^2 -> m^2 + 2 mu C, so
eps = w (1 + sqrt(m^2 + 2 mu C)); p = 1, C = 1, M = 1, w = 1 is Taut's 2-D
two-electron dot, E = 3.

    py -3 tests/references/inverse_power_trap_exact.py --omega 1 5 20 --strength 1 --exponent 3
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def radial_spectrum_h(m: int, omega: float, strength: float, exponent: float,
                      mu: float, n_grid: int, r_max: float, n_levels: int = 1) -> np.ndarray:
    """Lowest n_levels relative energies eps at one grid spacing h = r_max/n_grid.

    Finite volumes on the cell centres r_i = (i - 1/2) h with faces r_{i+-1/2}:
      -(1/2mu r) d/dr (r dR/dr) ~ -[r_{i+1/2}(R_{i+1}-R_i) - r_{i-1/2}(R_i-R_{i-1})]/(2mu r_i h^2)
    Scaling by sqrt(r_i) makes the matrix symmetric tridiagonal.
    """
    if (not isinstance(m, int) or not isinstance(n_grid, int) or n_grid < 2
            or not isinstance(n_levels, int) or not 1 <= n_levels <= n_grid
            or not math.isfinite(omega) or omega <= 0
            or not math.isfinite(strength) or not math.isfinite(exponent) or exponent <= 0
            or not math.isfinite(mu) or mu <= 0
            or not math.isfinite(r_max) or r_max <= 0):
        raise ValueError("radial spectrum needs integer sizes, omega, mu, r_max > 0, "
                         "finite strength and a positive exponent")
    if strength < 0 and exponent >= 2:
        raise ValueError("an attractive core with p >= 2 has no ground state")
    h = r_max / n_grid
    r = h * (np.arange(1, n_grid + 1) - 0.5)
    rp = r + 0.5 * h                       # r_{i+1/2}
    rm = r - 0.5 * h                       # r_{i-1/2}  (0 at the origin)
    k = 1.0 / (2.0 * mu * h * h)
    V = (m * m / (2.0 * mu * r * r) + 0.5 * mu * omega * omega * r * r
         + strength * r ** (-exponent))
    diag = k * (rp + rm) / r + V
    off = -k * rp[:-1] / np.sqrt(r[:-1] * r[1:])
    return eigh_tridiagonal(diag, off, select="i", select_range=(0, n_levels - 1),
                            eigvals_only=True)


def log_grid_spectrum_h(m: int, omega: float, strength: float, exponent: float,
                        mu: float, n_grid: int, r_min: float, r_max: float,
                        n_levels: int = 1) -> np.ndarray:
    """Lowest n_levels relative energies on a uniform grid in x = ln r.

    With r = e^x, R'' + R'/r = R_xx / r^2, so the radial equation becomes
      -(1/2mu)(R_xx - m^2 R) + r^2 V R = eps r^2 R,
    a generalized problem A R = eps B R with the positive weight B = r^2.  The
    grid spacing is uniform in ln r, so the steep contact wall
    exp(-kappa r^{-(p-2)/2}) is resolved as finely as the trap tail.  Dirichlet
    ends: valid when R vanishes at both ends, i.e. for a repulsive core with
    p >= 2 (exponentially or as a power at small r) and the trap (large r).

    Not reduced to a standard tridiagonal problem: B^{-1/2} A B^{-1/2} has
    entries ~ 1/r^2 near the core, and a dense-spectrum solver's absolute error
    ~ eps_mach |matrix| would swamp the O(1) ground level.  Shift-invert about
    0 (every level is positive) factorizes A itself and converges onto the
    lowest levels, which live where the entries are moderate.
    """
    if not (strength > 0 and exponent >= 2 and 0 < r_min < r_max):
        raise ValueError("the log grid needs a repulsive core with p >= 2 and 0 < r_min < r_max")
    x = np.linspace(math.log(r_min), math.log(r_max), n_grid + 2)[1:-1]
    h = x[1] - x[0]
    r = np.exp(x)
    k = 1.0 / (2.0 * mu * h * h)
    r2V = 0.5 * mu * omega * omega * r ** 4 + strength * r ** (2.0 - exponent)
    A = diags([np.full(n_grid - 1, -k), 2.0 * k + m * m / (2.0 * mu) + r2V,
               np.full(n_grid - 1, -k)], [-1, 0, 1], format="csc")
    B = diags(r * r, 0, format="csc")
    levels = eigsh(A, k=n_levels, M=B, sigma=0.0, which="LM", return_eigenvectors=False)
    return np.sort(levels)


def contact_r_min(strength: float, exponent: float, mu: float) -> float:
    """r where the contact factor exp(-kappa r^{-s}) of R is e^{-60} (s = (p-2)/2).

    Only for p > 2; at p = 2, R ~ r^{sqrt(m^2 + 2 mu C)} and r_min must be
    chosen so that power is negligible.
    """
    s = 0.5 * (exponent - 2.0)
    kappa = math.sqrt(2.0 * mu * strength) / s
    return (kappa / 60.0) ** (1.0 / s)


def relative_level(m: int, omega: float, strength: float, exponent: float, mu: float,
                   n_grid: int, r_max: float) -> float:
    """eps_|m| Richardson-extrapolated from h and h/2 (error O(h^4)).

    A repulsive core with p > 2 is solved on the ln r grid (log_grid_spectrum_h),
    everything else on the finite-volume grid in r (radial_spectrum_h).
    """
    if strength > 0 and exponent > 2:
        r_min = min(contact_r_min(strength, exponent, mu), 1e-3 * r_max)
        e1 = log_grid_spectrum_h(m, omega, strength, exponent, mu, n_grid, r_min, r_max)[0]
        e2 = log_grid_spectrum_h(m, omega, strength, exponent, mu, 2 * n_grid + 1,
                                 r_min, r_max)[0]
    else:
        e1 = radial_spectrum_h(m, omega, strength, exponent, mu, n_grid, r_max)[0]
        e2 = radial_spectrum_h(m, omega, strength, exponent, mu, 2 * n_grid, r_max)[0]
    return float((4.0 * e2 - e1) / 3.0)


def default_r_max(omega: float, mu: float) -> float:
    """14 relative oscillator lengths 1/sqrt(mu w): the tail exp(-mu w r^2/2) < 1e-42."""
    return 14.0 / math.sqrt(mu * omega)


def total_energy(omega: float, strength: float, exponent: float, mass: float = 1.0,
                 m: int = 0, n_grid: int = 20000, r_max: float = 0.0) -> float:
    """E = w (centre of mass) + eps_|m| (relative motion)."""
    mu = 0.5 * mass
    r_max = r_max or default_r_max(omega, mu)
    return omega + relative_level(m, omega, strength, exponent, mu, n_grid, r_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--omega", type=float, nargs="+", required=True)
    ap.add_argument("--strength", type=float, required=True, help="C in V = C/r^p")
    ap.add_argument("--exponent", type=float, required=True, help="p in V = C/r^p")
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=0, help="relative angular momentum")
    ap.add_argument("--n-grid", type=int, default=20000)
    ap.add_argument("--r-max", type=float, default=0.0,
                    help="radial box (default: 14 relative oscillator lengths)")
    ap.add_argument("--check", action="store_true",
                    help="also solve at half the spacing and print the difference")
    a = ap.parse_args()

    print(f"# C = {a.strength}  p = {a.exponent}  M = {a.mass}  m = {a.m}   E = w + eps_|m|")
    print("#     omega       E_total           (grid error)")
    for w in a.omega:
        e = total_energy(w, a.strength, a.exponent, a.mass, a.m, a.n_grid, a.r_max)
        extra = ""
        if a.check:
            e2 = total_energy(w, a.strength, a.exponent, a.mass, a.m, 2 * a.n_grid, a.r_max)
            extra = f"   ({abs(e2 - e):.1e})"
        print(f"  {w:10.5f}   {e:.10f}{extra}")


if __name__ == "__main__":
    main()
