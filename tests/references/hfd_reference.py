#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""hfd_reference.py -- reference numbers for tests/test_hfd.cpp, computed
independently of alea.

The HFD-B pair potential (R. A. Aziz, F. R. W. McCourt and C. C. K. Wong,
Mol. Phys. 61, 1487 (1987)), with x = r / r_m:

    V(r)  = eps [A exp(-alpha x + beta x^2) - F(x) (C6/x^6 + C8/x^8 + C10/x^10)],
    F(x)  = exp(-(D/x - 1)^2) for x < D, 1 otherwise.

HFD-B(HE) parameters (Table 1 of Ref. above, as reproduced in the OSTI report
MLM-3578 (1989)): A = 1.8443101e5, alpha = 10.43329537, beta = -2.27965105,
C6 = 1.36745214, C8 = 0.42123807, C10 = 0.17473318, D = 1.4826,
eps = 10.948 K, r_m = 2.963 Angstrom.

Prints
  1. V(r) and V'(r) at a few separations, by mpmath at 30 digits (V' by
     mpmath.diff of the same expression, not by the hand-derived slope);
  2. the continuum tail of 64 atoms at n = 0.02186 A^-3 in a cube (L = 14.3057 A),
         E_tail = (N^2 / 2 V) T,   T = int_{outside cube} V d^3 r
                = 4 pi int_h^inf V r^2 dr - 8 int_{[0,h]^3, r > h} V d^3 r,
     h = L/2: the radial integral over the exterior of the inscribed sphere
     minus the eight corners of the cube outside it, in Cartesian
     coordinates (scipy, relative accuracy ~1e-9) -- not the face
     parameterization alea uses;
  3. the exterior integral of r^-4 outside a cube of half-side 1,
         int_{outside} r^-4 d^3 r = 24 int_0^1 int_0^1 (1 + a^2 + b^2)^-2 da db,
     by mpmath.

Usage:  python tests/references/hfd_reference.py
"""
import mpmath as mp
import numpy as np
from scipy import integrate

mp.mp.dps = 30
HE = dict(eps=mp.mpf("10.948"), r_m=mp.mpf("2.963"), A=mp.mpf("1.8443101e5"),
          alpha=mp.mpf("10.43329537"), beta=mp.mpf("-2.27965105"), C6=mp.mpf("1.36745214"),
          C8=mp.mpf("0.42123807"), C10=mp.mpf("0.17473318"), D=mp.mpf("1.4826"))


def v_mp(r, q=HE):
    x = r / q["r_m"]
    F = mp.exp(-(q["D"] / x - 1) ** 2) if x < q["D"] else mp.mpf(1)
    return q["eps"] * (q["A"] * mp.exp(-q["alpha"] * x + q["beta"] * x * x)
                       - F * (q["C6"] / x**6 + q["C8"] / x**8 + q["C10"] / x**10))


def v_np(r):
    q = {k: float(val) for k, val in HE.items()}
    x = r / q["r_m"]
    F = np.exp(-(q["D"] / x - 1.0) ** 2) if x < q["D"] else 1.0
    return q["eps"] * (q["A"] * np.exp(-q["alpha"] * x + q["beta"] * x * x)
                       - F * (q["C6"] / x**6 + q["C8"] / x**8 + q["C10"] / x**10))


def main():
    print("1. V(r), V'(r) [K, K/A]")
    for r in ("1.0", "2.0", "2.963", "3.5", "4.0", "5.0", "8.0"):
        rr = mp.mpf(r)
        print("   r = %-6s V = %s   V' = %s" % (r, mp.nstr(v_mp(rr), 17),
                                                mp.nstr(mp.diff(v_mp, rr), 17)))

    n_atoms, density = 64, 0.02186
    L = (n_atoms / density) ** (1.0 / 3.0)
    h = 0.5 * L
    sphere, _ = integrate.quad(lambda r: v_np(r) * r * r, h, np.inf, epsabs=0, epsrel=1e-13,
                               limit=400)

    def column(y, z):
        x0 = np.sqrt(max(h * h - y * y - z * z, 0.0))
        val, _ = integrate.quad(lambda x: v_np(np.sqrt(x * x + y * y + z * z)), x0, h,
                                epsabs=0, epsrel=1e-12)
        return val

    corners, _ = integrate.dblquad(column, 0.0, h, 0.0, h, epsabs=0, epsrel=1e-10)
    T = 4.0 * np.pi * sphere - 8.0 * corners
    print("2. cube L = %.12f: T = %.12f K A^3, E_tail = %.10f K per cell" %
          (L, T, 0.5 * n_atoms * n_atoms / L**3 * T))

    face = mp.quad(lambda a, b: 1 / (1 + a * a + b * b) ** 2, [0, 1], [0, 1])
    print("3. int_{outside cube h = 1} r^-4 d^3 r = %s" % mp.nstr(24 * face, 20))


main()
