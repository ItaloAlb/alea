#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""inverse_power_trap_primitive.py -- exact finite-M path integral of two
identical bosons in a 2-D isotropic harmonic trap interacting through C / r^p,
in the primitive approximation that alea's PIMC samples.

Reference for the grand-canonical worm with an interaction: with mass 1 and
hbar = 1, the window {1, 2} has P(2)/P(1) = e^{beta mu} Z_2 / Z_1, i.e.

    F(2) - F(1) = -T ln(Z_2 / Z_1).

Primitive approximation (Ceperley, Rev. Mod. Phys. 67, 279 (1995), Sec. IV.A):
with M slices of tau = beta / M,

    Z_N = (1/N!) sum_P int prod_k dR_k  prod_k rho0(R_k, R_{k+1}; tau) e^{-tau V(R_k)},
    R_M = P R_0,  rho0 the normalized free propagator of each particle.

Centre of mass.  For two particles V = 1/2 w^2 (r_1^2 + r_2^2) + C / r^p
= w^2 X^2 + 1/4 w^2 r^2 + C / r^p with X = (r_1 + r_2)/2 (mass 2) and
r = r_1 - r_2 (mass mu = 1/2).  The quadratic link actions separate exactly
at every slice, and dr_1 dr_2 = dX dr, so even at finite M

    Z_2 = z_cm * Z_rel,    Z_rel = 1/2 (Z_rel^direct + Z_rel^exchange),

with the exchange closing the relative path on -r_0.  A discretized harmonic
ring does not depend on the mass (rescale x -> sqrt(m) x: the Jacobian cancels
the rho0 normalizations), so z_cm = Z_1 and

    Z_2 / Z_1 = Z_rel.

Partial waves.  e^{z cos theta} = sum_m I_m(z) e^{i m theta} (Abramowitz &
Stegun 9.6.34) splits the 2-D free propagator of mass mu,

    rho0(r, r') = mu/(2 pi tau) e^{-mu |r - r'|^2 / (2 tau)}
                = (1/2 pi) sum_m g_m(r, r') e^{i m (phi - phi')},
    g_m(r, r') = (mu/tau) e^{-mu (r - r')^2/(2 tau)} Ie_m(mu r r'/tau),

with Ie_m(x) = e^{-x} I_m(x) (scipy.special.ive).  The M angular integrals
each give 2 pi and keep m fixed, and closing on -r_0 multiplies by (-1)^m:

    Z_rel^direct = sum_m t_m,  Z_rel^exchange = sum_m (-1)^m t_m,
    Z_rel = sum_{m even} t_m,  t_m = Tr (G_m)^M,

where G_m is the radial transfer operator with measure r dr and
U(r) = 1/2 mu w^2 r^2 + C r^{-p}.  On Gauss-Legendre nodes r_i in [0, r_max]
with weights w_i it is the symmetric matrix

    S_ij = sqrt(w_i r_i) e^{-tau U_i/2} g_m(r_i, r_j) e^{-tau U_j/2} sqrt(w_j r_j),

and t_m = sum_n lambda_n^M over its eigenvalues.  The thermodynamic energy at
fixed M is E = -(1/M) d ln Z / d tau, by a central difference in tau on the
same nodes.

Checks (--check): at C = 0 the relative traces are the closed forms
Z_rel^direct = z(M) and Z_rel^exchange = z(2M)/z(M), with
z(K) = prod_j (4 sin^2(pi j/K) + tau^2 w^2)^{-1} the 2-D discretized ring;
the result is stable under doubling the nodes and enlarging r_max; and as
tau -> 0 Z_rel approaches the continuum sum over even m of e^{-beta eps} from
the radial spectrum of tests/references/inverse_power_trap_exact.py (an
independent solver).  The primitive action reaches its tau^2 regime slowly
with an r^-3 core: at beta = 2, C = 1 the difference in ln Z_rel falls by
2.85, 3.05 and 3.24 per halving of tau from M = 100 to M = 800
(5.7e-3 to 2.0e-4).  --check takes a few minutes.

    py -3 tests/references/inverse_power_trap_primitive.py --beta 2 --slices 16 --omega 1 --strength 1 --exponent 3
    py -3 tests/references/inverse_power_trap_primitive.py ... --check
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ive

MU_REL = 0.5  # reduced mass of two particles of mass 1


def ring_log_z(slices: int, tau: float, omega: float, dim: int = 2) -> float:
    """ln z(K) of one particle in the discretized harmonic ring (any mass)."""
    j = np.arange(slices)
    mode = 4.0 * np.sin(np.pi * j / slices) ** 2 + (tau * omega) ** 2
    return float(-0.5 * dim * np.sum(np.log(mode)))


def relative_traces(tau: float, slices: int, omega: float, strength: float,
                    exponent: float, nodes: int, r_max: float,
                    tolerance: float = 1e-15) -> list[float]:
    """t_m = Tr (G_m)^M for m = 0, 1, 2, ... until two consecutive t_m fall
    below tolerance * t_0."""
    if (slices < 1 or nodes < 2 or not tau > 0 or not omega > 0 or not r_max > 0
            or not math.isfinite(strength) or not exponent > 0):
        raise ValueError("relative_traces needs tau, omega, r_max > 0, slices >= 1, "
                         "nodes >= 2, finite strength and a positive exponent")
    if strength < 0 and exponent >= 2:
        raise ValueError("an attractive core with p >= 2 has no ground state")
    x, w = np.polynomial.legendre.leggauss(nodes)
    r = 0.5 * r_max * (x + 1.0)
    w = 0.5 * r_max * w
    u = 0.5 * MU_REL * omega ** 2 * r ** 2 + strength * r ** (-exponent)
    side = np.sqrt(w * r) * np.exp(-0.5 * tau * u)
    gauss = (MU_REL / tau) * np.exp(-MU_REL * (r[:, None] - r[None, :]) ** 2 / (2.0 * tau))
    arg = MU_REL * np.outer(r, r) / tau
    traces: list[float] = []
    small = 0
    m = 0
    while small < 2:
        s = side[:, None] * gauss * ive(m, arg) * side[None, :]
        lam = np.linalg.eigvalsh(s)
        if lam.min() < -1e-12 * max(lam.max(), 1e-300):
            raise RuntimeError(f"m = {m}: transfer matrix not positive (min eigenvalue "
                               f"{lam.min():.3e}); refine the nodes")
        t = float(np.sum(np.clip(lam, 0.0, None) ** slices))
        traces.append(t)
        small = small + 1 if t < tolerance * traces[0] else 0
        m += 1
    return traces


def relative_partition(traces: list[float]) -> dict[str, float]:
    """Direct, exchange and bosonic relative partition functions (m and -m both)."""
    t0 = traces[0]
    rest = np.asarray(traces[1:])
    sign = np.where(np.arange(1, len(traces)) % 2 == 0, 1.0, -1.0)
    direct = t0 + 2.0 * float(np.sum(rest))
    exchange = t0 + 2.0 * float(np.sum(sign * rest))
    return {"direct": direct, "exchange": exchange, "boson": 0.5 * (direct + exchange)}


def solve(beta: float, slices: int, omega: float, strength: float, exponent: float,
          nodes: int, r_max: float) -> dict[str, float]:
    """ln Z_rel (boson, direct, exchange), E(1), E(2) and F(2) - F(1) at finite M."""
    tau = beta / slices
    z = relative_partition(relative_traces(tau, slices, omega, strength, exponent, nodes, r_max))
    h = 1e-4 * tau
    zp = relative_partition(relative_traces(tau + h, slices, omega, strength, exponent, nodes, r_max))
    zm = relative_partition(relative_traces(tau - h, slices, omega, strength, exponent, nodes, r_max))
    e_rel = -(math.log(zp["boson"]) - math.log(zm["boson"])) / (2.0 * h) / slices
    e_cm = -(ring_log_z(slices, tau + h, omega) - ring_log_z(slices, tau - h, omega)) / (2.0 * h) / slices
    return {
        "tau": tau,
        "ln_z_rel_boson": math.log(z["boson"]),
        "ln_z_rel_direct": math.log(z["direct"]),
        "ln_z_rel_exchange": math.log(z["exchange"]) if z["exchange"] > 0 else float("nan"),
        "energy_1": e_cm,
        "energy_2": e_cm + e_rel,
        "delta_free_energy": -math.log(z["boson"]) / beta,
    }


def continuum_z_rel(beta: float, omega: float, strength: float, exponent: float,
                    n_grid: int, r_max: float, n_levels: int, m_max: int) -> float:
    """sum over even m (m and -m) and radial levels of e^{-beta eps}, tau = 0."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inverse_power_trap_exact import radial_spectrum_h  # noqa: E402
    total = 0.0
    for m in range(0, m_max + 1, 2):
        eps = radial_spectrum_h(m, omega, strength, exponent, MU_REL, n_grid, r_max, n_levels)
        total += (1.0 if m == 0 else 2.0) * float(np.sum(np.exp(-beta * eps)))
    return total


def check(args: argparse.Namespace) -> None:
    beta, M, omega = args.beta, args.slices, args.omega
    tau = beta / M
    # 1. Non-interacting closed forms (normalization, measure, exchange sign).
    ideal = relative_partition(relative_traces(tau, M, omega, 0.0, args.exponent, args.nodes, args.r_max))
    direct = math.exp(ring_log_z(M, tau, omega))
    exchange = math.exp(ring_log_z(2 * M, tau, omega)) / direct
    print(f"C = 0: direct {ideal['direct']:.12f} closed form {direct:.12f} "
          f"(rel. diff {ideal['direct'] / direct - 1:.2e})")
    print(f"C = 0: exchange {ideal['exchange']:.12f} closed form {exchange:.12f} "
          f"(rel. diff {ideal['exchange'] / exchange - 1:.2e})")
    # 2. Quadrature: twice the nodes, a larger box.
    base = solve(beta, M, omega, args.strength, args.exponent, args.nodes, args.r_max)
    finer = solve(beta, M, omega, args.strength, args.exponent, 2 * args.nodes, args.r_max)
    wider = solve(beta, M, omega, args.strength, args.exponent, args.nodes, 1.25 * args.r_max)
    for key in ("ln_z_rel_boson", "energy_2"):
        print(f"{key}: {base[key]:.12f}; 2x nodes {finer[key] - base[key]:+.2e}; "
              f"1.25x r_max {wider[key] - base[key]:+.2e}")
    # 3. tau -> 0 against the continuum radial spectrum (an independent solver).
    z_cont = continuum_z_rel(beta, omega, args.strength, args.exponent,
                             n_grid=8000, r_max=args.r_max, n_levels=40, m_max=40)
    print(f"continuum (radial spectrum): ln Z_rel = {math.log(z_cont):.8f}")
    previous = None
    for slices in (50, 100, 200):
        nodes = max(args.nodes, int(12 * args.r_max / math.sqrt(beta / slices / MU_REL)))
        res = solve(beta, slices, omega, args.strength, args.exponent, nodes, args.r_max)
        diff = res["ln_z_rel_boson"] - math.log(z_cont)
        ratio = "" if previous is None else f"  (previous / this = {previous / diff:.2f}; 4 once tau^2)"
        print(f"M = {slices:4d} (nodes {nodes}): ln Z_rel = {res['ln_z_rel_boson']:.8f}, "
              f"minus continuum {diff:+.3e}{ratio}")
        previous = diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--slices", type=int, required=True)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--exponent", type=float, default=3.0)
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--r-max", type=float, default=12.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    res = solve(args.beta, args.slices, args.omega, args.strength, args.exponent,
                args.nodes, args.r_max)
    for key, value in res.items():
        print(f"{key} = {value:.12f}")
    ideal = solve(args.beta, args.slices, args.omega, 0.0, args.exponent, args.nodes, args.r_max)
    print(f"ideal delta_free_energy = {ideal['delta_free_energy']:.12f}")
    print(f"no-exchange delta_free_energy = "
          f"{-(res['ln_z_rel_direct'] - math.log(2.0)) / args.beta:.12f}")
    if args.check:
        check(args)


if __name__ == "__main__":
    main()
