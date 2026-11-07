#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""moire_cells.py -- alea configs for electrons in a triangular moire superlattice.

For one point of the phase diagram (filling nu, electron count N_e, interaction
strength x = a_M/a_B*, moire depth kappa = V_0/Ha*, shape phi) this writes an
alea YAML for one competing state -- the pinned Wigner crystal or the fluid --
plus a JSON sidecar with the geometry, so the two states of a point are built in
the SAME periodic cell and differ only in the determinant.

Units are alea's effective atomic units (a_B*, Ha*): m* = 1, epsilon = 1, so
the moire period in the config IS x and the modulation amplitude IS kappa.

Hamiltonian (Wu, Lovorn & MacDonald, PRL 121, 026402 (2018)):

    H = sum_i -1/2 nabla_i^2 + sum_i V_M(r_i) + sum_{i<j} 1/r_ij
    V_M(r) = 2 V_0 sum_{j=1..3} cos(g_j . r + phi),   |g_j| = 4 pi/(sqrt3 a_M)

What the tool guarantees, and checks before writing anything:
  * the cell is a supercell of the moire lattice (integer combination of the
    moire vectors A1, A2), so V_M and the moire Jastrow are periodic in it;
  * it holds M = N_e/nu moire cells, i.e. the filling is exactly nu;
  * the crystal sites sit on the moire MINIMA (found at the three high-symmetry
    points and confirmed against a dense grid) and form the commensurate
    triangular sublattice of index 1/nu (sqrt3 x sqrt3 at 1/3, 2x2 at 1/4);
  * the cell is also a supercell of that crystal lattice;
  * N_e closes a reciprocal-lattice shell of the cell, so the plane-wave fluid
    determinant is non-degenerate (alea refuses an open shell anyway).

Only nu = 1/m with m a Loeschian number (1, 3, 4, 7, 9, 12, ...) and a
spin-polarized electron group are handled: that is where a triangular crystal
is commensurate with a hexagonal moire supercell.  Other fillings need
approximant cells (not here yet).

Usage
-----
    py -3 tools/moire_cells.py --list --nu 1/3
    py -3 tools/moire_cells.py --nu 1/3 --ne 19 --x 15 --kappa 0.02 \\
        --phi-deg 60 --state crystal --out runs/moire/demo/crystal_N19.yaml
    py -3 tools/moire_cells.py --selftest

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

SQRT3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi

# Einstein estimate of a 2-D Wigner crystal's Gaussian width, alpha_E = C/rs^1.5,
# the same number the configs/heg2d_crystal_*.yaml headers use as their scale.
EINSTEIN_C = 0.4


# ── small linear algebra ─────────────────────────────────────────────────────

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def add(a, b, s=1.0):
    return (a[0] + s * b[0], a[1] + s * b[1])


def scale(a, s):
    return (s * a[0], s * a[1])


def norm(a):
    return math.hypot(a[0], a[1])


def solve2(c1, c2, v):
    """Coefficients (n1, n2) with v = n1*c1 + n2*c2."""
    det = c1[0] * c2[1] - c1[1] * c2[0]
    return ((v[0] * c2[1] - v[1] * c2[0]) / det,
            (c1[0] * v[1] - c1[1] * v[0]) / det)


def is_int(x, tol=1e-8):
    return abs(x - round(x)) < tol


# ── moire geometry ───────────────────────────────────────────────────────────

def first_shell(a_M, theta0):
    """The three g_j, exactly as alea's moire_reciprocal_vectors builds them."""
    gm = 4.0 * math.pi / (SQRT3 * a_M)
    return [(gm * math.cos(theta0 + TWO_PI / 3 * i),
             gm * math.sin(theta0 + TWO_PI / 3 * i)) for i in range(3)]


def direct_lattice(g):
    """Moire lattice vectors dual to (g_0, g_1): g_i . A_j = 2 pi delta_ij.
    Same construction as App's commensurability check; A1, A2 are at 60 deg."""
    det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    k = TWO_PI / det
    return (k * g[1][1], -k * g[1][0]), (-k * g[0][1], k * g[0][0])


def loeschian_pairs(n):
    """All (p, q), p >= 0, q >= 0, with p^2 + pq + q^2 = n."""
    out = []
    for p in range(0, n + 1):
        for q in range(0, n + 1):
            if p * p + p * q + q * q == n:
                out.append((p, q))
    return out


def is_loeschian(n):
    return bool(loeschian_pairs(n))


def triangular_superlattice(A1, A2, p, q):
    """Hexagonal superlattice of index p^2+pq+q^2 (same convention as
    commensurate_triangular_sites in wavefunction/orbital_set.h)."""
    L1 = add(scale(A1, p), A2, q)
    L2 = add(scale(A1, -q), A2, p + q)
    return L1, L2


def moire_V(r, g, V0, phi):
    return 2.0 * V0 * sum(math.cos(dot(gj, r) + phi) for gj in g)


def moire_minimum(g, A1, A2, V0, phi):
    """Global minimum of V_M in one moire cell.

    The candidates are the three high-symmetry (stacking) points: the origin and
    the two points K, K' with g_j . K = +-2 pi/3 for every j, where
        V(0) = 6 V0 cos(phi),  V(K) = 6 V0 cos(phi + 2pi/3),  V(K') = 6 V0 cos(phi - 2pi/3).
    phi = 60 deg (V0 > 0) puts one deep minimum per cell at K: a triangular site
    lattice; phi = 0 gives two degenerate minima (honeycomb).  The winner is
    confirmed against a dense grid so an unexpected shape cannot slip through.
    """
    K = solve_for_phases(g, TWO_PI / 3)
    Kp = scale(K, -1.0)
    cands = [(0.0, 0.0), K, Kp]
    vals = [moire_V(c, g, V0, phi) for c in cands]
    order = sorted(range(3), key=lambda i: vals[i])
    if abs(vals[order[0]] - vals[order[1]]) < 1e-12 * max(1.0, abs(vals[order[0]])):
        raise SystemExit(
            f"phi = {math.degrees(phi):.3f} deg gives two degenerate minima per "
            "moire cell (honeycomb site lattice); this tool builds triangular "
            "crystals on a single-minimum lattice only.")
    rmin, vmin = cands[order[0]], vals[order[0]]
    n = 120
    grid_min = min(moire_V(add(scale(A1, i / n), A2, j / n), g, V0, phi)
                   for i in range(n) for j in range(n))
    if grid_min < vmin - 1e-9 * max(1.0, abs(vmin)):
        raise SystemExit("moire minimum is not at a high-symmetry point "
                         f"(grid {grid_min} < candidate {vmin}); unexpected phi/V0.")
    return rmin, vmin


def solve_for_phases(g, phase):
    """Point s with g_0 . s = g_1 . s = phase (then g_2 . s = -2 phase)."""
    det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    return ((phase * g[1][1] - phase * g[0][1]) / det,
            (g[0][0] * phase - g[1][0] * phase) / det)


def wrap(r, L1, L2):
    """Fold r into the centred cell, fractional coordinates in [-1/2, 1/2)."""
    f1, f2 = solve2(L1, L2, r)
    f1 -= math.floor(f1 + 0.5)
    f2 -= math.floor(f2 + 0.5)
    return add(scale(L1, f1), L2, f2)


def ws_inradius(L1, L2):
    best = float("inf")
    for i in range(-3, 4):
        for j in range(-3, 4):
            if i == 0 and j == 0:
                continue
            best = min(best, norm(add(scale(L1, i), L2, j)))
    return 0.5 * best


def closed_shell_counts(L1, L2, n_max=400):
    """Numbers of plane-wave orbitals that close a reciprocal shell of the cell
    (1, 7, 13, 19, 31, 37, ... for a hexagonal cell), as alea's
    reciprocal_shell_table counts them: one orbital per k-point."""
    area = abs(L1[0] * L2[1] - L1[1] * L2[0])
    b1 = (TWO_PI * L2[1] / area, -TWO_PI * L2[0] / area)
    b2 = (-TWO_PI * L1[1] / area, TWO_PI * L1[0] / area)
    # Sign fix: b_i . L_j = 2 pi delta_ij for either orientation of the cell.
    if dot(b1, L1) < 0:
        b1, b2 = scale(b1, -1), scale(b2, -1)
    m = int(math.isqrt(n_max)) + 6
    k2 = sorted(dot(k, k) for k in (add(scale(b1, i), b2, j)
                                    for i in range(-m, m + 1)
                                    for j in range(-m, m + 1)))
    counts, s = [], 0
    while s < len(k2) and s < n_max:
        e = s + 1
        while e < len(k2) and abs(k2[e] - k2[s]) <= 1e-9 * (1.0 + k2[s]):
            e += 1
        counts.append(e)
        s = e
    return counts


# ── the cell for one (nu, N_e) ───────────────────────────────────────────────

class Geometry:
    """Everything position-dependent for one point; shared by both states."""

    def __init__(self, nu: Fraction, ne: int, x: float, kappa: float,
                 phi: float, theta0: float = 0.0):
        if nu.numerator != 1 or not is_loeschian(nu.denominator):
            raise SystemExit(
                f"nu = {nu}: only nu = 1/m with m Loeschian (1, 3, 4, 7, 9, 12, "
                "...) has a triangular crystal commensurate with a hexagonal "
                "moire supercell.  Other fillings need approximant cells.")
        if kappa < 0.0:
            raise SystemExit("kappa must be >= 0 (sign conventions are folded into phi)")
        self.nu, self.ne, self.x, self.kappa = nu, ne, x, kappa
        self.phi, self.theta0 = phi, theta0
        self.m = nu.denominator
        self.M = ne * self.m
        if not is_loeschian(self.M):
            raise SystemExit(f"M = N_e/nu = {self.M} moire cells is not a Loeschian "
                             "number: no hexagonal supercell holds it.")

        self.g = first_shell(x, theta0)
        self.A1, self.A2 = direct_lattice(self.g)
        # Where the minima are depends on phi only, not on V0's size, so kappa = 0
        # (the free-gas edge of the phase diagram) places the crystal on the same
        # sites as any kappa > 0 and just switches the potential off.
        self.rmin, vmin_unit = moire_minimum(self.g, self.A1, self.A2, 1.0, phi)
        self.vmin = kappa * vmin_unit

        # A cell that is a supercell of the moire lattice AND of the crystal.
        choice = None
        for (p, q) in loeschian_pairs(self.M):
            L1, L2 = triangular_superlattice(self.A1, self.A2, p, q)
            for (pd, qd) in loeschian_pairs(self.m):
                D1, D2 = triangular_superlattice(self.A1, self.A2, pd, qd)
                if all(is_int(c) for L in (L1, L2) for c in solve2(D1, D2, L)):
                    choice = (p, q, pd, qd, L1, L2, D1, D2)
                    break
            if choice:
                break
        if choice is None:
            raise SystemExit(f"no hexagonal cell of {self.M} moire cells is also a "
                             f"supercell of the 1/nu = {self.m} crystal; try another N_e.")
        (self.p, self.q, self.pd, self.qd,
         self.L1, self.L2, self.D1, self.D2) = choice

        self.shells = closed_shell_counts(self.L1, self.L2, n_max=max(4 * ne, 64))
        self.area = abs(self.L1[0] * self.L2[1] - self.L1[1] * self.L2[0])
        self.rs = math.sqrt(self.area / (ne * math.pi))
        self.r_ws = ws_inradius(self.L1, self.L2)
        self.gmag = norm(self.g[0])
        self.E_M = 1.0 / (x * x)                 # hbar^2/(m* a_M^2) in Ha*

    @property
    def fluid_ok(self):
        return self.ne in self.shells

    def sites(self, shift=(0.0, 0.0)):
        """Crystal sites: r_min + shift + (D-lattice), one per electron, wrapped."""
        out = []
        n = 2 * self.ne + 2
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                r = wrap(add(add(self.rmin, shift), add(scale(self.D1, i), self.D2, j)),
                         self.L1, self.L2)
                if all(norm(wrap(add(r, s, -1.0), self.L1, self.L2)) > 1e-8 * self.x
                       for s in out):
                    out.append(r)
                if len(out) == self.ne:
                    return out
        raise SystemExit(f"generated {len(out)} crystal sites, expected {self.ne}")

    def alpha_seed(self):
        """Gaussian width exp(-alpha r^2) from a harmonic estimate: the moire well's
        curvature at the minimum, k_M = -V_min |g|^2 / 2 (Hessian of
        2V0 sum cos(g.r+phi) at a point where every phase is equal), plus the
        Einstein cage of the Wigner crystal, k_W = (2 C / rs^1.5)^2.  With m* = 1
        the oscillator ground state has alpha = sqrt(k)/2.  Only a seed: the
        optimizer's nodal stage moves it."""
        k_M = -0.5 * self.vmin * self.gmag ** 2
        k_W = (2.0 * EINSTEIN_C / self.rs ** 1.5) ** 2
        return 0.5 * math.sqrt(k_M + k_W)

    def j1_seed(self):
        """First-order perturbation theory, psi ~ 1 - sum_G V_G e^{iG.r}/(G^2/2):
        u = -(4 V0/|g|^2) sum_j cos(g_j.r + phi)
          = a sum cos(g.r) + b sum sin(g.r), a = -c cos phi, b = +c sin phi."""
        c = 4.0 * self.kappa / self.gmag ** 2
        return -c * math.cos(self.phi), c * math.sin(self.phi)


# ── YAML emission ────────────────────────────────────────────────────────────

def f(v):
    return repr(float(v))


def vec(v):
    return f"[{f(v[0])}, {f(v[1])}]"


def seed_from(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:12], 16) or 1


def qmc_block(geo, state, args, name):
    """Optimizer stage(s) + VMC (+ optional DMC chain).

    The imaginary time step is set from the crystal's localisation, identical
    for both states of a point: the DMC time-step bias must cancel in
    E_fluid - E_crystal, so the two runs share tau.  Scale: the HEG2D rs=30
    configs ran tau*omega = 0.006, 0.003, 0.0015 with omega = 2 alpha."""
    alpha = geo.alpha_seed()
    vmc_step = min(0.17 * geo.rs ** 2, 1.0 / (2.0 * alpha))
    nb = max(24, int(math.ceil(8 * math.sqrt(geo.M))))
    # rmax sits just inside the WS inradius: alea rejects rmax > its own inradius
    # with no tolerance, and its value can differ from ours in the last bit
    # (nu = 1/4, x = 27 died at startup on exactly this).
    r_max = 0.999 * geo.r_ws
    est = [f"      - {{ type: gofr, rmax: {f(r_max)}, n_bins: 100 }}",
           f"      - {{ type: density, rmax: {f(r_max)}, n_bins: {nb} }}"]

    def sampling(samples, eq, acc, steps, sub, tau, walk):
        return [f"    random_walk: {walk}",
                "    device: cpu",
                "    threads: 0",
                f"    samples: {samples}",
                f"    equilibration_blocks: {eq}",
                f"    accumulation_blocks: {acc}",
                f"    steps_per_block: {steps}",
                f"    sub_steps: {sub}",
                f"    time_step: {f(tau)}"]

    out = ["qmc:"]
    if args.opt_iters > 0:
        stages = (["nodal", "jastrow"] if state == "crystal" else ["jastrow"])
        for st in stages:
            out += ["  - method: optimizer",
                    "    type: linear",
                    f"    parameters: {st}",
                    f"    max_iterations: {args.opt_iters}"]
            out += sampling(args.samples, 20, args.opt_blocks, 10, 2, vmc_step,
                            "metropolis")
            out.append("")
    out += ["  - method: vmc"]
    out += sampling(args.samples, 20, args.vmc_blocks, 10, 2, vmc_step, "metropolis")
    out += ["    estimators:"] + est + [""]
    # Block length: a fixed imaginary time (--dmc-block-time) makes the steps per
    # block, hence the cost, grow as alpha (tau ~ 1/alpha): deep-moire points
    # cost ~20x the shallow ones, beyond a typical 72 h batch-job limit.  With
    # --dmc-block-omega the block is omega*T_block in the crystal's own time
    # unit 1/omega (the scale its correlation time follows), so every point
    # does the same steps per block, (omega*T)/(tau*omega).
    block_time = (args.dmc_block_omega / (2.0 * alpha)
                  if getattr(args, "dmc_block_omega", None) else args.dmc_block_time)
    for k, c in enumerate(args.dmc_tau_omega):
        tau = c / (2.0 * alpha)
        steps = max(50, int(round(block_time / tau)))
        out += ["  - method: dmc"]
        out += sampling(args.samples, 20 if k == 0 else 5, args.dmc_blocks, steps, 2,
                        tau, "drift_diffusion")
        out += [f"    walkers_out: {name}_dmc{k}.h5",
                "    estimators:"] + est + [""]
    return out


def build_yaml(geo, state, args, name):
    if state == "fluid" and not geo.fluid_ok:
        raise SystemExit(f"N_e = {geo.ne} does not close a reciprocal shell of this "
                         f"cell (closed: {geo.shells[:12]}); the fluid determinant "
                         "would be degenerate.")
    a1, b1 = geo.j1_seed()
    if state == "crystal":
        # The Gaussians already localise the crystal on the minima; the moire
        # Jastrow only refines the shape, so start it at a tenth of the
        # perturbative value rather than double-counting the localisation.
        a1, b1 = 0.1 * a1, 0.1 * b1
    L_drum = min((2.0 if state == "crystal" else 3.0) * geo.rs, 0.95 * geo.r_ws)
    alpha0 = 1.0 / (9.0 * L_drum ** 2)       # c*Gamma/(3L^2), c = 1, Gamma = 1/3

    lines = [
        "# " + "=" * 77,
        f"# MOIRE {state.upper()}  nu = {geo.nu}  N_e = {geo.ne}  M = {geo.M}  "
        f"x = a_M/a_B* = {geo.x:g}  kappa = V0/Ha* = {geo.kappa:g}  "
        f"phi = {math.degrees(geo.phi):g} deg",
        "#",
        f"# Generated by tools/moire_cells.py -- edit the generator, not this file.",
        f"# rs = {geo.rs:.6g}   kappa x^2 = V0/E_M = {geo.kappa * geo.x ** 2:.6g}   "
        f"cell = ({geo.p},{geo.q}) superlattice of the moire lattice",
        f"# moire minimum at {vec(geo.rmin)}, V_min = {geo.vmin:.6g} Ha*;  "
        f"crystal sublattice ({geo.pd},{geo.qd}) of index {geo.m}",
        f"# WS inradius {geo.r_ws:.6g};  closed shells {geo.shells[:10]}",
        "# " + "=" * 77,
        "project:",
        f"  name: {name}",
        f"  seed: {seed_from(name)}",
        "  log: true",
        "  output: true",
        "",
        "system:",
        "  lattice:",
        "    type: periodic",
        "    dimension: 2",
        f"    vectors: [{vec(geo.L1)}, {vec(geo.L2)}]",
        f"    ewald_tol: {args.ewald_tol:g}",
        "  particles:",
        "    - { name: e, group: electron_group, type: fermions, mass: 1.0, "
        f"charge: -1, up: {geo.ne}, down: 0, initialization: uniform }}",
        "",
        "wavefunction:",
        "  - slater:",
        "      group: electron_group",
    ]
    if state == "crystal":
        lines += ["      orbitals:",
                  "        type: gaussian",
                  f"        alpha: {f(geo.alpha_seed())}",
                  "        alpha_fixed: false",
                  "        centers:"]
        lines += [f"          - {vec(s)}" for s in geo.sites(args.shift)]
    else:
        lines += ["      orbitals: { type: plane_wave }"]
    lines += [
        "  - jastrow:",
        "      type: drummond",
        "      pair: [electron_group, electron_group]",
        f"      params: [{f(alpha0)}]",
        "      is_fixed: [false]",
        "      truncation: 3",
        f"      cutoff: {f(L_drum)}",
        "      cutoff_fixed: true",
        "  - moire_jastrow:",
        "      group: electron_group",
        "      shells: 1",
        f"      params: [{f(a1)}, {f(b1)}]",
        # No potential, nothing to follow: pinned at zero so kappa = 0 is exactly
        # the uniform-gas trial state.
        "      is_fixed: [true, true]" if geo.kappa == 0.0 else
        "      is_fixed: [false, false]",
        "",
        "hamiltonian:",
        "  terms:",
        "    - { type: coulomb, pair: [electron_group, electron_group], epsilon: 1.0 }",
        "    - type: moire",
        "      group: [electron_group]",
        "      V0: 0.0",
        f"      Vtilde: {f(geo.kappa)}",
        f"      phi: {f(geo.phi)}",
        f"      a_M: {f(geo.x)}",
        f"      theta0: {f(geo.theta0)}",
        "",
    ]
    lines += qmc_block(geo, state, args, name)
    return "\n".join(lines) + "\n"


def meta(geo, state, args, name):
    return {
        "name": name, "state": state, "nu": str(geo.nu), "ne": geo.ne, "M": geo.M,
        "x": geo.x, "kappa": geo.kappa, "phi_deg": math.degrees(geo.phi),
        "theta0": geo.theta0, "rs": geo.rs, "kappa_x2": geo.kappa * geo.x ** 2,
        "E_M": geo.E_M, "cell": [geo.L1, geo.L2], "r_ws": geo.r_ws,
        "moire_min": geo.rmin, "V_min": geo.vmin,
        "alpha_seed": geo.alpha_seed(), "j1_seed": geo.j1_seed(),
        "shift": list(args.shift),
        "sites": geo.sites(args.shift) if state == "crystal" else None,
    }


# ── self-test ────────────────────────────────────────────────────────────────

def selftest():
    fails = 0

    def check(cond, msg):
        nonlocal fails
        if not cond:
            fails += 1
            print("FAIL:", msg)

    for nu, ne in [(Fraction(1, 3), 13), (Fraction(1, 3), 19), (Fraction(1, 3), 37),
                   (Fraction(1, 4), 13), (Fraction(1, 4), 19), (Fraction(1, 7), 13)]:
        for theta0 in (0.0, 0.37):
            geo = Geometry(nu, ne, x=12.0, kappa=0.03, phi=math.radians(60.0),
                           theta0=theta0)
            A = geo.area
            check(abs(A - geo.M * SQRT3 / 2 * geo.x ** 2) < 1e-8 * A,
                  f"{nu} {ne}: cell area != M moire cells")
            for L in (geo.L1, geo.L2):
                for gj in geo.g:
                    check(is_int(dot(gj, L) / TWO_PI), f"{nu} {ne}: g.L not 2pi Z")
            sites = geo.sites()
            check(len(sites) == ne, f"{nu} {ne}: site count")
            for s in sites:
                check(abs(moire_V(s, geo.g, geo.kappa, geo.phi) - geo.vmin)
                      < 1e-9, f"{nu} {ne}: site not on a minimum")
            dmin = min(norm(wrap(add(a, b, -1.0), geo.L1, geo.L2))
                       for i, a in enumerate(sites) for b in sites[i + 1:])
            check(abs(dmin - math.sqrt(geo.m) * geo.x) < 1e-8 * geo.x,
                  f"{nu} {ne}: nearest-neighbour spacing != sqrt(1/nu) a_M")
            check(abs(geo.rs - geo.x * math.sqrt(SQRT3 / (2 * math.pi * float(nu))))
                  < 1e-10 * geo.rs, f"{nu} {ne}: rs formula")
            check(geo.fluid_ok, f"{nu} {ne}: fluid shell should close")
    check(closed_shell_counts((1.0, 0.0), (0.5, SQRT3 / 2))[:7] == [1, 7, 13, 19, 31, 37, 43],
          "hexagonal closed shells")
    geo = Geometry(Fraction(1, 3), 19, x=12.0, kappa=0.03, phi=math.radians(60.0))
    check(abs(geo.vmin + 6 * 0.03) < 1e-12, "phi=60: V_min = -6 V0")
    free = Geometry(Fraction(1, 3), 19, x=12.0, kappa=0.0, phi=math.radians(60.0))
    check(free.vmin == 0.0, "kappa=0: no potential")
    check(all(norm(add(a, b, -1.0)) < 1e-9 for a, b in zip(free.sites(), geo.sites())),
          "kappa=0 puts the crystal on the same sites as kappa>0")
    print("selftest:", "PASS" if fails == 0 else f"{fails} failure(s)")
    return fails == 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nu", type=Fraction, help="filling, e.g. 1/3")
    ap.add_argument("--ne", type=int, help="number of electrons")
    ap.add_argument("--x", type=float, help="a_M / a_B*")
    ap.add_argument("--kappa", type=float, help="V0 / Ha* (moire amplitude)")
    ap.add_argument("--phi-deg", type=float, default=60.0)
    ap.add_argument("--theta0", type=float, default=0.0, help="moire orientation (rad)")
    ap.add_argument("--state", choices=["crystal", "fluid"])
    ap.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0),
                    help="rigid crystal displacement (Cartesian, a_B*)")
    ap.add_argument("--out", type=Path, help="output YAML path")
    ap.add_argument("--name", help="project name (default: from --out)")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--opt-iters", type=int, default=12)
    ap.add_argument("--opt-blocks", type=int, default=50)
    ap.add_argument("--vmc-blocks", type=int, default=100)
    ap.add_argument("--dmc-tau-omega", type=float, nargs="*", default=[],
                    help="DMC chain as tau*omega values (e.g. 0.006 0.003 0.0015); "
                         "empty = VMC only")
    ap.add_argument("--dmc-blocks", type=int, default=80)
    ap.add_argument("--dmc-block-time", type=float, default=400.0,
                    help="imaginary time per DMC block (Ha*^-1)")
    ap.add_argument("--dmc-block-omega", type=float, default=None,
                    help="DMC block length as omega*T (omega = 2 alpha); overrides "
                         "--dmc-block-time and makes the cost independent of alpha")
    ap.add_argument("--ewald-tol", type=float, default=1e-6)
    ap.add_argument("--list", action="store_true",
                    help="list N_e <= 100 valid for --nu (moire + crystal + fluid)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if selftest() else 1
    if args.list:
        if args.nu is None:
            ap.error("--list needs --nu")
        rows = []
        for ne in range(1, 101):
            try:
                geo = Geometry(args.nu, ne, x=10.0, kappa=0.01,
                               phi=math.radians(args.phi_deg))
            except SystemExit:
                continue
            rows.append((ne, geo.M, geo.fluid_ok))
        print(f"nu = {args.nu}:  N_e  M(moire cells)  fluid shell closed")
        for ne, M, ok in rows:
            print(f"  {ne:4d}  {M:5d}   {'yes' if ok else 'no'}")
        return 0

    for req in ("nu", "ne", "x", "kappa", "state", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required")
    geo = Geometry(args.nu, args.ne, args.x, args.kappa,
                   math.radians(args.phi_deg), args.theta0)
    name = args.name or args.out.stem
    text = build_yaml(geo, args.state, args, name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta(geo, args.state, args, name), indent=1), encoding="utf-8")
    print(f"wrote {args.out}  (rs = {geo.rs:.4g}, kappa x^2 = {geo.kappa * geo.x**2:.4g}, "
          f"alpha seed = {geo.alpha_seed():.4g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
