#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""moire_cells.py -- alea configs for electrons in a triangular moire superlattice.

For one point of the phase diagram (filling nu, electron count N_e, interaction
strength x = a_M/a_B*, moire depth kappa = V_0/Ha*, shape phi) this writes an
alea YAML for one competing state -- the Wigner crystal pinned to the moire
minima, or the liquid -- plus a JSON sidecar with the geometry, so the two
states of a point are built in the SAME periodic cell and differ only in the
determinant.  It also reports the plane-wave shell structure of a cell, at Gamma
and on a twist grid, and the electron numbers that are valid for a filling.

Units are alea's effective atomic units (a_B*, Ha*): m* = 1, epsilon = 1, so
the moire period in the config IS x and the modulation amplitude IS kappa.

Hamiltonian (Wu, Lovorn & MacDonald, PRL 121, 026402 (2018)):

    H = sum_i -1/2 nabla_i^2 + sum_i V_M(r_i) + sum_{i<j} 1/r_ij
    V_M(r) = 2 V_0 sum_{j=1..3} cos(g_j . r + phi),   |g_j| = 4 pi/(sqrt3 a_M)

with no constant term (the code's V0 offset is written as 0).  phi = 225 deg is
the WSe2/WS2 potential of Zhang, Yuan & Fu (2020) and Geier et al. (2025), whose
-2V_0 with psi = 45 deg it equals; its deep wells sit at -4.24 V_0 and its
shallow wells at -1.55 V_0.  A material is given either by x and kappa (or x and
lambda = V_0/E_M, kappa = lambda/x^2), or by m*, a_M, V_0 and epsilon:
    x = a_M m*/(epsilon a_0),   lambda = V_0/E_M,   E_M = hbar^2/(m* a_M^2).

Cells.  `hex`: the hexagonal (p, q) supercells of the moire lattice, M = p^2 +
pq + q^2 moire cells.  `rect`: n x m copies of the rectangle a_M x sqrt3 a_M
(two moire cells), M = 2nm, written axis-aligned (theta0 = -30 deg by default) so
alea takes its orthorhombic fast path.

Crystals (the sites are the moire minima, found at the three high-symmetry
points and confirmed on a dense grid):
  * nu = 1/m, m Loeschian (1, 3, 4, 7, ...): the triangular sublattice of index
    m (sqrt3 x sqrt3 at 1/3, 2 x 2 at 1/4);
  * nu = 2/3: the honeycomb left by removing the sqrt3 x sqrt3 sublattice from
    the triangular lattice of minima.
The cell must be a supercell of the crystal's lattice.

Spin.  `polarized`: one group, `electron_group`.  `unpolarized`: groups `e_up`
and `e_dn` with N_e/2 each (backflow needs one spin per group); the crystal is
the collinear stripe state, alternate rows of the site lattice up and down,
which needs a cell commensurate with the stripes (it is refused otherwise).

What the tool guarantees, and checks before writing anything:
  * the cell is a supercell of the moire lattice (integer combination of the
    moire vectors A1, A2), so V_M and the moire Jastrow are periodic in it;
  * it holds M = N_e/nu moire cells, i.e. the filling is exactly nu;
  * the crystal sites sit on the deepest moire minima and form the pattern above;
  * the liquid: every spin group closes a reciprocal shell at Gamma, or at the
    requested twist no two plane waves tie at the occupation edge (alea refuses
    both otherwise); an OPEN shell at Gamma is written only with an explicit
    choice of its real orbitals (--pw-pick, or one file per symmetry class of
    choices with --pw-variants).

Usage
-----
    py -3 tools/moire_cells.py --selftest
    py -3 tools/moire_cells.py --list --nu 1/3 --spin unpolarized --cell rect
    py -3 tools/moire_cells.py --report shells.md --nu 1/3 --spin unpolarized \\
        --cells hex rect --n-max 72 --twist-grid 4 --phi-deg 225
    py -3 tools/moire_cells.py --nu 1/3 --ne 19 --x 15 --kappa 0.02 \\
        --phi-deg 60 --state crystal --out runs/moire/demo/crystal_N19.yaml
    py -3 tools/moire_cells.py --nu 2/3 --ne 18 --material geier2025 \\
        --inv-epsilon 0.11 --phi-deg 225 --state liquid --pw-variants \\
        --out runs/armA/liquid_e0p11.yaml

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import cmath
import functools
import hashlib
import itertools
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twist_grid import twist_grid  # noqa: E402  (the grid alea's twist runs use)

SQRT3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi

# Einstein estimate of a 2-D Wigner crystal's Gaussian width, alpha_E = C/rs^1.5,
# the same number the configs/heg2d_crystal_*.yaml headers use as their scale.
EINSTEIN_C = 0.4

# CODATA 2018: Bohr radius and Hartree energy.
A0_NM = 0.0529177210903
HA_MEV = 27211.386245988

# Material presets: effective mass (electron masses), moire period, depth.
MATERIALS = {
    # Geier, Nazaryan, Zaklama & Fu, PRB 112, 045119 (2025): WSe2/WS2.
    "geier2025": {"mstar": 0.35, "a_M_nm": 8.031, "V0_meV": 15.0},
}

# alea's tie test for plane-wave occupations (make_twisted_plane_wave_orbitals).
TIE_TOL = 1e-9


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


def reciprocal_basis(L1, L2):
    """b_i . L_j = 2 pi delta_ij for the vectors AS GIVEN: the basis alea reads
    lattice.twist and integer plane-wave k in (SimulationCell::given_reciprocal)."""
    det = L1[0] * L2[1] - L1[1] * L2[0]
    return ((TWO_PI * L2[1] / det, -TWO_PI * L2[0] / det),
            (-TWO_PI * L1[1] / det, TWO_PI * L1[0] / det))


# ── moire geometry ───────────────────────────────────────────────────────────

def first_shell(a_M, theta0):
    """The three g_j, exactly as alea's moire_reciprocal_vectors builds them."""
    gm = 4.0 * math.pi / (SQRT3 * a_M)
    return tuple((gm * math.cos(theta0 + TWO_PI / 3 * i),
                  gm * math.sin(theta0 + TWO_PI / 3 * i)) for i in range(3))


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


@functools.lru_cache(maxsize=None)
def moire_minimum(g, A1, A2, V0, phi):
    """Global minimum of V_M in one moire cell (cached: the grid check is slow).

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


# ── plane-wave shells ────────────────────────────────────────────────────────

def lowest_waves(L1, L2, n_need, theta=(0.0, 0.0)):
    """[(|k+theta|^2, (n1, n2))] for the n_need + 1 lowest k = n1 b1 + n2 b2 + theta
    (and their whole last shell), sorted by |k+theta|^2 then by index."""
    b1, b2 = reciprocal_basis(L1, L2)
    area_b = abs(b1[0] * b2[1] - b1[1] * b2[0])
    h_min = area_b / max(norm(b1), norm(b2))     # spacing of lattice lines
    t_norm = norm(theta)
    K = int(math.isqrt(4 * n_need + 16)) + 4
    while True:
        out = []
        for i in range(-K, K + 1):
            for j in range(-K, K + 1):
                q = add(add(scale(b1, i), b2, j), theta)
                out.append((dot(q, q), (i, j)))
        out.sort()
        r_safe = K * h_min - t_norm
        last = out[min(n_need, len(out) - 1)][0]
        if r_safe > 0 and last * (1 + 1e-6) < r_safe * r_safe * 0.99:
            return out
        K *= 2


def gamma_shells(L1, L2, n_need):
    """Closed shells at Gamma: [[(n1, n2), ...] per shell], through the shell
    that contains state n_need (counting one state per k-point, as alea's
    reciprocal_shell_table does)."""
    waves = lowest_waves(L1, L2, n_need)
    shells, s = [], 0
    while s < len(waves):
        e = s + 1
        while e < len(waves) and waves[e][0] - waves[s][0] <= TIE_TOL * (1.0 + waves[s][0]):
            e += 1
        shells.append([w[1] for w in waves[s:e]])
        if sum(len(x) for x in shells) > n_need:
            break
        s = e
    return shells


def closed_shell_counts(L1, L2, n_max=400):
    """Numbers of plane-wave orbitals that close a reciprocal shell of the cell
    (1, 7, 13, 19, 31, 37, ... for a hexagonal cell)."""
    counts, c = [], 0
    for sh in gamma_shells(L1, L2, n_max):
        c += len(sh)
        counts.append(c)
    return [x for x in counts if x <= n_max] or counts[:1]


def gamma_occupation(L1, L2, n):
    """How n plane waves fill the Gamma shells: (full shells, states in them,
    states taken from the next shell, its size).  Closed when the third is 0."""
    shells = gamma_shells(L1, L2, n)
    filled, k = 0, 0
    for sh in shells:
        if filled + len(sh) > n:
            return k, filled, n - filled, len(sh)
        filled += len(sh)
        k += 1
        if filled == n:
            return k, filled, 0, 0
    raise AssertionError("gamma_shells returned too few states")


def describe_gamma(L1, L2, n):
    k, filled, extra, size = gamma_occupation(L1, L2, n)
    return "closed" if extra == 0 else f"open: {filled} + {extra}/{size}"


def twist_tie(L1, L2, n, t):
    """True when the n lowest |k + theta| tie with the next one at twist t
    (fractional, in the cell's reciprocal basis): alea then refuses the twist."""
    b1, b2 = reciprocal_basis(L1, L2)
    theta = add(scale(b1, t[0]), b2, t[1])
    waves = lowest_waves(L1, L2, n, theta)
    return abs(waves[n][0] - waves[n - 1][0]) <= TIE_TOL * (1.0 + waves[n][0])


def grid2d(ns, offset):
    """The twists `tools/twist_grid.py make --grid ns --offset offset` writes for a
    2-D cell (one value applies to both axes), t and -t merged: [(t, weight)]."""
    ns = list(ns) if len(ns) == 2 else list(ns) * 2
    return twist_grid(ns, offset, True)


def real_orbitals(shell):
    """The real plane waves of one Gamma shell: [(n1, n2, is_sin)].  k = 0 gives
    the constant; each +-k pair gives cos and sin of its representative (first
    nonzero index positive), pairs ordered by angle."""
    if shell == [(0, 0)]:
        return [(0, 0, False)]
    reps = sorted({k if (k[0] > 0 or (k[0] == 0 and k[1] > 0)) else (-k[0], -k[1])
                   for k in shell})
    out = []
    for k in reps:
        out += [(k[0], k[1], False), (k[0], k[1], True)]
    return out


# ── cells and site patterns ──────────────────────────────────────────────────

class Cell:
    """A periodic supercell of the moire lattice."""

    def __init__(self, kind, shape, L1, L2, A1, A2):
        self.kind, self.shape, self.L1, self.L2 = kind, shape, L1, L2
        # Integer moire coordinates of the cell vectors: L = a A1 + b A2.
        a, b = solve2(A1, A2, L1)
        c, d = solve2(A1, A2, L2)
        if not all(is_int(v) for v in (a, b, c, d)):
            raise AssertionError("cell is not a supercell of the moire lattice")
        self.ab = (round(a), round(b))
        self.cd = (round(c), round(d))
        self.M = abs(self.ab[0] * self.cd[1] - self.ab[1] * self.cd[0])

    def label(self):
        return f"{self.kind} {self.shape[0]}x{self.shape[1]}" if self.kind == "rect" \
            else f"hex ({self.shape[0]},{self.shape[1]})"

    @property
    def aspect(self):
        """Longest over shortest cell vector: 1 for hex, >= 1 for rect.  A long
        thin cell has a small Wigner-Seitz inradius for its area, so its
        finite-size error is larger and anisotropic."""
        a, b = norm(self.L1), norm(self.L2)
        return max(a, b) / min(a, b)

    def moire_points(self):
        """Integer (i, j) of the M moire lattice points R = i A1 + j A2 in the
        cell, one per residue class modulo the cell, deterministic order."""
        (a, b), (c, d) = self.ab, self.cd
        det = a * d - b * c
        I = [0, a, c, a + c]
        J = [0, b, d, b + d]
        seen, out = set(), []
        for i in range(min(I), max(I) + 1):
            for j in range(min(J), max(J) + 1):
                key = ((d * i - c * j) % det, (-b * i + a * j) % det)
                if key not in seen:
                    seen.add(key)
                    out.append((i, j))
        if len(out) != self.M:
            raise AssertionError(f"found {len(out)} moire points, expected {self.M}")
        return sorted(out)


def hex_cells(A1, A2, M):
    out = []
    for p, q in loeschian_pairs(M):
        L1, L2 = triangular_superlattice(A1, A2, p, q)
        out.append(Cell("hex", (p, q), L1, L2, A1, A2))
    return out


def rect_cells(A1, A2, M):
    """n x m rectangles a_M x sqrt3 a_M: L1 = n A1, L2 = m (2 A2 - A1), the most
    compact (smallest aspect ratio) first."""
    out = []
    if M % 2:
        return out
    R2 = add(scale(A2, 2.0), A1, -1.0)
    for n in range(1, M // 2 + 1):
        if (M // 2) % n == 0:
            m = (M // 2) // n
            out.append(Cell("rect", (n, m), scale(A1, n), scale(R2, m), A1, A2))
    return sorted(out, key=lambda c: (c.aspect, c.shape))


class Pattern:
    """Which moire minima a crystal occupies: the moire points R that lie on the
    sublattice D (index m) or off it.  nu = 1/m on it, 2/3 off sqrt3 x sqrt3."""

    def __init__(self, nu: Fraction, pd, qd, on_sublattice: bool):
        self.nu, self.pd, self.qd, self.on = nu, pd, qd, on_sublattice
        self.m = pd * pd + pd * qd + qd * qd

    @property
    def bravais(self):
        return self.on

    def d_coords(self, i, j):
        """(u, v) with i A1 + j A2 = u D1 + v D2 when integer, else None."""
        p, q, m = self.pd, self.qd, self.m
        u_num, v_num = (p + q) * i + q * j, -q * i + p * j
        if u_num % m or v_num % m:
            return None
        return u_num // m, v_num // m

    def occupies(self, i, j):
        return (self.d_coords(i, j) is not None) == self.on

    def name(self):
        if self.on:
            return {1: "triangular (every minimum)", 3: "sqrt3 x sqrt3", 4: "2 x 2"}.get(
                self.m, f"triangular, index {self.m}")
        return "honeycomb (sqrt3 x sqrt3 vacancies)"


def patterns_for(nu: Fraction):
    if nu.numerator == 1 and is_loeschian(nu.denominator):
        return [Pattern(nu, p, q, True) for p, q in loeschian_pairs(nu.denominator)]
    if nu == Fraction(2, 3):
        return [Pattern(nu, 1, 1, False)]
    raise SystemExit(
        f"nu = {nu}: implemented are nu = 1/m with m Loeschian (1, 3, 4, 7, 9, 12, "
        "...) and nu = 2/3 (honeycomb).  nu = 1/2 (stripes) is not here yet.")


def stripe_parity(pattern, cell):
    """For a Bravais crystal, a parity vector c with c . (u, v) even for both cell
    vectors in D coordinates: the stripes (alternate rows) then tile the cell.
    None when no such c exists."""
    lu = [pattern.d_coords(*cell.ab), pattern.d_coords(*cell.cd)]
    if any(v is None for v in lu):
        return None
    for c in ((0, 1), (1, 0), (1, 1)):
        if all((c[0] * u + c[1] * v) % 2 == 0 for u, v in lu):
            return c
    return None


# ── the geometry of one (nu, N_e, cell) ──────────────────────────────────────

class Geometry:
    """Everything position-dependent for one point; shared by both states."""

    def __init__(self, nu: Fraction, ne: int, x: float, kappa: float,
                 phi: float, theta0=None, cell_kind: str = "hex",
                 spin: str = "polarized", shape=None):
        if kappa < 0.0:
            raise SystemExit("kappa must be >= 0 (sign conventions are folded into phi)")
        if spin not in ("polarized", "unpolarized"):
            raise SystemExit(f"spin '{spin}': expected polarized or unpolarized")
        if spin == "unpolarized" and ne % 2:
            raise SystemExit(f"N_e = {ne} is odd: an unpolarized state needs N_e/2 per spin")
        M = Fraction(ne) / nu
        if M.denominator != 1:
            raise SystemExit(f"N_e = {ne} at nu = {nu} is not a whole number of moire cells")
        self.nu, self.ne, self.x, self.kappa, self.phi = nu, ne, x, kappa, phi
        self.spin, self.cell_kind = spin, cell_kind
        self.M = int(M)
        if theta0 is None:
            theta0 = -math.pi / 6 if cell_kind == "rect" else 0.0
        self.theta0 = theta0
        self.n_spin = [ne // 2, ne // 2] if spin == "unpolarized" else [ne]

        self.g = first_shell(x, theta0)
        self.A1, self.A2 = direct_lattice(self.g)
        # Where the minima are depends on phi only, not on V0's size, so kappa = 0
        # (the free-gas edge of the phase diagram) places the crystal on the same
        # sites as any kappa > 0 and just switches the potential off.
        self.rmin, vmin_unit = moire_minimum(self.g, self.A1, self.A2, 1.0, phi)
        self.vmin_unit = vmin_unit
        self.vmin = kappa * vmin_unit

        if cell_kind == "hex":
            if not is_loeschian(self.M):
                raise SystemExit(f"M = N_e/nu = {self.M} moire cells is not a Loeschian "
                                 "number: no hexagonal supercell holds it.")
            cells = hex_cells(self.A1, self.A2, self.M)
        elif cell_kind == "rect":
            cells = rect_cells(self.A1, self.A2, self.M)
            if not cells:
                raise SystemExit(f"M = {self.M} moire cells is odd: no a_M x sqrt3 a_M "
                                 "rectangle (two moire cells) tiles it.")
        else:
            raise SystemExit(f"cell '{cell_kind}': expected hex or rect")
        if shape is not None:
            cells = [c for c in cells if c.shape == tuple(shape)]
            if not cells:
                raise SystemExit(f"no {cell_kind} cell of shape {tuple(shape)} holds "
                                 f"{self.M} moire cells")

        # A cell that is a supercell of the crystal (and of its stripes).
        choice, why = None, []
        for cell in cells:
            for pat in patterns_for(nu):
                if pat.d_coords(*cell.ab) is None or pat.d_coords(*cell.cd) is None:
                    why.append(f"{cell.label()}: not a supercell of the {pat.name()} lattice")
                    continue
                parity = None
                if spin == "unpolarized":
                    if not pat.bravais:
                        raise SystemExit(f"nu = {nu}: an unpolarized {pat.name()} crystal "
                                         "is not implemented (stripes need a Bravais "
                                         "site lattice); use --spin polarized.")
                    parity = stripe_parity(pat, cell)
                    if parity is None:
                        why.append(f"{cell.label()}: the spin stripes do not tile it")
                        continue
                choice = (cell, pat, parity)
                break
            if choice:
                break
        if choice is None:
            raise SystemExit(f"N_e = {ne}, nu = {nu}, {spin}, {cell_kind}: no valid cell ("
                             + "; ".join(why) + ")")
        self.cell, self.pattern, self.parity = choice
        self.L1, self.L2 = self.cell.L1, self.cell.L2
        self.p, self.q = self.cell.shape
        self.pd, self.qd, self.m = self.pattern.pd, self.pattern.qd, self.pattern.m

        self.area = abs(self.L1[0] * self.L2[1] - self.L1[1] * self.L2[0])
        self.rs = math.sqrt(self.area / (ne * math.pi))
        self.r_ws = ws_inradius(self.L1, self.L2)
        self.gmag = norm(self.g[0])
        self.E_M = 1.0 / (x * x)                 # hbar^2/(m* a_M^2) in Ha*
        self.shells = closed_shell_counts(self.L1, self.L2, n_max=max(4 * ne, 64))
        self._sites = self._build_sites()

    # -- crystal -------------------------------------------------------------

    def _build_sites(self):
        """[(position, spin)] in deterministic order; spin 0 = up, 1 = down."""
        out = []
        for i, j in self.cell.moire_points():
            if not self.pattern.occupies(i, j):
                continue
            r = wrap(add(self.rmin, add(scale(self.A1, i), self.A2, j)), self.L1, self.L2)
            s = 0
            if self.parity is not None:
                u, v = self.pattern.d_coords(i, j)
                s = (self.parity[0] * u + self.parity[1] * v) % 2
            out.append((r, s))
        if len(out) != self.ne:
            raise AssertionError(f"generated {len(out)} crystal sites, expected {self.ne}")
        if self.spin == "unpolarized" and sum(s for _, s in out) != self.ne // 2:
            raise AssertionError("stripes do not split the sites in halves")
        return out

    def sites(self, shift=(0.0, 0.0), spin=None):
        """Crystal sites (one per electron), wrapped; one spin's when given."""
        return [wrap(add(r, shift), self.L1, self.L2)
                for r, s in self._sites if spin is None or s == spin]

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

    # -- liquid --------------------------------------------------------------

    @property
    def fluid_ok(self):
        """Every spin group closes a Gamma shell."""
        return all(n in self.shells for n in self.n_spin)

    def open_shell(self):
        """None when every group closes a Gamma shell; else (closed orbitals,
        open-shell orbitals, how many of the latter each group takes) -- the
        groups have equal counts, so one choice serves both."""
        n = self.n_spin[0]
        k, filled, extra, size = gamma_occupation(self.L1, self.L2, n)
        if extra == 0:
            return None
        shells = gamma_shells(self.L1, self.L2, n)
        closed = [o for sh in shells[:k] for o in real_orbitals(sh)]
        return closed, real_orbitals(shells[k]), extra

    def twist_ok(self, t):
        return not any(twist_tie(self.L1, self.L2, n, t) for n in self.n_spin)

    # -- symmetry of open-shell choices ---------------------------------------

    def symmetry_ops(self):
        """[(O, R)]: r -> O (r - c) + c + R, with c the deep minimum, O in the
        point group of V_M about c that maps the cell lattice to itself, and R a
        moire lattice vector (one per moire cell): the symmetries of H in this
        cell, up to cell translations."""
        c = self.rmin
        rng = random.Random(7)
        probes = [(rng.uniform(-1, 1) * self.x, rng.uniform(-1, 1) * self.x) for _ in range(6)]
        # The twelve operations of C6v: rotations by 60k deg, and reflections
        # across the lines at 30k deg, [[cos 2a, sin 2a], [sin 2a, -cos 2a]].
        ops = []
        for k in range(6):
            a = math.pi / 3 * k
            ops.append(((math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))))
            ops.append(((math.cos(a), math.sin(a)), (math.sin(a), -math.cos(a))))
        keep = []
        for O in ops:
            def apply(v, O=O):
                return (O[0][0] * v[0] + O[0][1] * v[1], O[1][0] * v[0] + O[1][1] * v[1])
            if any(abs(moire_V(add(c, apply(add(r, c, -1.0))), self.g, 1.0, self.phi)
                       - moire_V(r, self.g, 1.0, self.phi)) > 1e-9 for r in probes):
                continue
            if not all(is_int(v, 1e-7) for L in (self.L1, self.L2)
                       for v in solve2(self.L1, self.L2, apply(L))):
                continue
            keep.append(O)
        trans = [add(scale(self.A1, i), self.A2, j) for i, j in self.cell.moire_points()]
        return [(O, R) for O in keep for R in trans]

    def choice_classes(self, limit=2000):
        """Symmetry classes of the ways to fill the open Gamma shell with real
        orbitals: [[choice, ...], ...], each choice a tuple of indices into the
        open-shell list.  Two choices are equivalent when a symmetry of H maps
        the span of one onto the span of the other (then |psi| of one is |psi| of
        the other at transformed coordinates, and every energy is equal)."""
        os_ = self.open_shell()
        if os_ is None:
            return []
        _, shell, r = os_
        ks = sorted({(o[0], o[1]) for o in shell} | {(-o[0], -o[1]) for o in shell})
        index = {k: a for a, k in enumerate(ks)}
        b1, b2 = reciprocal_basis(self.L1, self.L2)
        kvec = [add(scale(b1, k[0]), b2, k[1]) for k in ks]
        dim = len(ks)

        def vector(o):
            v = [0j] * dim
            plus, minus = index[(o[0], o[1])], index[(-o[0], -o[1])]
            if o[2]:   # sin = (e^{ik} - e^{-ik}) / 2i
                v[plus] += 1 / (2j)
                v[minus] -= 1 / (2j)
            else:
                v[plus] += 0.5
                v[minus] += 0.5
            n = math.sqrt(sum(abs(z) ** 2 for z in v))
            return [z / n for z in v]

        def projector(vs):
            return [[sum(v[a] * v[b].conjugate() for v in vs) for b in range(dim)]
                    for a in range(dim)]

        # Each symmetry as a monomial matrix: e_k -> phase * e_{Ok}, with
        # (g f)(r) = f(g^-1 r) and e^{ik.g^-1 r} = e^{i Ok.r} e^{i(k.c - Ok.(c+R))}.
        c = self.rmin
        mats = []
        for O, R in self.symmetry_ops():
            perm, phase = [0] * dim, [0j] * dim
            for a, k in enumerate(kvec):
                Ok = (O[0][0] * k[0] + O[0][1] * k[1], O[1][0] * k[0] + O[1][1] * k[1])
                n1, n2 = solve2(b1, b2, Ok)
                tgt = index.get((round(n1), round(n2)))
                if tgt is None:
                    raise AssertionError("a symmetry left the open shell")
                perm[a] = tgt
                phase[a] = cmath.exp(1j * (dot(k, c) - dot(Ok, add(c, R))))
            mats.append((perm, phase))

        def act(P, mat):
            perm, phase = mat
            out = [[0j] * dim for _ in range(dim)]
            for a in range(dim):
                for b in range(dim):
                    out[perm[a]][perm[b]] = phase[a] * P[a][b] * phase[b].conjugate()
            return out

        def same(P, Q):
            return all(abs(P[a][b] - Q[a][b]) < 1e-8 for a in range(dim) for b in range(dim))

        choices = list(itertools.combinations(range(len(shell)), r))
        if len(choices) > limit:
            raise SystemExit(f"{len(choices)} ways to fill the open shell; too many to "
                             "classify (pick one with --pw-pick)")
        proj = {ch: projector([vector(shell[i]) for i in ch]) for ch in choices}
        classes, assigned = [], {}
        for ch in choices:
            if ch in assigned:
                continue
            cls = [ch]
            assigned[ch] = len(classes)
            images = [act(proj[ch], mat) for mat in mats]
            for other in choices:
                if other in assigned:
                    continue
                if any(same(img, proj[other]) for img in images):
                    cls.append(other)
                    assigned[other] = len(classes)
            classes.append(cls)
        return classes


# ── units ────────────────────────────────────────────────────────────────────

def material_point(name, epsilon):
    """(x, lambda, kappa) for a material preset at dielectric constant epsilon."""
    mat = MATERIALS[name]
    x = mat["a_M_nm"] * mat["mstar"] / (epsilon * A0_NM)
    E_M = HA_MEV * (A0_NM / mat["a_M_nm"]) ** 2 / mat["mstar"]
    lam = mat["V0_meV"] / E_M
    return x, lam, lam / (x * x)


# ── YAML emission ────────────────────────────────────────────────────────────

def f(v):
    return repr(float(v))


def vec(v):
    return f"[{f(v[0])}, {f(v[1])}]"


def snapped(v, scale_):
    """Cell vectors written with their rounding noise removed, so an axis-aligned
    rectangle reaches alea as exactly axis-aligned (its orthorhombic fast path)."""
    return tuple(0.0 if abs(c) < 1e-12 * scale_ else c for c in v)


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


def groups_of(geo):
    """[(group name, up, down)] in the order the YAML lists them."""
    if geo.spin == "polarized":
        return [("electron_group", geo.ne, 0)]
    half = geo.ne // 2
    return [("e_up", half, 0), ("e_dn", 0, half)]


def pw_list(geo, pick):
    """The explicit real plane waves of one group: closed shells + picked ones."""
    closed, shell, r = geo.open_shell()
    if len(pick) != r or len(set(pick)) != r or not all(0 <= i < len(shell) for i in pick):
        raise SystemExit(f"--pw-pick needs {r} distinct indices in 0..{len(shell) - 1} "
                         "(the open shell's real orbitals; --report lists them)")
    return closed + [shell[i] for i in sorted(pick)]


def build_yaml(geo, state, args, name, pick=None, header=()):
    twist = getattr(args, "twist", None)
    if state == "fluid":
        if twist:
            if not geo.twist_ok(twist):
                raise SystemExit(f"twist {list(twist)}: the occupation of N/spin = "
                                 f"{geo.n_spin[0]} ties at the Fermi edge; alea refuses it.")
        elif pick is None and not geo.fluid_ok:
            closed, shell, r = geo.open_shell()
            raise SystemExit(
                f"N/spin = {geo.n_spin[0]} leaves a Gamma shell open ({len(closed)} + "
                f"{r}/{len(shell)}): choose its orbitals with --pw-pick, write one file "
                "per symmetry class with --pw-variants, or use a twist.")
    a1, b1 = geo.j1_seed()
    if state == "crystal":
        # The Gaussians already localise the crystal on the minima; the moire
        # Jastrow only refines the shape, so start it at a tenth of the
        # perturbative value rather than double-counting the localisation.
        a1, b1 = 0.1 * a1, 0.1 * b1
    L_drum = min((2.0 if state == "crystal" else 3.0) * geo.rs, 0.95 * geo.r_ws)
    # alpha_0 = c Gamma/(3 L^2) with c = 1, so the cusp needs no alpha_1 at the
    # start: Gamma = 1/3 for like spins (P-wave), 1 for unlike spins (2-D).
    alpha_like, alpha_unlike = 1.0 / (9.0 * L_drum ** 2), 1.0 / (3.0 * L_drum ** 2)
    scale_ = max(norm(geo.L1), norm(geo.L2))
    L1, L2 = snapped(geo.L1, scale_), snapped(geo.L2, scale_)
    groups = groups_of(geo)

    lines = [
        "# " + "=" * 77,
        f"# MOIRE {state.upper()}  nu = {geo.nu}  N_e = {geo.ne} ({geo.spin})  M = {geo.M}  "
        f"x = a_M/a_B* = {geo.x:g}  kappa = V0/Ha* = {geo.kappa:g}  "
        f"phi = {math.degrees(geo.phi):g} deg",
        "#",
        "# Generated by tools/moire_cells.py -- edit the generator, not this file.",
    ]
    lines += [f"# {h}" for h in header]
    lines += [
        f"# rs = {geo.rs:.6g}   lambda = kappa x^2 = V0/E_M = {geo.kappa * geo.x ** 2:.6g}   "
        f"cell = {geo.cell.label()}, theta0 = {geo.theta0:.6g}",
        f"# moire minimum at {vec(geo.rmin)}, V_min = {geo.vmin_unit:.6g} V0;  "
        f"crystal: {geo.pattern.name()}",
        f"# WS inradius {geo.r_ws:.6g};  Gamma shells per spin: "
        + ", ".join(describe_gamma(geo.L1, geo.L2, n) for n in sorted(set(geo.n_spin))),
    ]
    if geo.parity is not None:
        lines.append(f"# spin stripes: parity {geo.parity} of the site-lattice coordinates")
    lines += [
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
        f"    vectors: [{vec(L1)}, {vec(L2)}]",
        f"    ewald_tol: {args.ewald_tol:g}",
    ]
    if twist:
        lines.append(f"    twist: [{f(twist[0])}, {f(twist[1])}]")
    lines.append("  particles:")
    for gname, up, dn in groups:
        lines.append(f"    - {{ name: {gname}, group: {gname}, type: fermions, mass: 1.0, "
                     f"charge: -1, up: {up}, down: {dn}, initialization: uniform }}")
    lines += ["", "wavefunction:"]
    for s, (gname, _, _) in enumerate(groups):
        lines += ["  - slater:", f"      group: {gname}"]
        if state == "crystal":
            lines += ["      orbitals:",
                      "        type: gaussian",
                      f"        alpha: {f(geo.alpha_seed())}",
                      "        alpha_fixed: false",
                      "        centers:"]
            spin = None if geo.spin == "polarized" else s
            lines += [f"          - {vec(r)}" for r in geo.sites(args.shift, spin)]
        elif pick is not None and not twist:
            orbs = pw_list(geo, pick)
            lines += ["      orbitals:",
                      "        type: plane_wave",
                      "        k: [" + ", ".join(f"[{o[0]}, {o[1]}]" for o in orbs) + "]",
                      "        sin: [" + ", ".join("true" if o[2] else "false"
                                                   for o in orbs) + "]"]
        else:
            lines += ["      orbitals: { type: plane_wave }"]
    names = [gname for gname, _, _ in groups]

    def drummond(pairs_yaml, alpha0):
        return ["  - jastrow:",
                "      type: drummond",
                f"      {pairs_yaml}",
                f"      params: [{f(alpha0)}]",
                "      is_fixed: [false]",
                "      truncation: 3",
                f"      cutoff: {f(L_drum)}",
                "      cutoff_fixed: true"]

    if len(names) == 1:
        lines += drummond(f"pair: [{names[0]}, {names[0]}]", alpha_like)
    else:
        lines += drummond(f"pairs: [[{names[0]}, {names[0]}], [{names[1]}, {names[1]}]]",
                          alpha_like)
        lines += drummond(f"pair: [{names[0]}, {names[1]}]", alpha_unlike)
    for gname in names:
        lines += ["  - moire_jastrow:",
                  f"      group: {gname}",
                  "      shells: 1",
                  f"      params: [{f(a1)}, {f(b1)}]",
                  # No potential, nothing to follow: pinned at zero so kappa = 0 is
                  # exactly the uniform-gas trial state.
                  "      is_fixed: [true, true]" if geo.kappa == 0.0 else
                  "      is_fixed: [false, false]"]
    lines += ["", "hamiltonian:", "  terms:"]
    # Under PBC every pair interacts through one Ewald term; the pairs are
    # listed for the reader.
    for a, b in itertools.combinations_with_replacement(names, 2):
        lines.append(f"    - {{ type: coulomb, pair: [{a}, {b}], epsilon: 1.0 }}")
    lines += [
        "    - type: moire",
        f"      group: [{', '.join(names)}]",
        "      V0: 0.0",
        f"      Vtilde: {f(geo.kappa)}",
        f"      phi: {f(geo.phi)}",
        f"      a_M: {f(geo.x)}",
        f"      theta0: {f(geo.theta0)}",
        "",
    ]
    lines += qmc_block(geo, state, args, name)
    return "\n".join(lines) + "\n"


def meta(geo, state, args, name, pick=None, extra=None):
    out = {
        "name": name, "state": state, "nu": str(geo.nu), "ne": geo.ne, "M": geo.M,
        "spin": geo.spin, "cell": geo.cell.label(), "cell_vectors": [geo.L1, geo.L2],
        "x": geo.x, "kappa": geo.kappa, "phi_deg": math.degrees(geo.phi),
        "theta0": geo.theta0, "rs": geo.rs, "lambda": geo.kappa * geo.x ** 2,
        "E_M": geo.E_M, "r_ws": geo.r_ws,
        "moire_min": geo.rmin, "V_min": geo.vmin,
        "alpha_seed": geo.alpha_seed(), "j1_seed": geo.j1_seed(),
        "shift": list(args.shift),
        "twist": list(args.twist) if getattr(args, "twist", None) else None,
        "sites": ([{"r": r, "spin": "up" if s == 0 else "down"}
                   for r, s in zip(geo.sites(args.shift), (s for _, s in geo._sites))]
                  if state == "crystal" else None),
        "pw_pick": list(pick) if pick is not None else None,
    }
    if extra:
        out.update(extra)
    return out


# ── reports ──────────────────────────────────────────────────────────────────

def valid_rows(nu, spin, kind, n_max, phi, twist_ns, offset, x=10.0, max_aspect=None):
    """One row per (N, cell) where the crystal fits: the Gamma shells per spin and
    the twist grid's ties.  max_aspect drops cells longer than that ratio."""
    rows = []
    step = 2 if spin == "unpolarized" else 1
    for ne in range(step, n_max + 1, step):
        if (Fraction(ne) / nu).denominator != 1:
            continue
        M = int(Fraction(ne) / nu)
        g = first_shell(x, -math.pi / 6 if kind == "rect" else 0.0)
        A1, A2 = direct_lattice(g)
        cands = hex_cells(A1, A2, M) if kind == "hex" else rect_cells(A1, A2, M)
        seen = set()
        for cell in cands:
            if kind == "hex" and (cell.shape[1], cell.shape[0]) in seen:
                continue                       # the mirror image of a cell listed
            if max_aspect is not None and cell.aspect > max_aspect * (1 + 1e-12):
                continue
            seen.add(cell.shape)
            try:
                geo = Geometry(nu, ne, x, 0.01, phi, cell_kind=kind, spin=spin,
                               shape=cell.shape)
            except SystemExit:
                continue
            n = geo.n_spin[0]
            grid = grid2d(twist_ns, offset) if twist_ns else []
            ties = [i for i, (t, _) in enumerate(grid) if not geo.twist_ok(t)]
            rows.append({"ne": ne, "geo": geo, "n": n,
                         "gamma": describe_gamma(geo.L1, geo.L2, n),
                         "grid": grid, "ties": ties})
    return rows


def report(args, phi):
    nus = [Fraction(v) for v in args.nu]
    lines = ["# Plane-wave shells and valid electron numbers", "",
             "Generated by `tools/moire_cells.py --report` (alea-new).  Per spin: the "
             "Gamma occupation (`closed`, or `open: F + r/g` = F states in full shells "
             "and r of the g states of the next one) and, on the twist grid of "
             "`tools/twist_grid.py`, the twists whose occupation ties at the Fermi "
             "edge (alea refuses those).  A row exists only where the crystal fits "
             "the cell (with its spin stripes when unpolarized).", "",
             f"phi = {math.degrees(phi):g} deg; twist grid {args.twist_grid} per axis, "
             f"offset {args.twist_offset}, t and -t merged.", ""]
    summary = []
    for nu in nus:
        for spin in args.spin:
            for kind in args.cells:
                try:
                    patterns_for(nu)
                except SystemExit as e:
                    lines += [f"## nu = {nu}, {spin}, {kind}", "", str(e), ""]
                    continue
                rows = valid_rows(nu, spin, kind, args.n_max, phi,
                                  args.twist_grid, args.twist_offset,
                                  max_aspect=args.max_aspect)
                lines += [f"## nu = {nu}, {spin}, {kind} cells (N <= {args.n_max}"
                          + (f", aspect <= {args.max_aspect:g}" if kind == "rect" else "")
                          + ")", ""]
                if not rows:
                    lines += ["No cell holds the crystal.", ""]
                    continue
                n_grid = len(rows[0]["grid"]) if rows else 0
                lines += ["| N | cell | aspect | M | N/spin | Gamma | grid twists tied (of "
                          f"{n_grid}) |", "|---|---|---|---|---|---|---|"]
                for r in rows:
                    tied = ", ".join(str(i) for i in r["ties"]) or "none"
                    lines.append(f"| {r['ne']} | {r['geo'].cell.label()} | "
                                 f"{r['geo'].cell.aspect:.2f} | {r['geo'].M} | "
                                 f"{r['n']} | {r['gamma']} | {tied} |")
                closed = sorted({r["ne"] for r in rows if r["gamma"] == "closed"})
                crystal = sorted({r["ne"] for r in rows})
                twist_ok = sorted({r["ne"] for r in rows if not r["ties"]})
                lines += ["", f"- crystal fits: {crystal}",
                          f"- liquid closed at Gamma: {closed or 'none'}",
                          f"- liquid with no tie on the twist grid: {twist_ok or 'none'}", ""]
                summary.append((nu, spin, kind, crystal, closed, twist_ok))
    if args.detail:
        lines += ["## Details of chosen cells", ""]
        for spec in args.detail:
            nu_s, ne_s, kind, spin = spec.split(":")
            geo = Geometry(Fraction(nu_s), int(ne_s), 10.0, 0.01, phi, cell_kind=kind,
                           spin=spin)
            lines += detail_lines(geo, args.twist_grid, args.twist_offset)
    return "\n".join(lines) + "\n", summary


def detail_lines(geo, twist_ns, offset):
    out = [f"### nu = {geo.nu}, N = {geo.ne}, {geo.spin}, {geo.cell.label()} "
           f"(M = {geo.M})", ""]
    for n in sorted(set(geo.n_spin)):
        counts = closed_shell_counts(geo.L1, geo.L2, n_max=n + 40)
        out.append(f"- N/spin = {n}: Gamma {describe_gamma(geo.L1, geo.L2, n)}; "
                   f"closed counts {counts[:10]}")
    os_ = geo.open_shell()
    if os_ is not None:
        _, shell, r = os_
        out.append(f"- open shell, real orbitals (index: [n1, n2] cos|sin), take {r}:")
        for i, o in enumerate(shell):
            out.append(f"  - {i}: [{o[0]}, {o[1]}] {'sin' if o[2] else 'cos'}")
        try:
            classes = geo.choice_classes()
            out.append(f"- {len(classes)} symmetry class(es) of choices "
                       f"(symmetries of H in the cell: {len(geo.symmetry_ops())}):")
            for c, cls in enumerate(classes):
                out.append(f"  - class {c}: {len(cls)} choice(s), e.g. {list(cls[0])}")
        except SystemExit as e:
            out.append(f"- {e}")
    if twist_ns:
        out.append(f"- twist grid ({twist_ns} per axis):")
        for i, (t, w) in enumerate(grid2d(twist_ns, offset)):
            out.append(f"  - {i}: t = [{t[0]:.6g}, {t[1]:.6g}], weight {w}: "
                       + ("tie (refused)" if not geo.twist_ok(t) else "closed"))
    # Tied twists per twist_grid.py grid (n per axis, offset): a grid with none
    # can be averaged as it is.
    scan = []
    for n in (3, 4, 5, 6):
        for off in (0.5, 0.0, 0.25, 0.125):
            grid = grid2d([n], off)
            ties = sum(1 for t, _ in grid if not geo.twist_ok(t))
            scan.append(f"{n}x{n} @ {off:g}: {ties}/{len(grid)}")
    out.append("- tied twists per grid (n x n @ offset: tied/runs): " + "; ".join(scan))
    out.append("")
    return out


# ── self-test ────────────────────────────────────────────────────────────────

def log_abs_det(rows):
    """ln|det| of a square real matrix by Gaussian elimination."""
    a = [list(r) for r in rows]
    n, total = len(a), 0.0
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(a[i][k]))
        if a[p][k] == 0.0:
            return -math.inf
        a[k], a[p] = a[p], a[k]
        total += math.log(abs(a[k][k]))
        for i in range(k + 1, n):
            fct = a[i][k] / a[k][k]
            for j in range(k, n):
                a[i][j] -= fct * a[k][j]
    return total


def selftest():
    fails = 0

    def check(cond, msg):
        nonlocal fails
        if not cond:
            fails += 1
            print("FAIL:", msg)

    # The original battery: polarized nu = 1/m in hexagonal cells.
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

    phi225 = math.radians(225.0)
    # phi = 225 deg: the sites are the deep minima, -3 sqrt2 V0 = -4.24 V0, below
    # every other point of the cell; the shallow wells sit at -1.55 V0.
    for nu, ne, kind, spin in [(Fraction(1, 3), 28, "hex", "unpolarized"),
                               (Fraction(1, 3), 42, "rect", "unpolarized"),
                               (Fraction(2, 3), 18, "hex", "polarized")]:
        geo = Geometry(nu, ne, x=6.0, kappa=0.05, phi=phi225, cell_kind=kind, spin=spin)
        label = f"{nu} N={ne} {kind} {spin}"
        check(abs(geo.vmin_unit + 3 * math.sqrt(2.0)) < 1e-12, f"{label}: V_min != -4.24 V0")
        for s in geo.sites():
            check(abs(moire_V(s, geo.g, 1.0, geo.phi) + 3 * math.sqrt(2.0)) < 1e-9,
                  f"{label}: a site off the deep minimum")
        # N_e = nu x the number of moire cells.
        check(Fraction(geo.ne) == nu * geo.M, f"{label}: N_e != nu M")
        check(abs(geo.area - geo.M * SQRT3 / 2 * geo.x ** 2) < 1e-8 * geo.area,
              f"{label}: area != M moire cells")
        for L in (geo.L1, geo.L2):
            for gj in geo.g:
                check(is_int(dot(gj, L) / TWO_PI), f"{label}: g.L not 2pi Z")
        if spin == "unpolarized":
            ups, dns = geo.sites(spin=0), geo.sites(spin=1)
            check(len(ups) == len(dns) == ne // 2, f"{label}: stripes do not halve N")
            # Each spin's sites form a lattice: the difference of two up sites
            # translates the up set onto itself.
            d = add(ups[1], ups[0], -1.0)
            moved = {tuple(round(c, 6) for c in wrap(add(u, d), geo.L1, geo.L2)) for u in ups}
            check(moved == {tuple(round(c, 6) for c in u) for u in ups},
                  f"{label}: the up sites are not a lattice")
    sh_geo = Geometry(Fraction(2, 3), 18, x=6.0, kappa=0.05, phi=phi225)
    sites = sh_geo.sites()
    nn = [sorted(norm(wrap(add(a, b, -1.0), sh_geo.L1, sh_geo.L2)) for b in sites if b != a)
          for a in sites]
    check(all(abs(d[0] - sh_geo.x) < 1e-8 and abs(d[2] - sh_geo.x) < 1e-8 and
              d[3] > sh_geo.x * 1.5 for d in nn),
          "2/3: every site has three nearest neighbours at a_M (honeycomb)")
    rect = Geometry(Fraction(1, 3), 42, x=6.0, kappa=0.05, phi=phi225, cell_kind="rect",
                    spin="unpolarized")
    scale_ = max(norm(rect.L1), norm(rect.L2))
    check(snapped(rect.L1, scale_)[1] == 0.0 and snapped(rect.L2, scale_)[0] == 0.0,
          "rect cells are axis-aligned at the default theta0")
    check(rect.cell.shape == (9, 7), "N = 42 is the 9 x 7 rectangle")

    # Gamma shells of the unpolarized liquid at nu = 1/3 and of the polarized
    # liquid at nu = 2/3.
    hex13 = valid_rows(Fraction(1, 3), "unpolarized", "hex", 100, phi225, None, 0.5)
    check(hex13 and all(r["gamma"] != "closed" for r in hex13),
          "nu = 1/3 unpolarized: no hexagonal cell closes a Gamma shell")
    check({r["ne"] for r in hex13 if 12 <= r["ne"] <= 64} == {12, 16, 28, 36, 48, 52, 64},
          "nu = 1/3 unpolarized hexagonal N in [12, 64]: 12, 16, 28, 36, 48, 52, 64")
    # Compact rectangles (aspect <= 2) of 12 <= N <= 72: only 42 and 54 close.
    # (N = 2 and 6 close too, in cells of 6 and 18 moire cells; longer cells
    # close at many N.)
    rect13 = [r for r in valid_rows(Fraction(1, 3), "unpolarized", "rect", 72, phi225, None,
                                    0.5, max_aspect=2.0) if r["ne"] >= 12]
    check({(r["ne"], r["geo"].cell.shape) for r in rect13 if r["gamma"] == "closed"}
          == {(42, (9, 7)), (54, (9, 9))},
          "nu = 1/3 unpolarized compact rectangles: only 9 x 7 (N = 42) and 9 x 9 (N = 54) "
          "close at Gamma")
    check(describe_gamma(sh_geo.L1, sh_geo.L2, 18) == "open: 13 + 5/6",
          "27 cells, 18 polarized: open shell 13 + 5/6")
    check(sh_geo.M == 27 and sh_geo.cell.shape == (3, 3), "nu = 2/3, N = 18: the 27-cell cell")

    # The twist grid is the one twist_grid.py writes.
    # The twist grid is twist_grid.py's: 4 x 4 at offset 1/2, t and -t merged,
    # is 8 runs of weight 2, and one value means both axes.
    grid = grid2d([4], 0.5)
    check(len(grid) == 8 and all(w == 2 and len(t) == 2 for t, w in grid),
          "twist grid: 4 x 4 merged into 8 runs of weight 2")
    check(grid == twist_grid([4, 4], 0.5, True), "twist grid: one value means both axes")

    # Material units: Geier et al. (2025), x = 53.12/epsilon, lambda = 4.444.
    x, lam, kap = material_point("geier2025", 10.0)
    check(abs(x - 5.3117) < 1e-3 and abs(lam - 4.4437) < 1e-3 and
          abs(kap - lam / x ** 2) < 1e-15, "geier2025 units")

    # Open-shell choices: within a symmetry class, |psi| of one choice is |psi|
    # of the other at transformed coordinates (checked on the 27-cell cell).
    classes = sh_geo.choice_classes()
    check(sum(len(c) for c in classes) == 6, "27 cells: 6 ways to take 5 of 6 orbitals")
    b1, b2 = reciprocal_basis(sh_geo.L1, sh_geo.L2)
    closed, shell, r = sh_geo.open_shell()

    def orbital_value(o, pt):
        k = add(scale(b1, o[0]), b2, o[1])
        ph = dot(k, pt)
        return math.sin(ph) if o[2] else math.cos(ph)

    ops = sh_geo.symmetry_ops()
    rng = random.Random(3)
    pts = [(rng.uniform(-8, 8), rng.uniform(-8, 8)) for _ in range(18)]
    for cls in classes:
        if len(cls) < 2:
            continue
        orbs0 = closed + [shell[i] for i in cls[0]]
        found = False
        for other in cls[1:2]:
            orbs1 = closed + [shell[i] for i in other]
            target = log_abs_det([[orbital_value(o, p) for o in orbs1] for p in pts])
            for O, R in ops:
                # g^-1 r = O^T (r - c - R) + c
                c = sh_geo.rmin
                inv = [(O[0][0] * u[0] + O[1][0] * u[1], O[0][1] * u[0] + O[1][1] * u[1])
                       for u in (add(add(p, c, -1.0), R, -1.0) for p in pts)]
                back = [add(v, c) for v in inv]
                val = log_abs_det([[orbital_value(o, p) for o in orbs0] for p in back])
                if abs(val - target) < 1e-8 * max(1.0, abs(target)):
                    found = True
                    break
        check(found, f"class {cls}: no symmetry relates |psi| of its members")
    print("selftest:", "PASS" if fails == 0 else f"{fails} failure(s)")
    return fails == 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nu", type=str, nargs="+", help="filling(s), e.g. 1/3")
    ap.add_argument("--ne", type=int, help="number of electrons")
    ap.add_argument("--x", type=float, help="a_M / a_B*")
    ap.add_argument("--kappa", type=float, help="V0 / Ha* (moire amplitude)")
    ap.add_argument("--lam", type=float, help="lambda = V0/E_M; kappa = lambda/x^2")
    ap.add_argument("--material", choices=sorted(MATERIALS),
                    help="m*, a_M and V0 of a material; with --epsilon/--inv-epsilon "
                         "this sets x and kappa")
    ap.add_argument("--epsilon", type=float, help="dielectric constant (with --material)")
    ap.add_argument("--inv-epsilon", type=float, help="1/epsilon (with --material)")
    ap.add_argument("--phi-deg", type=float, default=60.0)
    ap.add_argument("--theta0", type=float, default=None,
                    help="moire orientation (rad); default 0 (hex), -pi/6 (rect)")
    ap.add_argument("--cell", choices=["hex", "rect"], default="hex")
    ap.add_argument("--shape", type=int, nargs=2, default=None,
                    help="the cell: (p, q) for hex, (n, m) for rect; default the first "
                         "that holds the crystal")
    ap.add_argument("--spin", nargs="+", default=["polarized"],
                    choices=["polarized", "unpolarized"])
    ap.add_argument("--state", choices=["crystal", "fluid", "liquid"])
    ap.add_argument("--twist", type=float, nargs=2, default=None,
                    help="twist in fractional reciprocal coordinates of the cell")
    ap.add_argument("--pw-pick", type=str, default=None,
                    help="liquid with an open Gamma shell: comma-separated indices of the "
                         "open shell's real orbitals to occupy (see --report --detail)")
    ap.add_argument("--pw-variants", action="store_true",
                    help="liquid with an open Gamma shell: one file per symmetry class of "
                         "choices, suffix _pw<class>")
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
    ap.add_argument("--report", type=Path, default=None,
                    help="write the shell / valid-N report (Markdown) for --nu x --spin "
                         "x --cells")
    ap.add_argument("--cells", nargs="+", default=["hex", "rect"], choices=["hex", "rect"])
    ap.add_argument("--n-max", type=int, default=72)
    ap.add_argument("--max-aspect", type=float, default=2.0,
                    help="report/list: skip cells longer than this ratio (rect only)")
    ap.add_argument("--twist-grid", type=int, nargs="+", default=None,
                    help="report: twists per reciprocal axis (twist_grid.py --grid)")
    ap.add_argument("--twist-offset", type=float, default=0.5)
    ap.add_argument("--detail", nargs="*", default=[],
                    help="report: per-twist detail for nu:N:cell:spin, e.g. 2/3:18:hex:polarized")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if selftest() else 1
    phi = math.radians(args.phi_deg)
    if args.report:
        if not args.nu:
            ap.error("--report needs --nu")
        text, summary = report(args, phi)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        for nu, spin, kind, crystal, closed, twist_ok in summary:
            print(f"nu = {nu} {spin} {kind}: crystal {crystal}; Gamma closed "
                  f"{closed or 'none'}")
        print(f"wrote {args.report}")
        return 0
    if args.list:
        if not args.nu:
            ap.error("--list needs --nu")
        nu = Fraction(args.nu[0])
        print(f"nu = {nu} ({args.spin[0]}, {args.cell}):  N_e  cell  aspect  M  "
              "Gamma per spin")
        for r in valid_rows(nu, args.spin[0], args.cell, 100, phi, None, 0.5,
                            max_aspect=args.max_aspect):
            print(f"  {r['ne']:4d}  {r['geo'].cell.label():12s} "
                  f"{r['geo'].cell.aspect:5.2f} {r['geo'].M:5d}   {r['gamma']}")
        return 0

    if args.material:
        if (args.epsilon is None) == (args.inv_epsilon is None):
            ap.error("--material needs one of --epsilon or --inv-epsilon")
        if args.x is not None or args.kappa is not None or args.lam is not None:
            ap.error("--material sets x and kappa; drop --x/--kappa/--lam")
        eps = args.epsilon if args.epsilon is not None else 1.0 / args.inv_epsilon
        args.x, lam, args.kappa = material_point(args.material, eps)
        header = [f"material {args.material}: {MATERIALS[args.material]}, epsilon = {eps:g}"
                  f" -> x = {args.x:.6g}, lambda = {lam:.6g}"]
    else:
        header = []
        if args.lam is not None:
            if args.kappa is not None or args.x is None:
                ap.error("--lam needs --x and replaces --kappa")
            args.kappa = args.lam / args.x ** 2
    state = "fluid" if args.state == "liquid" else args.state
    for req in ("nu", "ne", "x", "kappa", "state", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required")
    if len(args.nu) != 1 or len(args.spin) != 1:
        ap.error("writing a config takes one --nu and one --spin")
    geo = Geometry(Fraction(args.nu[0]), args.ne, args.x, args.kappa, phi, args.theta0,
                   args.cell, args.spin[0], args.shape)
    stem = args.name or args.out.stem
    jobs = [(None, args.out, stem)]
    if state == "fluid" and not args.twist and geo.open_shell() is not None:
        if args.pw_pick is not None:
            jobs = [(tuple(int(i) for i in args.pw_pick.split(",")), args.out, stem)]
        elif args.pw_variants:
            jobs = [(cls[0], args.out.with_name(f"{args.out.stem}_pw{c}{args.out.suffix}"),
                     f"{stem}_pw{c}") for c, cls in enumerate(geo.choice_classes())]
    for pick, out, name in jobs:
        text = build_yaml(geo, state, args, name, pick=pick, header=header)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".meta.json").write_text(
            json.dumps(meta(geo, state, args, name, pick), indent=1), encoding="utf-8")
        print(f"wrote {out}  (rs = {geo.rs:.4g}, lambda = {geo.kappa * geo.x**2:.4g}, "
              f"alpha seed = {geo.alpha_seed():.4g}"
              + (f", open-shell pick {list(pick)}" if pick is not None else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
