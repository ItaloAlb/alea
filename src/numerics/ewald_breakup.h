// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "numerics/long_range_spline.h"
#include "numerics/radial_table.h"
#include "population/simulation_cell.h"

namespace alea::numerics {

// Ewald splitting data for a 2-D periodic cell, built once at construction and
// read-only afterwards.
//
// ── The sum ─────────────────────────────────────────────────────────────────
// Particles live in a plane and interact through the 3-D Coulomb potential
// 1/|r| (the same convention CoulombTerm uses).  Splitting with the error
// function, 1/r = erfc(ar)/r + erf(ar)/r, and applying 2-D Poisson summation to
// the smooth half gives
//
//   E = Σ_{i<j} q_i q_j erfc(a r_ij)/r_ij            real space, nearest image
//     + (π/A) Σ_{k≠0} [erfc(k/2a)/k] |ρ(k)|²         reciprocal space
//     − (a/√π) Σ_i q_i²                              self-interaction
//
// with ρ(k) = Σ_i q_i exp(i k·r_i) and A the cell area.
//
// ── Why the real-space half is just the minimum image ───────────────────────
// `a` is chosen so that erfc(a·R)/R ≤ tol at R = the Wigner–Seitz inradius.
// The second-nearest image of any pair sits at least R away, so its erfc is
// already below tolerance — the lattice sum in real space collapses to the
// single nearest image, which Milestone 1 already computes and tests.  This is
// the concrete sense in which minimum image IS Ewald's real-space half.
//
// ── Potential-agnostic by way of the transform pair ─────────────────────────
// The only potential-specific input is the kernel's transform pair (V(r),
// Ṽ(k)), consulted only at setup.  For the in-plane potentials of this project
// Ṽ(k) = 2π/(k eps(k)):
//
//     coulomb           eps(k) = eps                    (constant)
//     rytova_keldysh    eps(k) = eps (1 + r0 k)
//     screened_coulomb  eps(k) = fsc_epsilon_k(ctx, k)  (the ETM library)
//
// but PotentialSpec carries Ṽ(k) itself, so a kernel need not be of that form
// (a pair in two planes carries e^{-kd}).  Nothing potential-specific survives
// into the reciprocal hot loop: it lives in the stored weights.
//
// The real-space half has no closed form for a general eps(k), so it is built
// numerically: the long-range part comes from the inverse Hankel transform
//
//     V_long(r) = integral_0^inf  J0(k r) erfc(k/2a) / eps(k)  dk
//
// (note the damping is erfc(k/2a), which IS the 2-D Fourier transform of the
// Ewald erf split — NOT a Gaussian), and then S(r) = r*(V(r) - V_long(r)) is
// fitted to a Chebyshev table.  For constant eps this reproduces erfc(a r)/eps
// analytically, which is how the generic machinery is validated.
//
// ── The k = 0 term ──────────────────────────────────────────────────────────
// Omitted here.  For a neutral cell ρ(0) = 0, so it vanishes.  A charged cell
// needs a neutralising background; that constant is a separate milestone step
// and `neutral` records which case the caller is in.
struct EwaldSetup {
    double r_cut    = 0.0;  // Wigner–Seitz inradius: the real-space reach
    double k_cut    = 0.0;  // reciprocal cutoff
    double area     = 0.0;  // cell volume (area in 2-D)

    // Half of k-space only: |ρ(k)|² = |ρ(−k)|², so each stored k carries the
    // weight of the pair {k, −k}.  Stored as Cartesian components plus the
    // integer indices, which the hot loop uses to index its exponential tables.
    std::vector<double> kx, ky;
    std::vector<int>    n1, n2;    // k = n1·b1 + n2·b2
    std::vector<double> weight;    // Ṽ_long(k)/A, the ×2 of the fold included

    int n1_max = 0, n2_max = 0;    // table extents implied by the k list

    // Long-range half in real space: the breakup's own quintic spline, which is
    // EXACT, not a fit.  A Chebyshev table cannot be substituted — the spline
    // is only C², so Chebyshev converges algebraically on it (NC = 32 leaves
    // 9.8e-10, NC = 256 still 5.4e-11), and any residual is not rounding but an
    // inconsistency with the reciprocal half, which no k-resolution removes.
    // Measured 5.7x faster than a 32-term Clenshaw besides.
    LongRangeSpline long_range;
    // max |Ṽ_long(k)| over the discrete shells just above k_cut: what the
    // truncated reciprocal sum actually drops.  This is the gate.
    double residual = 0.0;
    double condition = 0.0;   // cond(C) from the QR, not its square

    // V_long(0) = t_0, which sets the self-interaction constant.  Every basis
    // function but the first vanishes at the origin, so this is one coefficient.
    double v_long_zero = 0.0;

    // (1/A) ∫ V_short d²r: what a uniform neutralising background couples to.
    // Enters the energy as −½(Σq)²·sr_k0, so it is inert for a neutral cell.
    double sr_k0 = 0.0;

    [[nodiscard]] std::size_t n_k() const noexcept { return kx.size(); }
};

// A pair kernel, as build_ewald needs to see it: its real-space value and its
// 2-D Fourier transform.  Both are consulted at setup only, so std::function's
// indirection costs nothing where it is used.
//
// Ṽ(k) is supplied directly rather than through a dielectric function eps(k):
// every in-plane potential is 2π/(k eps(k)), but a pair in two different planes
// carries an extra e^{-kd}, which no natural eps(k) expresses.
struct PotentialSpec {
    // `kind` and the parameters below tell EwaldTerm how to evaluate V(r)
    // INLINE in its hot loop; v_real and v_fourier are the setup-time callables.
    // `epsilon` is also the long-range dielectric constant: every kernel tends
    // to 2π/(epsilon k) as k → 0.
    enum class Kind { Coulomb, Keldysh, Screened };
    Kind   kind    = Kind::Coulomb;
    double epsilon = 1.0;
    double r0      = 0.0;  // Keldysh only
    // Out-of-plane separation of the pair's two planes; 0 for an in-plane pair.
    // (Screened kernels bake their geometry into the table and keep this 0.)
    double d_layer = 0.0;

    // Screened only: V(r) = singular_coef / r + table(r).  The singular part is
    // 1/eps_source for a same-layer pair (the bulk-Coulomb core) and 0 across
    // layers; the table is the smooth remainder, exact to ~1e-12 on
    // [0, table->r_max()].  Shared, immutable, set up once.
    double                              singular_coef = 0.0;
    std::shared_ptr<const RadialTable>  table;

    std::function<double(double r)> v_real;     // V(r), r > 0
    std::function<double(double k)> v_fourier; // Ṽ(k) = 2π∫ r J0(kr) V(r) dr, k > 0

    // V' and V''.  Needed by the optimized breakup, which ASSIGNS them to the
    // outermost basis coefficients (the LPQHI coefficients are value, slope and
    // curvature at their knot) rather than fitting them.  Analytic, because a
    // central second difference carries eps/h^2 of round-off — about 1e-6 at
    // any sane h — and these three numbers are not averaged over.
    std::function<double(double r)> dv_real;
    std::function<double(double r)> d2v_real;

    // c = lim_{k→0} [Ṽ(k) − 2π/(ε k)]: what is left of the kernel at k = 0 once
    // the Coulomb tail every kernel shares is removed.  0 for Coulomb, −2π r0/ε
    // for Rytova-Keldysh.  In a neutral cell whose pairs all share ONE kernel it
    // multiplies (Σq)² = 0 and is inert; with a different kernel per pair it
    // survives as a finite k = 0 energy (see EwaldTerm).
    double k0_offset = 0.0;

    // Same interaction, compared by parameters (the callables are derived from
    // them).  Lets EwaldTerm build one breakup per DISTINCT kernel.
    [[nodiscard]] bool same_as(const PotentialSpec& o) const noexcept {
        return kind == o.kind && epsilon == o.epsilon && r0 == o.r0 &&
               d_layer == o.d_layer && table == o.table;
    }

    // Coulomb between two planes a distance d_layer apart (0: the plain in-plane
    // case, which is also the reference the generic path is validated against):
    //
    //   V(r) = 1/(eps sqrt(r² + d²)),   Ṽ(k) = 2π e^{-kd}/(eps k)
    //
    // an exact transform pair (Gradshteyn & Ryzhik 6.554.1).  For d > 0 the
    // kernel is finite at r = 0 (no cusp), and its k = 0 remainder
    // c = lim[Ṽ − 2π/(eps k)] = −2π d/eps is the parallel-plate capacitor term.
    [[nodiscard]] static PotentialSpec coulomb(double epsilon, double d_layer = 0.0);

    // Rytova-Keldysh: Ṽ(k) = 2π/(k eps (1 + r0 k)).  Verified numerically
    // against the repository's own K(x) = H0(x) - Y0(x) kernel:
    // inverse-Hankel transforming it reproduces (pi/2 eps r0) K(r/r0).
    // (The old fixed erfc split could not take it -- 1/(1 + r0 k) is not
    // analytic at k = 0 -- but the optimized breakup places no such demand.)
    //
    // With d_layer > 0 the pair sits in two planes and feels V_RK(sqrt(r²+d²)),
    // the open-boundary KeldyshTerm's definition.  That has no closed-form
    // transform; Ṽ(k) and k0_offset are computed by a 1-D quadrature derived in
    // ewald_breakup.cpp (setup-time only).
    [[nodiscard]] static PotentialSpec rytova_keldysh(double epsilon, double r0,
                                                      double d_layer = 0.0);
};

// Build the splitting for `cell` at relative tolerance `tol`.
//
// Follows build_screened_fit's sequence: validate the inputs, auto-tune the
// cutoffs, then gate on what was actually achieved rather than on what was
// asked for.  Throws std::runtime_error on a non-periodic cell, a non-finite or
// out-of-range tolerance, or a k-vector count that indicates a degenerate cell.
[[nodiscard]] EwaldSetup build_ewald(const SimulationCell& cell, double tol,
                                     const PotentialSpec& pot);

}  // namespace alea::numerics
