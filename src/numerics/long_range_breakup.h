// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "numerics/long_range_spline.h"
#include "numerics/lpqhi_basis.h"
#include "population/simulation_cell.h"

namespace alea::numerics {

// Optimized breakup of a radial function over a periodic cell, after
// Natoli & Ceperley, J. Comput. Phys. 117, 171 (1995).
//
// ── What it does ────────────────────────────────────────────────────────────
// Splits f(r) into a short-ranged part with compact support inside r_c, summed
// by minimum image, and a long-ranged part summed over a FEW reciprocal
// vectors:
//
//     f(r) = f_short(r) + f_long(r),   f_long(r) = Σ_n t_n h_n(r)  for r < r_c
//                                      f_long(r) = f(r)            for r ≥ r_c
//
// The basis carries the LONG-range part, which is smooth; the singularity of f
// lives entirely in f_short = f − f_long and is never expanded.  (Putting the
// singular part in the basis instead — with a hand-built cutoff function —
// costs large coefficients and a catastrophic cancellation between the
// real-space and self-energy terms.  Measured: |t| drops ~40% and the energy
// error halves when the roles are the right way round.)
//
// The coefficients are chosen to make f̃_long(k) ≈ 0 above k_cut, so truncating
// the reciprocal sum there is harmless.  That is the least-squares problem
//
//     minimise  Σ_{|k| > k_cut} w_k |X(k) − Σ_n t_n c_n(k)|²
//     X(k) = −∫_{r > r_c} f(r) e^{−ik·r} d²r = −f̃(k) + 2π ∫₀^{r_c} r J₀(kr) f(r) dr
//
// ── Two things that are easy to get wrong ───────────────────────────────────
// The sum runs to |k| = ∞, not to some convenient cutoff.  Stopping it leaves
// the fit unpenalised above that point and free to dump high-frequency content
// there; measured, that alone cost a factor of 300 in the Madelung constant.
// Above k_continuum the lattice shells are replaced by the continuum density
// of states, (A/2π)·k dk in 2-D, exactly as QMCPACK's SetupKVecs does.
//
// The normal equations must not be formed.  A = CᵀWC squares the condition
// number: measured cond(C) ~ 1e5–1e6, so cond(A) ~ 1e12 and half the digits
// are gone before the solve starts.  Householder QR on the rectangular system
// keeps them; measured, that is a factor of ~100 in the energy and it is what
// turns the error monotone in the number of knots.
struct RadialFunction {
    // f(r) for r > 0.  May be singular at the origin as long as r·f(r) is
    // integrable there (true for 1/r and for −log r in 2-D).
    std::function<double(double r)> value;
    // The 2-D Hankel transform f̃(k) = 2π ∫₀^∞ r J₀(kr) f(r) dr, in closed form.
    std::function<double(double k)> fourier;
    // f′ and f″ at r_c.  Analytic, not differenced: these two numbers are
    // ASSIGNED to the outermost basis coefficients rather than fitted, so an
    // error here is not averaged away.  A central second difference carries
    // ε/h² ≈ 1e-6 of round-off, which would become the accuracy floor.
    std::function<double(double r)> d1;
    std::function<double(double r)> d2;

    // Alternative to `fourier`, for functions with no closed-form transform
    // (a Padé Jastrow): the transform of the part BEYOND the cut,
    //     tail(k) = 2π ∫_{r_c}^∞ r J₀(kr) f(r) dr,
    // which is all the fit needs (X(k) = −tail(k)).  It is tied to one r_c, so
    // the radius it was built for travels with it and is checked.
    std::function<double(double k)> tail{};
    double                          tail_r_cut = 0.0;
};

struct BreakupOptions {
    int    n_knots     = 20;    // basis resolution over [0, r_c]
    double k_cut       = 0.0;   // reciprocal cutoff; 0 ⇒ 8·|b₁|
    double k_continuum = 0.0;   // discrete shells end here; 0 ⇒ 60π/r_c
    double k_max       = 0.0;   // continuum tail ends here; 0 ⇒ 6000/r_c
    int    n_continuum = 400;   // log-spaced points in the tail
};

class LongRangeBreakup {
public:
    LongRangeBreakup() = default;

    [[nodiscard]] double r_cut() const noexcept { return basis_.r_cut(); }
    [[nodiscard]] double k_cut() const noexcept { return k_cut_; }
    [[nodiscard]] const LpqhiBasis& basis() const noexcept { return basis_; }
    [[nodiscard]] const std::vector<double>& coefs() const noexcept { return coefs_; }

    // Retained reciprocal vectors (half space) and their weights.  The weight
    // already carries f̃_long(k)/(2·volume) with the ×2 for the folded half.
    [[nodiscard]] const std::vector<double>& kx() const noexcept { return kx_; }
    [[nodiscard]] const std::vector<double>& ky() const noexcept { return ky_; }
    [[nodiscard]] const std::vector<double>& weight() const noexcept { return weight_; }
    // Integer indices k = n1·b1 + n2·b2, which a consumer building exponential
    // ladders by recurrence needs in order to index them.
    [[nodiscard]] const std::vector<int>& n1() const noexcept { return n1_; }
    [[nodiscard]] const std::vector<int>& n2() const noexcept { return n2_; }
    [[nodiscard]] int n1_max() const noexcept { return n1_max_; }
    [[nodiscard]] int n2_max() const noexcept { return n2_max_; }

    // f_long(r) in real space: Σ_n t_n h_n(r) inside r_c, and f(r) beyond it.
    // The caller subtracts this from its own (vectorized) f to get f_short, so
    // no std::function survives into a hot loop.
    [[nodiscard]] double long_range_real(double r) const noexcept;

    // Hot-loop evaluator for f_long.  Inside a panel only six basis functions
    // are non-zero, so this costs six quintics instead of a sum over all 3N of
    // them; it agrees with long_range_real() to ~1e-14.  Measured 5.7x faster
    // than a 32-term Chebyshev Clenshaw of the same function, because the
    // Clenshaw's recurrence is a dependency chain of length 32 while the six
    // polynomials here are independent.
    [[nodiscard]] const LongRangeSpline& spline() const noexcept { return spline_; }

    // (1/A) ∫ f_short(r) d²r over the cell — the k = 0 Fourier component of the
    // short-range half, which is what a uniform neutralising background couples
    // to.  Zero-valued for a neutral cell's energy, since it enters multiplied
    // by (Σq)².  (QMCPACK calls the same quantity evaluateSR_k0.)
    [[nodiscard]] double short_range_k0() const noexcept { return sr_k0_; }

    // f_long(0) = t_0 — every other basis function vanishes at the origin.
    // This is the self-energy constant: −½ Σ_i q_i² f_long(0).
    [[nodiscard]] double long_range_zero() const noexcept { return long_range_zero_; }

    // ── Diagnostics.  Read the comment before using any of these as a gate ──
    // cond(C), from the QR — NOT cond(CᵀWC), which is its square.
    [[nodiscard]] double condition() const noexcept { return condition_; }
    // max |f̃_long(k)| over the DISCRETE shells just above k_cut.  This is the
    // honest gate: it is what the truncated reciprocal sum actually drops.
    [[nodiscard]] double residual_above_cut() const noexcept { return residual_; }
    // Weighted RMS of the fit residual.  NOT an error bar: its value depends on
    // how far in k the sum was taken (widening k_max adds shells where the
    // residual is ~0 and lowers it without changing any answer), and the
    // continuum weights (~1e3) swamp the discrete degeneracies (~4–16) just
    // above k_cut, which is exactly where the energy loses accuracy.  Use it to
    // compare fits of equal range, never as an absolute tolerance.
    [[nodiscard]] double chi() const noexcept { return chi_; }
    [[nodiscard]] double max_coefficient() const noexcept { return t_max_; }

    friend LongRangeBreakup build_long_range_breakup(const SimulationCell&,
                                                     const RadialFunction&,
                                                     const BreakupOptions&);

private:
    LpqhiBasis          basis_;
    std::vector<double> coefs_;
    std::vector<double> kx_, ky_, weight_;
    std::vector<int>    n1_, n2_;
    int                 n1_max_ = 0, n2_max_ = 0;
    LongRangeSpline     spline_;
    std::function<double(double)> value_;   // f(r), for r ≥ r_c
    double k_cut_           = 0.0;
    double long_range_zero_ = 0.0;
    double sr_k0_           = 0.0;
    double condition_       = 0.0;
    double residual_        = 0.0;
    double chi_             = 0.0;
    double t_max_           = 0.0;
};

// Throws std::invalid_argument if the cell is not periodic and 2-D, if r_c
// would exceed the Wigner-Seitz inradius (minimum image would then be wrong),
// or if the options are inconsistent.
[[nodiscard]] LongRangeBreakup build_long_range_breakup(
    const SimulationCell& cell, const RadialFunction& f,
    const BreakupOptions& opt);

}  // namespace alea::numerics
