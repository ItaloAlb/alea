// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Pair-Jastrow radial functions in the form the optimized breakup consumes, so
// that a pair Jastrow under periodic boundaries can be summed over ALL images
// (Ewald) rather than only the nearest one.
//
// Why the trial function needs this: Ceperley, "Introduction to QMC methods
// applied to the electron gas", §2.5 — "the Ewald image method is needed to
// represent the correct long-range behavior of the potential and the trial
// function", and Ψ must be properly periodic for the variational bound to hold.
// The splitting itself is Natoli & Ceperley, J. Comput. Phys. 117, 171 (1995),
// the same one the Coulomb term uses (numerics/long_range_breakup.h).
//
// A Jastrow has no closed-form Fourier transform in general, so these supply
// the breakup's `tail` instead of `fourier`:
//     tail(k) = 2π ∫_{r_c}^∞ r J₀(kr) f(r) dr.

#include <span>

#include "numerics/long_range_breakup.h"

namespace alea::numerics {

// Padé u(r) = (Σ_{k=1}^N a_k r^k) / (1 + Σ_{k=1}^M b_k r^k), as f = u − u∞.
//
// u∞ = lim u (a_N/b_M if N = M, 0 if N < M) only shifts ln Ψ by a constant, so
// it is dropped; what is summed over images is u − u∞, whose 1/r tail is then
// handled by the breakup exactly as Coulomb's is (the divergent k = 0 part is a
// configuration-independent constant in ln Ψ).
//
// tail(k): the large-r expansion u − u∞ = Σ_j e_j r^{−j} (power-series
// division of the rational function in x = 1/r) fixes a model
//     m(r) = Σ_{n=1}^{4} d_n (r² + s²)^{−n/2},
// matched so the remainder h = f − m is O(r^{−5}).  Each model term has a
// closed-form 2-D Hankel transform, from Gradshteyn & Ryzhik 6.565.4,
//     ∫₀^∞ r J₀(kr) (r² + s²)^{−μ−1} dr = (k/s)^μ K_μ(ks) / (2^μ Γ(μ+1)),
// with μ = n/2 − 1:
//     n = 1: e^{−ks}/k     n = 2: K₀(ks)     n = 3: e^{−ks}/s     n = 4: k K₁(ks)/(2s)
// (all ×2π).  The remainder is integrated numerically, split by a smooth
// window into a finite piece on [r_c, 2r_c], which carries the sharp edge at r_c
// and is needed at every k, and a smooth decaying piece whose transform is
// negligible beyond k·r_c ≈ 400.  `model_width` is s (0 → r_c/2); the result
// does not depend on it.
//
// Throws if N > M (u grows without bound: no image sum exists) or if the
// denominator vanishes at some r > 0 (a pole that the image sum would reach).
[[nodiscard]] RadialFunction pade_radial(std::span<const double> a,
                                         std::span<const double> b,
                                         double r_cut, double model_width = 0.0);

// u∞ = lim_{r→∞} of the same rational (a_N/b_M if N = M, else 0); what
// pade_radial subtracts.  Same validity rules (throws if N > M).
[[nodiscard]] double pade_limit(std::span<const double> a, std::span<const double> b);

}  // namespace alea::numerics
