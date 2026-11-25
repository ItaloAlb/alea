// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "numerics/quadrature.h"
#include "numerics/special_functions.h"
#include "population/simulation_cell.h"

namespace alea {

// ∫ over everything outside the Wigner–Seitz (WS) cell of a radial v(r):
//
//     ∫_{outside WS} v(|r|) d^d r = ∮ dΩ  f(ρ(Ω)),   f(ρ) = ∫_ρ^∞ v(r) r^{d−1} dr,
//
// with ρ(Ω) the distance from the cell's centre to the WS boundary along the
// direction Ω.  The caller supplies f, the radial tail of its potential.  This
// is the continuum (g → n) part of a pair sum whose pairs are taken at their
// minimum image (hamiltonian/inverse_power.h).  Setup only.
//
//   1-D: two directions, ρ = L/2:  2 f(L/2).
//   2-D: ρ(θ) = min over lattice vectors L with u·L > 0 of |L|²/(2 u·L),
//        u = (cos θ, sin θ); a midpoint rule in θ with 2^18 points, whose only
//        error comes from the ≤ 6 corners of the WS polygon, O(Δθ²) ≈ 1e−10
//        relative.  vectors() is the Gauss-reduced basis, so every face is the
//        bisector of one of ±a₁, ±a₂, ±(a₁ ± a₂); the range [−2, 2] is a
//        margin, and a translation that is not a face never wins the minimum.
//   3-D: only orthorhombic cells exist (SimulationCell), whose WS cell is the
//        box [−h_x, h_x] × [−h_y, h_y] × [−h_z, h_z].  A direction leaves it
//        through one face; on the face x = h_x at p = (h_x, b, c),
//        dΩ = h_x db dc / |p|³, so
//            ∮ dΩ f(ρ) = Σ_faces ∫∫ (h/|p|³) f(|p|) db dc.
//        The integrand is analytic on each face (|p| ≥ h > 0); the ± faces and
//        the four quadrants of a face are equal, and each quadrant takes a
//        product of 8 × 8 Gauss–Legendre panels of 20 points.
//
// Checks of the rule (test_inverse_power): f = 1 gives the full solid angle
// (2, 2π, 4π) and f(ρ) = ρ^d/d the cell volume, both exactly up to the rule's
// error, and C/r^p agrees with an independent angular midpoint rule.
template <class RadialTail>
double wigner_seitz_exterior_integral(const SimulationCell& cell, RadialTail&& f) {
    if (!cell.is_periodic())
        throw std::invalid_argument("wigner_seitz_exterior_integral: open cell");
    const std::size_t dim = cell.dim();
    const std::span<const double> a = cell.vectors();  // [i*dim + d]

    if (dim == 1) return 2.0 * f(0.5 * std::abs(a[0]));

    if (dim == 2) {
        std::vector<double> lx, ly;  // translations L = s a₁ + t a₂ ≠ 0
        for (int s = -2; s <= 2; ++s)
            for (int t = -2; t <= 2; ++t) {
                if (s == 0 && t == 0) continue;
                lx.push_back(s * a[0] + t * a[2]);
                ly.push_back(s * a[1] + t * a[3]);
            }
        constexpr std::size_t n_theta = std::size_t{1} << 18;
        const double dtheta = 2.0 * numerics::PI / static_cast<double>(n_theta);
        double sum = 0.0;
        for (std::size_t k = 0; k < n_theta; ++k) {
            const double theta = (static_cast<double>(k) + 0.5) * dtheta;
            const double ux = std::cos(theta), uy = std::sin(theta);
            // The bisector of L is {r : r·L = |L|²/2}; along u it is reached at
            // ρ = |L|²/(2 u·L), for the translations that u points towards.
            double rho = std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < lx.size(); ++c) {
                const double proj = ux * lx[c] + uy * ly[c];
                if (proj > 0.0)
                    rho = std::min(rho, 0.5 * (lx[c] * lx[c] + ly[c] * ly[c]) / proj);
            }
            sum += f(rho);
        }
        return sum * dtheta;
    }

    if (!cell.orthorhombic())
        throw std::logic_error("wigner_seitz_exterior_integral: a 3-D cell must be orthorhombic");
    const double half[3] = {0.5 * std::abs(a[0]), 0.5 * std::abs(a[4]), 0.5 * std::abs(a[8])};
    constexpr int panels = 8;
    double total = 0.0;
    for (std::size_t normal = 0; normal < 3; ++normal) {
        const double h  = half[normal];
        const double hb = half[(normal + 1) % 3], hc = half[(normal + 2) % 3];
        const double face = numerics::gauss20_panels(
            [&](double b) {
                return numerics::gauss20_panels(
                    [&](double c) {
                        const double p = std::sqrt(h * h + b * b + c * c);
                        return h / (p * p * p) * f(p);
                    },
                    0.0, hc, panels);
            },
            0.0, hb, panels);
        total += 8.0 * face;  // two faces × four quadrants
    }
    return total;
}

}  // namespace alea
