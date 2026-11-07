// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/inverse_power.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "numerics/special_functions.h"
#include "population/particle_batch.h"

namespace alea {

namespace {

// Exponents up to this value are evaluated by repeated multiplication.
constexpr int kMaxIntegerExponent = 16;

}  // namespace

double outside_wigner_seitz_integral(const SimulationCell& cell, double exponent) {
    if (!cell.is_periodic())
        throw std::invalid_argument("outside_wigner_seitz_integral: open cell");
    const std::size_t dim = cell.dim();
    const double d = static_cast<double>(dim);
    if (!(exponent > d))
        throw std::invalid_argument(
            "outside_wigner_seitz_integral: needs exponent > dimension");
    const std::span<const double> a = cell.vectors();  // [i*dim + d]

    // 1-D: the WS cell is [−L/2, L/2]; two directions, ρ = L/2 each.
    if (dim == 1) return 2.0 * std::pow(0.5 * std::abs(a[0]), 1.0 - exponent) / (exponent - 1.0);

    // 2-D.  vectors() is the Gauss-reduced basis (SimulationCell's ctor), so
    // every WS face is the bisector of one of ±a₁, ±a₂, ±(a₁ ± a₂); the range
    // [−2, 2] is a margin, and a translation that is not a face never wins the
    // minimum below.
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
        sum += std::pow(rho, d - exponent);
    }
    return sum * dtheta / (exponent - d);
}

InversePowerTerm::InversePowerTerm(ParticleGroup group_a, ParticleGroup group_b,
                                   double strength, double exponent,
                                   const SimulationCell& cell)
    : group_a_(std::move(group_a)),
      group_b_(std::move(group_b)),
      strength_(strength),
      exponent_(exponent),
      is_intragroup_(group_a_ == group_b_) {
    if (!std::isfinite(exponent_) || exponent_ <= 0.0)
        throw std::invalid_argument(
            "inverse_power: exponent must be finite and positive, got " +
            std::to_string(exponent_));
    if (!std::isfinite(strength_))
        throw std::invalid_argument("inverse_power: strength must be finite");
    // An attractive core at least as strong as 1/r² has no two-body ground
    // state: the energy is unbounded below as the pair collapses.
    if (strength_ < 0.0 && exponent_ >= 2.0)
        throw std::invalid_argument(
            "inverse_power: an attractive core (strength < 0) with exponent >= 2 "
            "falls to the centre: the two-body problem has no ground state.");

    const double rounded = std::round(exponent_);
    if (rounded == exponent_ && rounded <= kMaxIntegerExponent)
        integer_exponent_ = static_cast<int>(rounded);

    if (!cell.is_periodic()) return;

    if (exponent_ <= static_cast<double>(cell.dim()))
        throw std::invalid_argument(
            "inverse_power: in a periodic " + std::to_string(cell.dim()) +
            "-D cell the exponent must exceed the dimension (got " +
            std::to_string(exponent_) + "): only then does the sum over periodic "
            "images converge absolutely, which the minimum image plus continuum "
            "tail relies on.");
    const auto n_a = static_cast<double>(group_a_.size());
    const auto n_b = static_cast<double>(group_b_.size());
    const double pairs = is_intragroup_ ? 0.5 * n_a * n_a : n_a * n_b;
    tail_ = pairs / cell.volume() * strength_ *
            outside_wigner_seitz_integral(cell, exponent_);
}

void InversePowerTerm::evaluate_potential(const ParticleBatch& batch,
                                          std::span<double> v_out,
                                          std::size_t w_begin,
                                          std::size_t w_end) const {
    const std::size_t nw = batch.num_walkers();  // global stride
    if (w_end > nw) w_end = nw;
    const BatchedDistanceTable& dt = batch.dist_tables();
    double* __restrict v = v_out.data();
    const double c = strength_, p = exponent_;
    const int p_int = integer_exponent_;

    // Accumulate C/r^p over the contiguous walker row of pair (i,j); r is the
    // minimum-image distance in a periodic cell.
    auto add_pair = [&](std::size_t i, std::size_t j) {
        const double* __restrict r = dt.row(i, j).data();
        if (p_int > 0) {
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w) {
                const double inv_r = 1.0 / r[w];
                double inv_rp = inv_r;
                for (int k = 1; k < p_int; ++k) inv_rp *= inv_r;
                v[w] += c * inv_rp;
            }
        } else {
            for (std::size_t w = w_begin; w < w_end; ++w) v[w] += c * std::pow(r[w], -p);
        }
    };

    if (is_intragroup_) {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j) add_pair(i, j);
    } else {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j) add_pair(i, j);
    }

    if (tail_ != 0.0)
        for (std::size_t w = w_begin; w < w_end; ++w) v[w] += tail_;
}

double InversePowerTerm::coulomb_cusp_residue(const ParticleGroup& a,
                                              const ParticleGroup& b) const {
    return is_own_pair(a, b) && exponent_ == 1.0 ? strength_ : 0.0;
}

PowerLawCore InversePowerTerm::power_law_core(const ParticleGroup& a,
                                              const ParticleGroup& b) const {
    if (!is_own_pair(a, b) || exponent_ <= 1.0) return {};
    return {strength_, exponent_};
}

}  // namespace alea
