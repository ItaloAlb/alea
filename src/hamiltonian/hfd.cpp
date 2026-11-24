// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/hfd.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "numerics/quadrature.h"
#include "population/particle_batch.h"
#include "population/wigner_seitz_exterior.h"

namespace alea {

namespace {

// V*(x) of the header, branch-free in the damping (y = 0 above D).
inline double reduced_value(double x, const HfdParameters& q) noexcept {
    const double inv_x  = 1.0 / x;
    const double inv_x2 = inv_x * inv_x;
    const double inv_x6 = inv_x2 * inv_x2 * inv_x2;
    const double s = inv_x6 * (q.C6 + inv_x2 * (q.C8 + inv_x2 * q.C10));
    const double y = std::max(q.D * inv_x - 1.0, 0.0);
    return q.A * std::exp(x * (q.beta * x - q.alpha)) - std::exp(-y * y) * s;
}

// dV*/dx of the header.
inline double reduced_slope(double x, const HfdParameters& q) noexcept {
    const double inv_x  = 1.0 / x;
    const double inv_x2 = inv_x * inv_x;
    const double inv_x6 = inv_x2 * inv_x2 * inv_x2;
    const double s  = inv_x6 * (q.C6 + inv_x2 * (q.C8 + inv_x2 * q.C10));
    const double ds = -inv_x6 * inv_x * (6.0 * q.C6 + inv_x2 * (8.0 * q.C8 + inv_x2 * 10.0 * q.C10));
    const double y  = std::max(q.D * inv_x - 1.0, 0.0);
    const double f  = std::exp(-y * y);
    const double df = f * 2.0 * y * q.D * inv_x2;
    return q.A * (2.0 * q.beta * x - q.alpha) * std::exp(x * (q.beta * x - q.alpha)) - df * s -
           f * ds;
}

}  // namespace

HfdTerm::HfdTerm(ParticleGroup group_a, ParticleGroup group_b, const HfdParameters& parameters,
                 const SimulationCell& cell)
    : group_a_(std::move(group_a)),
      group_b_(std::move(group_b)),
      q_(parameters),
      is_intragroup_(group_a_ == group_b_) {
    const double all[] = {q_.epsilon, q_.r_m, q_.A, q_.alpha, q_.beta, q_.C6, q_.C8, q_.C10, q_.D};
    for (double v : all)
        if (!std::isfinite(v)) throw std::invalid_argument("hfd: every parameter must be finite");
    if (q_.epsilon <= 0.0 || q_.r_m <= 0.0 || q_.D <= 0.0)
        throw std::invalid_argument("hfd: epsilon, r_m and D must be positive");
    if (q_.A < 0.0) throw std::invalid_argument("hfd: A must not be negative");
    if (q_.C6 < 0.0 || q_.C8 < 0.0 || q_.C10 < 0.0)
        throw std::invalid_argument("hfd: C6, C8 and C10 must not be negative");
    // A exp(−αx + βx²) must decay as x → ∞, or V grows without bound with r.
    if (q_.beta > 0.0 || (q_.beta == 0.0 && q_.alpha <= 0.0))
        throw std::invalid_argument(
            "hfd: the repulsive exponential A exp(-alpha x + beta x^2) does not decay at large "
            "r; it needs beta < 0, or beta = 0 with alpha > 0");
    inv_r_m_ = 1.0 / q_.r_m;

    if (!cell.is_periodic()) return;
    const std::size_t dim = cell.dim();
    const auto n_a = static_cast<double>(group_a_.size());
    const auto n_b = static_cast<double>(group_b_.size());
    const double pairs = is_intragroup_ ? 0.5 * n_a * n_a : n_a * n_b;
    const double outside =
        wigner_seitz_exterior_integral(cell, [&](double rho) { return radial_tail(rho, dim); });
    tail_ = pairs / cell.volume() * outside;
    tail_pair_density_ = outside / cell.volume();
}

double HfdTerm::pair_value(double r) const noexcept {
    return q_.epsilon * reduced_value(r * inv_r_m_, q_);
}

double HfdTerm::slope_over_r(double r) const noexcept {
    const double x = r * inv_r_m_;
    return q_.epsilon * inv_r_m_ * inv_r_m_ * reduced_slope(x, q_) / x;
}

double HfdTerm::radial_tail(double rho, std::size_t dim, int panels) const {
    const double d = static_cast<double>(dim);
    const auto integrand = [&](double t) { return pair_value(rho / t) * std::pow(t, -d - 1.0); };
    const double t_kink = rho * inv_r_m_ / q_.D;  // x = D
    const double integral =
        t_kink < 1.0 ? numerics::gauss20_panels(integrand, 0.0, t_kink, panels) +
                           numerics::gauss20_panels(integrand, t_kink, 1.0, panels)
                     : numerics::gauss20_panels(integrand, 0.0, 1.0, panels);
    return std::pow(rho, d) * integral;
}

void HfdTerm::evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                                 std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw = batch.num_walkers();  // global stride
    if (w_end > nw) w_end = nw;
    const BatchedDistanceTable& dt = batch.dist_tables();
    double* __restrict v = v_out.data();
    const HfdParameters q = q_;
    const double inv_r_m = inv_r_m_;

    // Accumulate v(r) over the contiguous walker row of pair (i, j); r is the
    // minimum-image distance in a periodic cell.
    auto add_pair = [&](std::size_t i, std::size_t j) {
        const double* __restrict r = dt.row(i, j).data();
        for (std::size_t w = w_begin; w < w_end; ++w)
            v[w] += q.epsilon * reduced_value(r[w] * inv_r_m, q);
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

double HfdTerm::delta_potential_particle(ParticleBatch& configs, std::size_t p, std::size_t k,
                                         std::span<const double> proposed,
                                         std::span<double>) const {
    return radial_particle_delta(configs, p, k, proposed, group_a_, group_b_,
                                 [this](double r) { return pair_value(r); });
}

double HfdTerm::bead_energy(const ParticleBatch& configs, std::size_t w, std::size_t p,
                            std::span<const double> position, std::size_t n_rows,
                            std::size_t skip) const {
    if (!is_intragroup_)
        throw std::logic_error("hfd: a bead energy needs an intra-group term");
    if (!group_a_.contains(p)) return 0.0;
    std::size_t counted = 0;
    const double pairs = radial_bead_energy(configs, w, p, position, n_rows, skip, group_a_,
                                            [this](double r) { return pair_value(r); }, counted);
    return pairs + tail_pair_density_ * (static_cast<double>(counted) + 0.5);
}

void HfdTerm::configuration_gradient(const ParticleBatch& configs, std::size_t w, std::size_t p,
                                     std::size_t n_rows, std::size_t skip,
                                     std::span<double> out) const {
    radial_configuration_gradient(configs, w, p, n_rows, skip, group_a_, group_b_,
                                  [this](double r) { return slope_over_r(r); }, out);
}

void HfdTerm::bead_gradient(const ParticleBatch& configs, std::size_t w, std::size_t p,
                            std::span<const double> position, std::size_t n_rows,
                            std::size_t skip, std::span<double> self,
                            std::span<double> rows) const {
    radial_bead_gradient(configs, w, p, position, n_rows, skip, group_a_, group_b_,
                         [this](double r) { return slope_over_r(r); }, self, rows);
}

// The conditions of the torus identity are in the header (PIMC periodic
// virial); what depends on the run is only that the cell is periodic and the
// groups are rows of it.
bool HfdTerm::supports_periodic_virial(const SimulationCell& cell, std::size_t np) const noexcept {
    return cell.is_periodic() && group_a_.first <= group_a_.last &&
           group_b_.first <= group_b_.last && group_a_.last <= np && group_b_.last <= np;
}

std::unique_ptr<PotentialGradientWorkspace>
HfdTerm::make_potential_gradient_workspace(const ParticleBatch& batch) const {
    if (group_a_.first > group_a_.last || group_b_.first > group_b_.last ||
        group_a_.last > batch.n_particles() || group_b_.last > batch.n_particles())
        throw std::invalid_argument("hfd: the potential gradient groups do not fit the batch");
    return HamiltonianComponent::make_potential_gradient_workspace(batch);
}

void HfdTerm::evaluate_potential_gradient(const ParticleBatch& batch, std::span<double> out,
                                          PotentialGradientWorkspace& workspace,
                                          std::size_t w_begin, std::size_t w_end) const {
    w_end = workspace.checked_end(batch, *this, out, w_begin, w_end);
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    const auto positions = batch.positions();  // [np * dim * nw]
    const SimulationCell& cell = batch.cell();
    auto add_pair = [&](std::size_t i, std::size_t j) {
        for (std::size_t w = w_begin; w < w_end; ++w) {
            double delta[SimulationCell::kMaxDim];  // [dim] Δ = r_i − r_j, minimum image
            for (std::size_t d = 0; d < dim; ++d)
                delta[d] = positions[(i * dim + d) * nw + w] - positions[(j * dim + d) * nw + w];
            const double r2 = cell.min_image_dist2(delta);
            if (r2 == 0.0) continue;  // the tip of the cone V(Δ) (header)
            const double g = slope_over_r(std::sqrt(r2));
            for (std::size_t d = 0; d < dim; ++d) {
                out[(i * dim + d) * nw + w] += g * delta[d];
                out[(j * dim + d) * nw + w] -= g * delta[d];
            }
        }
    };
    if (is_intragroup_) {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j) add_pair(i, j);
    } else {
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j) add_pair(i, j);
    }
}

}  // namespace alea
