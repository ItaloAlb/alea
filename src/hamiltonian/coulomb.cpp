// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/coulomb.h"

#include <cmath>

#include "population/walker_set.h"

namespace alea {
std::vector<RadialPairPotential> CoulombTerm::radial_pair_potentials() const {
    if (!std::isfinite(epsilon_) || epsilon_<=0 || !std::isfinite(d2_))
        throw std::invalid_argument("coulomb pair action: invalid dielectric or layer separation");
    const double coefficient=group_a_.charge*group_b_.charge*inv_epsilon_,offset=d2_;
    if (!std::isfinite(coefficient)) throw std::invalid_argument("coulomb pair action: nonfinite charge coefficient");
    return {{group_a_,group_b_,[coefficient,offset](double r) {
        return coefficient==0 ? 0. : coefficient/std::sqrt(r*r+offset);
    },offset==0?coefficient:0.,"analytic Coulomb; cell integration of the 2D pole"}};
}

CoulombTerm::CoulombTerm(ParticleGroup group_a, ParticleGroup group_b,
                         double epsilon, double d_layer)
    : group_a_(std::move(group_a)),
      group_b_(std::move(group_b)),
      epsilon_(epsilon),
      inv_epsilon_(1.0 / epsilon),
      d2_(d_layer * d_layer),
      is_intragroup_(group_a_ == group_b_) {}

void CoulombTerm::evaluate_potential(const ParticleBatch& batch,
                                 std::span<double> v_out,
                                 std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw = batch.num_walkers();   // global stride
    if (w_end > nw) w_end = nw;
    const BatchedDistanceTable& dt = batch.dist_tables();
    double* __restrict le = v_out.data();

    // Accumulate coeff/r_ij over the contiguous walker row of pair (i,j).
    const double d2 = d2_;
    auto add_pair = [&](double coeff, std::size_t i, std::size_t j) {
        const double* __restrict r = dt.row(i, j).data();
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w)
            le[w] += coeff / std::sqrt(r[w] * r[w] + d2);
    };

    if (is_intragroup_) {
        // Intra-group: V = Σ_{i<j in A} q² / (ε r_ij).
        const double q2 = group_a_.charge * group_a_.charge * inv_epsilon_;
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j)
                add_pair(q2, i, j);
    } else {
        // Inter-group: V = Σ_{i∈A, j∈B} q_a q_b / (ε r_ij).
        const double qq = group_a_.charge * group_b_.charge * inv_epsilon_;
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j)
                add_pair(qq, i, j);
    }
}

std::unique_ptr<PotentialGradientWorkspace>
CoulombTerm::make_potential_gradient_workspace(const ParticleBatch& batch) const {
    const double coefficient = group_a_.charge *
        (is_intragroup_ ? group_a_.charge : group_b_.charge) * inv_epsilon_;
    if (group_a_.first > group_a_.last || group_b_.first > group_b_.last ||
        group_a_.last > batch.n_particles() || group_b_.last > batch.n_particles() ||
        !std::isfinite(epsilon_) || epsilon_ <= 0.0 || !std::isfinite(d2_) ||
        !std::isfinite(coefficient))
        throw std::invalid_argument("coulomb: invalid potential gradient groups or parameters");
    return HamiltonianComponent::make_potential_gradient_workspace(batch);
}

void CoulombTerm::evaluate_potential_gradient(const ParticleBatch& batch,
    std::span<double> out, PotentialGradientWorkspace& workspace,
    std::size_t w_begin, std::size_t w_end) const {
    w_end = workspace.checked_end(batch, *this, out, w_begin, w_end);
    const auto nw = batch.num_walkers(), dim = batch.dim();
    const auto positions = batch.positions(); // [np*dim*nw]
    const double coefficient = group_a_.charge *
        (is_intragroup_ ? group_a_.charge : group_b_.charge) * inv_epsilon_;
    if (coefficient == 0.0) return;
    for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
        for (std::size_t j = is_intragroup_ ? i+1 : group_b_.first; j < group_b_.last; ++j)
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double delta[SimulationCell::kMaxDim]; // [dim] Cartesian displacement
                for (std::size_t d = 0; d < dim; ++d)
                    delta[d] = positions[(i*dim+d)*nw+w]-positions[(j*dim+d)*nw+w];
                const double in_plane2 = batch.cell().min_image_dist2(delta);
                const double radius2 = in_plane2+d2_;
                if (!std::isfinite(radius2) || radius2 == 0.0)
                    throw std::domain_error("coulomb: potential gradient at singular or nonfinite separation");
                // The bounded interlayer coincidence is exactly stationary,
                // even when a tiny finite offset makes the radial scale large.
                if (in_plane2 == 0.0) continue;
                // Normalize Delta before multiplying by the inverse-square
                // strength. A 1/r^3 scale could underflow even when G~1/r^2
                // is representable (e.g. r=1e150, G~1e-300).
                const double strength = -coefficient/radius2;
                const double radius = std::sqrt(radius2);
                for (std::size_t d = 0; d < dim; ++d) {
                    const double gradient = strength*(delta[d]/radius);
                    if (!std::isfinite(gradient))
                        throw std::domain_error("coulomb: potential gradient exceeds floating-point range");
                    out[(i*dim+d)*nw+w] += gradient;
                    out[(j*dim+d)*nw+w] -= gradient;
                }
            }
}

// Changed pairs only: O(N) radial evaluations, committed batch untouched.
double CoulombTerm::delta_potential_particle(ParticleBatch& configs,
    std::size_t p, std::size_t k, std::span<const double> proposed,
    std::span<double>) const {
    const double coefficient = group_a_.charge *
        (is_intragroup_ ? group_a_.charge : group_b_.charge) * inv_epsilon_;
    return radial_particle_delta(configs, p, k, proposed, group_a_, group_b_,
        [&](double r) { return coefficient / std::sqrt(r*r + d2_); });
}

double CoulombTerm::coulomb_cusp_residue(const ParticleGroup& a,
                                         const ParticleGroup& b) const {
    // Match the requested pair against this term's groups in either order.
    const bool same = (a == group_a_ && b == group_b_) ||
                      (a == group_b_ && b == group_a_);
    if (!same) return 0.0;
    // A finite layer offset makes V smooth at r=0 (no 1/r), so no cusp.
    if (d2_ > 0.0) return 0.0;
    return group_a_.charge * group_b_.charge * inv_epsilon_;
}

}  // namespace alea
