// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/moire.h"

#include <cmath>
#include <limits>

#include "numerics/special_functions.h"
#include "population/walker_set.h"

namespace alea {

bool MoireTerm::supports_periodic_virial(const SimulationCell& cell, std::size_t np) const noexcept {
    if (!cell.is_periodic() || cell.dim() != dim_ || !std::isfinite(V0_) ||
        !std::isfinite(two_Vtilde_) || !std::isfinite(phi_)) return false;
    const auto vectors=cell.vectors();
    for (const auto& [first,last] : ranges_)
        if (first > last || last > np) return false;
    for (const auto& g : g_) {
        if (g.size() != dim_) return false;
        for (std::size_t a=0; a<dim_; ++a) {
            double phase=0.0,magnitude=0.0;
            for (std::size_t d=0; d<dim_; ++d) {
                const double term=g[d]*vectors[a*dim_+d]/(2.0*numerics::PI);
                phase+=term; magnitude+=std::abs(term);
            }
            // Cell faces cancel only for reciprocal vectors commensurate with
            // this lattice. Admit roundoff, never a physical mismatch.
            const double tolerance=std::min(1e-10,64.0*std::numeric_limits<double>::epsilon()*
                std::max(1.0,magnitude));
            if (!std::isfinite(phase) || std::abs(phase-std::round(phase)) > tolerance) return false;
        }
    }
    return true;
}

// Triangular first-shell reciprocal vectors: a single magnitude
// |g| = 4π/(√3 a_M), three directions 120° apart from theta0.
std::vector<std::vector<double>>
moire_reciprocal_vectors(double a_M, double theta0, std::size_t dim) {
    const double g_mag = 4.0 * numerics::PI / (std::sqrt(3.0) * a_M);
    const double third = 2.0 * numerics::PI / 3.0;  // 120°

    std::vector<std::vector<double>> g(3, std::vector<double>(dim, 0.0));
    for (std::size_t i = 0; i < 3; ++i) {
        const double angle = theta0 + third * static_cast<double>(i);
        if (dim > 0) g[i][0] = g_mag * std::cos(angle);
        if (dim > 1) g[i][1] = g_mag * std::sin(angle);
    }
    return g;
}

MoireTerm::MoireTerm(std::size_t dim, double V0, double Vtilde, double phi,
                     std::vector<std::vector<double>> g_vectors,
                     std::vector<std::pair<std::size_t, std::size_t>> ranges)
    : dim_(dim),
      V0_(V0),
      two_Vtilde_(2.0 * Vtilde),
      phi_(phi),
      g_(std::move(g_vectors)),
      ranges_(std::move(ranges)) {}

void MoireTerm::evaluate_potential(const ParticleBatch& batch,
                               std::span<double> v_out,
                               std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw  = batch.num_walkers();  // global stride
    const std::size_t dim = dim_;
    if (w_end > nw) w_end = nw;
    const double* __restrict pos = batch.positions().data();
    double* __restrict le = v_out.data();

    // V(r_p) = V0 + 2Ṽ Σ_i cos(g_i·r_p + φ), summed over every particle p of the
    // listed groups.  The walker loop is the contiguous, SIMD-friendly inner loop
    // (positions are SoA with w varying fastest); the per-g cos pass over walkers
    // dominates the cost.
    const double two_v = two_Vtilde_;
    const double phi   = phi_;
    for (const auto& [first, last] : ranges_) {
        for (std::size_t p = first; p < last; ++p) {
            const double* __restrict rp = pos + p * dim * nw;  // r_p, SoA
            for (std::size_t w = w_begin; w < w_end; ++w) le[w] += V0_;
            for (const std::vector<double>& gv : g_) {
                const double* __restrict g = gv.data();
                for (std::size_t w = w_begin; w < w_end; ++w) {
                    double gr = 0.0;
                    for (std::size_t d = 0; d < dim; ++d)
                        gr += g[d] * rp[d * nw + w];
                    le[w] += two_v * std::cos(gr + phi);
                }
            }
        }
    }
}

std::unique_ptr<PotentialGradientWorkspace>
MoireTerm::make_potential_gradient_workspace(const ParticleBatch& batch) const {
    if (dim_ != batch.dim() || !std::isfinite(V0_) || !std::isfinite(two_Vtilde_) ||
        !std::isfinite(phi_))
        throw std::invalid_argument("moire: invalid potential gradient shape or parameters");
    for (const auto& g : g_) {
        if (g.size() != dim_) throw std::invalid_argument("moire: gradient reciprocal-vector dimension mismatch");
        for (double x : g)
            if (!std::isfinite(x)) throw std::invalid_argument("moire: nonfinite gradient reciprocal vector");
    }
    for (const auto& [first,last] : ranges_)
        if (first > last || last > batch.n_particles())
            throw std::invalid_argument("moire: potential gradient particle range is invalid");
    return HamiltonianComponent::make_potential_gradient_workspace(batch);
}

void MoireTerm::evaluate_potential_gradient(const ParticleBatch& batch,
    std::span<double> out, PotentialGradientWorkspace& workspace,
    std::size_t w_begin, std::size_t w_end) const {
    w_end = workspace.checked_end(batch, *this, out, w_begin, w_end);
    const auto nw = batch.num_walkers();
    const auto positions = batch.positions(); // [np*dim*nw]
    for (const auto& [first,last] : ranges_)
        for (std::size_t p = first; p < last; ++p)
            for (const auto& g : g_) {
                for (std::size_t w = w_begin; w < w_end; ++w) {
                    double phase = phi_;
                    for (std::size_t d = 0; d < dim_; ++d)
                        phase += g[d]*positions[(p*dim_+d)*nw+w];
                    const double scale = -two_Vtilde_*std::sin(phase);
                    for (std::size_t d = 0; d < dim_; ++d)
                        out[(p*dim_+d)*nw+w] += scale*g[d];
                }
            }
}

// O(number of first-shell vectors) independent of particle count.
double MoireTerm::delta_potential_particle(ParticleBatch& configs,
    std::size_t p, std::size_t k, std::span<const double> proposed,
    std::span<double>) const {
    double candidate[SimulationCell::kMaxDim];
    for (std::size_t d = 0; d < dim_; ++d) candidate[d] = proposed[d];
    configs.cell().wrap(candidate);
    double result = 0.0;
    for (const auto& [first, last] : ranges_) {
        if (p < first || p >= last) continue;
        for (const auto& g : g_) {
            double old_phase = phi_, next_phase = phi_;
            for (std::size_t d = 0; d < dim_; ++d) {
                old_phase += g[d] * configs.positions()[p * dim_ * configs.num_walkers() + d * configs.num_walkers() + k];
                next_phase += g[d] * candidate[d];
            }
            result += two_Vtilde_ * (std::cos(next_phase) - std::cos(old_phase));
        }
    }
    return result;
}

}  // namespace alea
