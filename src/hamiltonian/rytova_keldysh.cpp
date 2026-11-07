// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "hamiltonian/rytova_keldysh.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include "numerics/keldysh_kernel.h"
#include "numerics/special_functions.h"
#include "population/walker_set.h"

namespace alea {
std::vector<RadialPairPotential> KeldyshTerm::radial_pair_potentials() const {
    const double coefficient=group_a_.charge*group_b_.charge*prefactor_,offset=d2_,inverse=inv_r0_;
    if (!std::isfinite(epsilon_) || epsilon_<=0 || !std::isfinite(r0_) || r0_<=0 ||
        !std::isfinite(coefficient) || !std::isfinite(inverse) || !std::isfinite(offset))
        throw std::invalid_argument("RK pair action: invalid radial scaling");
    const auto* kernel=cheb_;
    return {{group_a_,group_b_,[coefficient,offset,inverse,kernel](double r) {
        return coefficient*kernel->eval(std::sqrt(r*r+offset)*inverse);
    },0.,potential_approximation_notes()}};
}

KeldyshTerm::KeldyshTerm(ParticleGroup group_a, ParticleGroup group_b,
                         double epsilon, double r0, double d_layer)
    : group_a_(std::move(group_a)),
      group_b_(std::move(group_b)),
      epsilon_(epsilon),
      r0_(r0),
      inv_r0_(1.0 / r0),
      prefactor_(numerics::PI / (2.0 * epsilon * r0)),
      d2_(d_layer * d_layer),
      is_intragroup_(group_a_ == group_b_),
      cheb_(&numerics::keldysh_cheb()) {}

bool KeldyshTerm::supports_open_centroid_virial() const noexcept {
    const double charge_product=group_a_.charge*(is_intragroup_ ? group_a_.charge : group_b_.charge);
    if (!std::isfinite(epsilon_) || epsilon_ <= 0 || !std::isfinite(r0_) || r0_ <= 0 ||
        !std::isfinite(inv_r0_) || !std::isfinite(prefactor_) || prefactor_ <= 0 || !std::isfinite(d2_) ||
        !std::isfinite(charge_product*prefactor_) ||
        group_a_.first > group_a_.last || group_b_.first > group_b_.last ||
        (!is_intragroup_ && std::max(group_a_.first,group_b_.first)<std::min(group_a_.last,group_b_.last))) return false;
    // Keep the existing conservative primitive collision rule even though the
    // retained low clamp makes the represented potential finite at zero.
    if (!(d2_ > 0 || charge_product >= 0)) return false;
    // |T_n|<=1 and |T_n'|<=n^2 on the fitted interval bound the
    // represented value and derivative. The blend adds s'*(tail-fit), with
    // max |s'|=15/64. Reject parameter scaling that can overflow either.
    double value_bound=0.0,derivative_bound=0.0;
    for (std::size_t n=0; n<cheb_->coeffs.size(); ++n) {
        value_bound+=std::abs(cheb_->coeffs[n]);
        derivative_bound+=double(n*n)*std::abs(cheb_->coeffs[n]);
    }
    derivative_bound*=cheb_->map_a*std::pow(cheb_->x_lo,1.0/cheb_->kP)/(cheb_->kP*cheb_->x_lo);
    const double x=cheb_->kXjoin,u=1/(x*x);
    const double tail_value=(2/numerics::PI)/x*(1+u*(1+u*(9+u*(225+u*(11025+893025*u)))));
    const double tail_derivative=(2/numerics::PI)/(x*x)*(1+u*(3+u*(45+u*(1575+u*(99225+9823275*u)))));
    const double gradient_bound=std::max(derivative_bound,tail_derivative)+(15.0/64.0)*(value_bound+tail_value);
    const double coefficient=std::abs(charge_product*prefactor_);
    // The gradient multiplies coefficient*K' before inv_r0. Bound that
    // intermediate too; a small inv_r0 cannot undo an earlier overflow.
    return coefficient <= std::numeric_limits<double>::max()/std::max(value_bound,tail_value) &&
        coefficient <= std::numeric_limits<double>::max()/gradient_bound &&
        coefficient*inv_r0_ <= std::numeric_limits<double>::max()/gradient_bound;
}
bool KeldyshTerm::supports_periodic_virial(const SimulationCell& cell, std::size_t np) const noexcept {
    // Equal Euclidean radii on Voronoi faces imply equal scalar values. The
    // force can jump there; the continuous torus weight still has zero flux.
    return cell.is_periodic() && supports_open_centroid_virial() &&
        group_a_.last <= np && group_b_.last <= np;
}
std::string KeldyshTerm::potential_approximation_notes() const {
    std::ostringstream out;
    out << std::setprecision(17) << "sampled RK; rho/r0<0.001 frozen (rho_core=" << r0_*cheb_->x_lo
        << "); C2 tail blend at rho/r0=32..40; fit/core error excluded from statistical uncertainty";
    return out.str();
}

void KeldyshTerm::evaluate_potential(const ParticleBatch& batch,
                                 std::span<double> v_out,
                                 std::size_t w_begin, std::size_t w_end) const {
    const std::size_t nw = batch.num_walkers();   // global stride
    if (w_end > nw) w_end = nw;
    const BatchedDistanceTable& dt = batch.dist_tables();
    double* __restrict le = v_out.data();

    // Accumulate coeff·K(r_ij/r₀) over the contiguous walker row of pair (i,j).
    const double d2 = d2_;
    auto add_pair = [&](double coeff, std::size_t i, std::size_t j) {
        const double* __restrict r = dt.row(i, j).data();
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            const double x  = std::sqrt(r[w] * r[w] + d2) * inv_r0_;
            le[w] += coeff * cheb_->eval(x);
        }
    };

    if (is_intragroup_) {
        // Intra-group: V = Σ_{i<j in A} (π q²)/(2 ε r₀) · K(r_ij/r₀).
        const double q2 = group_a_.charge * group_a_.charge * prefactor_;
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = i + 1; j < group_a_.last; ++j)
                add_pair(q2, i, j);
    } else {
        // Inter-group: V = Σ_{i∈A, j∈B} (π q_a q_b)/(2 ε r₀) · K(r_ij/r₀).
        const double qq = group_a_.charge * group_b_.charge * prefactor_;
        for (std::size_t i = group_a_.first; i < group_a_.last; ++i)
            for (std::size_t j = group_b_.first; j < group_b_.last; ++j)
                add_pair(qq, i, j);
    }
}

std::unique_ptr<PotentialGradientWorkspace>
KeldyshTerm::make_potential_gradient_workspace(const ParticleBatch& batch) const {
    const double coefficient = group_a_.charge *
        (is_intragroup_ ? group_a_.charge : group_b_.charge) * prefactor_;
    if (group_a_.first > group_a_.last || group_b_.first > group_b_.last ||
        group_a_.last > batch.n_particles() || group_b_.last > batch.n_particles() ||
        !std::isfinite(epsilon_) || epsilon_ <= 0.0 || !std::isfinite(r0_) || r0_ <= 0.0 ||
        !std::isfinite(inv_r0_) || !std::isfinite(d2_) || !std::isfinite(coefficient))
        throw std::invalid_argument("rytova_keldysh: invalid gradient groups or parameters");
    return HamiltonianComponent::make_potential_gradient_workspace(batch);
}

void KeldyshTerm::evaluate_potential_gradient(const ParticleBatch& batch,
    std::span<double> out, PotentialGradientWorkspace& workspace,
    std::size_t w_begin, std::size_t w_end) const {
    w_end = workspace.checked_end(batch, *this, out, w_begin, w_end);
    const auto nw=batch.num_walkers(), dim=batch.dim();
    const auto pos=batch.positions(); // [np*dim*nw]
    const double coefficient=group_a_.charge *
        (is_intragroup_ ? group_a_.charge : group_b_.charge)*prefactor_;
    if (coefficient == 0.0) return;
    for (std::size_t i=group_a_.first; i<group_a_.last; ++i)
        for (std::size_t j=is_intragroup_ ? i+1 : group_b_.first; j<group_b_.last; ++j)
            for (std::size_t w=w_begin; w<w_end; ++w) {
                double delta[SimulationCell::kMaxDim]; // [dim]
                for (std::size_t d=0; d<dim; ++d)
                    delta[d]=pos[(i*dim+d)*nw+w]-pos[(j*dim+d)*nw+w];
                const double r2=batch.cell().min_image_dist2(delta);
                const double rho=std::sqrt(r2+d2_);
                if (!std::isfinite(rho))
                    throw std::domain_error("rytova_keldysh: nonfinite gradient separation");
                // Coincidence lies in the constant low clamp or is softened.
                // Its in-plane derivative is zero without dividing by rho.
                if (r2 == 0.0) continue;
                const double radial=coefficient*cheb_->d1(rho*inv_r0_)*inv_r0_;
                for (std::size_t d=0; d<dim; ++d) {
                    const double gradient=radial*(delta[d]/rho);
                    if (!std::isfinite(gradient))
                        throw std::domain_error("rytova_keldysh: nonfinite spatial gradient");
                    out[(i*dim+d)*nw+w]+=gradient;
                    out[(j*dim+d)*nw+w]-=gradient;
                }
            }
}

// Same fitted/clamped kernel as evaluate_potential; changed pairs only.
double KeldyshTerm::delta_potential_particle(ParticleBatch& configs,
    std::size_t p, std::size_t k, std::span<const double> proposed,
    std::span<double>) const {
    const double coefficient = group_a_.charge *
        (is_intragroup_ ? group_a_.charge : group_b_.charge) * prefactor_;
    auto radial = [&](double r) {
        const double x = std::sqrt(r*r + d2_) * inv_r0_;
        return coefficient*cheb_->eval(x);
    };
    return radial_particle_delta(configs, p, k, proposed, group_a_, group_b_, radial);
}

}  // namespace alea
