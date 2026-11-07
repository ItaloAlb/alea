// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "estimators/momentum_distribution.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace alea {
namespace {
// Volume of the radial shell [b, b+1) * delta in d dimensions (d = 1 counts
// both signs of r).
double shell_volume(std::size_t dim, double delta, std::size_t b) {
    const double r0 = static_cast<double>(b) * delta, r1 = r0 + delta;
    switch (dim) {
    case 1: return 2.0 * delta;
    case 2: return std::numbers::pi * (r1 * r1 - r0 * r0);
    default: return 4.0 / 3.0 * std::numbers::pi * (r1 * r1 * r1 - r0 * r0 * r0);
    }
}
// Angular average of exp(-i k.r) over directions of r in d dimensions.
double angular_average(std::size_t dim, double kr) {
    switch (dim) {
    case 1: return std::cos(kr);
    case 2: return std::cyl_bessel_j(0.0, kr);
    default: return kr < 1e-8 ? 1.0 - kr * kr / 6.0 : std::sin(kr) / kr;
    }
}
// Angular average of [sum_d k_d²/(2m_d)] exp(-ik.r), from the Hessian
// of the radial Fourier kernel f. f'' = -k²f-(d-1)f'/r gives
// 0.5 k² m_proj^-1 f + 0.5(d m_proj^-1-Tr m^-1) f'/r.
double angular_kinetic(std::size_t dim, double k, double radius, double kernel,
                       double projected_inverse_mass, double trace_inverse_mass) {
    const double x = k * radius;
    double derivative_over_r;
    // The 3D expression subtracts two O(x) terms to obtain O(x^3).
    // At x=1e-4 it already loses ~1e-8 relative precision. The omitted
    // x^6 term of this series is <=2e-16 for |x|<1e-2 (all d=1..3).
    if (std::abs(x) < 1e-2) {
        const double d = static_cast<double>(dim);
        derivative_over_r = k*k * (-1.0/d + x*x/(2.0*d*(d+2.0))
                                   - x*x*x*x/(8.0*d*(d+2.0)*(d+4.0)));
    } else if (dim == 1) {
        derivative_over_r = -k * std::sin(x) / radius;
    } else if (dim == 2) {
        derivative_over_r = -k * std::cyl_bessel_j(1.0,x) / radius;
    } else {
        derivative_over_r = k*k * (x*std::cos(x)-std::sin(x)) / (x*x*x);
    }
    return 0.5*k*k*projected_inverse_mass*kernel +
           0.5*(static_cast<double>(dim)*projected_inverse_mass-trace_inverse_mass)*derivative_over_r;
}
}  // namespace

MomentumDistribution::MomentumDistribution(
    std::span<const ParticleGroup> groups, const std::string& group, double rmax,
    int n_bins, double kmax, int n_k, int samples, const SimulationCell& cell,
    std::size_t dim)
    : group_(group), rmax_(rmax), n_bins_(static_cast<std::size_t>(std::max(n_bins, 0))),
      dim_(dim), samples_(static_cast<std::size_t>(std::max(samples, 0))), cell_(cell),
      periodic_(cell.is_periodic()) {
    const auto group_it = std::find_if(groups.begin(), groups.end(),
                                      [&](const ParticleGroup& g) { return g.name == group; });
    if (group_it == groups.end())
        throw std::runtime_error("momentum_distribution: unknown group '" + group + "'");
    if (dim < 1 || dim > 3)
        throw std::runtime_error("momentum_distribution: dimension must be 1, 2 or 3");
    if (!(std::isfinite(rmax) && rmax > 0.0) || n_bins < 1)
        throw std::runtime_error("momentum_distribution: rmax > 0 and n_bins >= 1 required");
    if (!(std::isfinite(kmax) && kmax > 0.0))
        throw std::runtime_error("momentum_distribution: kmax must be finite and > 0");
    if (!std::isfinite(kmax*kmax))
        throw std::runtime_error("momentum_distribution: momentum cutoff overflows squared momentum");
    if (samples < 1)
        throw std::runtime_error("momentum_distribution: samples must be >= 1");
    delta_ = rmax / static_cast<double>(n_bins_);
    if (group_it->inv_mass.size() != dim ||
        !std::all_of(group_it->inv_mass.begin(), group_it->inv_mass.end(),
                     [](double m) { return std::isfinite(m) && m > 0.0; }))
        throw std::runtime_error("momentum_distribution: kinetic sums need positive "
                                 "finite inverse masses in every dimension");
    inverse_mass_ = group_it->inv_mass;
    for (double m : inverse_mass_) trace_inverse_mass_ += m;
    if (!std::isfinite(trace_inverse_mass_))
        throw std::runtime_error("momentum_distribution: inverse-mass sum overflows");
    isotropic_mass_ = std::all_of(inverse_mass_.begin(),inverse_mass_.end(),
                                [&](double m){ return m == inverse_mass_[0]; });
    if (periodic_) {
        // Beyond the inradius the minimum-image displacement is not unique.
        if (rmax > cell.ws_radius())
            throw std::runtime_error("momentum_distribution: rmax exceeds the cell's "
                                     "Wigner-Seitz inradius " + std::to_string(cell.ws_radius()));
        if (n_k != 0)
            throw std::runtime_error("momentum_distribution: n_k applies to open cells; "
                                     "periodic cells use the reciprocal-lattice shells up to kmax");
        // Every reciprocal vector with |k| <= kmax: |n_i| = |k . a_i| / 2 pi
        // <= kmax |a_i| / 2 pi bounds each integer coordinate.
        const auto a = cell.vectors(), b = cell.reciprocal();
        std::vector<int> bound(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            double length = 0.0;
            for (std::size_t d = 0; d < dim; ++d) length += a[i * dim + d] * a[i * dim + d];
            bound[i] = static_cast<int>(std::ceil(kmax * std::sqrt(length) / (2.0 * std::numbers::pi)));
        }
        std::vector<std::pair<double, std::vector<double>>> ks;
        std::vector<int> n(dim);
        for (std::size_t i = 0; i < dim; ++i) n[i] = -bound[i];
        while (true) {
            std::vector<double> k(dim, 0.0);
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t d = 0; d < dim; ++d) k[d] += n[i] * b[i * dim + d];
            double norm = 0.0;
            for (const double v : k) norm += v * v;
            norm = std::sqrt(norm);
            if (norm <= kmax * (1.0 + 1e-12)) ks.emplace_back(norm, std::move(k));
            // Odometer over the integer box [-bound, bound]^dim.
            std::size_t i = 0;
            for (; i < dim; ++i) {
                if (++n[i] <= bound[i]) break;
                n[i] = -bound[i];
            }
            if (i == dim) break;
        }
        std::stable_sort(ks.begin(), ks.end(),
                         [](const auto& x, const auto& y) { return x.first < y.first; });
        const double tol = 1e-9 * (1.0 + kmax);
        for (const auto& [norm, k] : ks) {
            if (k_.empty() || norm - k_.back() > tol) {
                k_.push_back(norm);
                shell_count_.push_back(0.0);
            }
            shell_count_.back() += 1.0;
            shell_of_.push_back(k_.size() - 1);
            kvec_.insert(kvec_.end(), k.begin(), k.end());
            double kinetic = 0.0;
            for (std::size_t d = 0; d < dim; ++d)
                kinetic += 0.5 * k[d] * k[d] * group_it->inv_mass[d];
            if (!std::isfinite(kinetic))
                throw std::runtime_error("momentum_distribution: reciprocal kinetic energy overflows");
            k_energy_.push_back(kinetic);
        }
    } else {
        if (n_k < 2)
            throw std::runtime_error("momentum_distribution: n_k >= 2 required in an open cell");
        for (int j = 0; j < n_k; ++j)
            k_.push_back(kmax * static_cast<double>(j) / static_cast<double>(n_k - 1));
        const double sphere = dim==1 ? 2.0 : dim==2 ? 2.0*std::numbers::pi : 4.0*std::numbers::pi;
        const double measure = sphere / std::pow(2.0*std::numbers::pi,static_cast<double>(dim));
        const double dk = kmax / static_cast<double>(n_k-1);
        for (int j=0; j<n_k; ++j) {
            const double weight=measure*dk*std::pow(k_[j],static_cast<double>(dim-1)) *
                                (j==0 || j==n_k-1 ? 0.5 : 1.0);
            if (!std::isfinite(weight) || !std::isfinite(k_[j]*k_[j]*trace_inverse_mass_))
                throw std::runtime_error("momentum_distribution: radial momentum quadrature overflows");
            radial_weights_.push_back(weight);
        }
    }
    block_.assign(n_bins_ + k_.size() + 2, 0.0);
}

void MomentumDistribution::configure_open_paths(std::size_t worm_particles) {
    if (worm_particles < 2)
        throw std::runtime_error("momentum_distribution: the worm needs two or more particles");
    worm_particles_ = static_cast<double>(worm_particles);
}

void MomentumDistribution::start_block() {
    std::fill(block_.begin(), block_.end(), 0.0);
    block_sign_ = 0.0;
    block_count_ = 0.0;
}

void MomentumDistribution::accumulate(const ParticleBatch&, std::span<const double> weights) {
    if (weights.empty())
        throw std::runtime_error("momentum_distribution: a measurement needs slice weights");
    // Every slice of a closed path carries the path sign s.
    double sum = 0.0;
    for (const double w : weights) sum += w;
    block_sign_ += sum / static_cast<double>(weights.size());
    block_count_ += 1.0;
}

void MomentumDistribution::skip_measurement(std::size_t) {
    if (!reblocking_.signed_mode())
        throw std::logic_error("momentum_distribution: skip_measurement needs signed weights");
    block_count_ += 1.0;
}

void MomentumDistribution::accumulate_open(std::span<const double> displacement,
                                           std::span<const double> weight) {
    if (displacement.size() != weight.size() * dim_)
        throw std::runtime_error("momentum_distribution: open-path sample shape mismatch");
    const std::size_t n_shells = k_.size();
    for (std::size_t i = 0; i < weight.size(); ++i) {
        const double* r = displacement.data() + i * dim_;
        const double c = weight[i];
        double r2 = 0.0;
        for (std::size_t d = 0; d < dim_; ++d) r2 += r[d] * r[d];
        const double radius = std::sqrt(r2);
        if (radius < rmax_)
            block_[std::min(static_cast<std::size_t>(radius / delta_), n_bins_ - 1)] += c;
        double* nk = block_.data() + n_bins_;
        if (periodic_) {
            for (std::size_t q = 0; q < shell_of_.size(); ++q) {
                double phase = 0.0;
                for (std::size_t d = 0; d < dim_; ++d) phase += kvec_[q * dim_ + d] * r[d];
                const double kernel = c * std::cos(phase);
                nk[shell_of_[q]] += kernel;
                // Aggregate kernels before reblocking; shell errors share
                // samples and a denominator and cannot be added in quadrature.
                block_[n_bins_ + n_shells] += kernel;
                block_[n_bins_ + n_shells + 1] += k_energy_[q] * kernel;
            }
        } else {
            double projected=inverse_mass_[0];
            if (!isotropic_mass_) {
                projected=0.0;
                for (std::size_t d=0; d<dim_; ++d)
                    projected += inverse_mass_[d]*r[d]*r[d];
                projected = r2>0.0 ? projected/r2 : trace_inverse_mass_/static_cast<double>(dim_);
            }
            for (std::size_t j = 0; j < n_shells; ++j) {
                const double angular=angular_average(dim_,k_[j]*radius);
                const double kinetic=isotropic_mass_ ? 0.5*inverse_mass_[0]*k_[j]*k_[j]*angular :
                    angular_kinetic(dim_,k_[j],radius,angular,projected,trace_inverse_mass_);
                nk[j] += c*angular;
                block_[n_bins_ + n_shells] += radial_weights_[j]*c*angular;
                block_[n_bins_ + n_shells + 1] += radial_weights_[j]*c*kinetic;
            }
        }
    }
    block_count_ += 1.0;
}

void MomentumDistribution::stop_block() {
    if (!(block_count_ > 0.0))
        throw std::runtime_error("momentum_distribution: block has no measurement");
    std::vector<double> values(block_.size());
    for (std::size_t i = 0; i < block_.size(); ++i) values[i] = block_[i] / block_count_;
    reblocking_.append(values, block_sign_ / block_count_);
}

void MomentumDistribution::configure_reblocking(std::size_t num_blocks) {
    reblocking_.configure(num_blocks, block_.size());
}

std::size_t MomentumDistribution::reblocking_storage_bytes(std::size_t num_blocks) const {
    return EstimatorReblocking::storage_bytes(num_blocks, block_.size());
}

void MomentumDistribution::collect(const Estimator& other) {
    reblocking_.collect(dynamic_cast<const MomentumDistribution&>(other).reblocking_);
}

void MomentumDistribution::finalize(RunResult& out) const {
    if (!reblocking_.enabled() || !(worm_particles_ > 0.0))
        throw std::runtime_error("momentum_distribution: needs signed PIMC reblocking and a worm");
    std::vector<double> r_axis(n_bins_), norms(n_bins_);
    for (std::size_t b = 0; b < n_bins_; ++b) {
        r_axis[b] = (static_cast<double>(b) + 0.5) * delta_;
        norms[b] = shell_volume(dim_, delta_, b);
    }
    out.add_series("obdm_" + group_ + "_r", std::move(r_axis));
    reblocking_.emit(out, "obdm_" + group_, 0, n_bins_, norms);
    // <n_k> = (N_w / V) n~(k) per state, averaged over each shell (PBC).
    std::vector<double> k_norms(k_.size(), 1.0);
    if (periodic_)
        for (std::size_t j = 0; j < k_.size(); ++j)
            k_norms[j] = shell_count_[j] * cell_.volume() / worm_particles_;
    out.add_series("nk_" + group_ + "_k", k_);
    reblocking_.emit(out, "nk_" + group_, n_bins_, k_.size(), k_norms);
    {
        const std::vector<double> norm{(periodic_ ? cell_.volume() : 1.0) / worm_particles_};
        reblocking_.emit(out, "nk_" + group_ + "_number", n_bins_ + k_.size(), 1, norm);
        reblocking_.emit(out, "nk_" + group_ + "_kinetic", n_bins_ + k_.size() + 1, 1, norm);
        reblocking_.emit_ratio_history(out, "nk_" + group_ + "_number", n_bins_ + k_.size(),
                                      norm[0], "nk_" + group_ + "_closed_denominator_blocks");
        reblocking_.emit_ratio_history(out, "nk_" + group_ + "_kinetic", n_bins_ + k_.size() + 1,
                                      norm[0]);
        // First six radial bins permit a normalization extrapolation with
        // bin-volume moments and their covariance, rather than calling the
        // first finite shell n(0). Larger cutoffs remain declared separately.
        for (std::size_t b = 0; b < std::min(n_bins_, std::size_t{6}); ++b)
            reblocking_.emit_ratio_history(out, "obdm_" + group_ + "_near_zero_" + std::to_string(b),
                                          b, norms[b]);
        if (periodic_) out.add_series("nk_" + group_ + "_degeneracy", shell_count_);
        out.add_metadata("nk_" + group_ + "_sum_scope",
                         periodic_ ? "finite reciprocal cutoff; joint signed-ratio reblocking; "
                                     "statistical errors exclude the omitted momentum tail" :
                                     "finite radial momentum cutoff/grid; trapezoidal integration; "
                                     "joint signed-ratio reblocking; statistical errors exclude tail/grid errors");
    }
}

}  // namespace alea
