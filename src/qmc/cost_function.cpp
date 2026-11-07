// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/cost_function.h"

#include <cmath>

#include <algorithm>
#include <numeric>

namespace alea {

void CostFunction::clear() {
    n_samples_ = 0;
    energy_sum_ = 0.0;
    energy_sq_sum_ = 0.0;
    deriv_sums_.clear();
    deriv_prods_.clear();
    energy_derivs_.clear();
    energy_deriv_prods_.clear();
    el_deriv_sums_.clear();
    deriv_el_deriv_prods_.clear();
}

void CostFunction::ensure_capacity(std::size_t n_params) {
    if (deriv_sums_.size() == n_params) return;
    deriv_sums_.assign(n_params, 0.0);
    deriv_prods_.assign(n_params * n_params, 0.0);
    energy_derivs_.assign(n_params, 0.0);
    energy_deriv_prods_.assign(n_params * n_params, 0.0);
    el_deriv_sums_.assign(n_params, 0.0);
    deriv_el_deriv_prods_.assign(n_params * n_params, 0.0);
}

void CostFunction::add_samples(std::span<const double> local_energies,
                               std::span<const double> param_derivs,
                               std::span<const double> el_derivs,
                               std::size_t n_params,
                               std::size_t n_samples) {
    if (n_samples == 0) return;
    if (n_params == 0) {
        // No parameters to track, but we still accumulate the energy.
        for (std::size_t w = 0; w < n_samples; ++w) {
            energy_sum_ += local_energies[w];
            energy_sq_sum_ += local_energies[w] * local_energies[w];
        }
        n_samples_ += n_samples;
        return;
    }
    ensure_capacity(n_params);

    for (std::size_t w = 0; w < n_samples; ++w) {
        const double e = local_energies[w];
        energy_sum_ += e;
        energy_sq_sum_ += e * e;

        // First pass: accumulate linear terms and diagonal product terms.
        for (std::size_t i = 0; i < n_params; ++i) {
            const double di = param_derivs[i * n_samples + w];
            deriv_sums_[i] += di;
            energy_derivs_[i] += e * di;

            // Diagonal of product matrices.
            deriv_prods_[i * n_params + i] += di * di;
            energy_deriv_prods_[i * n_params + i] += e * di * di;

            // Off-diagonal (both triangles).
            for (std::size_t j = i + 1; j < n_params; ++j) {
                const double dj = param_derivs[j * n_samples + w];
                const double prod = di * dj;
                const double eprod = e * prod;
                deriv_prods_[i * n_params + j] += prod;
                deriv_prods_[j * n_params + i] += prod;
                energy_deriv_prods_[i * n_params + j] += eprod;
                energy_deriv_prods_[j * n_params + i] += eprod;
            }
        }

        // Curvature term: Σ ∂_j E_L and Σ d_i ∂_j E_L (not symmetric in i,j).
        if (!el_derivs.empty()) {
            for (std::size_t j = 0; j < n_params; ++j) {
                const double dej = el_derivs[j * n_samples + w];
                el_deriv_sums_[j] += dej;
                for (std::size_t i = 0; i < n_params; ++i)
                    deriv_el_deriv_prods_[i * n_params + j] +=
                        param_derivs[i * n_samples + w] * dej;
            }
        }
    }
    n_samples_ += n_samples;
}

double CostFunction::mean_energy() const noexcept {
    if (n_samples_ == 0) return 0.0;
    return energy_sum_ / static_cast<double>(n_samples_);
}

double CostFunction::mean_energy_error() const noexcept {
    if (n_samples_ < 2) return 0.0;
    const double n = static_cast<double>(n_samples_);
    const double m = energy_sum_ / n;
    const double var = energy_sq_sum_ / n - m * m;
    return var > 0.0 ? std::sqrt(var / n) : 0.0;
}

void CostFunction::merge(const CostFunction& other) {
    if (other.n_samples_ == 0) return;
    if (n_samples_ == 0) {
        *this = other;
        return;
    }

    n_samples_ += other.n_samples_;
    energy_sum_ += other.energy_sum_;
    energy_sq_sum_ += other.energy_sq_sum_;

    const std::size_t np = deriv_sums_.size();
    for (std::size_t i = 0; i < np; ++i) {
        deriv_sums_[i] += other.deriv_sums_[i];
        energy_derivs_[i] += other.energy_derivs_[i];
        el_deriv_sums_[i] += other.el_deriv_sums_[i];
        for (std::size_t j = 0; j < np; ++j) {
            deriv_prods_[i * np + j] += other.deriv_prods_[i * np + j];
            energy_deriv_prods_[i * np + j] += other.energy_deriv_prods_[i * np + j];
            deriv_el_deriv_prods_[i * np + j] +=
                other.deriv_el_deriv_prods_[i * np + j];
        }
    }
}

void CostFunction::build_matrices(Eigen::MatrixXd& Sbar,
                                  Eigen::MatrixXd& Hbar,
                                  std::size_t n_params) const {
    const std::size_t N = n_params + 1;
    Sbar.resize(N, N);
    Hbar.resize(N, N);
    Sbar.setZero();
    Hbar.setZero();

    if (n_samples_ == 0) return;

    const double inv_n = 1.0 / static_cast<double>(n_samples_);
    const double mean_e = energy_sum_ * inv_n;

    // Row/col 0 (the wavefunction itself).
    Sbar(0, 0) = 1.0;
    Hbar(0, 0) = mean_e;

    // Everything below is written in the CENTRED deviations
    //     D_i = d_i - <d_i>,     dE = E_L - <E_L>,
    // which is the semi-orthogonalised basis Psi_bar_i = Psi_i - <d_i> Psi.
    // Each mean is expanded back onto the raw accumulators, so no extra
    // accumulation pass is needed:
    //
    //   <D_i D_j>    = <d_i d_j> - m_i m_j
    //   <D_i dE>     = <E_L d_i> - m_i <E_L>
    //   <D_i dE D_j> = [<d_i E_L d_j> - <E_L><d_i d_j>]
    //                  - m_j [<d_i E_L> - <E_L> m_i]
    //                  - m_i [<E_L d_j> - <E_L> m_j]
    for (std::size_t i = 0; i < n_params; ++i) {
        const double m_i = deriv_sums_[i] * inv_n;
        // <D_i dE>.  Hbar(0,i) equals Hbar(i,0) because <dE_L/dp_i> vanishes
        // identically for a real Psi and a hermitian H — the same identity that
        // makes the energy gradient 2<dE D_i>.
        const double cov_i = energy_derivs_[i] * inv_n - m_i * mean_e;
        Sbar(0, i + 1) = 0.0;
        Sbar(i + 1, 0) = 0.0;
        Hbar(0, i + 1) = cov_i;
        Hbar(i + 1, 0) = cov_i;

        for (std::size_t j = 0; j < n_params; ++j) {
            const double m_j = deriv_sums_[j] * inv_n;
            const double cov_j = energy_derivs_[j] * inv_n - m_j * mean_e;
            const double mean_prod = deriv_prods_[i * n_params + j] * inv_n;
            const double mean_eprod = energy_deriv_prods_[i * n_params + j] * inv_n;

            // <D_i ∂_j E_L>.  <∂_j E_L> is zero in exact expectation; subtracting
            // the SAMPLE mean is what keeps the gauge invariance exact in-sample.
            const double curv =
                deriv_el_deriv_prods_[i * n_params + j] * inv_n
                - m_i * el_deriv_sums_[j] * inv_n;

            // Hbar_ij = <D_i D_j E_L> + <D_i ∂_j E_L>, UNSHIFTED: the identity
            // (H Psi_j)/Psi = ∂_j E_L + E_L d_j leaves no -<E_L> on it, and
            // Hbar_00 is <E_L>, so shifting only this block mixes conventions.
            const double s_ij = mean_prod - m_i * m_j;
            Sbar(i + 1, j + 1) = s_ij;
            Hbar(i + 1, j + 1) = (mean_eprod - mean_e * mean_prod)
                                 - m_j * cov_i - m_i * cov_j
                                 + mean_e * s_ij + curv;
        }
    }
}

}  // namespace alea
