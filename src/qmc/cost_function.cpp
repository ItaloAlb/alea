// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/cost_function.h"

#include <cmath>

#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <stdexcept>

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
    complex_ = false;
    c_energy_sum_ = cplx{};
    for (std::vector<cplx>* v : {&c_o_, &c_oo_, &c_oe_, &c_eo_, &c_oeo_, &c_del_, &c_odel_})
        v->clear();
}

void CostFunction::ensure_complex_capacity(std::size_t n_params) {
    if (c_o_.size() == n_params) return;
    for (std::vector<cplx>* v : {&c_o_, &c_oe_, &c_eo_, &c_del_}) v->assign(n_params, cplx{});
    for (std::vector<cplx>* v : {&c_oo_, &c_oeo_, &c_odel_})
        v->assign(n_params * n_params, cplx{});
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
    if (complex_)
        throw std::logic_error("CostFunction: real samples added to a complex accumulation");
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

void CostFunction::add_complex_samples(std::span<const double> local_energies,
                                       std::span<const double> local_energies_imag,
                                       std::span<const double> param_derivs,
                                       std::span<const double> param_phase_derivs,
                                       std::span<const double> el_derivs,
                                       std::span<const double> el_derivs_imag,
                                       std::size_t n_params,
                                       std::size_t n_samples) {
    if (n_samples == 0) return;
    if (n_params == 0) {
        add_samples(local_energies, {}, {}, 0, n_samples);
        return;
    }
    if (!complex_ && n_samples_ > 0)
        throw std::logic_error("CostFunction: complex samples added to a real accumulation");
    complex_ = true;
    ensure_complex_capacity(n_params);
    const bool curvature = !el_derivs.empty();
    std::vector<cplx> o(n_params), del(n_params);   // one sample's O and ∂E_L
    for (std::size_t w = 0; w < n_samples; ++w) {
        const cplx e(local_energies[w], local_energies_imag[w]);
        energy_sum_ += e.real();
        energy_sq_sum_ += e.real() * e.real();
        c_energy_sum_ += e;
        for (std::size_t i = 0; i < n_params; ++i) {
            o[i] = cplx(param_derivs[i * n_samples + w], param_phase_derivs[i * n_samples + w]);
            del[i] = curvature ? cplx(el_derivs[i * n_samples + w], el_derivs_imag[i * n_samples + w])
                               : cplx{};
        }
        for (std::size_t i = 0; i < n_params; ++i) {
            const cplx oi_conj = std::conj(o[i]);
            const cplx oi_conj_e = oi_conj * e;
            c_o_[i] += o[i];
            c_oe_[i] += oi_conj_e;
            c_eo_[i] += e * o[i];
            c_del_[i] += del[i];
            for (std::size_t j = 0; j < n_params; ++j) {
                c_oo_[i * n_params + j] += oi_conj * o[j];
                c_oeo_[i * n_params + j] += oi_conj_e * o[j];
                c_odel_[i * n_params + j] += oi_conj * del[j];
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
    if (complex_ != other.complex_)
        throw std::logic_error("CostFunction: merging a real and a complex accumulation");

    n_samples_ += other.n_samples_;
    energy_sum_ += other.energy_sum_;
    energy_sq_sum_ += other.energy_sq_sum_;
    if (complex_) {
        c_energy_sum_ += other.c_energy_sum_;
        const auto add = [](std::vector<cplx>& to, const std::vector<cplx>& from) {
            for (std::size_t k = 0; k < to.size(); ++k) to[k] += from[k];
        };
        add(c_o_, other.c_o_);
        add(c_oo_, other.c_oo_);
        add(c_oe_, other.c_oe_);
        add(c_eo_, other.c_eo_);
        add(c_oeo_, other.c_oeo_);
        add(c_del_, other.c_del_);
        add(c_odel_, other.c_odel_);
        return;
    }

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
    if (complex_) {
        build_complex_matrices(Sbar, Hbar, n_params);
        return;
    }

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

// The complex case of build_matrices (see the header), each centred mean
// expanded onto the raw sums as in the real case, m_i = <O_i>:
//   <D_i* D_j>     = <O_i* O_j> − m_i* m_j
//   <D_i* E_L>     = <O_i* E_L> − m_i* <E_L>
//   <D_i* E_L D_j> = <O_i* E_L O_j> − m_i* <E_L O_j> − <O_i* E_L> m_j + m_i* <E_L> m_j
//   <D_i* ∂_j E_L> = <O_i* ∂_j E_L> − m_i* <∂_j E_L>
void CostFunction::build_complex_matrices(Eigen::MatrixXd& Sbar, Eigen::MatrixXd& Hbar,
                                          std::size_t n_params) const {
    const double inv_n = 1.0 / static_cast<double>(n_samples_);
    const cplx mean_e = c_energy_sum_ * inv_n;
    Sbar(0, 0) = 1.0;
    Hbar(0, 0) = mean_e.real();
    for (std::size_t i = 0; i < n_params; ++i) {
        const cplx m_i_conj = std::conj(c_o_[i] * inv_n);
        const cplx oe_i = c_oe_[i] * inv_n;
        const double cov_i = (oe_i - m_i_conj * mean_e).real();
        Sbar(0, i + 1) = Sbar(i + 1, 0) = 0.0;
        Hbar(0, i + 1) = Hbar(i + 1, 0) = cov_i;
        for (std::size_t j = 0; j < n_params; ++j) {
            const std::size_t ij = i * n_params + j;
            const cplx m_j = c_o_[j] * inv_n;
            const cplx eo_j = c_eo_[j] * inv_n;
            Sbar(i + 1, j + 1) = (c_oo_[ij] * inv_n - m_i_conj * m_j).real();
            const cplx d_e_d = c_oeo_[ij] * inv_n - m_i_conj * eo_j - oe_i * m_j +
                               m_i_conj * mean_e * m_j;
            const cplx curv = c_odel_[ij] * inv_n - m_i_conj * (c_del_[j] * inv_n);
            Hbar(i + 1, j + 1) = (d_e_d + curv).real();
        }
    }
}

}  // namespace alea
