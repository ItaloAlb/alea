// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <span>
#include <vector>

namespace alea {

// Accumulates VMC samples (local energies and parameter derivatives) and builds
// the overlap matrix Sbar and Hamiltonian matrix Hbar for the linear method.
//
// The basis is SEMI-ORTHOGONALISED (Psi_bar_i = Psi_i - <d_i> Psi).  Psi is only
// defined up to a p-dependent normalisation, so d_i -> d_i + c must not change
// the step; projecting out the component along Psi is what guarantees that.
// Skipping it silently divides the step by ~(1 + <d>²/Var(d)), which for an
// extensive parameter is ~N.
//
// Thread safety: each thread should own its own CostFunction instance;
// merge() combines them with a simple sum of accumulators.
class CostFunction {
public:
    CostFunction() = default;

    // Reset all accumulated statistics.
    void clear();

    // Add a batch of samples at once.
    //   local_energies: [n_samples]
    //   param_derivs:   [n_params * n_samples], SoA: derivs[k * nw + w]
    //   el_derivs:      ∂E_L/∂p_k, same layout.  Empty (or zero columns for
    //                   parameters whose component cannot supply it) drops the
    //                   <D_i ∂_j E_L> term from Hbar for those parameters.
    void add_samples(std::span<const double> local_energies,
                     std::span<const double> param_derivs,
                     std::span<const double> el_derivs,
                     std::size_t n_params,
                     std::size_t n_samples);

    void add_samples(std::span<const double> local_energies,
                     std::span<const double> param_derivs,
                     std::size_t n_params,
                     std::size_t n_samples) {
        add_samples(local_energies, param_derivs, {}, n_params, n_samples);
    }

    // Number of accumulated samples.
    [[nodiscard]] std::size_t n_samples() const noexcept { return n_samples_; }

    // Mean local energy.
    [[nodiscard]] double mean_energy() const noexcept;

    // Standard error of that mean.  A FLOOR, not an honest bar: the samples are
    // correlated, so the true error is larger (measured 3x in this code).
    [[nodiscard]] double mean_energy_error() const noexcept;

    // Build the (N+1) x (N+1) Sbar and Hbar matrices, in the semi-orthogonalised
    // basis (see above).  With D_i = d_i - <d_i> and dE = E_L - <E_L>:
    //
    //   Sbar_00 = 1        Sbar_0i = Sbar_i0 = 0     Sbar_ij = <D_i D_j>
    //   Hbar_00 = <E_L>    Hbar_0i = Hbar_i0 = <D_i dE>
    //   Hbar_ij = <D_i D_j E_L> + <D_i ∂_j E_L>
    //
    // Hbar_0i equals Hbar_i0 because <∂E_L/∂p_i> vanishes identically for a real
    // Psi and a hermitian H.  The <D_i ∂_j E_L> term is the curvature; without
    // it the step direction is right but its length carries no curvature.  It is
    // dropped per-parameter when el_derivs is not supplied.
    //   N = n_params.
    void build_matrices(Eigen::MatrixXd& Sbar,
                        Eigen::MatrixXd& Hbar,
                        std::size_t n_params) const;

    // Thread-safe merge: add another CostFunction's accumulators into this one.
    void merge(const CostFunction& other);

private:
    std::size_t n_samples_ = 0;
    double      energy_sum_ = 0.0;
    double      energy_sq_sum_ = 0.0;

    // Σ_w d_i(w)           — size: n_params (resized on first add)
    // Σ_w d_i(w) d_j(w)    — symmetric, stored flat [i * n_params + j]
    // Σ_w E_L(w) d_i(w)    — size: n_params
    // Σ_w E_L(w) d_i(w) d_j(w) — symmetric, stored flat
    std::vector<double> deriv_sums_;
    std::vector<double> deriv_prods_;
    std::vector<double> energy_derivs_;
    std::vector<double> energy_deriv_prods_;
    // Σ_w ∂_j E_L(w)  and  Σ_w d_i(w) ∂_j E_L(w)  — the curvature term.
    std::vector<double> el_deriv_sums_;
    std::vector<double> deriv_el_deriv_prods_;

    void ensure_capacity(std::size_t n_params);
};

}  // namespace alea
