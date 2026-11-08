// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Eigen/Dense>

#include <complex>
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

    // A complex trial ψ = e^{u + iφ} (twisted boundaries), whose real
    // parameters p_k have complex log-derivatives and local energies:
    //   O_k = ∂ ln ψ/∂p_k = d_k + i d̃_k,   E_L = e + i f,   ∂_k E_L = c_k + i c̃_k.
    //   local_energies / local_energies_imag:  e, f        [n_samples]
    //   param_derivs / param_phase_derivs:     d, d̃        [n_params * n_samples]
    //   el_derivs / el_derivs_imag:            c, c̃        same layout; empty
    //                                          drops the curvature term
    // A CostFunction holds samples of one kind: mixing real and complex ones
    // throws.  mean_energy() and its error are those of Re E_L.
    void add_complex_samples(std::span<const double> local_energies,
                             std::span<const double> local_energies_imag,
                             std::span<const double> param_derivs,
                             std::span<const double> param_phase_derivs,
                             std::span<const double> el_derivs,
                             std::span<const double> el_derivs_imag,
                             std::size_t n_params,
                             std::size_t n_samples);

    [[nodiscard]] bool is_complex() const noexcept { return complex_; }

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
    //
    // Complex samples.  The parameters are real, so a step Δp is real, and the
    // energy of ψ + Σ_k Δp_k ψ̄_k is ΔpᵀHΔp / ΔpᵀSΔp over the complex matrices
    // H_kl = ⟨ψ̄_k|H|ψ̄_l⟩, S_kl = ⟨ψ̄_k|ψ̄_l⟩ (here with ψ̄_0 = ψ).  For real Δp
    // only their real parts count: S is Hermitian, so Im S is antisymmetric,
    // and so is Im H in expectation.  The matrices are therefore the real
    // parts of the same per-sample estimators, with D_i = O_i − <O_i>:
    //
    //   Sbar_ij = Re<D_i* D_j>
    //   Hbar_00 = Re<E_L>    Hbar_0i = Hbar_i0 = Re<D_i* E_L>
    //   Hbar_ij = Re<D_i* E_L D_j> + Re<D_i* ∂_j E_L>
    //
    // Taking Re of each sample's product (not of each factor) keeps the
    // property of the non-symmetric estimator that makes it exact for a basis
    // that H maps into itself (here: with real coefficients), M. P.
    // Nightingale and V. Melik-Alaverdian, PRL 87, 043401 (2001).  Hbar_0i =
    // Hbar_i0 again holds in expectation: for a hermitian H,
    //   <∂_j E_L> = <E_L* O_j> − <E_L O_j> = −2i <Im E_L · O_j>,
    // so Re<E_L D_j + ∂_j E_L> = Re<D_j* E_L>, and 2 Hbar_i0 is the energy
    // gradient ∂E/∂p_i = 2 Re<D_i* (E_L − E)>.  With d̃ = f = c̃ = 0 every
    // entry is the real case's.
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

    // Complex samples (add_complex_samples), with O = d + i d̃ and the complex
    // E_L and ∂_j E_L.  [n] or [n * n], flat [i * n + j].
    using cplx = std::complex<double>;
    bool              complex_ = false;
    cplx              c_energy_sum_{};    // Σ E_L
    std::vector<cplx> c_o_;               // Σ O_i
    std::vector<cplx> c_oo_;              // Σ O_i* O_j
    std::vector<cplx> c_oe_;              // Σ O_i* E_L
    std::vector<cplx> c_eo_;              // Σ E_L O_i
    std::vector<cplx> c_oeo_;             // Σ O_i* E_L O_j
    std::vector<cplx> c_del_;             // Σ ∂_j E_L
    std::vector<cplx> c_odel_;            // Σ O_i* ∂_j E_L

    void ensure_capacity(std::size_t n_params);
    void ensure_complex_capacity(std::size_t n_params);
    void build_complex_matrices(Eigen::MatrixXd& Sbar, Eigen::MatrixXd& Hbar,
                                std::size_t n_params) const;
};

}  // namespace alea
