// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "wavefunction_component.h"

namespace alea {

// Reciprocal vectors of the first `n_shells` shells of a triangular moiré
// lattice, grouped three per shell and built from the first shell g_0,g_1,g_2
// (moire_reciprocal_vectors in hamiltonian/moire.h):
//
//   shell 1:  g_j                       |k| = |g|
//   shell 2:  g_j − g_{j+1}             |k| = √3 |g|
//   shell 3:  2 g_j                     |k| = 2 |g|
//
// Each shell is closed under the C3 rotation that maps g_j → g_{j+1}, so one
// coefficient pair per shell keeps the factor C3-symmetric, like V_M itself.
// Every vector is an integer combination of the g_j, so a cell commensurate
// with the moiré potential is commensurate with every shell.
// Returned flat: k[(s*3 + j)*dim + d].  n_shells ∈ {1, 2, 3}.
[[nodiscard]] std::vector<double>
moire_jastrow_shell_vectors(std::span<const std::vector<double>> g_first_shell,
                            std::size_t n_shells, std::size_t dim);

// One-body moiré Jastrow factor
//
//     log ψ_M = Σ_i Σ_s Σ_{j=1}^{3} [ a_s cos(k_sj · r_i) + b_s sin(k_sj · r_i) ]
//
// over every particle i of the listed groups.
//
// ── Why it is needed ─────────────────────────────────────────────────────────
// Plane waves times a pair Jastrow depend only on r_i − r_j, so that trial
// state has a UNIFORM density.  In a moiré potential the true fluid piles up
// in the minima; a uniform trial state misses that energy, which biases a VMC
// fluid–crystal comparison toward the crystal and inflates DMC variance.  This
// factor lets the density follow V_M.  It is strictly positive, so it moves no
// node and leaves the fixed-node DMC energy unchanged in exact arithmetic.
//
// ── Parameterization ─────────────────────────────────────────────────────────
// Linear in (a_s, b_s) rather than an amplitude and phase c·cos(k·r + φ_J):
// log ψ is then linear in every parameter, its derivatives are exact basis
// functions, and a zero start is harmless (no sign lock, no degenerate phase at
// c = 0).  First-order perturbation theory for H = −½∇²/m + 2Ṽ Σ cos(g·r + φ)
// gives the natural seed a_1 = −(4Ṽm/|g|²) cos φ, b_1 = +(4Ṽm/|g|²) sin φ
// (ψ ≈ 1 − Σ_G V_G e^{iG·r}/(G²/2m)); the generator writes that.
//
// ── Derivatives (per particle, per dimension d) ─────────────────────────────
//   ∂_d  u = Σ k_d [ −a sin(k·r) + b cos(k·r) ]
//   ∂²_d u = −Σ k_d² [ a cos(k·r) + b sin(k·r) ]
//   ∂u/∂a_s = Σ_j cos(k_sj·r),   ∂u/∂b_s = Σ_j sin(k_sj·r)
// One-body, so a move of particle p changes only p's own terms: the committed
// share u(r_p) (the base value_c_ cache) makes a move cost one evaluation, at
// the proposed position.  An accepted position is then wrapped into the cell;
// the cached u, like the committed gradient and laplacian, stays the value at
// the unwrapped point, which differs from u at the wrapped one by rounding
// only (k · a_j ∈ 2πZ, below).  The factor is
// evaluated at the particle's stored (wrapped) position; cos(k·r) is invariant
// under a cell translation because every k is on the cell's reciprocal
// lattice (enforced in App's check_cell_support).
//
// params layout: [a_1, b_1, a_2, b_2, ...], 2 per shell.
class MoireJastrow final : public WavefunctionComponent {
public:
    // k_vectors: flat [n_k * dim] as from moire_jastrow_shell_vectors, n_k = 3·n_shells.
    // ranges   : half-open particle index ranges [first,last) the factor acts on.
    // params   : 2 per shell.
    MoireJastrow(std::size_t dim, std::vector<double> k_vectors,
                 std::vector<std::pair<std::size_t, std::size_t>> ranges,
                 std::vector<Parameter> params);

    std::string name() const override { return "moire_jastrow"; }

    // Every per-walker cache, the per-particle shares u(r_p) included, is a base
    // cache (enable_particle_values), so the base prepare and permute_caches
    // carry all of it.
    void evaluate_batch(WalkerSet& batch) override;

    void ratio_batch(WalkerSet& batch, std::size_t p,
                     std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;

    void apply_move_batch(WalkerSet& batch,
                          std::size_t p,
                          std::span<const bool> accept_mask,
                          std::size_t w_begin = 0,
                          std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<MoireJastrow>(*this);
    }

    std::span<Parameter>       parameters()       override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }

    // Coefficients are read straight from params_, so there is nothing to rebuild.
    void rebuild() override {}

    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap) const override;

    [[nodiscard]] std::size_t n_shells() const noexcept { return params_.size() / 2; }

private:
    [[nodiscard]] bool acts_on(std::size_t p) const noexcept;

    // log ψ contribution u, gradient and per-dimension laplacian of ONE particle
    // at positions x (SoA stride nw, component d at x[d*nw + w]) over [w0, w1).
    void particle_terms(const double* x, std::size_t nw, std::size_t w0,
                        std::size_t w1, double* u, double* grad, double* lap) const;

    std::size_t                                      dim_;
    std::vector<double>                              k_;       // [n_k * dim]
    std::vector<std::pair<std::size_t, std::size_t>> ranges_;
    std::vector<Parameter>                           params_;  // [a_1, b_1, ...]
};

}  // namespace alea
