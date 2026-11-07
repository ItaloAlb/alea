// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Reciprocal-space pair sum: for two particle groups A, B of a 2-D periodic
// cell and given weights w_k on a set of half-space reciprocal vectors,
//     same group  (A = B):  U = Σ_k w_k |ρ_A(k)|²
//     two groups  (A ≠ B):  U = Σ_k w_k 2 Re[ρ_A(k) ρ_B(k)*]
// with ρ_g(k) = Σ_{i∈g} e^{ik·r_i}, as a term of ln Ψ: value, per-particle
// gradient and per-dimension laplacian, incremental ratio/accept, branching,
// and the derivatives in parameters that enter only through the weights.
//
// Users: PeriodicPairSum (weights from the optimized breakup of an image-summed
// pair Jastrow) and KSpaceJastrow (free shell coefficients).
//
// ── Derivatives (e_i = e^{ik·r_i} = c + is) ─────────────────────────────────
//   same group, i ∈ A:  ∂_d T = 2k_d (c ρ_im − s ρ_re)
//                       ∂²_d T = 2k_d² (1 − (c ρ_re + s ρ_im))
//   two groups, i ∈ A:  ∂_d T = 2k_d (c ρ_B,im − s ρ_B,re)
//                       ∂²_d T = −2k_d² (c ρ_B,re + s ρ_B,im)     (A ↔ B for i ∈ B)
// A move of p changes ρ of p's group, so the gradient of every particle that
// sees that ρ changes: all of A (same group) or all of the OTHER group.  That
// is O(N·n_k) per accepted move — the rank-one cost of the determinant, for
// n_k ~ N.  Every particle's phases are cached per walker, so a move computes
// only the moved particle's new ones, from integer ladders e^{in₁θ₁}·e^{in₂θ₂}
// (θ = b·r): one complex multiply per k instead of a sincos.
//
// ── Cost ────────────────────────────────────────────────────────────────────
// The phase cache holds 2·N·n_k doubles per walker — for N = 61, n_k ≈ 45,
// 1024 walkers, ~45 MB.  Measured on periodic_2d_ehgas (three image-summed
// Padés): caching the phases took the run from 2m25 to 1m15, bit-identical.
//
// ── Layout and threads ──────────────────────────────────────────────────────
// ρ is stored walker-major, [w·n_k + k]: every loop here runs over k for one
// walker, so k is the contiguous index (unlike the walker-fastest SoA of the
// real-space kernels, whose inner loop is over walkers).  Concurrent DMC
// crowds write disjoint walker ranges; the ladder scratch is thread_local, as
// in EwaldTerm.  No new parallel region: strong and weak scaling follow the
// existing walker loop.

#include <cstddef>
#include <span>
#include <vector>

#include "numerics/reciprocal_phases.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

class WalkerSet;

class KSpacePairSum {
public:
    KSpacePairSum() = default;

    // The half-space vectors k = n1·b1 + n2·b2 (with their Cartesian (kx, ky)),
    // their weights, and, per parameter, ∂w_k/∂p.
    KSpacePairSum(const SimulationCell& cell, ParticleGroup a, ParticleGroup b,
                  std::vector<int> n1, std::vector<int> n2, std::vector<double> kx,
                  std::vector<double> ky, std::vector<double> weights,
                  std::vector<std::vector<double>> param_weights = {});

    // New weights on the SAME k set (an optimiser step).  The cached ρ and
    // phases stay valid; ln Ψ and its derivatives must be re-evaluated.
    void set_weights(std::vector<double> weights,
                     std::vector<std::vector<double>> param_weights = {});

    [[nodiscard]] std::size_t n_k() const noexcept { return kx_.size(); }
    [[nodiscard]] std::span<const double> weights() const noexcept { return w_; }

    void prepare(std::size_t nw);
    void permute(std::span<const std::size_t> src);

    // Full evaluation: build ρ and ADD Σ w T and its gradient/laplacian into the
    // [nw], [np·dim·nw] arrays.
    void evaluate(const WalkerSet& batch, std::span<double> log_psi,
                  std::span<double> grad, std::span<double> lap);

    // Particle p proposed at batch.new_pos(): ADD the change of Σ w T to
    // log_ratio [nw], and p's k-space gradient/laplacian at the new position
    // to new_grad / new_lap [dim·nw].  Walkers [w_begin, w_end).  Keeps p's
    // new phases for accept().
    void ratio(const WalkerSet& batch, std::size_t p, std::size_t w_begin,
               std::size_t w_end, std::span<double> log_ratio,
               std::span<double> new_grad, std::span<double> new_lap);

    // Commit p's move where mask[w] = 1: update ρ, ADD Δ(Σ w T) to each of
    // `log_psi_targets`, and the gradient/laplacian change of every particle
    // OTHER than p to each of `grad_targets`/`lap_targets` (p's own new values
    // come from ratio()).  Must follow ratio() for the same p and walkers.
    void accept(const WalkerSet& batch, std::size_t p, std::span<const double> mask,
                std::size_t w_begin, std::size_t w_end,
                std::span<const std::span<double>> log_psi_targets,
                std::span<const std::span<double>> grad_targets,
                std::span<const std::span<double>> lap_targets);

    // Optimiser support, from the CURRENT ρ (after evaluate):
    //   d_logpsi[w] += Σ ∂w/∂p_k T,  and the same gradient/laplacian forms.
    //
    // ∂u/∂p is split by its own breakup.  The breakup is linear, but its fit is
    // ill-conditioned (cond ~1e5), so the split of ∂u/∂p and the derivative of
    // the split of u are different members of a family of near-equivalent
    // splits.  They agree in everything configuration-dependent (measured: ∇,
    // ∇² and walker-to-walker contrasts of d_logpsi to 1e-5 against finite
    // differences through rebuild) but carry different constants — the self-
    // image and k = 0 terms — so d_logpsi is defined up to a walker-independent
    // constant: the normalisation gauge the linear method projects out.
    void param_log_psi(std::size_t k, std::size_t nw, std::span<double> out) const;
    void param_grad_lap(const WalkerSet& batch, std::size_t k, std::span<double> dgrad,
                        std::span<double> dlap) const;

private:
    // Phases e^{ik·r} of one position for every retained k, into (c, s).
    void phases(double x, double y, double* c, double* s) const;
    // Add Σ_k wts T and the gradient/laplacian of every particle, for walker w,
    // from the cached ρ and phases.
    void accumulate(const WalkerSet& batch, std::size_t w, std::span<const double> wts,
                    double* log_psi, double* grad, double* lap) const;
    // Cached phases of particle i (global index) in walker w: (c, s) rows.
    [[nodiscard]] std::size_t slot(std::size_t i) const noexcept {
        return a_.contains(i) ? i - a_.first : a_.size() + (i - b_.first);
    }
    double* cos_row(std::size_t w, std::size_t i) noexcept {
        return pc_.data() + (w * n_cached_ + slot(i)) * kx_.size();
    }
    double* sin_row(std::size_t w, std::size_t i) noexcept {
        return ps_.data() + (w * n_cached_ + slot(i)) * kx_.size();
    }
    const double* cos_row(std::size_t w, std::size_t i) const noexcept {
        return pc_.data() + (w * n_cached_ + slot(i)) * kx_.size();
    }
    const double* sin_row(std::size_t w, std::size_t i) const noexcept {
        return ps_.data() + (w * n_cached_ + slot(i)) * kx_.size();
    }

    ParticleGroup a_, b_;
    bool same_group_ = true;

    // Retained k (half space) and their weights.
    std::vector<double> kx_, ky_, w_;
    std::vector<double> kx2_, ky2_;   // k_x², k_y²
    std::vector<std::vector<double>> param_w_;
    numerics::ReciprocalPhases phases_;   // e^{ik·r} for exactly these k

    // ρ_A (and ρ_B for two groups), walker-major [w·n_k + k].
    std::size_t nw_ = 0;
    std::vector<double> ra_re_, ra_im_, rb_re_, rb_im_;

    // e^{ik·r_i} of every particle of the group(s), walker-major
    // [(w·n_cached + slot)·n_k + k]: a move recomputes only the moved
    // particle's phases, not those of every particle that sees the changed ρ.
    // pend_c_/pend_s_: the moved particle's NEW phases from ratio(), [w·n_k + k].
    std::size_t n_cached_ = 0;
    std::vector<double> pc_, ps_, pend_c_, pend_s_;
};

}  // namespace alea
