// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The sum over periodic images of a pair Jastrow, U = Σ_{pairs} Σ_L u(|r_ij + L|),
// evaluated by the optimized breakup (Natoli & Ceperley, J. Comput. Phys. 117,
// 171 (1995)) — the trial-function counterpart of EwaldTerm.
//
// Why: under periodic boundaries Ψ must be periodic and SMOOTH (∇Ψ continuous
// at the edges of the box) or the variational bound is lost; a long-ranged u
// read at the minimum-image distance has a kink on the Wigner–Seitz boundary.
// Ceperley, "Introduction to QMC methods applied to the electron gas", §2.2
// and §2.5: "the Ewald image method is needed to represent the correct
// long-range behavior of the potential and the trial function".
//
// ── The split ───────────────────────────────────────────────────────────────
//     u(r) − u∞ = u_short(r) + u_long(r)
// u_short has compact support inside r_c (the Wigner–Seitz inradius) and is
// summed by minimum image — the pair Jastrow's own real-space kernel, with u
// replaced through to_short_range().  u_long is summed in reciprocal space:
//     same group  (A = B):  Σ_{k>0} w_k |ρ_A(k)|²
//     two groups  (A ≠ B):  Σ_{k>0} w_k 2 Re[ρ_A(k) ρ_B(k)*]
// with ρ_g(k) = Σ_{i∈g} e^{ik·r_i}, the sum over half the retained k (±k fold)
// and w_k = ũ_long(k)/A from the breakup.  u∞ and the k = 0 term are
// configuration-independent constants in ln Ψ and are dropped.  The split is
// exact for ANY weights (both halves are smooth and periodic), so the breakup's
// accuracy only decides how close Ψ is to the true image sum, never whether Ψ
// is valid.
//
// The reciprocal half is a KSpacePairSum with these weights (see
// kspace_pair_sum.h for its derivatives, cost and threading).

#include <cstddef>
#include <span>
#include <vector>

#include "numerics/long_range_breakup.h"
#include "numerics/long_range_spline.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "wavefunction/kspace_pair_sum.h"

namespace alea {

class WalkerSet;

class PeriodicPairSum {
public:
    PeriodicPairSum() = default;   // inactive: the plain minimum-image Jastrow

    // u_minus_limit = u − u∞ (with u′, u″ and a tail transform); u_inf = u∞.
    // param_minus_limit[k] / param_inf[k]: the same for ∂u/∂p of the k-th
    // optimisable parameter, which the optimiser's derivatives need.
    PeriodicPairSum(const SimulationCell& cell, ParticleGroup a, ParticleGroup b,
                    const numerics::RadialFunction& u_minus_limit, double u_inf,
                    const std::vector<numerics::RadialFunction>& param_minus_limit = {},
                    const std::vector<double>& param_inf = {});

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] double r_cut() const noexcept { return spline_.r_cut; }
    [[nodiscard]] std::size_t n_k() const noexcept { return k_.n_k(); }

    // In place, element-wise: (u, u′, u″) of the full u at distance r[i] →
    // the short-range part.  Zero at and beyond r_c.
    void to_short_range(std::span<const double> r, std::span<double> u,
                        std::span<double> du, std::span<double> d2u) const noexcept;
    // The same for ∂u/∂p of optimisable parameter `k` (value and two radial
    // derivatives at one distance).
    void param_to_short_range(std::size_t k, double r, double& v, double& d1,
                              double& d2) const noexcept;

    // ── Reciprocal-space half: a KSpacePairSum with the breakup's weights ───
    void prepare(std::size_t nw) { k_.prepare(nw); }
    void permute(std::span<const std::size_t> src) { k_.permute(src); }
    void evaluate(const WalkerSet& batch, std::span<double> log_psi,
                  std::span<double> grad, std::span<double> lap) {
        k_.evaluate(batch, log_psi, grad, lap);
    }
    void ratio(const WalkerSet& batch, std::size_t p, std::size_t w_begin,
               std::size_t w_end, std::span<double> log_ratio,
               std::span<double> new_grad, std::span<double> new_lap) {
        k_.ratio(batch, p, w_begin, w_end, log_ratio, new_grad, new_lap);
    }
    void accept(const WalkerSet& batch, std::size_t p, std::span<const double> mask,
                std::size_t w_begin, std::size_t w_end,
                std::span<const std::span<double>> log_psi_targets,
                std::span<const std::span<double>> grad_targets,
                std::span<const std::span<double>> lap_targets) {
        k_.accept(batch, p, mask, w_begin, w_end, log_psi_targets, grad_targets, lap_targets);
    }

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
    void param_log_psi(std::size_t k, std::size_t nw, std::span<double> out) const {
        k_.param_log_psi(k, nw, out);
    }
    void param_grad_lap(const WalkerSet& batch, std::size_t k, std::span<double> dgrad,
                        std::span<double> dlap) const {
        k_.param_grad_lap(batch, k, dgrad, dlap);
    }

private:
    bool active_ = false;
    numerics::LongRangeSpline spline_;
    double u_inf_ = 0.0;
    std::vector<numerics::LongRangeSpline> param_splines_;
    std::vector<double> param_inf_;
    KSpacePairSum k_;
};

}  // namespace alea
