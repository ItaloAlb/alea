// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "wavefunction/kspace_pair_sum.h"
#include "wavefunction/wavefunction_component.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"

namespace alea {

// Long-range Jastrow in reciprocal space, one free coefficient per |k| shell
// (CASINO's p-term; Drummond, Towler & Needs, PRB 70, 235119 (2004), §III B):
//
//     ln Ψ ⊃ Σ_{s=1}^{n} p_s Σ_{k∈s} T_k,
//     T_k = |ρ_A(k)|²  (one group)   or   2 Re[ρ_A(k) ρ_B(k)*]  (two groups),
//
// over the first n shells of half-space reciprocal vectors of the cell.  It is
// the correlation at wavelengths beyond a short-range Jastrow's cutoff — the
// plasmon zero-point motion that sets S(k) ∝ k^{3/2} at small k in 2-D — and
// it is smooth at r = 0, so it leaves the cusp to the short-range term.
//
// In terms of a periodic pair function, p_k = ũ(k)/A (A the cell area): the
// same group's Σ_{i<j} u(r_ij) is (1/A) Σ_{k>0} ũ(k)|ρ_k|² up to a constant.
// That is how the builder seeds the p_s from the Gaskell RPA form.
//
// Every p_s is a linear parameter: ∂ ln Ψ/∂p_s = Σ_{k∈s} T_k exactly, with the
// curvature terms from the same expressions (KSpacePairSum).
class KSpaceJastrow final : public WavefunctionComponent {
public:
    // params: one per shell, in shell order.  Throws if the cell is not
    // periodic 2-D or the count does not match n_shells.
    KSpaceJastrow(const SimulationCell& cell, ParticleGroup group_a, ParticleGroup group_b,
                  std::size_t n_shells, std::vector<Parameter> params);

    std::string name() const override { return "kspace_jastrow"; }

    // |k| of each shell (half-space vectors), for seeding and diagnostics.
    [[nodiscard]] const std::vector<double>& shell_k() const noexcept { return shell_k_; }
    [[nodiscard]] const std::vector<std::size_t>& shell_count() const noexcept {
        return shell_count_;
    }

    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;
    void permute_caches(std::span<const std::size_t> src) override;
    void evaluate_batch(WalkerSet& batch) override;
    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;
    void apply_move_batch(WalkerSet& batch, std::size_t p, std::span<const bool> accept_mask,
                          std::size_t w_begin = 0, std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<KSpaceJastrow>(*this);
    }

    std::span<Parameter>       parameters()       override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }
    void rebuild() override;

    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;
    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap) const override;

private:
    // Weights w_k = p_{shell(k)} and, per free shell, its indicator ∂w_k/∂p_s.
    void push_weights();

    ParticleGroup group_a_, group_b_;
    std::vector<Parameter> params_;
    std::vector<std::size_t> shell_of_;   // per k vector
    std::vector<double> shell_k_;
    std::vector<std::size_t> shell_count_;
    KSpacePairSum sum_;
    std::vector<double> mask_;   // accept mask as 0/1, [nw]
};

}  // namespace alea
