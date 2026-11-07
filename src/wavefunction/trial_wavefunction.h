// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "wavefunction_component.h"

namespace alea {

class WalkerSet;

class TrialWavefunction {
public:
    void add_component(std::unique_ptr<WavefunctionComponent> component);

    // Size all components' per-walker caches/scratch up front (no compute), so
    // range-restricted ratio_batch/apply_move_batch never resize.
    void prepare(std::size_t nw, std::size_t np, std::size_t dim);

    // Full evaluation for all walkers in batch.
    // Calls evaluate_batch on each component and sums contributions into
    // batch.gradients(), batch.laplacians(), and batch.log_psi().
    void evaluate_batch(WalkerSet& batch);

    // Compute log(ψ(R'_w)/ψ(R_w)) for particle p at batch.new_pos() over the
    // walker range [w_begin, w_end) (default = whole batch).  Zeros the range of
    // batch.new_grad()/new_lap()/log_ratios() first, then accumulates each
    // component's contribution.
    void ratio_batch(WalkerSet& batch, std::size_t p,
                     std::size_t w_begin = 0,
                     std::size_t w_end = static_cast<std::size_t>(-1));

    // Commit accepted / reject discarded moves for particle p over [w_begin, w_end).
    void apply_move_batch(WalkerSet& batch,
                          std::size_t p,
                          std::span<const bool> accept_mask,
                          std::size_t w_begin = 0,
                          std::size_t w_end = static_cast<std::size_t>(-1));

    // Same commit order as apply_move_batch; components may defer partner
    // derivatives. Refresh p before drift; refresh all before H, estimators,
    // parameter derivatives or switching back to the eager interface.
    // Calls on disjoint walker ranges remain independent.
    void apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
        std::span<const bool> accept, std::size_t w_begin = 0,
        std::size_t w_end = static_cast<std::size_t>(-1));
    void refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
        std::size_t w_begin = 0, std::size_t w_end = static_cast<std::size_t>(-1));
    void refresh_derivatives(WalkerSet& batch, std::size_t w_begin = 0,
        std::size_t w_end = static_cast<std::size_t>(-1));
    // WavefunctionComponent::set_proposal_laplacian on every component.
    void set_proposal_laplacian(bool needed) {
        for (const auto& c : components_) c->set_proposal_laplacian(needed);
    }

    // Σ of the components' committed per-walker state, valid after
    // evaluate_batch; sizes cache-blocked sweeps (never changes results).
    [[nodiscard]] std::size_t walker_state_bytes() const noexcept;

    // True when any component carries a phase (ψ complex).  Then evaluate_batch
    // enables and fills the batch's phase planes, and DMC runs fixed-phase.
    [[nodiscard]] bool is_complex() const;

    // Ratios of every pair (i ∈ [a_first, a_last), j ∈ [b_first, b_last))
    // moved together by shift_w: Σ_c of the components'
    // shifted_pair_ratios_batch (WavefunctionComponent), into log_ratios and
    // phases [(i′ N_b + j′) nw + w], which it zeroes first.
    void shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                   std::size_t a_last, std::size_t b_first, std::size_t b_last,
                                   std::span<const double> shift, std::span<double> log_ratios,
                                   std::span<double> phases);
    // Names of the components that cannot supply those ratios (empty: all can).
    [[nodiscard]] std::vector<std::string> components_without_shifted_pairs() const;

    // Produce an independent copy for use in a separate thread.
    [[nodiscard]] TrialWavefunction clone() const;

    // Reorder every component's committed per-walker caches so column c becomes
    // a copy of old column src[c] (reorder/duplicate/remove).  Called alongside
    // WalkerSet::apply_permutation during branching so the cached wavefunction
    // state travels with the walker — no recomputation.
    void permute_caches(std::span<const std::size_t> src);

    // ── Optimisation support ──────────────────────────────────────────────

    // Expose components for read access (parameter collection, etc.).
    [[nodiscard]] std::span<const std::unique_ptr<WavefunctionComponent>>
    components() const noexcept { return components_; }

    // Mutable access for parameter updates during optimisation.
    [[nodiscard]] std::span<std::unique_ptr<WavefunctionComponent>>
    components() noexcept { return components_; }

    // Rebuild internal kernels of all components from their current parameter
    // values.  Call after modifying parameters via the components' mutable
    // parameter spans.
    void rebuild();

    // Total number of non-fixed parameters across all components.
    [[nodiscard]] std::size_t total_optimizable_params() const;

    // Compute ∂ log Ψ / ∂ p_k for all walkers, accumulating across components.
    // `derivs` is sized [total_optimizable_params() * nw].
    // The batch must already have been fully evaluated.
    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const;

    // ∂(∇ log Ψ)/∂p_k and ∂(∇² log Ψ)/∂p_k, [n_opt * np * dim * nw] each.
    // `supported` is [n_opt]: 1 where the owning component supplied them, 0
    // where it could not (those slices stay zero).  Returns true if any did.
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap,
                                                std::span<char> supported) const;

private:
    std::vector<std::unique_ptr<WavefunctionComponent>> components_;
};

}  // namespace alea
