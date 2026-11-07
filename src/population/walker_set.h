// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "particle_batch.h"

namespace alea {

class Walker;
namespace numerics { class XoshiroRandom; }

// Working state for a crowd of nw walkers processed by a single thread: the
// configurations and distance tables of ParticleBatch, plus everything a
// ψ-sampled walker carries (log ψ, its derivatives, local energies, DMC
// bookkeeping).
//
// All per-walker arrays use SoA layout with the walker index varying fastest:
//   array[p * stride_p + d * nw + w]   (positions, gradients)
//   array[p * nw + w]                  (laplacians, local_energies)
//   array[d * nw + w]                  (new_pos, new_grad — one particle at a time)
//
// This layout makes the inner loop over walkers contiguous and SIMD-friendly.
class WalkerSet : public ParticleBatch {
public:
    // The cell is held by value (small POD, no lifetime coupling when a
    // WalkerSet is cloned per OpenMP thread).  Defaulting it to an open cell
    // keeps every existing construction site and its behaviour unchanged.
    WalkerSet(std::size_t nw, std::size_t np, std::size_t dim,
              const SimulationCell& cell = SimulationCell{});

    WalkerSet(const WalkerSet&)            = delete;
    WalkerSet& operator=(const WalkerSet&) = delete;
    WalkerSet(WalkerSet&&) noexcept        = default;
    ~WalkerSet()                           = default;

    void add_walker(Walker& w)  { walkers_.push_back(&w); }
    void clear_walkers()         { walkers_.clear(); }

    std::span<Walker*> walkers()      { return walkers_; }

    // Load all walker positions into SoA arrays and build distance tables.
    void load_all();
    // Save SoA positions and local energies back into walker objects.
    void save_all();
    // Append every walker's configuration to `out` as AoS rows [w][p][d]
    // (np·dim doubles per walker) — the layout of walker checkpoints and of
    // the driver-to-driver carry-over.  Cold path: runs once per driver.
    void append_configurations(std::vector<double>& out) const;

    // propose_move_batch (ParticleBatch) must be called before ratio_batch
    // and apply_move.

    // Apply accepted moves for particle p after accept/reject decision, over
    // the walker range [w_begin, w_end).  Commits pending_dist_rows_ for
    // accepted walkers.
    void apply_move(std::size_t p,
                    std::span<const double> new_pos,
                    std::span<const double> new_grad,
                    std::span<const double> new_lap,
                    std::span<const bool>   accept_mask,
                    std::size_t w_begin = 0, std::size_t w_end = npos);

    // Rebuild the per-walker SoA for a new set of walkers: new column c is a
    // copy of old column src[c].  Supports reorder, duplication and removal
    // (the basis of branching).  All committed per-walker state (positions,
    // gradients, laplacians, log_psi, local/old energies, rr_*, distance
    // tables) travels with the walker — no recomputation.  Transient scratch
    // (new_pos/new_grad/new_lap/log_ratios/pending rows/accept_mask) is just
    // resized.  After the call num_walkers() == src.size().
    void apply_permutation(std::span<const std::size_t> src);

    // QMCPACK-batched dynamic branching.  For each walker, the integer number
    // of copies is m = floor(weight + ξ), ξ~U[0,1): m==0 dies, m>1 fissions.
    // The population is clipped to n_max = 2·target+1 and lifted to
    // n_min = target/5+1 (a lone survivor is kept on total extinction).
    // Copies get a fresh id and parent_id = source id; weights/multiplicities
    // reset to 1.  Returns the source map (dst column ← src[dst]) so the caller
    // can carry wavefunction caches with TrialWavefunction::permute_caches.
    [[nodiscard]] std::vector<std::size_t> branch(numerics::XoshiroRandom& rng,
                                                  std::size_t target);
    // Reuse caller-owned offspring storage across DMC steps.
    void branch(numerics::XoshiroRandom& rng, std::size_t target, std::vector<std::size_t>& src);

    // ── SoA working state ────────────────────────────────────────────────
    // positions (ParticleBatch) / gradients: [p * dim * nw + d * nw + w]
    std::span<double>       gradients()        { return gradients_; }
    std::span<const double> gradients()  const { return gradients_; }
    // laplacians: [p * dim * nw + d * nw + w]
    // log_psi / local_energies: [nw]
    std::span<double>       laplacians()       { return laplacians_; }
    std::span<const double> laplacians() const { return laplacians_; }
    std::span<double>       log_psi()          { return log_psi_; }
    std::span<const double> log_psi()    const { return log_psi_; }
    std::span<double>       local_energies()       { return local_energies_; }
    std::span<const double> local_energies() const { return local_energies_; }

    // ── Phase planes of a complex trial ψ = e^{u + iφ} (fixed phase) ─────
    // The arrays above always hold the MODULUS part: log_psi = u = ln|ψ|,
    // gradients = ∇u, laplacians = ∂²_d u.  A complex trial also needs ∇φ and
    // ∂²_d φ, stored here as separate real planes (split re/im keeps the walker
    // loop contiguous and SIMD-friendly).  They exist only after enable_phase(),
    // which TrialWavefunction calls when a component is complex; a real trial
    // never allocates them and runs exactly the pre-fixed-phase code.
    // Layout as gradients: [p * dim * nw + d * nw + w].
    void enable_phase();
    [[nodiscard]] bool has_phase() const noexcept { return has_phase_; }

    // Bytes of per-walker SoA state a sweep touches: the distance table,
    // positions and ∇/∇² log ψ (doubled for a complex trial's phase planes).
    // Sizes cache-blocked sweeps together with the trial's own state.
    [[nodiscard]] std::size_t walker_state_bytes() const noexcept {
        const std::size_t planes = has_phase_ ? 5 : 3;
        return (np_ * np_ + planes * np_ * dim_) * sizeof(double);
    }
    std::span<double>       phase_gradients()        { return phase_gradients_; }
    std::span<const double> phase_gradients()  const { return phase_gradients_; }
    std::span<double>       phase_laplacians()       { return phase_laplacians_; }
    std::span<const double> phase_laplacians() const { return phase_laplacians_; }
    // Im E_L, [nw].  local_energies() holds Re E_L — the fixed-phase local
    // energy that DMC branches on.  Im E_L averages to 0 under |ψ|² for any
    // Hermitian H, so its mean is a correctness diagnostic, not an observable.
    std::span<double>       local_energies_imag()       { return local_energies_imag_; }
    std::span<const double> local_energies_imag() const { return local_energies_imag_; }

    // ── Per-walker scalar columns (the SoA, not the Walker objects, is the
    //    source of truth during a run — so branching can duplicate columns
    //    independently). All [nw]; carried by apply_permutation. ────────────
    std::span<double>            weights()              { return weights_; }
    std::span<const double>      weights()        const { return weights_; }
    std::span<double>            multiplicities()       { return multiplicities_; }
    std::span<const double>      multiplicities() const { return multiplicities_; }
    std::span<std::size_t>       ages()                 { return ages_; }
    std::span<const std::size_t> ages()           const { return ages_; }
    std::span<std::size_t>       ids()                  { return ids_; }
    std::span<const std::size_t> ids()            const { return ids_; }
    std::span<std::size_t>       parent_ids()           { return parent_ids_; }
    std::span<const std::size_t> parent_ids()     const { return parent_ids_; }

    // ── Proposed-move scratch (one particle p at a time) ─────────────────
    // new_pos / new_grad / new_lap: [d * nw + w]    log_ratios: [nw]
    // accept_mask: [nw] — written by the acceptance step, read by apply_move.
    std::span<bool>   accept_mask() { return {accept_mask_.get(), nw_}; }
    std::span<double> new_grad()    { return new_grad_; }
    std::span<double> new_lap()     { return new_lap_; }
    std::span<double> log_ratios()  { return log_ratios_; }
    // Accumulated phase of the proposed-move ratio, imag(log(ψ'/ψ)): [nw].
    // Sign-changing components add their contribution in ratio_batch; for the
    // nodeless (positive) wavefunctions it stays 0.  Read by the DMC fixed-node
    // gate (crosses_node, qmc/dmc.h).  Transient scratch like log_ratios.
    std::span<double> phase_diffs() { return phase_diffs_; }
    // Proposed-position ∇φ / ∂²_d φ of the moved particle, [d * nw + w]; the
    // phase counterparts of new_grad/new_lap (complex trial only).  apply_move
    // commits them into the phase planes for accepted walkers.
    std::span<double> new_phase_grad() { return new_phase_grad_; }
    std::span<double> new_phase_lap()  { return new_phase_lap_; }

    // ── DMC-specific fields ──────────────────────────────────────────────
    // rr_proposed: Σ_p τ/m_p·|ΔR|² (all proposed moves, for age tracking)
    // rr_accepted: Σ_p τ/m_p·|ΔR|² (accepted moves only, for branching)
    // old_energies: E_L before the step (snapshot for branch_weight).
    std::span<double>       rr_proposed()       { return rr_proposed_; }
    std::span<const double> rr_proposed() const { return rr_proposed_; }
    std::span<double>       rr_accepted()       { return rr_accepted_; }
    std::span<const double> rr_accepted() const { return rr_accepted_; }
    std::span<double>       old_energies()       { return old_energies_; }
    std::span<const double> old_energies() const { return old_energies_; }

    // ── Statistics ───────────────────────────────────────────────────────
    void        reset_statistics();
    void        record_accepts(std::size_t n) { n_accepted_ += n; }
    void        record_rejects(std::size_t n) { n_rejected_ += n; }
    std::size_t n_accepted() const noexcept { return n_accepted_; }
    std::size_t n_rejected() const noexcept { return n_rejected_; }

private:
    std::vector<Walker*> walkers_;

    std::vector<double> gradients_;        // [np * dim * nw]
    std::vector<double> laplacians_;       // [np * dim * nw]
    std::vector<double> log_psi_;          // [nw]
    std::vector<double> local_energies_;   // [nw]

    bool                has_phase_ = false;
    std::vector<double> phase_gradients_;   // [np * dim * nw] ∇φ   (complex ψ only)
    std::vector<double> phase_laplacians_;  // [np * dim * nw] ∂²_d φ
    std::vector<double> new_phase_grad_;    // [dim * nw] proposed-move scratch
    std::vector<double> new_phase_lap_;     // [dim * nw]
    std::vector<double> local_energies_imag_;  // [nw] Im E_L

    std::vector<double>      weights_;        // [nw] DMC weight
    std::vector<double>      multiplicities_; // [nw] DMC multiplicity
    std::vector<std::size_t> ages_;           // [nw] steps since last accepted move
    std::vector<std::size_t> ids_;            // [nw] walker id
    std::vector<std::size_t> parent_ids_;     // [nw] parent id (genealogy)

    std::vector<double> new_grad_;         // [dim * nw]
    std::vector<double> new_lap_;          // [dim * nw]
    std::vector<double> log_ratios_;       // [nw]
    std::vector<double> phase_diffs_;      // [nw] imag(log(ψ'/ψ)); fixed-node gate

    std::vector<double> rr_proposed_;      // [nw]
    std::vector<double> rr_accepted_;      // [nw]
    std::vector<double> old_energies_;     // [nw]

    std::unique_ptr<bool[]> accept_mask_;    // [nw]
    std::size_t accept_capacity_ = 0;
    std::vector<double> gather_double_;       // reusable SoA permutation destination
    std::vector<std::size_t> gather_size_;    // reusable scalar permutation destination
    std::vector<Walker*> gather_walkers_;    // [nw], handles only
    std::vector<char> branch_seen_, branch_copy_; // [old_nw], [new_nw]

    std::size_t n_accepted_ = 0;
    std::size_t n_rejected_ = 0;

    std::size_t next_id_ = 0;  // id counter for walkers spawned by branching
};

}  // namespace alea
