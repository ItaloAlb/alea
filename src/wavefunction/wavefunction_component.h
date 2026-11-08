// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace alea {

class WalkerSet;
class OrbitalSet;

// One determinant block of a nodal component: particles [first, last) fill the
// rows of det[φ_i(r_j)] with the given orbitals.  A view into the component
// (clone the orbitals to keep them); see WavefunctionComponent::nodal_blocks.
struct NodalBlockView {
    std::size_t       first    = 0;
    std::size_t       last     = 0;
    const OrbitalSet* orbitals = nullptr;
};

struct Parameter {
    std::string name;
    double      value;
    bool        is_fixed;
    // can_flip_sign == false → optimise as p = s·exp(θ) so the sign is locked.
    // Default initialiser keeps all positional 3-field brace-inits valid.
    bool        sign_locked = false;
    // Largest value an optimizer step may give it (a cutoff held within the
    // Wigner–Seitz inradius); the step is clamped there, see stepped_value.
    double      upper_bound = std::numeric_limits<double>::infinity();
};

// The value after an optimizer step: additive, or multiplicative for a
// sign-locked parameter (Δθ in p = s·exp(θ) gives p·exp(Δθ)), then clamped at
// upper_bound.  A step past the bound becomes the step to it; the optimizer
// scores every candidate by correlated sampling at the parameters actually
// applied, so a clamped step that raises the energy significantly is refused
// like any other.
[[nodiscard]] inline double stepped_value(const Parameter& p, double step) noexcept {
    const double value = p.sign_locked ? p.value * std::exp(step) : p.value + step;
    return std::min(value, p.upper_bound);
}

// Base class for a single factor in the trial wavefunction ψ = Π_c ψ_c.
//
// All methods operate on the full crowd (nw walkers) simultaneously.
// Data layout for all arrays: SoA with walker index varying fastest.
//   gradient: [p * dim * nw + d * nw + w]
//   laplacian: [p * dim * nw + d * nw + w]
class WavefunctionComponent {
public:
    virtual ~WavefunctionComponent() = default;

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    virtual std::string name() const = 0;

    // Size all per-walker caches/scratch for nw walkers, np particles, dim
    // dimensions, WITHOUT computing anything.  Must be called (directly or via
    // evaluate_batch) before any range-restricted ratio_batch/apply_move_batch,
    // so those never resize — which keeps them safe to call concurrently on
    // disjoint walker ranges of one shared component.
    virtual void prepare(std::size_t nw, std::size_t np, std::size_t dim) {
        resize_caches(nw, np, dim);
    }

    // Full evaluation for all walkers in batch.
    // Populates internal caches (log_psi_c_, grad_c_, lap_c_) and
    // accumulates this component's contribution into batch's
    // gradients(), laplacians(), and log_psi().
    virtual void evaluate_batch(WalkerSet& batch) = 0;

    // Compute log(ψ_c(R'_w) / ψ_c(R_w)) for particle p at new positions batch.new_pos().
    // Writes into batch.new_grad() (new ∇_p log ψ_c at the proposed position: [dim * nw]),
    //         batch.new_lap()  (new ∂²_d log ψ_c at the proposed position: [dim * nw]),
    //         batch.log_ratios() (log|ψ'_c/ψ_c|: [nw]).
    // A component that can change sign also accumulates the phase of its ratio,
    //         imag(log(ψ'_c/ψ_c)),  into batch.phase_diffs() ([nw]); for a real
    // wavefunction that is 0 (positive ratio) or π (the move crossed a node).
    // Nodeless (strictly positive) components leave phase_diffs() untouched.
    // The DMC fixed-node gate rejects a move when the accumulated phase signals a
    // node crossing (crosses_node, qmc/dmc.h).
    // The results are accumulated (+=), so the caller must zero these before
    // the first component call.  [w_begin, w_end) restricts work to a contiguous
    // walker range (a crowd); w_end == npos means "to nw".  Stride is the global
    // nw; scratch must already be sized via prepare().
    virtual void ratio_batch(WalkerSet& batch, std::size_t p,
                             std::size_t w_begin = 0,
                             std::size_t w_end = npos) = 0;

    // Commit accepted moves and discard rejected ones, over the walker range
    // [w_begin, w_end).
    // accept_mask[w] == true  → accept the pending move for walker w
    // accept_mask[w] == false → reject
    // Updates internal caches and accumulates the delta into batch's
    // total gradients()/laplacians().
    virtual void apply_move_batch(WalkerSet& batch,
                                   std::size_t p,
                                   std::span<const bool> accept_mask,
                                   std::size_t w_begin = 0,
                                   std::size_t w_end = npos) = 0;

    // Deep copy: each thread owns one set of components.
    [[nodiscard]] virtual std::unique_ptr<WavefunctionComponent> clone() const = 0;

    // Does this factor set the nodes of psi?  Only a sign-changing factor does,
    // and only its parameters can move a fixed-node energy.  Used to scope an
    // optimisation stage (nodes, then everything else).
    [[nodiscard]] virtual bool is_nodal() const { return false; }

    // Does this factor carry a non-trivial phase, ψ_c = e^{u_c + iφ_c}?  A
    // complex component additionally accumulates ∇φ_c / ∂²_d φ_c into the
    // batch's phase planes (WalkerSet::enable_phase), and the arg of its move
    // ratio into phase_diffs().  Any complex component turns the whole trial
    // complex, and DMC then runs fixed-phase instead of fixed-node.
    [[nodiscard]] virtual bool is_complex() const { return false; }

    // The determinant blocks whose signs make up this factor's nodes, for a
    // consumer that evaluates the nodal surface on its own configurations (the
    // PIMC fixed-node restriction, which has no WalkerSet).  Empty by default:
    // a nodal factor that returns nothing cannot be used as PIMC nodes.
    [[nodiscard]] virtual std::vector<NodalBlockView> nodal_blocks() const { return {}; }

    // ── A pair moved together (two-body density matrices) ───────────────────
    // For every i of particles [a_first, a_last) and j of [b_first, b_last)
    // (disjoint ranges), add log|ψ_c(R_ij)/ψ_c(R)| into log_ratios and
    // arg(ψ_c(R_ij)/ψ_c(R)) into phases, both [(i′ N_b + j′) nw + w] with
    // i′ = i − a_first, j′ = j − b_first; R_ij is R with r_i and r_j both
    // shifted by the same shift_w ([d * nw + w]).  From the committed state
    // of `batch` alone: no configuration with only one of the two moved is
    // formed, so a node there (a determinant row of zeros) does no harm.  It
    // must not run between a ratio_batch and its apply_move_batch, whose
    // scratch it may reuse.  A component that does not implement it says so
    // below, and the default throws.
    [[nodiscard]] virtual bool supports_shifted_pairs() const { return false; }
    virtual void shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                           std::size_t a_last, std::size_t b_first,
                                           std::size_t b_last, std::span<const double> shift,
                                           std::span<double> log_ratios,
                                           std::span<double> phases);

    // Parameter management (for wavefunction optimization).
    virtual std::span<Parameter>       parameters()       = 0;
    virtual std::span<const Parameter> parameters() const = 0;

    // Number of non-fixed parameters. Returns 0 if this component has nothing
    // to optimise.
    [[nodiscard]] std::size_t num_optimizable_params() const;

    // Rebuild internal kernels from current parameter values. Called by the
    // optimiser after modifying parameters through the mutable span returned by
    // parameters(). Subclasses that construct helper objects from their
    // Parameter values must override this.
    virtual void rebuild() = 0;

    // Compute ∂ log ψ_c / ∂ p_k for every non-fixed parameter.
    // The batch must already have been evaluated (evaluate_batch or equivalent).
    // Writes into `derivs`, which is sized [n_opt * nw], where n_opt is
    // num_optimizable_params() and nw is batch.num_walkers().
    // Layout: derivs[k * nw + w] where k indexes the k-th non-fixed parameter.
    virtual void compute_parameter_derivatives(const WalkerSet& batch,
                                               std::span<double> derivs) const = 0;

    // ∂(∂_d log ψ_c)/∂p_k and ∂(∂²_d log ψ_c)/∂p_k for every non-fixed
    // parameter.  Both are [n_opt * np * dim * nw], layout k*np*dim*nw plus
    // grad_c_'s own index.  The optimiser needs them for ∂E_L/∂p, the curvature
    // term of the linear method's Hbar.
    // Returns false when this component cannot supply them; the optimiser then
    // drops that term for ITS parameters and keeps the older approximation,
    // whose step length is unreliable (it overshot ~2.5× for Drummond α0).
    // Whether this component implements the method below.  Queried up front so
    // the optimiser can say which parameters lose the curvature term.
    [[nodiscard]] virtual bool supports_parameter_gradient_derivatives() const {
        return false;
    }

    virtual bool compute_parameter_gradient_derivatives(
        const WalkerSet& batch, std::span<double> dgrad,
        std::span<double> dlap) const {
        (void)batch; (void)dgrad; (void)dlap;
        return false;
    }

    // ── Complex factors: both parts of the parameter derivatives ────────────
    // For ψ_c = e^{u_c + iφ_c} the log-derivative is complex,
    //   O_k = ∂ ln ψ_c/∂p_k = ∂u_c/∂p_k + i ∂φ_c/∂p_k,
    // and the linear method of a complex trial needs both parts (CostFunction).
    // derivs takes the real part (what compute_parameter_derivatives writes),
    // phase_derivs the imaginary part, same layout [n_opt * nw].  The default
    // is a real factor: Im O = 0.
    virtual void compute_complex_parameter_derivatives(const WalkerSet& batch,
                                                       std::span<double> derivs,
                                                       std::span<double> phase_derivs) const {
        compute_parameter_derivatives(batch, derivs);
        std::fill(phase_derivs.begin(), phase_derivs.end(), 0.0);
    }
    // The same for the curvature term: ∂(∂_d u_c)/∂p_k, ∂(∂²_d u_c)/∂p_k into
    // dgrad/dlap and ∂(∂_d φ_c)/∂p_k, ∂(∂²_d φ_c)/∂p_k into dpgrad/dplap, all
    // in the layout of compute_parameter_gradient_derivatives.  Returns false,
    // as that does, when the component cannot supply them.
    virtual bool compute_complex_parameter_gradient_derivatives(
        const WalkerSet& batch, std::span<double> dgrad, std::span<double> dlap,
        std::span<double> dpgrad, std::span<double> dplap) const {
        std::fill(dpgrad.begin(), dpgrad.end(), 0.0);
        std::fill(dplap.begin(), dplap.end(), 0.0);
        return compute_parameter_gradient_derivatives(batch, dgrad, dlap);
    }

    // Read-only access to this component's cached contribution.
    // Populated by evaluate_batch and updated incrementally by apply_move_batch.
    // After a deferred commit, grad_c/lap_c may need the corresponding refresh.
    std::span<const double> log_psi_c()  const { return log_psi_c_; }   // [nw]
    std::span<const double> grad_c()     const { return grad_c_; }      // [np * dim * nw]
    std::span<const double> lap_c()      const { return lap_c_; }  // [np * dim * nw]

    // Reorder the committed per-walker caches (log_psi_c_, value_c_, grad_c_,
    // lap_c_) so column c becomes a copy of old column src[c] (reorder /
    // duplicate / remove).  This is how the cached
    // wavefunction state travels with the walker during branching — no
    // recomputation.  The transient pending_* caches are not carried (they are
    // rebuilt by the next ratio_batch).  After the call the caches hold
    // src.size() walkers.  Components with extra per-walker committed state
    // must override and also permute it.
    virtual void permute_caches(std::span<const std::size_t> src);

    // Optional deferred commit with the same ratio/accept/WalkerSet commit order.
    // Values, phases and inverse state remain current; partner derivatives may
    // be stale. Refresh a particle before reading its drift and all derivatives
    // before H, estimators, parameter derivatives or returning to eager calls.
    // Clone/permutation must carry the matching stale totals and contributions.
    // Defaults preserve the eager contract for components without deferral.
    virtual void apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
        std::span<const bool> accept, std::size_t w_begin = 0, std::size_t w_end = npos) {
        apply_move_batch(batch, p, accept, w_begin, w_end);
    }
    virtual void refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
        std::size_t w_begin = 0, std::size_t w_end = npos) {
        (void)batch; (void)p; (void)w_begin; (void)w_end;
    }
    virtual void refresh_derivatives(WalkerSet& batch,
        std::size_t w_begin = 0, std::size_t w_end = npos) {
        (void)batch; (void)w_begin; (void)w_end;
    }

    // A driver that commits with apply_move_batch_deferred and calls
    // refresh_derivatives before reading any Laplacian (VMC and DMC: the
    // Hamiltonian, estimators and parameter derivatives all follow a refresh)
    // passes false: a component may then leave the moved particle's
    // contribution to new_lap at zero and its own Laplacian stale, like a
    // partner's, until the refresh.  new_grad and the ratio stay exact.
    // Default true: ratio_batch writes the exact new_lap, as documented above.
    // For components whose proposal Laplacian is expensive (backflow, O(N³)).
    virtual void set_proposal_laplacian(bool needed) { (void)needed; }

    // Bytes of committed per-walker state a sweep reads or writes, valid after
    // evaluate_batch.  Drivers size cache-blocked sweeps from it; it never
    // changes results.  Default: the base caches (log ψ_c, ∇ and ∇² log ψ_c);
    // components with large extra state (determinant matrices) add theirs.
    [[nodiscard]] virtual std::size_t walker_state_bytes() const noexcept {
        if (log_psi_c_.empty()) return 0;
        return (log_psi_c_.size() + value_c_.size() + grad_c_.size() + lap_c_.size())
               / log_psi_c_.size() * sizeof(double);
    }

protected:
    // Resize caches for a batch of nw walkers, np particles, dim dimensions.
    void resize_caches(std::size_t nw, std::size_t np, std::size_t dim);

    // ── Opt-in per-particle share of log ψ_c ───────────────────────────────
    // value_c_[p*nw + w] = s_p, the part of log ψ_c that depends on particle
    // p; for a one-body factor log ψ_c = Σ_p u(r_p), s_p = u(r_p).  A move of
    // p changes log ψ_c by s_p' − s_p, so ratio_batch evaluates the proposed
    // position only and reads s_p instead of recomputing it at the old one; it
    // leaves s_p' in pending_value_c_, and apply_move_batch commits it with
    // commit_particle_value.  A pair factor's share Σ_k u(r_pk) gains nothing
    // from this: every accepted move of p also changes the partners' shares,
    // which needs the old pair values anyway.
    // A component opts in from its constructor.  The base then sizes the
    // cache in resize_caches, carries it in permute_caches and counts it in
    // walker_state_bytes; it stays empty for components that do not, so
    // determinants and collective (k-space) factors pay nothing.  Like every
    // committed cache it is valid after evaluate_batch until the parameters
    // change.
    void enable_particle_values() noexcept { particle_values_ = true; }

    // For accepted walkers in [w_begin, w_end): add s_p' − s_p to log_psi_c_
    // and to the batch total log ψ, and make value_c_[p] the pending share
    // itself (assigned, not incremented: s + (s' − s) can round away from the
    // s' that ratio_batch evaluated).
    void commit_particle_value(std::span<double> batch_log_psi, std::size_t p,
                               std::span<const bool> accept, std::size_t w_begin,
                               std::size_t w_end) noexcept;

    // Committed contribution of this component.
    std::vector<double> log_psi_c_;   // [nw]
    std::vector<double> value_c_;     // [np * nw] if enable_particle_values(), else empty
    std::vector<double> grad_c_;      // [np * dim * nw]
    std::vector<double> lap_c_;       // [np * dim * nw]

    // Pending contribution for the in-flight ratio_batch call.
    // Committed by apply_move_batch (accept) or discarded (reject).
    std::vector<double> pending_value_c_;    // [nw]        (particle p only)
    std::vector<double> pending_grad_c_;     // [dim * nw]  (particle p only)
    std::vector<double> pending_lap_c_;      // [dim * nw]  (particle p only)

    // Minimum-image displacement scratch for the periodic path, [dim * nw] each
    // (fold_delta_old_ holds the pre-move pair when a kernel needs both).
    //
    // Gradients need the folded DIRECTION Δ_d/r, not just the folded |r| the
    // distance table stores, and on a general Bravais lattice the fold mixes
    // Cartesian components — so it cannot be applied inside the per-dimension
    // loops and is materialised once per pair here instead.  Sized in
    // resize_caches (never inside a batch call); concurrent DMC crowds write
    // disjoint [w_begin, w_end) slices, so sharing across a crowd split is
    // race-free.  Untouched on the open path.
    std::vector<double> fold_delta_;      // [dim * nw]
    std::vector<double> fold_delta_old_;  // [dim * nw]

private:
    bool particle_values_ = false;   // value_c_ is sized (enable_particle_values)
};

}  // namespace alea
