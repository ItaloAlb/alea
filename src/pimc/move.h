// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Move components for PIMC.
//
// Each Move proposes a change to a subset of beads, computes the action
// change, and accepts/rejects via Metropolis.
//
// SingleSliceMove:  displace one particle at one random slice.
// BisectionMove:    Levy construction over 2^levels slices for one particle.
// CentroidMove:     rigid translation of one particle's permutation cycle.

#include <array>
#include <cstddef>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "pimc/path_set.h"
#include "pimc/action.h"
#include "numerics/random.h"

namespace alea {

class Move {
public:
    virtual ~Move() = default;

    // Attempt one move on `path` using `action` for Metropolis.
    // Path distance tables must initially match the coordinates; this invariant
    // is preserved on acceptance and rejection. Scratch is not thread-shared:
    // concurrent chains need distinct Move instances (e.g. clone()).
    // Returns true if accepted.
    virtual bool attempt(PathSet& path, Action& action,
                         numerics::Random& rng) = 0;

    // Allocate reusable move-owned scratch for this path shape. Attempts call
    // this lazily too, so existing direct callers need no setup migration.
    virtual void prepare(const PathSet&) {}

    // A necessary sampling capability when M > 1: rigid translations alone
    // preserve the initial ring shape. This does not guarantee mixing times.
    [[nodiscard]] virtual bool changes_internal_beads() const noexcept { return false; }

    // Whether the move can change the path permutation (sample exchange).
    [[nodiscard]] virtual bool changes_permutation() const noexcept { return false; }

    // Move-specific counters for the run record, name -> value; summed over
    // replicas by the driver.  Called once per replica after sampling.
    [[nodiscard]] virtual std::vector<std::pair<std::string, double>>
    extra_diagnostics() const { return {}; }

    [[nodiscard]] virtual std::string name() const = 0;

    [[nodiscard]] virtual std::unique_ptr<Move> clone() const = 0;

    // Selection weight (for weighted random choice among moves).
    double weight() const noexcept { return weight_; }
    // Throws invalid_argument unless w is finite and strictly positive.
    void set_weight(double w);

    // Proposals rejected because the proposed path has infinite action
    // (Delta S = +inf, e.g. a pair leaving a finite pair-action domain) or
    // overflowing coordinates.  Rejecting them is exact — exp(-S) = 0 there —
    // but a nonzero count means a domain wall truncates the sampled
    // distribution, so the driver reports it instead of folding it silently
    // into the acceptance ratio.
    [[nodiscard]] std::size_t infinite_action_rejections() const noexcept {
        return infinite_action_rejections_;
    }
    virtual void reset_diagnostics() noexcept { infinite_action_rejections_ = 0; }

protected:
    // Classify a nonfinite action change Delta S of particle p at slice k
    // before anything is committed.  +inf is a counted rejection.  NaN (a
    // component bug) and -inf (an unbounded weight that the action admission
    // missed) throw: neither is a valid Metropolis outcome, and rejecting them
    // would silently cut configurations out of the sampled distribution.
    void reject_nonfinite(double delta_s, std::size_t p, std::size_t k);
    // An overflowing proposal (nonfinite coordinate) is an infinite-action
    // rejection, never a NaN propagated into the path.
    void reject_overflow() noexcept { ++infinite_action_rejections_; }

    double weight_ = 1.0;
    std::size_t infinite_action_rejections_ = 0;
};

// Single-slice move: displace one particle at one random slice by a Gaussian.
class SingleSliceMove final : public Move {
public:
    // sigma: finite positive Cartesian proposal width; throws otherwise.
    explicit SingleSliceMove(double sigma = 0.5);

    bool attempt(PathSet& path, Action& action,
                 numerics::Random& rng) override;

    [[nodiscard]] bool changes_internal_beads() const noexcept override { return true; }

    std::string name() const override { return "single_slice"; }

    std::unique_ptr<Move> clone() const override {
        return std::make_unique<SingleSliceMove>(*this);
    }

private:
    double sigma_;
};

// Bisection move: Levy bridge construction over 2^levels contiguous slices
// for one particle.  The endpoints are fixed; the interior beads are sampled
// from a free-particle bridge. Under PBC, symmetric relative lattice-frame
// increments sample winding; acceptance includes potential and frame terms.
class BisectionMove final : public Move {
public:
    // levels must be in [1,30], before any integer shifts or allocation.
    explicit BisectionMove(int levels = 3);

    void prepare(const PathSet& path) override;

    bool attempt(PathSet& path, Action& action,
                 numerics::Random& rng) override;

    [[nodiscard]] bool changes_internal_beads() const noexcept override { return true; }

    std::string name() const override { return "bisection"; }

    std::unique_ptr<Move> clone() const override {
        return std::make_unique<BisectionMove>(*this);
    }

private:
    int levels_;  // 2^levels slices are rebuilt
    std::vector<double> bridge_;   // [(2^levels + 1) * dim]
};

// Centroid move: translate the permutation cycle containing one random
// particle (its ring when pi(p) = p) by a Gaussian displacement.  The kinetic
// springs cancel, so only the potential action change matters.
class CentroidMove final : public Move {
public:
    // step: finite positive Cartesian proposal width; throws otherwise.
    explicit CentroidMove(double step = 0.5);

    void prepare(const PathSet& path) override;

    bool attempt(PathSet& path, Action& action,
                 numerics::Random& rng) override;

    std::string name() const override { return "centroid"; }

    std::unique_ptr<Move> clone() const override {
        return std::make_unique<CentroidMove>(*this);
    }

private:
    double step_;
    // Selected permutation cycle, worldline order [L*M*dim] (L = 1: one ring).
    std::vector<double> old_pos_;
    std::vector<double> proposed_pos_; // staging before mutating the path
    std::vector<std::size_t> members_; // [L_max], pi^j(p) in cycle order
};

// Endpoint table of one exchange class for a window [k0, k0 + n] that
// straddles the permuted closing link (shared by PermutationMove and
// JointPermutationMove).  Members are the class's particle indices
// (increasing); sigma(i) is the member reached by member i's worldline at the
// window's right end.  With L_i = r_i(k0) and R_j = r_j(k1), k1 = k0 + n - M:
//
//   t(i, j) = exp(-sum_d m_d D_d^2 / (2 n tau)),  D = minimage(R_j - L_i),
//   W(C)    = prod_m t(i_m, sigma(i_{m+1})) / t(i_m, sigma(i_m))
//
// for a cycle sequence C = (i_1 .. i_L) whose members move to the next
// member's target.  A link is admissible when t >= epsilon = 1e-3 (|D| beyond
// ~3.7 window widths sqrt(n tau / m) is pruned: such a cycle carries < 0.3% of
// a typical cycle's weight); a cycle is admissible when all its old and new
// links are, which is the same condition for C and for its reverse after the
// move, so pruning keeps detailed balance exact.
class CycleTable {
public:
    static constexpr double kEpsilon = 1e-3;
    static constexpr double kMinusLogEpsilon = 6.907755278982137;

    // Scratch for classes of up to max_members particles.
    void prepare(std::size_t max_members, std::size_t max_cycle);
    // Table, targets and adjacency for `members` (cycles up to max_cycle).
    void build(const PathSet& path, std::span<const std::size_t> members,
               std::size_t k0, std::size_t k1, std::size_t n, const double* mass,
               std::size_t max_cycle);
    std::size_t size() const noexcept { return n_; }
    std::size_t sigma(std::size_t i) const noexcept { return sigma_[i]; }
    void set_sigma(std::size_t i, std::size_t target) noexcept { sigma_[i] = target; }
    double t(std::size_t i, std::size_t j) const noexcept { return table_[i * n_ + j]; }
    bool old_ok(std::size_t i) const noexcept { return old_ok_[i] != 0; }
    // New link from i to j's current target, both old links admissible.
    bool edge(std::size_t i, std::size_t j) const noexcept {
        return i != j && old_ok_[i] && old_ok_[j] && t(i, sigma_[j]) >= kEpsilon;
    }
    // Recompute old-link admissibility and the CSR adjacency after sigma changes.
    void build_adjacency();
    // Q_s = 1 + sum of W over admissible cycles whose smallest member is s.
    double cycle_sum_from(std::size_t root);
    // Visit each admissible cycle rooted at root once, as the sequence that
    // starts at root: visit(sequence, length, W) returns true to stop.
    template <class Visit>
    bool for_each_cycle_from(std::size_t root, Visit&& visit);
    // W(C) of the sequence (product of new/old link ratios).
    double rotation_weight(const std::size_t* sequence, std::size_t length) const noexcept;

private:
    template <class Visit>
    bool extend(std::size_t depth, double weight, Visit& visit);

    std::size_t n_ = 0, length_limit_ = 0;
    std::vector<double> table_;          // [n * n], t(left i, right j)
    std::vector<std::size_t> sigma_;     // [n]
    std::vector<char> old_ok_;           // [n], t(i, sigma(i)) >= epsilon
    std::vector<std::size_t> adj_start_; // [n + 1]
    std::vector<std::size_t> adj_;       // [n * n], CSR edges
    std::vector<std::size_t> stack_;     // [max_cycle], DFS sequence
};

// Permutation move (Ceperley, Rev. Mod. Phys. 67, 279 (1995), Sec. V.F).
// Reconnects the worldlines of one
// exchange class across a window of n = 2^levels slices that contains the
// permuted closing link; the endpoints L_a, R_b stay fixed (CycleTable).
//
// Rooted heat bath: a uniform member s, then a cycle rooted at s (s its
// smallest member) with probability W(C)/Q_s (the identity, no change, with
// probability 1/Q_s), so P(C|x) = W(C) / (N_c Q_s(x)).  The members get free
// Levy bridges from L_a to R_{sigma'(a)} in the minimum-image frame; the
// bridges cancel the kinetic action, so Metropolis-Hastings leaves
//
//   A = min(1, exp(-Delta V) Q_s(x) / (W(C) Q_s(x'))),
//
// with Q_s(x') from the same endpoint table and sigma'.  Cost: O(N_c^2) for
// the table plus three rooted enumerations of O(k^{max_cycle-1}) cycles (k
// admissible neighbours).  Every member roots the cycles above it, so all
// cycles remain reachable.  Reversal also needs each old window's frame to be
// minimum image (the frame the reverse bridge would draw); otherwise the
// proposal is rejected.  The class and the window are drawn uniformly,
// independent of the state.  The chain samples |weight|; the fermion sign is
// carried by the PathSet.
class PermutationMove final : public Move {
public:
    // levels in [1, 30] (2^levels <= M, checked on use); max_cycle in [2, 8].
    PermutationMove(int levels, int max_cycle);

    void prepare(const PathSet& path) override;

    bool attempt(PathSet& path, Action& action,
                 numerics::Random& rng) override;

    [[nodiscard]] bool changes_permutation() const noexcept override { return true; }

    std::string name() const override { return "permutation"; }

    std::unique_ptr<Move> clone() const override {
        return std::make_unique<PermutationMove>(*this);
    }

    // Per cycle length L = 2..max_cycle: proposed / accepted non-identity
    // cycles; plus identity draws and frame (reversibility) rejections.
    std::vector<std::pair<std::string, double>> extra_diagnostics() const override;
    void reset_diagnostics() noexcept override;

private:
    int levels_, max_cycle_;
    CycleTable table_;
    std::vector<std::size_t> chosen_;     // [max_cycle], the drawn cycle
    std::vector<double> bridge_;          // [(2^levels + 1) * dim]
    std::vector<PathSet::Bead> beads_;    // [max_cycle * (2^levels - 1)]
    std::vector<double> positions_;       // [max_cycle * (2^levels - 1) * dim]
    std::vector<std::size_t> pi_new_;     // [np]
    // Diagnostics, [L] for L = 0..max_cycle (0, 1 unused).
    std::vector<std::size_t> proposed_, accepted_;
    std::size_t identity_draws_ = 0, frame_rejections_ = 0;
};

// Joint exchange move:
// exchanges bound pairs, e.g. two excitons, by permuting a class A (electrons)
// and a class B (holes) together in one window.  Swapping two excitons is one
// e-e and one h-h transposition; separate single-species moves would have to
// pass through a state with an electron detached from its hole.
//
//   1. A class pair (one exchangeable class of each group), the window and an
//      A-root s are drawn uniformly; an A-cycle C_A = (a_1 .. a_L) is drawn by
//      the rooted heat bath of PermutationMove (identity: no change).
//   2. A B-cycle of the same length is drawn as a sequence (b_1 .. b_L)
//      matched to (a_1 .. a_L), with weight W_B(b) prod_m g(a_m, b_m),
//      g = exp(-|minimage(L_b - L_a)|^2 / (2 lambda^2)) at the left endpoints
//      (lambda = pair_radius); pairs with g < 1e-3 are pruned (symmetric).
//      Z_B(x) = the sum of that weight over all admissible sequences.
//   3. All 2L members get free Levy bridges; one Metropolis decision.
//
// The rotations of b give the same final state and their g-sum is identical
// for the reverse move (the reverse pairs the reversed A sequence with the
// reversed B sequence, the same pairs), so it cancels:
//
//   A = min(1, exp(-Delta V) Q_s(x) Z_B(x) / (W_A W_B Q_s(x') Z_B(x'))),
//
// with Q_s, Z_B at x' from the same endpoint tables and the new targets.
// Joint moves alone keep parity(P_A) = parity(P_B); combine them with
// single-species permutation moves to sample every sector.
class JointPermutationMove final : public Move {
public:
    // One entry per admissible class pair: members of the A class and of the
    // B class (increasing particle indices, two or more each).
    using ClassPair = std::pair<std::vector<std::size_t>, std::vector<std::size_t>>;
    // levels in [1, 30]; max_cycle in [2, 8]; pair_radius finite and > 0.
    JointPermutationMove(int levels, int max_cycle, double pair_radius,
                         std::vector<ClassPair> class_pairs);

    void prepare(const PathSet& path) override;

    bool attempt(PathSet& path, Action& action,
                 numerics::Random& rng) override;

    [[nodiscard]] bool changes_permutation() const noexcept override { return true; }

    std::string name() const override { return "joint_permutation"; }

    std::unique_ptr<Move> clone() const override {
        return std::make_unique<JointPermutationMove>(*this);
    }

    // Per cycle length L: proposed / accepted joint cycles; identity draws,
    // A-cycles without an admissible B partner, frame rejections.
    std::vector<std::pair<std::string, double>> extra_diagnostics() const override;
    void reset_diagnostics() noexcept override;

private:
    // Candidate B members g(a, b) >= 1e-3 for each position of `sequence`.
    void build_candidates(const PathSet& path, const ClassPair& pair,
                          const std::size_t* sequence, std::size_t length,
                          std::size_t k0);
    // Visit every admissible B sequence matched to the candidates:
    // visit(sequence, W_B, G) returns true to stop.
    template <class Visit>
    bool for_each_partner_cycle(std::size_t length, Visit&& visit);
    template <class Visit>
    bool extend_partner(std::size_t depth, std::size_t length, double g, Visit& visit);
    // Z_B: sum of W_B * G over the admissible B sequences.
    double partner_sum(std::size_t length);

    int levels_, max_cycle_;
    double pair_radius_;
    std::vector<ClassPair> class_pairs_;
    CycleTable table_a_, table_b_;
    std::vector<std::size_t> chosen_a_, chosen_b_, reversed_;  // [max_cycle]
    std::vector<std::size_t> partner_;                          // [max_cycle], DFS
    std::vector<std::size_t> candidate_start_;                  // [max_cycle + 1]
    std::vector<std::size_t> candidate_;                        // [max_cycle * N_B]
    std::vector<double> candidate_g_;                           // [max_cycle * N_B]
    std::vector<double> bridge_;                                // [(2^levels + 1) * dim]
    std::vector<PathSet::Bead> beads_;      // [2 * max_cycle * (2^levels - 1)]
    std::vector<double> positions_;         // [2 * max_cycle * (2^levels - 1) * dim]
    std::vector<std::size_t> pi_new_;       // [np]
    std::vector<std::size_t> proposed_, accepted_;
    std::size_t identity_draws_ = 0, no_partner_ = 0, frame_rejections_ = 0;
};

// Canonical worm with one cut link (Boninsegni, Prokof'ev, Svistunov, Phys. Rev. E 74, 036701 (2006), at a fixed
// particle count and without missing beads).  Extended space: closed paths Z
// and paths G in which the link leaving a head bead h of a worm class is cut,
// weight W_G = C W_Z / rho0(h, t; tau), t = bead_after(h) the tail.  In Z the
// driver selects this move with probability p_sel = weight / sum of weights
// and it proposes `open`; while the path is open the driver calls only this
// move, which draws swap / head / tail / close with probabilities
// kSwap / kHead / kTail / kClose.  Every regrowth is a free random walk or a
// free bridge, which cancels the primitive kinetic action, so with m uniform
// in [1, max_length], N_b = N_w M worm beads, a the bead m slices before the
// head and rho0 the normalized free propagator:
//   open   A = C N_b kClose exp(-dV) / (p_sel rho0(a, t; (m+1) tau)),
//   close  A = p_sel rho0(a, t; (m+1) tau) exp(-dV) / (C N_b kClose),
//   head / tail regrowth  A = exp(-dV),
//   swap   A = exp(-dV) Sigma_h / Sigma_zeta,
// Sigma the heat-bath sums of rho0(., alpha; m_s tau) over the class at slice
// k_h + m_s (pruned symmetrically like CycleTable).  A swap reconnects the head
// to alpha; zeta, the bead m_s slices before alpha, becomes the head, and
// PathSet::exchange_successors restores identity in-array links.  Under PBC a
// proposal whose reverse needs a non-minimum-image frame, or whose new links
// are not their own minimum image, is rejected.  The chain samples |weight|;
// the fermion sign is that of the virtually closed path.
class WormMove final : public Move {
public:
    static constexpr double kSwap = 0.4, kHead = 0.2, kTail = 0.2, kClose = 0.2;

    // classes: members (increasing) of each worm exchange class, two or more
    // each; max_length >= 1 (<= M - 2 on use); swap_length >= 2 (<= M - 1 on
    // use); worm_constant > 0, or NaN to start from an estimate and tune it.
    WormMove(std::vector<std::vector<std::size_t>> classes, int max_length,
             int swap_length, double worm_constant);

    void prepare(const PathSet& path) override;
    bool attempt(PathSet& path, Action& action,
                 numerics::Random& rng) override;
    [[nodiscard]] bool changes_permutation() const noexcept override { return true; }
    std::string name() const override { return "worm"; }
    std::unique_ptr<Move> clone() const override {
        return std::make_unique<WormMove>(*this);
    }
    std::vector<std::pair<std::string, double>> extra_diagnostics() const override;
    void reset_diagnostics() noexcept override;

    // ── Driver hooks ──
    // p_sel in (0, 1]: the probability that the driver picks this move in Z.
    void set_selection_probability(double p);
    // The path is open in one of this worm's classes.
    [[nodiscard]] bool owns(const PathSet& path) const noexcept;
    [[nodiscard]] double worm_constant() const noexcept { return constant_; }
    [[nodiscard]] bool tunes_constant() const noexcept { return tune_; }
    // After an equilibration block: closed / own-open fractions of the
    // attempts; scales C by the square root of the ratio that would give
    // open/closed = target_ratio (clamped to [1/4, 4]).  No-op when C was given.
    void tune_constant(double closed_fraction, double open_fraction, double target_ratio);
    [[nodiscard]] std::size_t worm_particles() const noexcept { return particles_.size(); }
    // One-body density matrix at an open path (design doc): per sample draw
    // x ~ rho0(h, .; tau), write the minimum-image x - t to displacement[i*d..]
    // and s exp(-(S_V(t -> x) - S_V(t)) / 2) / (n C N_w M) to weight[i].
    void sample_open_ends(PathSet& path, Action& action, numerics::Random& rng,
                          std::span<double> displacement, std::span<double> weight);

private:
    bool attempt_open(PathSet& path, Action& action, numerics::Random& rng);
    bool attempt_close(PathSet& path, Action& action, numerics::Random& rng);
    bool attempt_regrow(PathSet& path, Action& action, numerics::Random& rng, bool head);
    bool attempt_swap(PathSet& path, Action& action, numerics::Random& rng);
    // Metropolis on log A after a staged regrowth of beads_[0..count).
    bool accept_staged(PathSet& path, Action& action, numerics::Random& rng,
                       std::size_t count, double log_ratio, double& delta_v);
    void load_mass(const Action& action);
    double log_rho0(const double* separation, std::size_t links, const PathSet& path) const;

    std::vector<std::vector<std::size_t>> classes_;
    std::vector<std::size_t> particles_;    // all worm particles, increasing
    std::vector<std::size_t> class_of_;     // [np], worm class or npos
    int max_length_, swap_length_;
    double constant_;
    bool tune_;
    double p_select_ = 0.0;
    double mass_[SimulationCell::kMaxDim] = {};
    std::vector<PathSet::Bead> beads_;      // [max(max_length, swap_length - 1)]
    std::vector<double> positions_;         // [same * dim], staged coordinates
    std::vector<double> walk_;              // [(max(max_length, swap_length) + 2) * dim]
    std::vector<double> heat_;              // [largest worm class], rho0 weights
    // Diagnostics: proposals / accepts per sub-move, rejection causes, relabels.
    enum Kind { kOpen, kCloseMove, kSwapMove, kHeadMove, kTailMove, kKinds };
    std::array<std::size_t, kKinds> proposed_{}, accepted_{};
    std::size_t tail_hits_ = 0, no_candidate_ = 0, frame_rejections_ = 0;
    std::size_t relabel_slices_ = 0;
    double relabel_seconds_ = 0.0;
};

}  // namespace alea
