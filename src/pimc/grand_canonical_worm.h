// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Grand-canonical worm (Boninsegni, Prokof'ev & Svistunov, Phys. Rev. E 74,
// 036701 (2006); "BPS" below), for one bosonic species that fills every row
// of the PathSet capacity.  The configuration space and the update pairs
// (open/close, insert/remove, advance/recede, swap) are BPS's; the acceptance
// ratios below are derived here from the proposal probabilities this code
// uses, so their constants differ from the paper's choice of proposals.
//
// ── Configuration space and weights ─────────────────────────────────────────
// Z: closed paths of N = R active rows (PathSet::active_particles),
//      W_Z = exp(Phi(N M)) prod_links rho0 exp(-S_V),
// G: one open worldline whose head is followed by a gap of g in [0, M - 1]
//    missing beads (PathSet::gap; at most one per slice), n_b = R M - g beads,
//      W_G = C exp(Phi(n_b)) prod_{present links} rho0 exp(-S_V)
// (BPS's Z and G sectors at discrete tau).  Phi is the chemical-potential
// weight of the bead count, piecewise linear with slope tau mu_R on the
// beads of the R-th row, n_b in ((R - 1) M, R M]:
//      Phi(n_b) - Phi(n_b') = tau mu_R (n_b - n_b')   for both in that range,
// so the closed sectors weigh exp(beta sum_{R <= N} mu_R).  BPS's single
// mu_in is mu_R = mu_in for every R, Phi = tau mu_in n_b.  One mu per pair
// of sectors (mu_R for N = R - 1 <-> R) is the multicanonical preweighting
// of N (Berg & Neuhaus, Phys. Rev. Lett. 68, 9 (1992); Torrie & Valleau,
// J. Comput. Phys. 23, 187 (1977)): with mu_R near F(R) - F(R - 1) every
// sector of a wide window is visited, and the driver removes the known
// weights exactly.  G configurations with R = n_min rows (n_b below n_min M)
// take the first pair's mu: any weight of G is valid, since G carries no
// measurement, and P(N) does not depend on it.  rho0 is the normalized free propagator
// of one link, S_V = tau sum_k V(R_k) over the beads R_k present at slice k
// (primitive action), with every N-dependent constant of V taken at the n_k
// beads of that slice (HamiltonianComponent::bead_energy).  The weights are
// those of unlabeled worldline configurations: the 1/N! of Z_N cancels the
// N! labelings of the rows, and every move below acts the same way on every
// labeling, so no N-dependent label factor appears.  Counting convention:
// beads (a free walk of m links from a new tail adds m + 1 beads).
//
// ── Moves ───────────────────────────────────────────────────────────────────
// The driver selects this move in Z with probability p_sel; it then proposes
// open (kOpenZ) or insert (1 - kOpenZ).  In G the driver calls only this
// move, which draws swap / advance / recede / close / remove with kSwap,
// kAdvance, kRecede, kClose, kRemove.  m is uniform in [1, L], L = max_length;
// the open/close gap g is uniform in [0, L].  Walks and bridges are drawn
// from rho0, so their link factors cancel; dS = S_V(new) - S_V(old) is the
// sum of bead actions of the changed beads, which lie on distinct slices.
// With A the cell area (periodic; an open cell draws the tail from the
// Gaussian density q(r) of width insertion_width and replaces 1/A by q):
//
// Every sub-move that changes n_b stays inside one row's range of Phi, so
// its factor is e^{tau mu_R dn_b} with R the row count of its G configuration
// (R = N + 1 for an insert from N):
//
//   open   (head = uniform bead of R M, the g beads after it go missing)
//          A = C e^{-tau mu_R g} e^{-dS} kClose R M (L + 1) / (p_sel kOpenZ rho0(t - h; g + 1))
//   close  (free bridge h -> t over g + 1 links fills the gap, g <= L)
//          A = p_sel kOpenZ rho0(t - h; g + 1) e^{tau mu_R g} e^{-dS} / (C kClose R M (L + 1))
//   insert (new row; tail uniform in the cell and in slice; walk of m links)
//          A = C A M L e^{tau mu_{N+1} (m + 1)} e^{-dS} kRemove / (p_sel (1 - kOpenZ))
//   remove (a single-row worm of m + 1 <= L + 1 beads)
//          A = p_sel (1 - kOpenZ) e^{-tau mu_R (m + 1)} e^{-dS} / (C A M L kRemove)
//   advance (m <= g gap beads drawn as a free walk from the head)
//          A = e^{tau mu_R m} e^{-dS} kRecede / kAdvance
//   recede (the head and its m - 1 predecessors go missing, g + m <= M - 1)
//          A = e^{-tau mu_R m} e^{-dS} kAdvance / kRecede
//   swap   (BPS swap, as WormMove; alpha by heat bath among the beads present
//          at slice k_h + m_s)   A = e^{-dS} Sigma_h / Sigma_zeta
//
// Each pair is the Metropolis-Hastings ratio W(y) T(y -> x) / (W(x) T(x -> y))
// with T the product of the sub-move selection, the uniform draws (bead or
// slice, length, gap, tail position) and the rho0 densities of the drawn
// walk or bridge (a bridge over n links has density prod rho0 / rho0(end; n)).
// Window: G keeps R in [max(n_min, 1), n_max], so every G configuration can
// close inside the window; an insert from N = n_max and a remove that would
// leave R - 1 < n_min are rejected.  The row count changes only through
// insert/remove: an advance never passes the tail (m <= g) and a recede never
// makes the gap a full row (g + m <= M - 1).  Under PBC a proposal whose new
// links are not their own minimum image, or whose reverse bridge would need a
// non-minimum-image frame, is rejected, as in WormMove.
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pimc/move.h"

namespace alea {

// The grand-canonical block of a worm move.
struct GrandCanonicalSpec {
    // mu_in in the run's energy unit: one value for every pair of sectors, or
    // n_max - n_min values, the mu of each pair N <-> N + 1 from N = n_min.
    std::vector<double> chemical_potential;
    std::size_t n_min = 0, n_max = 0; // window of the closed particle count
    // Open cells only: the tail of an inserted worm is drawn from a Gaussian
    // of this width (> 0) around insertion_center ([dim]); periodic cells draw
    // it uniformly in the cell and must leave the width at 0.
    double insertion_width = 0.0;
    std::vector<double> insertion_center;
};

class GrandCanonicalWormMove final : public WormBase {
public:
    static constexpr double kOpenZ = 0.5;
    static constexpr double kSwap = 0.3, kAdvance = 0.2, kRecede = 0.2,
                            kClose = 0.15, kRemove = 0.15;

    // max_length >= 1 (<= M - 2 on use); swap_length >= 2 (<= M - 1 on use);
    // worm_constant > 0, or NaN to start from an estimate and tune it; the
    // window 1 <= n_max, n_min <= n_max; finite chemical potentials, one or
    // n_max - n_min of them.
    GrandCanonicalWormMove(int max_length, int swap_length, double worm_constant,
                           GrandCanonicalSpec spec);

    void prepare(const PathSet& path) override;
    // `action` must offer bead actions for every row of the path capacity
    // (Action::supports_bead_action, built for capacity() rows).
    bool attempt(PathSet& path, Action& action, numerics::Random& rng) override;
    [[nodiscard]] bool changes_permutation() const noexcept override { return true; }
    std::string name() const override { return "worm"; }
    std::unique_ptr<Move> clone() const override {
        return std::make_unique<GrandCanonicalWormMove>(*this);
    }
    std::vector<std::pair<std::string, double>> extra_diagnostics() const override;
    void reset_diagnostics() noexcept override;

    void set_selection_probability(double p) override;
    [[nodiscard]] double worm_constant() const noexcept override { return constant_; }
    [[nodiscard]] bool tunes_constant() const noexcept override { return tune_; }
    void tune_constant(double closed_fraction, double open_fraction,
                       double target_ratio) override;
    // Use C = c (finite, > 0) from now on: the driver hands every replica the
    // one C it tunes from their pooled closed fraction.  Throws invalid_argument.
    void set_worm_constant(double c);
    [[nodiscard]] const GrandCanonicalSpec& spec() const noexcept { return spec_; }
    // mu_R, the slope of Phi on the beads of row R (configurations with R
    // rows in G, i.e. the pair N = R - 1 <-> R); R = n_min takes the first pair's.
    [[nodiscard]] double row_chemical_potential(std::size_t rows) const noexcept;
    // dS_V of the last accepted proposal (tests compare it with an
    // independent evaluation of the potential action before and after).
    [[nodiscard]] double last_delta_action() const noexcept { return last_delta_s_; }
    // ln r of the last attempt, r the Metropolis ratio W(y) T(y -> x) /
    // (W(x) T(x -> y)) of its proposal, or NaN when the attempt was rejected
    // before the Metropolis test (window, gap length, frame, no candidate).
    // Tests compare it with the weights and proposal densities written out
    // independently.
    [[nodiscard]] double last_log_acceptance() const noexcept { return last_log_a_; }

    // ln of the tail's insertion density at r ([dim]): -ln A in a periodic
    // cell, the Gaussian log-density in an open one.
    [[nodiscard]] double log_insertion_density(const PathSet& path, const double* r) const;

private:
    bool attempt_open(PathSet&, Action&, numerics::Random&);
    bool attempt_close(PathSet&, Action&, numerics::Random&);
    bool attempt_insert(PathSet&, Action&, numerics::Random&);
    bool attempt_remove(PathSet&, Action&, numerics::Random&);
    bool attempt_advance(PathSet&, Action&, numerics::Random&);
    bool attempt_recede(PathSet&, Action&, numerics::Random&);
    bool attempt_swap(PathSet&, Action&, numerics::Random&);
    // Metropolis on log A; false (and counted) for a nonfinite dS.
    bool accept(double log_a, double delta_s, numerics::Random& rng, PathSet::Bead where);
    // tau sum of bead energies of beads_[0..count) at positions (staged) or at
    // their current coordinates (staged == false), against the beads present
    // at each slice.
    double bead_actions(const PathSet& path, const Action& action, std::size_t count,
                        bool staged, std::size_t n_rows) const;
    void commit(PathSet& path, std::size_t count) const;
    double log_rho0(const double* separation, std::size_t links, const PathSet& path) const;
    void load_mass(const Action& action);
    void initial_constant(const PathSet& path);

    int max_length_, swap_length_;
    double constant_;
    bool tune_;
    GrandCanonicalSpec spec_;
    double p_select_ = 0.0;
    double mass_[SimulationCell::kMaxDim] = {};
    std::vector<PathSet::Bead> beads_;  // [max(max_length + 1, swap_length - 1)]
    std::vector<double> positions_;     // [same * dim], staged coordinates
    std::vector<double> walk_;          // [(max(max_length, swap_length) + 2) * dim]
    std::vector<double> heat_;          // [capacity], swap heat-bath weights
    enum Kind { kOpen, kCloseMove, kInsert, kRemoveMove, kAdvanceMove, kRecedeMove,
                kSwapMove, kKinds };
    std::array<std::size_t, kKinds> proposed_{}, accepted_{};
    std::size_t window_rejections_ = 0, frame_rejections_ = 0, no_candidate_ = 0;
    double last_delta_s_ = 0.0;
    double last_log_a_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace alea
