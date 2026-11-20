// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Closed imaginary-time paths in SoA layout [particle * dim * M + axis * M + slice].
//
// Connectivity: every link inside
// the arrays is the identity, bead (p, k) -> (p, k+1), and only the closing
// link is permuted, (p, M-1) -> (pi(p), 0).  A path with permutation pi then
// represents the worldlines of Ceperley's permuted density matrix
// rho(R, P R; beta) (Ceperley, Rev. Mod. Phys. 67, 279 (1995), Sec. V); a
// fermionic class contributes (-1)^{P_c}.  Because a slice holds the same
// labels whatever pi is, the per-slice distance tables never depend on pi.
//
// pi may only connect particles of one exchange class.  By default every
// particle is its own class, so only the identity is admissible and the
// boltzmannon stage cannot acquire exchange by accident.
//
// Active rows (grand-canonical runs): the storage holds capacity() =
// n_particles() rows, of which rows [0, active_particles()) are particles.
// Unused rows are fixed points of pi with finite placeholder coordinates that
// no action term reads; cycles, classes and moves see the active rows only.
// A canonical path has active_particles() == capacity().
#include <cstddef>
#include <span>
#include <vector>
#include "population/particle_batch.h"

namespace alea {
class PathSet : public ParticleBatch {
public:
    // One bead: a particle label and an imaginary-time slice.
    struct Bead {
        std::size_t particle;
        std::size_t slice;
    };

    // Construct an M-slice path at time step tau (atomic units). Invalid sizes,
    // nonfinite tau, and overflowing storage dimensions throw runtime_error.
    PathSet(std::size_t M, std::size_t np, std::size_t dim, double tau,
            const SimulationCell& cell = SimulationCell{});
    std::size_t num_slices() const noexcept { return num_walkers(); }
    double tau() const noexcept { return tau_; }
    double beta() const noexcept { return tau_ * static_cast<double>(num_slices()); }
    double& pos(std::size_t p, std::size_t d, std::size_t k) {
        return positions()[p * dim() * num_slices() + d * num_slices() + k];
    }
    double pos(std::size_t p, std::size_t d, std::size_t k) const {
        return positions()[p * dim() * num_slices() + d * num_slices() + k];
    }
    std::size_t next(std::size_t k) const noexcept { return next_slice_[k]; }
    std::size_t prev(std::size_t k) const noexcept { return prev_slice_[k]; }

    // pi: the particle whose slice-0 bead follows particle p's slice M-1.
    std::span<const std::size_t> permutation() const noexcept { return permutation_; }
    std::size_t next_particle(std::size_t p) const noexcept { return permutation_[p]; }
    std::size_t prev_particle(std::size_t p) const noexcept { return inverse_[p]; }
    bool identity_permutation() const noexcept { return identity_; }
    // Neighbouring beads along the worldline through (p, k).
    Bead bead_after(std::size_t p, std::size_t k) const noexcept {
        return k + 1 < num_slices() ? Bead{p, k + 1} : Bead{permutation_[p], 0};
    }
    Bead bead_before(std::size_t p, std::size_t k) const noexcept {
        return k > 0 ? Bead{p, k - 1} : Bead{inverse_[p], num_slices() - 1};
    }

    // Exchange classes: class_id[p] labels the class of row p and
    // fermionic[p] != 0 marks fermions (uniform within a class), for every
    // row of the capacity.  The current permutation must already respect the
    // classes.  Throws invalid_argument.
    void set_exchange_classes(std::span<const std::size_t> class_id,
                              std::span<const char> fermionic);
    // Replace pi ([capacity], unused rows fixed points) after validating it is
    // a class-preserving bijection; the path is unchanged on failure.
    // Coordinates and tables are not touched.
    void set_permutation(std::span<const std::size_t> pi);
    // Product over fermionic cycles of (-1)^(length-1); +1 for bosons/identity.
    double permutation_sign() const noexcept { return sign_; }
    // The smallest particle index of every cycle of pi over the active rows,
    // increasing.
    std::span<const std::size_t> cycle_leaders() const noexcept { return cycle_leaders_; }
    std::size_t exchange_class(std::size_t p) const noexcept { return class_id_[p]; }
    // Rows in the largest exchange class at full capacity (1 when nothing can
    // exchange): the bound that sizes cycle scratch, never exceeded later.
    std::size_t largest_exchange_class() const noexcept { return largest_class_; }
    // Active members (increasing) of every class that has more than one row
    // at full capacity, in order of first member: the sets a permutation move
    // may reconnect.  The list itself does not change with the active count,
    // so a move drawing a class uniformly draws it state-independently; a
    // class may hold fewer than two active members in a grand-canonical run.
    std::span<const std::vector<std::size_t>> exchangeable_classes() const noexcept {
        return exchangeable_;
    }

    // ── Active rows ──
    std::size_t capacity() const noexcept { return n_particles(); }
    std::size_t active_particles() const noexcept { return active_; }
    // Setup only: make rows [0, n) the particles of a closed path whose
    // permutation leaves rows >= n fixed.  Throws invalid_argument.
    void set_active_particles(std::size_t n);
    // Activate row active_particles() as a new particle, a fixed point of pi
    // in the class set_exchange_classes gave that row, and return its index.
    // Coordinates are the caller's: it writes them and refreshes the tables.
    // Closed paths only; throws logic_error at capacity.  O(N).
    std::size_t append_row();
    // Delete particle p, a fixed point of pi, from a closed path: the last
    // active row moves into row p (coordinates, tables, pi and its class), so
    // the active rows stay contiguous.  Throws logic_error otherwise.
    // O(capacity * M) for the moved row's tables.
    void remove_row(std::size_t p);

    // Refresh all pairs, or only a selected contiguous slice range [begin,end).
    // Particle-only refresh is O(np * (end-begin)) and reuses batch scratch.
    void rebuild_tables() { dist_tables().build_all(positions()); }
    void rebuild_tables(std::size_t begin, std::size_t end);
    void update_particle_tables(std::size_t p, std::size_t begin, std::size_t end) {
        refresh_particle_tables(p, begin, end);
    }
    // Initialize every slice from a finite AoS [np * dim] configuration. Periodic
    // coordinates are canonicalized before tables are rebuilt; throws on mismatch.
    void collapse_to(std::span<const double> config_aos);
    std::vector<double> all_positions() const {
        const auto p = positions();
        return {p.begin(), p.end()};
    }
    void wrap_positions() { wrap_all_positions(); }
    // Wrap only particle p over [begin,end); never alters unselected coordinates.
    void wrap_particle(std::size_t p, std::size_t begin = 0, std::size_t end = npos);

    // ── Worm ──
    // An open path lacks the g >= 0 beads that follow the head bead along its
    // worldline (in-array links, then pi), and the g + 1 links between head and
    // tail; the tail is the bead after that gap.  The canonical worm has g = 0:
    // one cut link, tail = bead_after(head).  A gap of g <= M - 1 beads visits
    // each slice at most once, so at most one bead is missing per slice: in the
    // head's row after the head slice, then (when the gap crosses the closing
    // link) in row pi(head) before the tail slice.  Missing beads keep their
    // last coordinates as placeholders.  pi stays a bijection, so
    // permutation_sign() is the sign of the virtually closed path.  Only the
    // worm moves and the driver read or change this state.
    bool is_open() const noexcept { return open_; }
    Bead head() const noexcept { return head_; }
    std::size_t gap() const noexcept { return gap_; }
    Bead tail() const noexcept {
        const std::size_t s = head_.slice + gap_ + 1, M = num_slices();
        return s < M ? Bead{head_.particle, s} : Bead{permutation_[head_.particle], s - M};
    }
    // The row whose bead at slice k is missing, or npos.  O(1).
    std::size_t missing_row(std::size_t k) const noexcept {
        if (!open_ || gap_ == 0) return npos;
        const std::size_t M = num_slices();
        const std::size_t offset = (k + M - head_.slice - 1) % M;  // gap index of slice k
        if (offset >= gap_) return npos;
        return head_.slice + 1 + offset < M ? head_.particle : permutation_[head_.particle];
    }
    // Cut the link leaving `head`; throws invalid_argument if the path is
    // already open or the bead is outside the path.
    void open_at(Bead head);
    // Open with a gap: the g beads after `head` go missing (g <= M - 1).
    void open_with_gap(Bead head, std::size_t g);
    // Move the cut of an open path to the link leaving `head`.
    void set_head(Bead head);
    // Change the gap of an open path (g <= M - 1), keeping the head.
    void set_gap(std::size_t g);
    void close() noexcept { open_ = false; gap_ = 0; }
    // Exchange the successors of beads (p, k) and (s, k) of one exchange class
    // (p != s): the bead at (p, k) then continues into the old successor of
    // (s, k) and vice versa (the reconnection of a worm swap).  Every in-array
    // link stays the identity: the two rows' coordinates are swapped on the
    // shorter side of slice k — slices k+1..M-1 with pi <- pi o (p s), or
    // slices 0..k with pi <- (p s) o pi — and those slices' tables refreshed,
    // O(min(k+1, M-1-k) * np * dim).  Returns the row now holding the bead
    // that was at (p, k): p, or s when slices 0..k were swapped.
    std::size_t exchange_successors(std::size_t p, std::size_t s, std::size_t k);
private:
    // Recompute the inverse, cycle leaders, sign and identity flag of pi.
    void refresh_cycles();

    double tau_;
    std::vector<std::size_t> next_slice_, prev_slice_;
    std::vector<std::size_t> permutation_, inverse_; // [np]
    std::vector<std::size_t> class_id_;              // [capacity]
    std::vector<char> fermionic_;                    // [capacity]
    std::vector<std::size_t> cycle_leaders_;         // [n_cycles], reserved to capacity
    // [class][active member], each reserved to its capacity members.
    std::vector<std::vector<std::size_t>> exchangeable_;
    std::vector<std::size_t> exchangeable_of_;       // [capacity], entry or npos
    std::vector<char> visited_;                      // [capacity], cycle scratch
    double sign_ = 1.0;
    bool identity_ = true;
    std::size_t largest_class_ = 1;
    std::size_t active_ = 0;     // rows [0, active_) are particles
    bool open_ = false;          // worm: the links after head_ are cut
    std::size_t gap_ = 0;        // worm: missing beads after head_
    Bead head_{0, 0};
};
} // namespace alea
