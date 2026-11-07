// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Composite Euclidean action S[R] = sum_c S_c[R] over independently cloned
// components (primitive kinetic springs, primitive or pair-product potential,
// ...).  Moves sample exp(-S) through component deltas; an energy estimator
// sums each component's contribution, e.g. the thermodynamic
// -d_beta ln(normalization) + <d_beta S> at fixed M (see primitive_action.h).
// YAML interpretation belongs to action_builders.h.
#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include "pimc/action_component.h"
#include "pimc/path_set.h"
#include "pimc/periodic_virial.h"
#include "hamiltonian/hamiltonian.h"
#include "population/particle_group.h"

namespace alea {
class Action {
public:
    Action() = default;
    // Compatibility construction: install the two independent primitive pieces.
    Action(std::span<const ParticleGroup> groups, const Hamiltonian& ham,
           std::size_t dim, bool kinetic = true, bool potential = true);
    Action(const Action& other);
    Action(Action&&) noexcept = default;
    Action& operator=(const Action&) = delete;
    Action& operator=(Action&&) noexcept = default;
    void add_component(std::unique_ptr<ActionComponent> component);
    std::span<const std::unique_ptr<ActionComponent>> components() const {
        return components_;
    }
    void prepare(std::size_t M);
    void validate_sampling(const SimulationCell&, double tau) const;
    void prepare_time_step(double tau);
    // Also reserves staging scratch for exchange moves: a whole permutation
    // cycle (L*M beads) or a joint window of two classes (2 L (n-1) beads),
    // both bounded by 2 * largest_class * M.  Inline: every bead delta calls
    // it, so the common no-growth case must be cheap.
    void prepare(const PathSet& path) {
        prepare(path.num_slices());
        if (window_beads_.size() < 2 * path.largest_exchange_class() * path.num_slices() ||
            window_saved_.size() < window_beads_.size() * dim_)
            reserve_cycle_window(path);
    }
    [[nodiscard]] Action clone() const;
    [[nodiscard]] bool supports_free_bridge() const noexcept;
    // Every component's per-bead deltas are slice-local (no link coupling):
    // a worm's cut link then changes no potential term.
    [[nodiscard]] bool slice_local_deltas() const noexcept;
    [[nodiscard]] std::string potential_approximation_notes() const;
    // Every component's nodal-pocket particle ranges, in component order.
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> restricted_blocks() const;
    // Same-particle spring between slices k1 and k2 of particle p (ignores
    // the permutation: an explicit pair of beads, as in the legacy tests).
    double spring_action(const PathSet& path, std::size_t p, std::size_t k1,
                         std::size_t k2) const {
        return spring_action(path, p, k1, p, k2);
    }
    // Spring between beads (p1, k1) and (p2, k2) of one worldline.
    double spring_action(const PathSet& path, std::size_t p1, std::size_t k1,
                         std::size_t p2, std::size_t k2) const;
    // The link leaving bead (p, k): to (p, k+1), or to (pi(p), 0) at k = M-1.
    double link_action(const PathSet& path, std::size_t p, std::size_t k) const {
        const auto after = path.bead_after(p, k);
        return spring_action(path, p, k, after.particle, after.slice);
    }
    double spring_action_all(const PathSet& path, std::size_t k1,
                             std::size_t k2) const;
    double potential_action_slice(const PathSet& path, std::size_t k);
    // Change tau*V without committing the proposal; generic components restore
    // positions and tables exactly, including when evaluation throws.
    // Requires current distance tables; disposable pending buffers may change.
    double delta_potential_particle(PathSet& path, std::size_t p, std::size_t k,
                                    std::span<const double> proposed);
    // Staged worldline window: `count` consecutive beads starting at (p, first)
    // and following the worldline, which continues on pi(p) after slice M-1;
    // positions [count*dim].  count <= M visits each slice once; a whole
    // permutation cycle (count = L*M, from slice 0) visits each slice L times.
    // Coupled link actions observe the entire proposal, never a sum of
    // isolated beads.
    double delta_potential_window(PathSet&, std::size_t p, std::size_t first,
                                  std::span<const double> positions);
    // Action change when the listed beads ([n], each at most once, possibly
    // several particles on one slice) move to `positions` ([n*dim]).  Slice-
    // local components telescope bead by bead; link-coupled components take
    // full before/after differences.  Coordinates and tables are restored
    // before returning or throwing.  n <= 2 * largest_exchange_class() * M.
    double delta_potential_beads(PathSet&, std::span<const PathSet::Bead> beads,
                                 std::span<const double> positions);
    double total_action(PathSet& path);
    double total_potential(PathSet& path);
    using ThermoMeasure = ActionThermoMeasure;
    [[nodiscard]] ThermoMeasure measure_thermo(PathSet& path);
    void validate_open_centroid(const SimulationCell& cell, std::size_t np,
                                std::size_t dim) const;
    void prepare_open_centroid(const PathSet& path);
    [[nodiscard]] ThermoMeasure measure_open_centroid(PathSet& path);
    void validate_periodic_virial(const SimulationCell&, std::size_t np, std::size_t dim) const;
    void prepare_periodic_virial(const PathSet&);
    [[nodiscard]] ThermoMeasure measure_periodic_virial(PathSet&);
    double thermodynamic_energy(PathSet& path) { return measure_thermo(path).energy; }
    [[nodiscard]] double mass(std::size_t p, std::size_t d) const;
    std::size_t dim() const noexcept { return dim_; }
    std::size_t np() const noexcept { return np_; }
private:
    std::vector<std::unique_ptr<ActionComponent>> components_;
    std::size_t dim_ = 0, np_ = 0, prepared_M_ = 0;
    std::vector<double> centroids_; // [np*dim], arithmetic centroid per particle/axis
    std::size_t centroid_M_ = 0;
    PeriodicVirialField periodic_field_;
    std::vector<double> window_saved_; // [L_max*M*dim], proposal/restore scratch
    // Staged beads [L_max*M]: particle and slice of each proposed bead.
    std::vector<PathSet::Bead> window_beads_;
    // Grow the window scratch to the largest permutation cycle of `path`.
    void reserve_cycle_window(const PathSet& path);
    // delta_potential_particle without prepare(): the path is prepared.
    double delta_particle_prepared(PathSet&, std::size_t p, std::size_t k,
                                   std::span<const double> proposed);
    // Exact telescoping difference for staged beads (each bead at most once);
    // the caller prepared the path and saved nothing yet.
    double delta_beads_prepared(PathSet&, std::span<const PathSet::Bead> beads,
                                std::span<const double> proposed);
    // Exact difference for a window that visits a slice more than once.
    double delta_cycle_window(PathSet&, std::size_t p, std::size_t first,
                              std::span<const double> positions);
};
} // namespace alea
