// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Action components own their physics, thermal derivatives and prepared scratch.
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace alea {
class PathSet;
class SimulationCell;

struct ActionThermoMeasure {
    double kinetic = 0.0;
    double potential = 0.0;
    double energy = 0.0;
};

class ActionComponent {
public:
    virtual ~ActionComponent() = default;
    // Allocate setup scratch once before the sampling loop.
    virtual void prepare(std::size_t) {}
    virtual void validate_sampling(const SimulationCell&, double, std::size_t) const {}
    virtual void prepare_time_step(double) {}
    // Opt in only when per-slice deltas add without cross-slice coupling.
    virtual bool slice_local_deltas() const noexcept { return false; }
    virtual double total_action(const PathSet& path) = 0;
    // Energy pieces at fixed M: -d_beta log(normalization) and d_beta S.
    // Components can expose these independently, including nonprimitive actions.
    virtual double normalization_energy(const PathSet&) const { return 0.0; }
    virtual double beta_derivative(const PathSet&) { return 0.0; }
    virtual ActionThermoMeasure measure_thermo(const PathSet& path) {
        return {0.0, 0.0, normalization_energy(path) + beta_derivative(path)};
    }
    virtual void validate_open_centroid(std::size_t, std::size_t) const {
        throw std::invalid_argument("centroid_virial: action component has no open centroid contribution");
    }
    virtual bool confines_open_coordinate(std::size_t, std::size_t, std::size_t) const noexcept {
        return false;
    }
    virtual void prepare_open_centroid(const PathSet&) {}
    // Centroids are [np*dim]; the contribution owns its prepared measurement scratch.
    virtual ActionThermoMeasure measure_open_centroid(const PathSet&, std::span<const double>) {
        throw std::logic_error("centroid_virial: action component is unsupported");
    }
    virtual void validate_periodic_virial(const SimulationCell&, std::size_t, std::size_t) const {
        throw std::invalid_argument("periodic_virial: action component has no periodic contribution");
    }
    virtual void prepare_periodic_virial(const PathSet&) {}
    // Field layout [np*dim*M]. Only the kinetic contributor consumes div F.
    virtual ActionThermoMeasure measure_periodic_virial(const PathSet&, std::span<const double>, double) {
        throw std::logic_error("periodic_virial: action component is unsupported");
    }
    // Kinetic link between beads (p1, k1) and (p2, k2) of one worldline; the
    // two particles share an exchange class, hence a mass tensor.
    virtual double spring_action(const PathSet&, std::size_t, std::size_t,
                                 std::size_t, std::size_t) const { return 0.0; }
    virtual double potential_action_slice(const PathSet&, std::size_t) { return 0.0; }
    virtual double total_potential(const PathSet&) { return 0.0; }
    // One-bead action change without committing coordinates or tables.
    // Default: evaluate/restore total_action, including on exception; an explicit
    // slice_local_deltas capability permits the cheaper potential_action_slice.
    // An optimized override must preserve the same state invariant.
    virtual double delta_potential_particle(PathSet&, std::size_t, std::size_t,
                                            std::span<const double>);
    // Action change when particle p's beads on the cyclic window
    // [first, first+count) move to `positions` ([count*dim]), for a component
    // that couples slices through links.  Opt in with link_local_window_deltas
    // and sum only the links touching the window — O(N * count) instead of a
    // before/after evaluation of the whole path.  Read-only on the path.
    // Without the opt-in, Action falls back to the exact full evaluation.
    virtual bool link_local_window_deltas() const noexcept { return false; }
    virtual double delta_potential_window(const PathSet&, std::size_t,
                                          std::size_t, std::span<const double>) {
        throw std::logic_error("action component has no incremental window delta");
    }
    // Arbitrary components must opt in: a free Levy bridge only cancels the
    // primitive Gaussian spring. Coupled corrections must participate in the
    // complete staged-window difference used by bisection/centroid moves.
    virtual bool supports_free_bridge() const noexcept { return false; }
    virtual bool provides_free_kinetic() const noexcept { return false; }
    virtual std::string potential_approximation_notes() const { return {}; }
    // Particle ranges [first, last) whose paths this component confines to one
    // nodal pocket (fixed-node PIMC, fixed_node_restriction.h).  The driver
    // matches them against the fermion exchange classes.  Default: none.
    virtual std::vector<std::pair<std::size_t, std::size_t>> restricted_blocks() const {
        return {};
    }
    virtual double mass(std::size_t, std::size_t) const {
        throw std::logic_error("action component has no free kinetic mass");
    }
    virtual std::unique_ptr<ActionComponent> clone() const = 0;
};
} // namespace alea
