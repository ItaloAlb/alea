// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Composite primitive action: component-owned normalization and thermal pieces.
#include "pimc/action.h"
#include "pimc/primitive_action.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace alea {
double
ActionComponent::delta_potential_particle(PathSet& path, std::size_t p,
                                          std::size_t k,
                                          std::span<const double> proposed) {
    const auto dim = path.dim();
    if (p >= path.n_particles() || k >= path.num_slices() ||
        proposed.size() != dim)
        throw std::invalid_argument(
            "action delta: invalid particle, slice or position");
    double saved[SimulationCell::kMaxDim], candidate[SimulationCell::kMaxDim];
    for (std::size_t d = 0; d < dim; ++d) {
        saved[d] = path.pos(p, d, k);
        candidate[d] = proposed[d];
    }
    path.cell().wrap(candidate);
    // Safe default for a link-coupled correction: moving one bead changes both
    // incident links. Slice-local components explicitly opt into the smaller
    // sum.
    const auto evaluate = [&]() {
        return slice_local_deltas() ? potential_action_slice(path, k)
                                    : total_action(path);
    };
    const double old_value = evaluate();
    struct Restore {
        PathSet& path;
        std::size_t p, k;
        const double* saved;
        ~Restore() {
            for (std::size_t d = 0; d < path.dim(); ++d)
                path.pos(p, d, k) = saved[d];
            path.update_particle_tables(p, k, k + 1);
        }
    } restore{path, p, k, saved};
    for (std::size_t d = 0; d < dim; ++d)
        path.pos(p, d, k) = candidate[d];
    path.update_particle_tables(p, k, k + 1);
    return evaluate() - old_value;
}

PrimitiveKineticAction::PrimitiveKineticAction(
    std::span<const ParticleGroup> groups, std::size_t dim)
    : dim_(dim) {
    if (dim == 0 || dim > SimulationCell::kMaxDim)
        throw std::invalid_argument(
            "primitive action: invalid spatial dimension");
    for (const auto& g : groups) {
        if (g.last < g.first || (g.mass.size() != 1 && g.mass.size() != dim))
            throw std::invalid_argument(
                "primitive action: mass must be scalar or dimension-sized");
        for (double m : g.mass)
            if (!std::isfinite(m) || m <= 0.0)
                throw std::invalid_argument(
                    "primitive action: masses must be finite and positive");
        np_ = std::max(np_, g.last);
    }
    // A particle must have exactly one mass tensor. Overlapping groups would
    // silently let their order choose the kinetic action for that particle.
    for (std::size_t a = 0; a < groups.size(); ++a)
        for (std::size_t b = a + 1; b < groups.size(); ++b)
            if (std::max(groups[a].first, groups[b].first) <
                std::min(groups[a].last, groups[b].last))
                throw std::invalid_argument(
                    "primitive action: overlapping particle groups");
    if (np_ > std::numeric_limits<std::size_t>::max() / sizeof(double) / dim_)
        throw std::invalid_argument(
            "primitive action: mass storage dimensions overflow");
    mass_.assign(np_ * dim, 0.0);
    for (const auto& g : groups)
        for (std::size_t p = g.first; p < g.last; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                mass_[p * dim + d] = g.mass[g.mass.size() == 1 ? 0 : d];
    for (double m : mass_)
        if (m == 0.0)
            throw std::invalid_argument(
                "primitive action: uncovered particle mass");
}

// One minimum-image Gaussian link; its normalization derivative is below.
// The beads may belong to two particles of one exchange class (the closing
// link of a permuted path), which share the mass tensor of p1.
double PrimitiveKineticAction::spring_action(const PathSet& path, std::size_t p1,
                                             std::size_t k1, std::size_t p2,
                                             std::size_t k2) const {
    double delta[SimulationCell::kMaxDim];
    for (std::size_t d = 0; d < dim_; ++d)
        delta[d] = path.pos(p2, d, k2) - path.pos(p1, d, k1);
    path.cell().min_image(delta);
    double result = 0.0;
    for (std::size_t d = 0; d < dim_; ++d)
        result += mass(p1, d) * delta[d] * delta[d] / (2.0 * path.tau());
    return result;
}

// Sum over every link (p, k) -> bead_after(p, k); the closing link of p goes
// to (pi(p), 0).  With pi = identity this is the same sum, in the same order,
// as the ring (p, k) -> (p, k+1 mod M).
double PrimitiveKineticAction::total_action(const PathSet& path) {
    const std::size_t M = path.num_slices();
    double result = 0.0;
    for (std::size_t k = 0; k + 1 < M; ++k)
        for (std::size_t p = 0; p < np_; ++p)
            result += spring_action(path, p, k, p, k + 1);
    for (std::size_t p = 0; p < np_; ++p)  // closing links, last in the sum
        result += spring_action(path, p, M - 1, path.next_particle(p), 0);
    return result;
}

// -d_beta log normalization + d_beta S, with M held fixed.
double PrimitiveKineticAction::normalization_energy(const PathSet& path) const {
    return static_cast<double>(dim_ * np_) / (2.0 * path.tau());
}
double PrimitiveKineticAction::beta_derivative(const PathSet& path) {
    return -total_action(path) / path.beta();
}
ActionThermoMeasure
PrimitiveKineticAction::measure_thermo(const PathSet& path) {
    const double kinetic = normalization_energy(path) + beta_derivative(path);
    return {kinetic, 0.0, kinetic};
}
std::unique_ptr<ActionComponent> PrimitiveKineticAction::clone() const {
    return std::make_unique<PrimitiveKineticAction>(*this);
}

void PrimitiveKineticAction::validate_open_centroid(std::size_t np,
                                                    std::size_t dim) const {
    if (np != np_ || dim != dim_)
        throw std::invalid_argument(
            "centroid_virial: kinetic dimensions do not match path");
}
ActionThermoMeasure
PrimitiveKineticAction::measure_open_centroid(const PathSet& path,
                                              std::span<const double>) {
    // One free centroid per closed loop: d/(2 beta) each (N loops for the
    // identity permutation, fewer when particles share a cycle).
    const double kinetic = static_cast<double>(path.cycle_leaders().size()) *
                           static_cast<double>(dim_) / (2.0 * path.beta());
    return {kinetic, 0.0, kinetic};
}

void PrimitiveKineticAction::validate_periodic_virial(
    const SimulationCell& cell, std::size_t np, std::size_t dim) const {
    if (np != np_ || dim != dim_)
        throw std::invalid_argument(
            "periodic_virial: kinetic dimensions do not match path");
    // Euclidean nearest-image faces preserve isotropic spring values. A skew
    // cell with anisotropic masses needs a different mass-metric image search.
    if (!cell.orthorhombic())
        for (std::size_t p = 0; p < np_; ++p)
            for (std::size_t d = 1; d < dim_; ++d)
                if (mass(p, d) != mass(p, 0))
                    throw std::invalid_argument(
                        "periodic_virial: skew cell requires isotropic masses "
                        "for a continuous minimum-image action");
}
ActionThermoMeasure PrimitiveKineticAction::measure_periodic_virial(
    const PathSet& path, std::span<const double> field, double divergence) {
    const auto M = path.num_slices();
    double spring = 0.0, contraction = 0.0;
    for (std::size_t p = 0; p < np_; ++p)
        for (std::size_t k = 0; k < M; ++k) {
            const auto next = path.next(k);
            double delta[SimulationCell::kMaxDim];
            for (std::size_t d = 0; d < dim_; ++d)
                delta[d] = path.pos(p, d, next) - path.pos(p, d, k);
            path.cell().min_image(delta);
            for (std::size_t d = 0; d < dim_; ++d) {
                const auto base = (p * dim_ + d) * M;
                spring += mass(p, d) * delta[d] * delta[d] / (2.0 * path.tau());
                contraction += mass(p, d) * delta[d] *
                               (field[base + next] - field[base + k]) /
                               path.tau();
            }
        }
    const double kinetic = normalization_energy(path) - spring / path.beta() +
                           (contraction - divergence) / (2.0 * path.beta());
    return {kinetic, 0.0, kinetic};
}

PrimitivePotentialAction::PrimitivePotentialAction(
    std::span<const ParticleGroup> groups, const Hamiltonian& ham)
    : ham_(ham.clone()) {
    // Test actual nonempty group pairs; a one-particle self-group has no pair.
    // Attractive 1/r yields exp(+tau*a/r), nonintegrable for every positive
    // tau. Attractive RK logs are conservatively unsupported: a pair requires
    // tau*|qa qb|/(epsilon*r0)<d, and clusters impose stronger conditions.
    for (const auto& component : ham_.components()) {
        if (!component->is_potential())
            continue;
        for (std::size_t a = 0; a < groups.size(); ++a)
            for (std::size_t b = a; b < groups.size(); ++b) {
                if (groups[a].size() == 0 || groups[b].size() == 0 ||
                    (a == b && groups[a].size() < 2))
                    continue;
                if (component->coulomb_cusp_residue(groups[a], groups[b]) < 0.0)
                    throw std::invalid_argument(
                        "primitive action: attractive 1/r pole in " +
                        component->name() +
                        "; use a finite layer separation or pair action");
                if (!component->primitive_pair_admissible(groups[a], groups[b]))
                    throw std::invalid_argument(
                        "primitive action: unsupported attractive singularity "
                        "in " +
                        component->name() +
                        "; use a finite layer separation or pair action");
            }
    }
}
PrimitivePotentialAction::PrimitivePotentialAction(
    const PrimitivePotentialAction& other)
    : ham_(other.ham_.clone()), scratch_(other.scratch_) {}
void PrimitivePotentialAction::prepare(std::size_t M) {
    if (scratch_.size() != M)
        scratch_.resize(M);
}
double PrimitivePotentialAction::potential_action_slice(const PathSet& path,
                                                        std::size_t k) {
    prepare(path.num_slices());
    ham_.evaluate_potential(path, scratch_, k, k + 1);
    return path.tau() * scratch_[k];
}
double PrimitivePotentialAction::total_potential(const PathSet& path) {
    prepare(path.num_slices());
    ham_.evaluate_potential(path, scratch_);
    double result = 0.0;
    for (double v : scratch_)
        result += v;
    return result;
}
double PrimitivePotentialAction::total_action(const PathSet& path) {
    return path.tau() * total_potential(path);
}
double PrimitivePotentialAction::normalization_energy(const PathSet&) const {
    return 0.0;
}
double PrimitivePotentialAction::beta_derivative(const PathSet& path) {
    return total_potential(path) / static_cast<double>(path.num_slices());
}
ActionThermoMeasure
PrimitivePotentialAction::measure_thermo(const PathSet& path) {
    const double potential = beta_derivative(path);
    return {0.0, potential, potential};
}

void PrimitivePotentialAction::validate_open_centroid(std::size_t,
                                                      std::size_t) const {
    for (const auto& component : ham_.components()) {
        if (!component->is_potential())
            continue;
        if (!component->supports_potential_gradient())
            throw std::invalid_argument(
                "centroid_virial: missing potential gradient in " +
                component->name());
        if (!component->supports_open_centroid_virial())
            throw std::invalid_argument(
                "centroid_virial: open identity/accuracy is not admitted for " +
                component->name());
    }
}
bool PrimitivePotentialAction::confines_open_coordinate(
    std::size_t p, std::size_t d, std::size_t dim) const noexcept {
    for (const auto& component : ham_.components())
        if (component->is_potential() &&
            component->confines_open_coordinate(p, d, dim))
            return true;
    return false;
}
void PrimitivePotentialAction::prepare_open_centroid(const PathSet& path) {
    prepare(path.num_slices());
    gradient_.resize(path.positions().size());
    gradient_workspace_ = std::make_unique<HamiltonianGradientWorkspace>(
        ham_.make_potential_gradient_workspace(path));
}
ActionThermoMeasure PrimitivePotentialAction::measure_open_centroid(
    const PathSet& path, std::span<const double> centroids) {
    if (!gradient_workspace_ || gradient_.size() != path.positions().size())
        throw std::logic_error(
            "centroid_virial: potential measurement is not prepared");
    ham_.evaluate_potential(path, scratch_);
    ham_.evaluate_potential_gradient(path, gradient_, *gradient_workspace_);
    const auto M = path.num_slices();
    const auto positions = path.positions(); // [np*dim*M]
    double potential = 0.0, virial = 0.0;
    for (double value : scratch_)
        potential += value;
    for (std::size_t axis = 0; axis < centroids.size(); ++axis)
        for (std::size_t k = 0; k < M; ++k)
            virial += (positions[axis * M + k] - centroids[axis]) *
                      gradient_[axis * M + k];
    potential /= static_cast<double>(M);
    const double kinetic = virial / (2.0 * static_cast<double>(M));
    return {kinetic, potential, kinetic + potential};
}

void PrimitivePotentialAction::validate_periodic_virial(
    const SimulationCell& cell, std::size_t np, std::size_t) const {
    for (const auto& component : ham_.components()) {
        if (!component->is_potential())
            continue;
        if (!component->supports_potential_gradient())
            throw std::invalid_argument(
                "periodic_virial: missing potential gradient in " +
                component->name());
        if (!component->supports_periodic_virial(cell, np))
            throw std::invalid_argument(
                "periodic_virial: periodic identity/accuracy is not admitted "
                "for " +
                component->name());
    }
}
void PrimitivePotentialAction::prepare_periodic_virial(const PathSet& path) {
    prepare_open_centroid(
        path); // Shared gradient scratch, without open admission.
}
std::string PrimitivePotentialAction::potential_approximation_notes() const {
    std::string result;
    for (const auto& component : ham_.components()) {
        if (!component->is_potential())
            continue;
        const auto notes = component->potential_approximation_notes();
        if (notes.empty())
            continue;
        if (!result.empty())
            result += "; ";
        result += component->name() + ": " + notes;
    }
    return result;
}
ActionThermoMeasure PrimitivePotentialAction::measure_periodic_virial(
    const PathSet& path, std::span<const double> field, double) {
    if (!gradient_workspace_ || gradient_.size() != path.positions().size())
        throw std::logic_error(
            "periodic_virial: potential measurement is not prepared");
    ham_.evaluate_potential(path, scratch_);
    ham_.evaluate_potential_gradient(path, gradient_, *gradient_workspace_);
    double potential = 0.0, contraction = 0.0;
    for (double value : scratch_)
        potential += value;
    for (std::size_t i = 0; i < field.size(); ++i)
        contraction += field[i] * gradient_[i];
    potential /= static_cast<double>(path.num_slices());
    const double kinetic =
        contraction / (2.0 * static_cast<double>(path.num_slices()));
    return {kinetic, potential, kinetic + potential};
}
double PrimitivePotentialAction::delta_potential_particle(
    PathSet& path, std::size_t p, std::size_t k,
    std::span<const double> proposed) {
    prepare(path.num_slices());
    return path.tau() *
           ham_.delta_potential_particle(path, p, k, proposed, scratch_);
}
std::unique_ptr<ActionComponent> PrimitivePotentialAction::clone() const {
    return std::make_unique<PrimitivePotentialAction>(*this);
}

Action::Action(std::span<const ParticleGroup> groups, const Hamiltonian& ham,
               std::size_t dim, bool kinetic, bool potential)
    : dim_(dim) {
    for (const auto& g : groups)
        np_ = std::max(np_, g.last);
    if (kinetic)
        add_component(std::make_unique<PrimitiveKineticAction>(groups, dim));
    if (potential)
        add_component(std::make_unique<PrimitivePotentialAction>(groups, ham));
}
Action::Action(const Action& other)
    : dim_(other.dim_), np_(other.np_), prepared_M_(other.prepared_M_),
      window_saved_(other.window_saved_), window_beads_(other.window_beads_) {
    components_.reserve(other.components_.size());
    for (const auto& component : other.components_)
        components_.push_back(component->clone());
}
void Action::add_component(std::unique_ptr<ActionComponent> component) {
    if (!component)
        throw std::invalid_argument("action: null component");
    if (prepared_M_ != 0)
        component->prepare(prepared_M_);
    components_.push_back(std::move(component));
    centroid_M_ = 0; // Composition changes invalidate all centroid preparation.
    periodic_field_.invalidate();
}
void Action::prepare(std::size_t M) {
    if (M == prepared_M_)
        return;
    if (dim_ && M > std::numeric_limits<std::size_t>::max() / dim_)
        throw std::length_error("action: staged window dimensions overflow");
    centroid_M_ = 0;
    periodic_field_.invalidate();
    for (const auto& component : components_)
        component->prepare(M);
    window_saved_.resize(M * dim_);
    prepared_M_ = M;
}
void Action::validate_sampling(const SimulationCell& cell, double tau) const {
    for (const auto& component : components_)
        component->validate_sampling(cell, tau, dim_);
}
void Action::prepare_time_step(double tau) {
    for (const auto& component : components_)
        component->prepare_time_step(tau);
}
void Action::reserve_cycle_window(const PathSet& path) {
    // A centroid move of a whole permutation cycle stages L*M beads and a
    // joint exchange move 2 L (n-1) beads of two classes: both fit in
    // 2 * largest_class * M.  Grow each buffer on its own: prepare(M) resets
    // window_saved_ to M*dim when M changes, while window_beads_ may still be
    // large from an earlier M.
    const auto beads = 2 * path.largest_exchange_class() * path.num_slices();
    if (window_beads_.size() < beads) window_beads_.resize(beads);
    if (window_saved_.size() < window_beads_.size() * dim_)
        window_saved_.resize(window_beads_.size() * dim_);
}
Action Action::clone() const { return Action(*this); }
std::string Action::potential_approximation_notes() const {
    std::string result;
    for (const auto& component : components_) {
        const auto notes = component->potential_approximation_notes();
        if (notes.empty())
            continue;
        if (!result.empty())
            result += "; ";
        result += notes;
    }
    return result;
}
std::vector<std::pair<std::size_t, std::size_t>> Action::restricted_blocks() const {
    std::vector<std::pair<std::size_t, std::size_t>> blocks;
    for (const auto& component : components_)
        for (const auto& block : component->restricted_blocks())
            blocks.push_back(block);
    return blocks;
}
bool Action::slice_local_deltas() const noexcept {
    return std::all_of(components_.begin(), components_.end(),
                       [](const auto& c) { return c->slice_local_deltas(); });
}
bool Action::supports_free_bridge() const noexcept {
    std::size_t kinetic_components = 0;
    for (const auto& component : components_) {
        if (!component->supports_free_bridge())
            return false;
        kinetic_components += component->provides_free_kinetic();
    }
    return kinetic_components == 1;
}
double Action::mass(std::size_t p, std::size_t d) const {
    for (const auto& component : components_)
        if (component->provides_free_kinetic())
            return component->mass(p, d);
    throw std::logic_error("action: no free kinetic component");
}
double Action::spring_action(const PathSet& path, std::size_t p1, std::size_t k1,
                             std::size_t p2, std::size_t k2) const {
    double result = 0.0;
    for (const auto& component : components_)
        result += component->spring_action(path, p1, k1, p2, k2);
    return result;
}
double Action::spring_action_all(const PathSet& path, std::size_t k1,
                                 std::size_t k2) const {
    double result = 0.0;
    for (std::size_t p = 0; p < path.n_particles(); ++p)
        result += spring_action(path, p, k1, k2);
    return result;
}
double Action::potential_action_slice(const PathSet& path, std::size_t k) {
    prepare(path);
    double result = 0.0;
    for (const auto& component : components_)
        result += component->potential_action_slice(path, k);
    return result;
}
double Action::delta_potential_particle(PathSet& path, std::size_t p,
                                        std::size_t k,
                                        std::span<const double> proposed) {
    prepare(path);
    return delta_particle_prepared(path, p, k, proposed);
}
// Per-bead sum for callers that already prepared this path (window loops):
// a bead delta can cost only tens of ns, so no per-bead bookkeeping here.
double Action::delta_particle_prepared(PathSet& path, std::size_t p,
                                       std::size_t k,
                                       std::span<const double> proposed) {
    double result = 0.0;
    for (const auto& component : components_)
        result += component->delta_potential_particle(path, p, k, proposed);
    return result;
}
namespace {
// Restores the first `committed` staged beads (positions and tables),
// including when an evaluation throws.  saved [n*dim], beads [n].
struct BeadRestore {
    PathSet& path;
    const PathSet::Bead* beads;
    const std::vector<double>& saved;
    std::size_t committed = 0;
    ~BeadRestore() {
        const auto dim = path.dim();
        for (std::size_t i = 0; i < committed; ++i) {
            const auto [p, k] = beads[i];
            for (std::size_t d = 0; d < dim; ++d)
                path.pos(p, d, k) = saved[i * dim + d];
            path.update_particle_tables(p, k, k + 1);
        }
    }
};
} // namespace

double Action::delta_potential_window(PathSet& path, std::size_t p,
                                      std::size_t first,
                                      std::span<const double> proposed) {
    prepare(path);
    const auto M = path.num_slices(), dim = path.dim(),
               count = proposed.size() / dim;
    if (dim != dim_ || p >= np_ || first >= M || proposed.size() % dim)
        throw std::invalid_argument("action window: invalid staged dimensions");
    if (count > M)
        return delta_cycle_window(path, p, first, proposed);
    // Bead i sits at slice (first+i) mod M on p, or on pi(p) once the
    // worldline has passed slice M-1.  With count <= M every slice is
    // visited at most once, so independent per-bead deltas are exact.
    const std::size_t after = path.next_particle(p);
    const auto owner = [&](std::size_t i) { return first + i < M ? p : after; };
    if (std::all_of(components_.begin(), components_.end(),
                    [](const auto& c) { return c->slice_local_deltas(); })) {
        // Preserve the primitive summation order and its existing RNG path.
        double local = 0;
        for (std::size_t i = 0; i < count; ++i) {
            local += delta_particle_prepared(path, owner(i), (first + i) % M,
                                              proposed.subspan(i * dim, dim));
            if (!std::isfinite(local))
                return local;
        }
        return local;
    }
    // A link-local component sums one particle's links against its partners'
    // own beads; a permuted closing link (M-1 -> 0) touched by the window
    // joins other particles' beads, so it takes the generic route instead.
    const bool closing_link_permuted =
        !path.identity_permutation() && (first == 0 || first + count >= M);
    double result = 0;
    for (const auto& component : components_) {
        if (component->slice_local_deltas()) {
            for (std::size_t i = 0; i < count; ++i)
                result += component->delta_potential_particle(
                    path, owner(i), (first + i) % M,
                    proposed.subspan(i * dim, dim));
        } else if (component->link_local_window_deltas() &&
                   !closing_link_permuted) {
            // Only links touching the window change; the component sums them
            // without mutating the path.
            result +=
                component->delta_potential_window(path, p, first, proposed);
        } else {
            // Generic coupled component: exact before/after evaluation of the
            // whole path, with the accepted coordinates restored on exit.
            const double old = component->total_action(path);
            for (std::size_t i = 0; i < count; ++i) {
                window_beads_[i] = {owner(i), (first + i) % M};
                for (std::size_t d = 0; d < dim; ++d)
                    window_saved_[i * dim + d] =
                        path.pos(window_beads_[i].particle, d, window_beads_[i].slice);
            }
            BeadRestore restore{path, window_beads_.data(), window_saved_};
            for (std::size_t i = 0; i < count; ++i) {
                const auto [q, k] = window_beads_[i];
                for (std::size_t d = 0; d < dim; ++d)
                    path.pos(q, d, k) = proposed[i * dim + d];
                path.update_particle_tables(q, k, k + 1);
                restore.committed = i + 1;
            }
            result += component->total_action(path) - old;
        }
    }
    return result;
}

// A window covering a whole permutation cycle (count = L*M) moves L beads on
// every slice, so isolated per-bead deltas would miss the changed pair terms
// between moved beads.  Slice-local components telescope instead: each bead's
// delta is taken with the previously staged beads already committed, so the
// sum is exactly V(all new) - V(all old).  Link-coupled components are
// evaluated before and after the complete commit.  Every bead and table is
// restored before returning or throwing.
double Action::delta_cycle_window(PathSet& path, std::size_t p,
                                  std::size_t first,
                                  std::span<const double> proposed) {
    const auto M = path.num_slices(), dim = path.dim(),
               count = proposed.size() / dim;
    std::size_t length = 1;
    for (std::size_t q = path.next_particle(p); q != p; q = path.next_particle(q))
        ++length;
    if (count != length * M)
        throw std::invalid_argument(
            "action window: a window longer than M must cover its whole "
            "permutation cycle exactly once");
    for (std::size_t i = 0, owner = p; i < count; ++i) {
        const auto k = (first + i) % M;
        if (i > 0 && k == 0) owner = path.next_particle(owner);
        window_beads_[i] = {owner, k};
    }
    return delta_beads_prepared(
        path, std::span<const PathSet::Bead>(window_beads_.data(), count), proposed);
}

double Action::delta_potential_beads(PathSet& path,
                                     std::span<const PathSet::Bead> beads,
                                     std::span<const double> proposed) {
    prepare(path);
    const auto dim = path.dim();
    if (dim != dim_ || proposed.size() != beads.size() * dim)
        throw std::invalid_argument("action beads: staged positions do not match the beads");
    if (beads.size() > window_beads_.size())
        throw std::invalid_argument(
            "action beads: " + std::to_string(beads.size()) +
            " staged beads exceed the capacity " + std::to_string(window_beads_.size()) +
            " (2 * largest exchange class * M)");
    for (const auto& bead : beads)
        if (bead.particle >= np_ || bead.slice >= path.num_slices())
            throw std::invalid_argument("action beads: bead outside the path");
    return delta_beads_prepared(path, beads, proposed);
}

// Several beads may share a slice (a permutation cycle moves L particles on
// every slice), so isolated per-bead deltas would miss the changed pair terms
// between moved beads.  Slice-local components telescope instead: each bead's
// delta is taken with the previously staged beads already committed, so the
// sum is exactly V(all new) - V(all old).  Link-coupled components are
// evaluated before and after the complete commit.  Every bead and table is
// restored before returning or throwing.
double Action::delta_beads_prepared(PathSet& path,
                                    std::span<const PathSet::Bead> beads,
                                    std::span<const double> proposed) {
    const auto dim = path.dim();
    for (std::size_t i = 0; i < beads.size(); ++i)
        for (std::size_t d = 0; d < dim; ++d)
            window_saved_[i * dim + d] = path.pos(beads[i].particle, d, beads[i].slice);
    double coupled_old = 0.0;
    for (const auto& component : components_)
        if (!component->slice_local_deltas())
            coupled_old += component->total_action(path);
    BeadRestore restore{path, beads.data(), window_saved_};
    double result = 0.0;
    for (std::size_t i = 0; i < beads.size(); ++i) {
        const auto [q, k] = beads[i];
        const auto bead = proposed.subspan(i * dim, dim);
        for (const auto& component : components_)
            if (component->slice_local_deltas())
                result += component->delta_potential_particle(path, q, k, bead);
        if (!std::isfinite(result))
            return result;
        for (std::size_t d = 0; d < dim; ++d)
            path.pos(q, d, k) = bead[d];
        path.update_particle_tables(q, k, k + 1);
        restore.committed = i + 1;
    }
    double coupled_new = 0.0;
    for (const auto& component : components_)
        if (!component->slice_local_deltas())
            coupled_new += component->total_action(path);
    return result + (coupled_new - coupled_old);
}
double Action::total_action(PathSet& path) {
    prepare(path);
    path.rebuild_tables();
    double result = 0.0;
    for (const auto& component : components_)
        result += component->total_action(path);
    return result;
}
double Action::total_potential(PathSet& path) {
    prepare(path);
    path.rebuild_tables();
    double result = 0.0;
    for (const auto& component : components_)
        result += component->total_potential(path);
    return result;
}
Action::ThermoMeasure Action::measure_thermo(PathSet& path) {
    prepare(path);
    path.rebuild_tables();
    ThermoMeasure result;
    for (const auto& component : components_) {
        const auto contribution = component->measure_thermo(path);
        result.kinetic += contribution.kinetic;
        result.potential += contribution.potential;
        result.energy += contribution.energy;
    }
    return result;
}

void Action::validate_open_centroid(const SimulationCell& cell, std::size_t np,
                                    std::size_t dim) const {
    if (cell.is_periodic())
        throw std::invalid_argument(
            "centroid_virial: periodic minimum-image boundary contributions "
            "are not implemented");
    if (np != np_ || dim != dim_ || np == 0 || dim == 0)
        throw std::invalid_argument(
            "centroid_virial: action dimensions do not match path");
    std::size_t kinetic_count = 0;
    for (const auto& component : components_) {
        component->validate_open_centroid(np, dim);
        kinetic_count += component->provides_free_kinetic();
    }
    if (kinetic_count != 1)
        throw std::invalid_argument("centroid_virial: requires exactly one "
                                    "primitive kinetic contribution");
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d) {
            const bool confined = std::any_of(
                components_.begin(), components_.end(),
                [&](const auto& component) {
                    return component->confines_open_coordinate(p, d, dim);
                });
            if (!confined)
                throw std::invalid_argument(
                    "centroid_virial: requires absolute open confinement for "
                    "particle " +
                    std::to_string(p) + ", axis " + std::to_string(d));
        }
}
void Action::prepare_open_centroid(const PathSet& path) {
    validate_open_centroid(path.cell(), path.n_particles(), path.dim());
    prepare(path);
    centroids_.resize(np_ * dim_);
    for (const auto& component : components_)
        component->prepare_open_centroid(path);
    centroid_M_ = path.num_slices();
}

// E_cv=Nd/(2 beta)+mean(V)+(1/(2M))*sum (r_pk-centroid_p).grad_p V.
// Yamamoto, J. Chem. Phys. 123, 104101 (2005), equations 6-10: scale
// non-centroid fluctuations, keeping M fixed.
// With exchange every closed loop is a permutation cycle of L particles, one
// ring of L*M beads: scaling each loop's deviations from ITS centroid by
// sqrt(beta) leaves the springs invariant and gives beta^{-d/2} per loop from
// the normalization and Jacobian, so
//   E_cv = d n_loops / (2 beta) + <V> + (1/(2M)) sum_{p,k} (r_pk - c_loop(p)).grad_p V,
// with c_loop the centroid of all L*M beads of p's cycle (n_loops = N and
// c_loop = the ring centroid when pi is the identity). Open confinement eliminates
// boundary flux; fixed anisotropic masses do not change the degree count.
// K_cv=E_cv-V_cv. This measurement never changes coordinates or consumes RNG.
Action::ThermoMeasure Action::measure_open_centroid(PathSet& path) {
    if (path.cell().is_periodic() || path.n_particles() != np_ ||
        path.dim() != dim_ || centroid_M_ != path.num_slices())
        throw std::logic_error(
            "centroid_virial: measurement is not prepared for this open path");
    path.rebuild_tables();
    const auto M = path.num_slices();
    if (path.identity_permutation()) {
        for (std::size_t axis = 0; axis < centroids_.size(); ++axis) {
            // Anchor the sum to bead zero to avoid summing a large common offset.
            const double origin = path.positions()[axis * M];
            double sum = 0.0;
            for (std::size_t k = 0; k < M; ++k)
                sum += path.positions()[axis * M + k] - origin;
            centroids_[axis] = origin + sum / static_cast<double>(M);
        }
    } else {
        // Cycle centroids over all L*M beads (open cell: worldlines are
        // continuous across the permuted closing links), shared by members.
        for (const std::size_t leader : path.cycle_leaders())
            for (std::size_t d = 0; d < dim_; ++d) {
                const double origin = path.pos(leader, d, 0);
                double sum = 0.0;
                std::size_t length = 0, p = leader;
                do {
                    for (std::size_t k = 0; k < M; ++k) sum += path.pos(p, d, k) - origin;
                    ++length;
                    p = path.next_particle(p);
                } while (p != leader);
                const double centroid =
                    origin + sum / static_cast<double>(length * M);
                p = leader;
                do {
                    centroids_[p * dim_ + d] = centroid;
                    p = path.next_particle(p);
                } while (p != leader);
            }
    }
    ThermoMeasure result;
    for (const auto& component : components_) {
        const auto contribution =
            component->measure_open_centroid(path, centroids_);
        result.kinetic += contribution.kinetic;
        result.potential += contribution.potential;
        result.energy += contribution.energy;
    }
    return result;
}

void Action::validate_periodic_virial(const SimulationCell& cell,
                                      std::size_t np, std::size_t dim) const {
    if (!cell.is_periodic() || cell.dim() != dim)
        throw std::invalid_argument(
            "periodic_virial: requires a periodic cell matching the path");
    if (np != np_ || dim != dim_ || np == 0 || dim == 0)
        throw std::invalid_argument(
            "periodic_virial: action dimensions do not match path");
    std::size_t kinetic_count = 0;
    for (const auto& component : components_) {
        component->validate_periodic_virial(cell, np, dim);
        kinetic_count += component->provides_free_kinetic();
    }
    if (kinetic_count != 1)
        throw std::invalid_argument("periodic_virial: requires exactly one "
                                    "primitive kinetic contribution");
}
void Action::prepare_periodic_virial(const PathSet& path) {
    validate_periodic_virial(path.cell(), path.n_particles(), path.dim());
    for (std::size_t p = 0; p < np_; ++p)
        if (path.permutation()[p] != p)
            throw std::invalid_argument(
                "periodic_virial: requires identity connectivity");
    prepare(path);
    periodic_field_.invalidate();
    for (const auto& component : components_)
        component->prepare_periodic_virial(path);
    periodic_field_.prepare(path);
}
Action::ThermoMeasure Action::measure_periodic_virial(PathSet& path) {
    // The kinetic contraction follows same-particle links, so only the
    // identity permutation is supported.
    if (!path.identity_permutation())
        throw std::logic_error(
            "periodic_virial: exchange (a non-identity permutation) is not "
            "supported yet");
    periodic_field_.evaluate(path);
    path.rebuild_tables();
    ThermoMeasure result;
    for (const auto& component : components_) {
        const auto contribution = component->measure_periodic_virial(
            path, periodic_field_.field(), periodic_field_.divergence());
        result.kinetic += contribution.kinetic;
        result.potential += contribution.potential;
        result.energy += contribution.energy;
    }
    return result;
}
} // namespace alea
