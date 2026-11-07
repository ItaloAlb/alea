// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "pimc/pimc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <omp.h>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "io/logger.h"

namespace alea {

namespace {

// Rethrow the lowest-index replica failure captured inside a parallel region,
// naming the replica and phase (the original message is kept).
void rethrow_replica_error(std::span<const std::exception_ptr> errors,
                           const char* phase) {
    for (std::size_t s = 0; s < errors.size(); ++s) {
        if (!errors[s]) continue;
        const std::string where = "pimc: replica " + std::to_string(s) +
                                  " failed during " + phase + ": ";
        try {
            std::rethrow_exception(errors[s]);
        } catch (const std::exception& e) {
            throw std::runtime_error(where + e.what());
        } catch (...) {
            throw std::runtime_error(where + "unknown exception");
        }
    }
}

// "fermions", "bosons" or "fermions+bosons" for the classes that can exchange.
std::string statistics_label(std::span<const ExchangeClass> classes) {
    bool fermions = false, bosons = false;
    for (const ExchangeClass& c : classes) {
        if (c.particles.size() < 2) continue;
        fermions = fermions || c.statistics == "fermions";
        bosons = bosons || c.statistics == "bosons";
    }
    return fermions && bosons ? "fermions+bosons" : fermions ? "fermions" : "bosons";
}

// Squared winding number, summed over closed loops and divided by the number
// of particles and lattice directions: W_c = fractional( sum of min-image
// link vectors around loop c ), an integer vector.  A loop is a ring when
// pi = identity and a whole permutation cycle otherwise (its L*M links); the
// superfluid estimator of Pollock & Ceperley, PRB 36, 8343 (1987) uses the
// same sum, <W^2> = sum_c <W_c^2>, since cross terms average to zero.  Zero
// for open cells (winding is not defined there).
double winding2_of(const PathSet& path) {
    if (!path.cell().is_periodic()) return 0.0;
    const std::size_t M   = path.num_slices();
    const std::size_t np  = path.n_particles();
    const std::size_t dim = path.dim();
    double w2 = 0.0;
    double sum[SimulationCell::kMaxDim];
    double del[SimulationCell::kMaxDim];
    if (!path.identity_permutation()) {
        for (const std::size_t leader : path.cycle_leaders()) {
            for (std::size_t i = 0; i < dim; ++i) sum[i] = 0.0;
            std::size_t p = leader;
            do {
                for (std::size_t k = 0; k < M; ++k) {
                    const auto after = path.bead_after(p, k);
                    for (std::size_t i = 0; i < dim; ++i)
                        del[i] = path.pos(after.particle, i, after.slice) -
                                 path.pos(p, i, k);
                    path.cell().min_image(del);
                    for (std::size_t i = 0; i < dim; ++i) sum[i] += del[i];
                }
                p = path.next_particle(p);
            } while (p != leader);
            double f[SimulationCell::kMaxDim];
            path.cell().fractional(sum, f);
            for (std::size_t i = 0; i < dim; ++i) w2 += f[i] * f[i];
        }
        return w2 / static_cast<double>(np * dim);
    }
    for (std::size_t p = 0; p < np; ++p) {
        for (std::size_t i = 0; i < dim; ++i) sum[i] = 0.0;
        for (std::size_t k = 0; k < M; ++k) {
            const std::size_t kn = (k + 1) % M;
            for (std::size_t i = 0; i < dim; ++i)
                del[i] = path.pos(p, i, kn) - path.pos(p, i, k);
            path.cell().min_image(del);
            for (std::size_t i = 0; i < dim; ++i) sum[i] += del[i];
        }
        double f[SimulationCell::kMaxDim];
        path.cell().fractional(sum, f);
        for (std::size_t i = 0; i < dim; ++i) w2 += f[i] * f[i];
    }
    return w2 / static_cast<double>(np * dim);
}

}  // namespace

Pimc::Pimc(const PimcConfig& cfg,
           std::span<const ParticleGroup> groups,
           const Hamiltonian& ham,
           const SimulationCell& cell,
           std::size_t dim,
           std::size_t np,
           WalkerEnsemble* ensemble,
           const Action* built_action)
    : cfg_(cfg), dim_(dim), np_(np), cell_(cell),
      groups_(groups.begin(), groups.end()), ensemble_(ensemble) {

    // Statistics: an
    // exchange class of more than one particle keeps quantum statistics
    // unless the YAML opts into distinguishable particles explicitly.  The
    // move set is checked below: without a move that changes the
    // permutation, a quantum run would silently stay in its starting sector.
    std::size_t largest_class =
        static_cast<std::size_t>(std::max(cfg_.exchange_class_max, 1));
    for (const ExchangeClass& c : cfg_.exchange_classes)
        largest_class = std::max(largest_class, c.particles.size());
    // fixed_node samples the identity sector of one nodal pocket: exchange
    // classes are not kept, and the action's restriction supplies the
    // antisymmetry (checked against the classes below).
    fixed_node_ = cfg_.statistics == "fixed_node";
    quantum_ = largest_class > 1 && cfg_.statistics != "boltzmannon" && !fixed_node_;

    if (!std::isfinite(cfg_.beta) || cfg_.beta <= 0 ||
        !std::isfinite(cfg_.time_step) || cfg_.time_step <= 0)
        throw std::runtime_error("pimc: beta and time_step must be positive and finite");
    if (cfg_.samples <= 0 || cfg_.threads < 0 || cfg_.equilibration_blocks < 0 ||
        cfg_.accumulation_blocks <= 0 || cfg_.steps_per_block <= 0 || cfg_.sub_steps <= 0)
        throw std::runtime_error("pimc: invalid replica, thread, block, step or sweep count");
    if (!np || !dim || dim > SimulationCell::kMaxDim)
        throw std::runtime_error("pimc: nonzero particle count and valid dimension required");
    if (!cfg_.statistics.empty() && cfg_.statistics != "boltzmannon" && !fixed_node_)
        throw std::runtime_error("pimc: unknown statistics '" + cfg_.statistics +
                                 "' (implemented: boltzmannon, fixed_node)");
    if (cfg_.energy == "thermodynamic")
        energy_estimator_ = EnergyEstimator::thermodynamic;
    else if (cfg_.energy == "centroid_virial")
        energy_estimator_ = EnergyEstimator::centroid_virial;
    else if (cfg_.energy == "periodic_virial")
        energy_estimator_ = EnergyEstimator::periodic_virial;
    else
        throw std::runtime_error("pimc: unknown energy estimator '"+cfg_.energy+"'");
    if (cfg_.compare_thermodynamic && energy_estimator_ == EnergyEstimator::thermodynamic)
        throw std::runtime_error("pimc: compare_thermodynamic requires energy: centroid_virial or periodic_virial");
    // The open centroid virial uses cycle centroids under exchange; the
    // periodic torus field is not derived for permuted paths yet.
    if (quantum_ && energy_estimator_ == EnergyEstimator::periodic_virial)
        throw std::runtime_error("pimc: the periodic virial estimator with exchange "
                                 "is not available yet; use energy: thermodynamic");
    if (quantum_ && !(std::isfinite(cfg_.sign_threshold) && cfg_.sign_threshold > 0))
        throw std::runtime_error("pimc: sign_threshold must be finite and positive");

    // Validate before floating-to-integer conversion or any path allocation.
    const double M_exact = cfg_.beta / cfg_.time_step;
    if (!std::isfinite(M_exact) || M_exact < 1 ||
        M_exact >= static_cast<double>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("pimc: beta/time_step is outside the supported slice range");
    const double rounded = std::round(M_exact);
    if (std::abs(rounded - M_exact) >
        8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, M_exact))
        throw std::runtime_error(
            "pimc: beta/time_step = " + std::to_string(M_exact) +
            " is not an integer; adjust beta or time_step");
    M_ = static_cast<std::size_t>(rounded);

    // Validate the resolved move set before allocating any replica paths.
    auto base_moves = build_moves();
    const bool samples_exchange = std::any_of(base_moves.begin(), base_moves.end(),
        [](const auto& move) { return move->changes_permutation(); });
    if (quantum_ && !samples_exchange && !cfg_.fixed_permutation_sector)
        throw std::runtime_error(
            "pimc: the system has an exchange class with " +
            std::to_string(largest_class) + " identical particles, so "
            "Boltzmann statistics would change the result.  Add a "
            "`permutation` move to sample exchange, or set `statistics: "
            "boltzmannon` explicitly to run them as distinguishable particles.");
    if (samples_exchange && fixed_node_)
        throw std::runtime_error(
            "pimc: statistics: fixed_node samples the identity permutation inside "
            "one nodal pocket (odd permutations cross the node); remove the "
            "permutation and worm moves");
    if (samples_exchange && !quantum_)
        throw std::runtime_error(
            "pimc: a permutation move needs quantum statistics: an exchange "
            "class of two or more particles and no `statistics: boltzmannon`");

    if (!cell.is_periodic() && ham.translation_invariant())
        cm_energy_correction_ = static_cast<double>(dim) / (2.0 * cfg_.beta);
    // A YAML action is an ordered composite: retain every component and clone
    // its prepared physics independently per replica.  Without one, sample
    // the default primitive kinetic + potential action of `ham`.
    Action base_action = built_action ? built_action->clone() :
        Action(groups, ham, dim);
    // SoA coordinates and per-axis masses must use the same spatial stride.
    // Equal particle counts alone can hide an action reading another particle's
    // coordinates as an axis, or reading past the end of the path storage.
    if (base_action.dim() != dim)
        throw std::runtime_error("pimc: action dimension does not match path dimension");
    if (!base_action.supports_free_bridge())
        throw std::runtime_error("pimc: current moves require one primitive free kinetic component and compatible potential components");
    if (base_action.np() != np)
        throw std::runtime_error("pimc: particle groups do not match particle count");
    base_action.validate_sampling(cell,cfg_.time_step);

    // A nodal restriction and fixed_node statistics come together: each fermion
    // class of two or more particles must be exactly one restricted determinant
    // block, or its antisymmetry would be lost; bosons have no nodes.
    const auto restricted = base_action.restricted_blocks();
    if (fixed_node_) {
        if (restricted.empty())
            throw std::runtime_error(
                "pimc: statistics: fixed_node needs the fixed-node restriction in "
                "the action ('- nodes: wavefunction')");
        for (const ExchangeClass& ec : cfg_.exchange_classes) {
            if (ec.particles.size() < 2) continue;
            const std::string name = "class " + ec.group + " (flavour " +
                                     std::to_string(ec.flavour) + ")";
            if (ec.statistics != "fermions")
                throw std::runtime_error(
                    "pimc: statistics: fixed_node has no nodes for the " + ec.statistics +
                    " " + name + "; identical bosons need permutation moves");
            std::vector<std::size_t> members = ec.particles;
            std::sort(members.begin(), members.end());
            const bool one_block = std::any_of(
                restricted.begin(), restricted.end(), [&](const auto& block) {
                    if (block.second - block.first != members.size()) return false;
                    for (std::size_t i = 0; i < members.size(); ++i)
                        if (members[i] != block.first + i) return false;
                    return true;
                });
            if (!one_block)
                throw std::runtime_error(
                    "pimc: statistics: fixed_node: the fermion " + name +
                    " is not one determinant block of the nodes, so its "
                    "antisymmetry would be lost; give its group a slater item");
        }
        // Conversely, a block of two or more particles must be one fermion
        // class: nodes on distinguishable particles have no physical meaning.
        for (const auto& [first, last] : restricted) {
            if (last - first < 2) continue;
            const bool fermion_class = std::any_of(
                cfg_.exchange_classes.begin(), cfg_.exchange_classes.end(),
                [&](const ExchangeClass& ec) {
                    if (ec.statistics != "fermions" || ec.particles.size() != last - first)
                        return false;
                    std::vector<std::size_t> members = ec.particles;
                    std::sort(members.begin(), members.end());
                    for (std::size_t i = 0; i < members.size(); ++i)
                        if (members[i] != first + i) return false;
                    return true;
                });
            if (!fermion_class)
                throw std::runtime_error(
                    "pimc: statistics: fixed_node: the determinant block of particles [" +
                    std::to_string(first) + ", " + std::to_string(last) +
                    ") is not one fermion exchange class; declare those particles "
                    "as fermions of one group and spin");
        }
    } else if (!restricted.empty()) {
        throw std::runtime_error(
            "pimc: the action restricts paths to a nodal pocket ('nodes'), which "
            "only 'statistics: fixed_node' samples correctly");
    }

    // Exchange labels for every replica path.  Distinguishable particles keep
    // their own singleton class (only the identity can touch them); members
    // of a class share id np + c.  The closing link of a permuted path joins
    // two class members, so they must share one mass tensor.
    constexpr std::size_t no_class = static_cast<std::size_t>(-1);
    class_id_.resize(np);
    std::iota(class_id_.begin(), class_id_.end(), std::size_t{0});
    class_index_.assign(np, no_class);
    fermionic_.assign(np, 0);
    if (quantum_) {
        if (cfg_.exchange_classes.empty())
            throw std::runtime_error("pimc: quantum statistics need the exchange "
                                     "classes of the system");
        for (std::size_t c = 0; c < cfg_.exchange_classes.size(); ++c) {
            const ExchangeClass& ec = cfg_.exchange_classes[c];
            if (ec.statistics != "fermions" && ec.statistics != "bosons")
                throw std::runtime_error("pimc: exchange class statistics must be "
                                         "fermions or bosons, got '" + ec.statistics + "'");
            for (std::size_t p : ec.particles) {
                if (p >= np || class_index_[p] != no_class)
                    throw std::runtime_error("pimc: exchange classes must be "
                                             "disjoint sets of system particles");
                class_index_[p] = c;
                class_id_[p] = np + c;
                fermionic_[p] = ec.statistics == "fermions";
                for (std::size_t d = 0; d < dim; ++d)
                    if (base_action.mass(p, d) != base_action.mass(ec.particles.front(), d))
                        throw std::runtime_error("pimc: an exchange class must share "
                                                 "one mass tensor");
            }
            signed_ = signed_ || (ec.statistics == "fermions" && ec.particles.size() > 1);
        }
    }
    weighted_ = signed_ || worm_;
    // The cut link of an open worm carries no kinetic factor; an action that
    // couples slices through links (pair product) would still count it.
    if (worm_ && !base_action.slice_local_deltas())
        throw std::runtime_error("pimc: the worm needs an action whose potential terms are "
                                 "slice-local (primitive); the pair-product action couples "
                                 "the cut link");
    if (energy_estimator_ == EnergyEstimator::centroid_virial)
        base_action.validate_open_centroid(cell,np,dim);
    if (energy_estimator_ == EnergyEstimator::periodic_virial)
        base_action.validate_periodic_virial(cell,np,dim);
    if (cell.is_periodic()) {
        double widest_sigma = 0.0;
        for (std::size_t p = 0; p < np; ++p) {
            for (std::size_t d = 0; d < dim; ++d) {
                widest_sigma = std::max(widest_sigma,
                    std::sqrt(cfg_.time_step / base_action.mass(p, d)));
                if (!cell.orthorhombic() &&
                    base_action.mass(p,d) != base_action.mass(p,0))
                    throw std::runtime_error("pimc: anisotropic masses in a skew periodic cell require a mass-metric kinetic kernel");
            }
        }
        // Planned small-width domain: sigma <= L_shortest/10. This controls
        // discarded Gaussian images, not finite-tau interaction bias.
        const double shortest_length = 2.0 * cell.ws_radius();
        if (!(shortest_length > 0) || widest_sigma > shortest_length / 10.0)
            throw std::runtime_error("pimc: minimum-image kinetic approximation requires sqrt(tau/m) <= L_shortest/10; reduce time_step");
        periodic_width_ratio_ = widest_sigma / shortest_length;
        // Coordinate union bound for a free Gaussian link in an orthogonal
        // cell; the omitted-image probability for an interacting/skew-cell
        // path is not bounded by this quantity. Do not present it as such.
        if (cell.orthorhombic())
            periodic_image_tail_bound_ =
                std::min(1.0, static_cast<double>(M_) * static_cast<double>(np_) *
                    static_cast<double>(dim_) *
                    std::erfc(cell.ws_radius() / (std::sqrt(2.0) * widest_sigma)));
    }

    // One PathSet/Action/RNG per replica; OpenMP distributes replicas in
    // run(), so the result depends only on (config, seed) — never on the
    // thread count. Expected strong scaling saturates near min(threads,
    // samples): independent replicas offer no more parallel tasks than samples.
    // Expected weak scaling at fixed replicas/thread is limited by memory and
    // the serial reduction/reblocking of block histories. Neither scaling
    // curve nor wall-time speedup has been measured.
    const int samples = cfg_.samples;
    // Build immutable time-step-dependent pair tables once, then share them
    // read-only through replica clones. No setup work belongs in a move.
    base_action.prepare_time_step(cfg_.time_step);
    paths_.reserve(static_cast<std::size_t>(samples));
    actions_.reserve(static_cast<std::size_t>(samples));
    rngs_.reserve(static_cast<std::size_t>(samples));

    std::vector<double> init(np * dim, 0.0); // [np*dim] AoS, setup scratch
    const auto lattice = cell.vectors();      // [dim*dim], rows a_i
    for (int s = 0; s < samples; ++s) {
        const auto replica = static_cast<std::uint64_t>(s);
        paths_.emplace_back(M_, np, dim, cfg_.time_step, cell);
        if (quantum_) paths_.back().set_exchange_classes(class_id_, fermionic_);
        actions_.push_back(base_action.clone());
        if (energy_estimator_ == EnergyEstimator::centroid_virial)
            actions_.back().prepare_open_centroid(paths_.back());
        if (energy_estimator_ == EnergyEstimator::periodic_virial)
            actions_.back().prepare_periodic_virial(paths_.back());
        rngs_.emplace_back(
            numerics::stream_seed(cfg_.seed, kPimcReplicaStream, replica));

        // Fallback start when no ensemble was supplied (unit tests): every
        // bead at one configuration, uniform in the cell (r = sum_i f_i a_i,
        // f_i in [0,1)), or in the unit box [-1/2,1/2)^d when open — the
        // equilibration spreads it.  Drawn from its own stream, so the
        // replica's sampling stream starts untouched.  With an ensemble, run()
        // overwrites this with collapsed walker configurations.
        numerics::Random ring_rng(
            numerics::stream_seed(cfg_.seed, kPimcRingInitStream, replica));
        for (std::size_t p = 0; p < np; ++p) {
            double* r = init.data() + p * dim;
            if (cell.is_periodic()) {
                std::fill(r, r + dim, 0.0);
                for (std::size_t i = 0; i < dim; ++i) {
                    const double f = ring_rng.uniform();
                    for (std::size_t d = 0; d < dim; ++d)
                        r[d] += f * lattice[i * dim + d];
                }
            } else {
                for (std::size_t d = 0; d < dim; ++d)
                    r[d] = ring_rng.uniform(-0.5, 0.5);
            }
        }
        paths_.back().collapse_to(init);  // wraps and rebuilds the tables
    }

    // Clone the already validated move set for independent replicas.
    moves_.resize(static_cast<std::size_t>(samples));
    for (int s = 0; s < samples; ++s) {
        for (const auto& m : base_moves)
            moves_[static_cast<std::size_t>(s)].push_back(m->clone());
    }
}

std::vector<std::unique_ptr<Move>> Pimc::build_moves() {
    // An omitted move list resolves to the shared default (bisection as deep
    // as the ring allows + centroid translation), validated like any other.
    move_specs_ = cfg_.moves.empty() ? io::default_pimc_moves(M_) : cfg_.moves;
    std::vector<std::unique_ptr<Move>> moves;
    double total_weight = 0.0;
    for (const PimcConfig::MoveSpec& ms : move_specs_) {
        total_weight += ms.weight;
        if (!std::isfinite(total_weight))
            throw std::runtime_error("pimc: sum of move weights must be finite");
        if (!std::isfinite(ms.weight) || !(ms.weight > 0.0))
            throw std::runtime_error(
                "pimc: move weight must be > 0 (got " +
                std::to_string(ms.weight) + " for '" + ms.type + "')");
        std::unique_ptr<Move> m;
        if (ms.type == "bisection") {
            // 2^levels contiguous slices are rebuilt per attempt; they must
            // fit in the ring.  Checked at build time so a bad config fails
            // before the first sweep, not mid-run.
            if (ms.levels < 1 || ms.levels > 30 ||
                static_cast<std::size_t>(1) << ms.levels > M_)
                throw std::runtime_error(
                    "pimc: bisection levels=" + std::to_string(ms.levels) +
                    " needs 2^levels <= M = " + std::to_string(M_));
            m = std::make_unique<BisectionMove>(ms.levels);
        } else if (ms.type == "centroid") {
            if (!std::isfinite(ms.step) || !(ms.step > 0.0))
                throw std::runtime_error("pimc: centroid step must be > 0");
            m = std::make_unique<CentroidMove>(ms.step);
        } else if (ms.type == "single_slice") {
            if (!std::isfinite(ms.step) || !(ms.step > 0.0))
                throw std::runtime_error("pimc: single_slice step must be > 0");
            m = std::make_unique<SingleSliceMove>(ms.step);
        } else if (ms.type == "permutation") {
            // The window of 2^levels slices must fit in the ring.
            if (ms.levels < 1 || ms.levels > 30 ||
                static_cast<std::size_t>(1) << ms.levels > M_)
                throw std::runtime_error(
                    "pimc: permutation levels=" + std::to_string(ms.levels) +
                    " needs 2^levels <= M = " + std::to_string(M_));
            if (ms.max_cycle < 2 || ms.max_cycle > 8)
                throw std::runtime_error("pimc: permutation max_cycle must be in [2, 8]");
            m = std::make_unique<PermutationMove>(ms.levels, ms.max_cycle);
        } else if (ms.type == "joint_permutation") {
            if (ms.levels < 1 || ms.levels > 30 ||
                static_cast<std::size_t>(1) << ms.levels > M_)
                throw std::runtime_error(
                    "pimc: joint_permutation levels=" + std::to_string(ms.levels) +
                    " needs 2^levels <= M = " + std::to_string(M_));
            if (ms.max_cycle < 2 || ms.max_cycle > 8)
                throw std::runtime_error("pimc: joint_permutation max_cycle must be in [2, 8]");
            if (!std::isfinite(ms.pair_radius) || !(ms.pair_radius > 0.0))
                throw std::runtime_error("pimc: joint_permutation pair_radius must be finite and > 0");
            if (ms.pair[0] == ms.pair[1])
                throw std::runtime_error("pimc: joint_permutation pairs two different groups");
            // Every pairing of an exchangeable class of group A with one of
            // group B (e.g. all spin/valley combinations of an exciton).
            std::vector<JointPermutationMove::ClassPair> pairs;
            for (const ExchangeClass& a : cfg_.exchange_classes) {
                if (a.group != ms.pair[0] || a.particles.size() < 2) continue;
                for (const ExchangeClass& b : cfg_.exchange_classes)
                    if (b.group == ms.pair[1] && b.particles.size() >= 2)
                        pairs.emplace_back(a.particles, b.particles);
            }
            if (pairs.empty())
                throw std::runtime_error(
                    "pimc: joint_permutation needs an exchange class of two or more "
                    "particles in both groups '" + ms.pair[0] + "' and '" + ms.pair[1] + "'");
            m = std::make_unique<JointPermutationMove>(ms.levels, ms.max_cycle,
                                                       ms.pair_radius, std::move(pairs));
        } else if (ms.type == "worm") {
            if (ms.max_length < 1 || static_cast<std::size_t>(ms.max_length) + 2 > M_)
                throw std::runtime_error("pimc: worm max_length must be in [1, M - 2], M = " +
                                         std::to_string(M_));
            if (ms.swap_length < 2 || static_cast<std::size_t>(ms.swap_length) + 1 > M_)
                throw std::runtime_error("pimc: worm swap_length must be in [2, M - 1], M = " +
                                         std::to_string(M_));
            if (!std::isnan(ms.worm_constant) &&
                !(std::isfinite(ms.worm_constant) && ms.worm_constant > 0.0))
                throw std::runtime_error("pimc: worm C must be finite and > 0");
            // Every exchangeable class (flavour) of the group.
            std::vector<std::vector<std::size_t>> classes;
            for (const ExchangeClass& ec : cfg_.exchange_classes)
                if (ec.group == ms.group && ec.particles.size() >= 2)
                    classes.push_back(ec.particles);
            if (classes.empty())
                throw std::runtime_error("pimc: worm needs an exchange class of two or more "
                                         "particles in group '" + ms.group + "'");
            m = std::make_unique<WormMove>(std::move(classes), ms.max_length,
                                           ms.swap_length, ms.worm_constant);
        } else {
            throw std::runtime_error("pimc: unknown move type '" + ms.type + "'");
        }
        m->set_weight(ms.weight);
        moves.push_back(std::move(m));
    }
    // Worms: the driver picks a worm in Z with probability weight / total
    // (part of its open/close acceptance) and calls only the open worm in G.
    worm_moves_.clear();
    worm_of_particle_.assign(np_, std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; i < moves.size(); ++i) {
        if (move_specs_[i].type != "worm") continue;
        auto& worm = static_cast<WormMove&>(*moves[i]);
        worm.set_selection_probability(move_specs_[i].weight / total_weight);
        for (const ExchangeClass& ec : cfg_.exchange_classes) {
            if (ec.group != move_specs_[i].group || ec.particles.size() < 2) continue;
            for (const std::size_t p : ec.particles) {
                if (worm_of_particle_[p] != std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("pimc: two worm moves act on group '" +
                                             ec.group + "'");
                worm_of_particle_[p] = i;
            }
        }
        worm_moves_.push_back(i);
    }
    worm_ = !worm_moves_.empty();
    // With M > 1, centroid shifts preserve every r_k-r_{k+1}. They cannot
    // sample the internal degrees of freedom in exp(-S), even if their
    // Metropolis acceptance obeys detailed balance within that frozen shape.
    // M = 1 has no internal beads. This gate is necessary, not a proof of
    // equilibration or efficient winding-sector mixing (Ceperley, RMP 67,
    // 279 (1995), sections II-V).
    if (M_ > 1 && std::none_of(moves.begin(), moves.end(),
            [](const auto& move) { return move->changes_internal_beads(); }))
        throw std::runtime_error(
            "pimc: M > 1 requires a move that changes internal beads; "
            "centroid-only sampling leaves the ring shape frozen. "
            "Add bisection or single_slice to moves");
    return moves;
}

void Pimc::import_paths(const io::PathCheckpoint& checkpoint) {
    io::require_path_compatibility(checkpoint, paths_.size(), M_, np_, dim_,
                                   cfg_.beta, cfg_.time_step, cell_);
    // Validation completed before any replica changes; preserve full binary64
    // beads, not a bead-0 walker projection or a collapsed initialization.
    const std::size_t path_size = M_ * np_ * dim_;
    // Compatibility alone cannot establish finite action for a numerical
    // pair domain (or an overflowing spring). Validate every candidate on a
    // disposable path/action before committing any replica or its caches.
    PathSet candidate(M_,np_,dim_,cfg_.time_step,cell_);
    if (quantum_) candidate.set_exchange_classes(class_id_, fermionic_);
    Action validation=actions_.front().clone();
    const auto permutation_of = [&](std::size_t s) {
        return std::span<const std::size_t>(checkpoint.permutation).subspan(s * np_, np_);
    };
    for (std::size_t s=0; s<paths_.size(); ++s) {
        check_permutation(permutation_of(s));
        candidate.set_permutation(permutation_of(s));
        std::copy_n(checkpoint.positions.begin()+s*path_size,path_size,candidate.positions().begin());
        if (!std::isfinite(validation.total_action(candidate)))
            throw std::runtime_error("pimc: checkpoint action is nonfinite for replica "+std::to_string(s));
    }
    for (std::size_t s = 0; s < paths_.size(); ++s) {
        paths_[s].set_permutation(permutation_of(s));
        auto dest = paths_[s].positions();
        std::copy_n(checkpoint.positions.begin() + s * path_size, path_size,
                    dest.begin());
        paths_[s].rebuild_tables();
    }
    imported_paths_ = true;
}

void Pimc::check_permutation(std::span<const std::size_t> pi) const {
    // Distinguishable runs are identity-only; a quantum run may connect members
    // of one exchange class (PathSet::set_permutation validates the classes).
    if (!quantum_)
        for (std::size_t p = 0; p < pi.size(); ++p)
            if (pi[p] != p)
                throw std::runtime_error(
                    "pimc: a non-identity checkpoint permutation needs quantum "
                    "statistics; this run samples distinguishable particles");
}

io::PathCheckpoint Pimc::export_paths() const {
    io::PathCheckpoint checkpoint;
    checkpoint.replicas = paths_.size();
    checkpoint.m = M_;
    checkpoint.np = np_;
    checkpoint.dim = dim_;
    checkpoint.beta = cfg_.beta;
    checkpoint.tau = cfg_.time_step;
    if (cell_.is_periodic()) {
        const auto vectors = cell_.vectors();
        checkpoint.lattice.assign(vectors.begin(), vectors.end());
    }
    checkpoint.positions.reserve(paths_.size() * M_ * np_ * dim_);
    checkpoint.permutation.reserve(paths_.size() * np_);
    for (const PathSet& path : paths_) {
        const auto positions = path.positions();
        checkpoint.positions.insert(checkpoint.positions.end(),
                                    positions.begin(), positions.end());
        const auto permutation = path.permutation();
        checkpoint.permutation.insert(checkpoint.permutation.end(),
                                      permutation.begin(), permutation.end());
    }
    return checkpoint;
}

void Pimc::set_estimators(std::vector<std::unique_ptr<Estimator>> ests) {
    constexpr std::size_t kMaxAllReplicaHistoryBytes = 256 * 1024 * 1024;
    const auto blocks = static_cast<std::size_t>(cfg_.accumulation_blocks);
    std::size_t per_replica = 0;
    for (const auto& est : ests) {
        if (!est) throw std::runtime_error("pimc: null estimator");
        const auto bytes = est->reblocking_storage_bytes(blocks);
        if (bytes > kMaxAllReplicaHistoryBytes - per_replica)
            throw std::runtime_error("pimc: optional-observable history exceeds aggregate 256 MiB budget; reduce bins, blocks or replicas");
        per_replica += bytes;
    }
    if (per_replica > kMaxAllReplicaHistoryBytes /
            (static_cast<std::size_t>(cfg_.samples) + 1))
        throw std::runtime_error("pimc: optional-observable histories exceed aggregate 256 MiB budget; reduce bins, blocks or replicas");
    // A signed run measures <O s>/<s>: each estimator keeps sum|w|-normalized
    // blocks and their mean signs (EstimatorReblocking signed mode).
    if (weighted_)
        for (auto& est : ests) est->configure_signed_weights();
    // Off-diagonal estimators follow the worm of their group.
    est_worm_.assign(ests.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t e = 0; e < ests.size(); ++e) {
        const std::string group = ests[e]->open_path_group();
        if (group.empty()) continue;
        for (const std::size_t w : worm_moves_)
            if (move_specs_[w].group == group) est_worm_[e] = w;
        if (est_worm_[e] == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("pimc: estimator '" + ests[e]->type() +
                                     "' needs a worm move on group '" + group + "'");
        ests[e]->configure_open_paths(
            static_cast<const WormMove&>(*moves_.front()[est_worm_[e]]).worm_particles());
    }
    for (auto& est : ests) est->configure_reblocking(blocks);
    est_templates_ = std::move(ests);
}

void Pimc::run() {
    const int n_threads = cfg_.threads > 0
                              ? cfg_.threads
                              : omp_get_max_threads();
    const int samples = cfg_.samples;
    const int n_eq = cfg_.equilibration_blocks;
    const int n_acc = cfg_.accumulation_blocks;
    const int steps = cfg_.steps_per_block;
    const int sub_steps = cfg_.sub_steps;

    // Weighted move selection: compute cumulative weights.
    // All replicas have the same move types, so use replica 0 as template.
    std::vector<double> cum_weights;
    {
        double total = 0.0;
        for (const auto& m : moves_[0]) {
            total += m->weight();
            cum_weights.push_back(total);
        }
        for (double& w : cum_weights) w /= total;
    }

    const std::size_t n_moves = cum_weights.size();
    const auto n_replicas = static_cast<std::size_t>(samples);
    move_proposals_.assign(n_moves, 0);
    move_accepts_.assign(n_moves, 0);
    move_infinite_rejections_.assign(n_moves, 0);
    // Per-replica move counters in one block, layout
    // [replica * stride + {proposals[n_moves], accepts[n_moves], padding}].
    // Every attempt increments them, so a row is padded by at least one
    // 64-byte line: counters of replicas on different threads never share a
    // cache line (small per-replica heap rows sat adjacent: false sharing).
    constexpr std::size_t kCountersPerLine = 64 / sizeof(std::size_t);
    const std::size_t counter_stride =
        (2 * n_moves + kCountersPerLine - 1) / kCountersPerLine *
            kCountersPerLine + kCountersPerLine;
    std::vector<std::size_t> move_counters(n_replicas * counter_stride, 0);

    // An exception must not leave an OpenMP structured block (that calls
    // std::terminate: no message, no partial output).  Each replica captures
    // its own; the others stop at their next step; the lowest-index failure
    // is rethrown after the region.
    std::vector<std::exception_ptr> replica_errors(n_replicas);
    std::atomic<bool> stop_replicas{false};
    const auto stopped = [&stop_replicas]() {
        return stop_replicas.load(std::memory_order_relaxed);
    };

    auto select_move = [&](numerics::Random& rng) -> std::size_t {
        double u = rng.uniform();
        for (std::size_t i = 0; i < cum_weights.size(); ++i)
            if (u < cum_weights[i])
                return i;
        return cum_weights.size() - 1;
    };

    io::Logger::milestone(
        "pimc: M=" + std::to_string(M_) + " tau=" +
        std::to_string(cfg_.time_step) + " beta=" +
        std::to_string(cfg_.beta) + " samples=" + std::to_string(samples) +
        " threads=" + std::to_string(n_threads));

    // ── Initial rings: collapsed from the ensemble ──
    // Read INSIDE run() so a driver-to-driver carry-over (VMC -> PIMC) that
    // seeds the ensemble before run() is picked up for free.  Every slice of
    // a replica starts at the same configuration; the equilibration blocks
    // inflate each ring to its thermal width.
    if (ensemble_ && !imported_paths_) {
        const auto walkers = ensemble_->walkers();
        if (walkers.empty())
            throw std::runtime_error(
                "pimc: the ensemble handed to the driver has no walkers");
        for (int s = 0; s < samples; ++s)  // collapse_to wraps and rebuilds
            paths_[static_cast<std::size_t>(s)].collapse_to(
                walkers[static_cast<std::size_t>(s) % walkers.size()]
                    .positions());
    }

    // Metropolis samples exp(-S); a committed path must have finite S so
    // proposal differences cannot become inf-inf (Ceperley, RMP 67, 279
    // (1995), section II). Validate after ensemble carry-over or path import,
    // before parallel sampling. Do not regularize singular interactions here.
    for (std::size_t s = 0; s < paths_.size(); ++s)
        if (!std::isfinite(actions_[s].total_action(paths_[s])))
            throw std::runtime_error(
                "pimc: initial action is nonfinite for replica " +
                std::to_string(s) + "; provide a finite-action initial path");

    // ── Equilibration ───────────────────────────────────────────────────
    io::Logger::milestone("pimc: equilibration " + std::to_string(n_eq) +
                          " blocks");

    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (int s = 0; s < samples; ++s) {
        const auto r = static_cast<std::size_t>(s);
        try {
            auto& path   = paths_[r];
            auto& action = actions_[r];
            auto& rng    = rngs_[r];
            // Worm runs count, per block, the attempts made on closed paths
            // and on paths open in each worm, then tune each tunable C
            // towards a 50 % closed fraction shared equally by the worms.
            std::vector<std::size_t> open_attempts(worm_moves_.size(), 0);
            for (int b = 0; b < n_eq && !stopped(); ++b) {
                std::size_t closed_attempts = 0;
                std::fill(open_attempts.begin(), open_attempts.end(), 0);
                for (int st = 0; st < steps && !stopped(); ++st)
                    for (int ss = 0; ss < sub_steps; ++ss) {
                        if (!path.is_open()) {
                            ++closed_attempts;
                            moves_[r][select_move(rng)]->attempt(path, action, rng);
                            continue;
                        }
                        const std::size_t owner = worm_of_particle_[path.head().particle];
                        for (std::size_t w = 0; w < worm_moves_.size(); ++w)
                            open_attempts[w] += worm_moves_[w] == owner;
                        moves_[r][owner]->attempt(path, action, rng);
                    }
                const double total = static_cast<double>(steps) * sub_steps;
                for (std::size_t w = 0; w < worm_moves_.size(); ++w)
                    static_cast<WormMove&>(*moves_[r][worm_moves_[w]]).tune_constant(
                        static_cast<double>(closed_attempts) / total,
                        static_cast<double>(open_attempts[w]) / total,
                        1.0 / static_cast<double>(worm_moves_.size()));
            }
        } catch (...) {
            replica_errors[r] = std::current_exception();
            stop_replicas.store(true, std::memory_order_relaxed);
        }
    }
    rethrow_replica_error(replica_errors, "equilibration");

    // ── Accumulation ────────────────────────────────────────────────────
    io::Logger::milestone("pimc: accumulation " + std::to_string(n_acc) +
                          " blocks");

    // Per-replica block averages of {s E, s K, s V, s W^2, s * optional paired
    // E difference, s}, flat [replica * n_acc + block].  s is the permutation
    // sign of a signed run and exactly 1.0 otherwise, so unsigned sums are
    // bit-identical to unweighted ones (x * 1.0 == x).
    // Index 6: the fraction of measurements on closed paths (worm runs).
    using Block = std::array<double, 7>;
    const auto n_blocks = static_cast<std::size_t>(n_acc);
    std::vector<Block> sample_blocks(n_replicas * n_blocks, Block{});

    // Cycle-length counts per replica, [replica][class_offset[c] + L - 1]:
    // particles of class c found in cycles of length L (quantum runs only).
    std::vector<std::size_t> class_offset(cfg_.exchange_classes.size() + 1, 0);
    for (std::size_t c = 0; c < cfg_.exchange_classes.size(); ++c)
        class_offset[c + 1] = class_offset[c] + cfg_.exchange_classes[c].particles.size();
    std::vector<std::vector<double>> replica_cycles(
        n_replicas, std::vector<double>(quantum_ ? class_offset.back() : 0, 0.0));
    std::vector<std::size_t> replica_measurements(n_replicas, 0);

    // Infinite-action rejections are reported for the accumulation phase,
    // like the proposal and acceptance counts.
    for (auto& replica_moves : moves_)
        for (auto& move : replica_moves) move->reset_diagnostics();

    // Per-replica estimator clones (reduced after the parallel region).
    std::vector<std::vector<std::unique_ptr<Estimator>>> replica_ests(
        static_cast<std::size_t>(samples));
    if (!est_templates_.empty())
        for (int s = 0; s < samples; ++s)
            for (const auto& tmpl : est_templates_)
                replica_ests[static_cast<std::size_t>(s)].push_back(
                    tmpl->clone());

    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (int s = 0; s < samples; ++s) {
        const auto r = static_cast<std::size_t>(s);
        try {
            auto& path   = paths_[r];
            auto& action = actions_[r];
            auto& rng    = rngs_[r];
            auto& ests   = replica_ests[r];
            std::size_t* proposals = move_counters.data() + r * counter_stride;
            std::size_t* accepts   = proposals + n_moves;
            // PIMC measures on slices: every slice samples the diagonal
            // density matrix with weight 1 (Estimator docs, estimator.h).
            std::vector<double> weights(path.num_slices(), 1.0);
            // Open-path samples (worm runs): [samples * dim] and [samples].
            std::size_t open_samples = 0;
            for (const auto& e : ests) open_samples = std::max(open_samples, e->open_path_samples());
            std::vector<double> open_displacement(open_samples * path.dim()), open_weight(open_samples);

            for (int b = 0; b < n_acc && !stopped(); ++b) {
                for (auto& e : ests) e->start_block();
                Block sum{};
                int e_count = 0;
                for (int st = 0; st < steps && !stopped(); ++st) {
                    for (int ss = 0; ss < sub_steps; ++ss) {
                        // An open path belongs to its worm (worm moves only in G).
                        const std::size_t move_index = path.is_open()
                            ? worm_of_particle_[path.head().particle] : select_move(rng);
                        ++proposals[move_index];
                        if (moves_[r][move_index]->attempt(path, action, rng))
                            ++accepts[move_index];
                    }
                    // Measure: one pass gives E, K and V of the same path,
                    // plus the ring winding (the quantity frame-jump
                    // bisection exists to keep ergodic under PBC).
                    // Diagonal observables need a closed path; an open one
                    // still counts as a measurement with weight 0.
                    const bool closed = !path.is_open();
                    if (!weighted_) {
                        const Action::ThermoMeasure m = measure_energy(action, path);
                        sum[0] += m.energy;
                        sum[1] += m.kinetic;
                        sum[2] += m.potential;
                        sum[3] += winding2_of(path);
                        if (cfg_.compare_thermodynamic)
                            sum[4] += m.energy - action.measure_thermo(path).energy;
                        sum[5] += 1.0;
                    } else if (closed) {
                        // Ratio estimator <O s [closed]>/<s [closed]>: the
                        // chain samples |weight|; s = sign(pi) restricted to
                        // fermions (1 for bosons).
                        const Action::ThermoMeasure m = measure_energy(action, path);
                        const double s = path.permutation_sign();
                        sum[0] += s * m.energy;
                        sum[1] += s * m.kinetic;
                        sum[2] += s * m.potential;
                        sum[3] += s * winding2_of(path);
                        if (cfg_.compare_thermodynamic)
                            sum[4] += s * (m.energy - action.measure_thermo(path).energy);
                        sum[5] += s;
                        sum[6] += 1.0;
                    }
                    if (quantum_ && closed) {
                        auto& counts = replica_cycles[r];
                        for (const std::size_t leader : path.cycle_leaders()) {
                            const std::size_t c = class_index_[leader];
                            if (c >= cfg_.exchange_classes.size()) continue;
                            std::size_t length = 1;
                            for (std::size_t q = path.next_particle(leader);
                                 q != leader; q = path.next_particle(q))
                                ++length;
                            counts[class_offset[c] + length - 1] +=
                                static_cast<double>(length);
                        }
                        ++replica_measurements[r];
                    }
                    ++e_count;
                    if (closed) {
                        // Every slice of a closed path carries the path's sign.
                        if (weighted_)
                            std::fill(weights.begin(), weights.end(), path.permutation_sign());
                        for (auto& e : ests)
                            e->accumulate(path, std::span<const double>(weights));
                    } else {
                        // Open path: its worm's off-diagonal estimators get
                        // extra-bead samples; every other estimator counts
                        // the measurement without binning it.
                        const std::size_t owner = worm_of_particle_[path.head().particle];
                        for (std::size_t e = 0; e < ests.size(); ++e) {
                            if (est_worm_[e] != owner) {
                                ests[e]->skip_measurement(weights.size());
                                continue;
                            }
                            const std::size_t n = ests[e]->open_path_samples();
                            const std::span<double> displacement(open_displacement.data(), n * path.dim());
                            const std::span<double> weight(open_weight.data(), n);
                            static_cast<WormMove&>(*moves_[r][owner]).sample_open_ends(
                                path, action, rng, displacement, weight);
                            ests[e]->accumulate_open(displacement, weight);
                        }
                    }
                }
                for (auto& e : ests) e->stop_block();
                const double inv = 1.0 / static_cast<double>(e_count);
                for (double& value : sum) value *= inv;
                sample_blocks[r * n_blocks + static_cast<std::size_t>(b)] = sum;
            }
        } catch (...) {
            replica_errors[r] = std::current_exception();
            stop_replicas.store(true, std::memory_order_relaxed);
        }
    }
    rethrow_replica_error(replica_errors, "accumulation");

    for (std::size_t r = 0; r < n_replicas; ++r)
        for (std::size_t i = 0; i < n_moves; ++i) {
            move_proposals_[i] += move_counters[r * counter_stride + i];
            move_accepts_[i] += move_counters[r * counter_stride + n_moves + i];
            move_infinite_rejections_[i] +=
                moves_[r][i]->infinite_action_rejections();
        }
    // Move-specific counters (e.g. permutation cycle lengths), summed in
    // replica order.
    move_extra_.assign(n_moves, {});
    for (std::size_t i = 0; i < n_moves; ++i)
        for (std::size_t r = 0; r < n_replicas; ++r) {
            const auto extra = moves_[r][i]->extra_diagnostics();
            if (r == 0) move_extra_[i] = extra;
            else
                for (std::size_t e = 0; e < extra.size(); ++e)
                    move_extra_[i][e].second += extra[e].second;
        }
    if (worm_) {
        // C per worm after equilibration (frozen during accumulation).
        worm_constants_.assign(worm_moves_.size(), {0.0, 0.0, 0.0});
        for (std::size_t w = 0; w < worm_moves_.size(); ++w)
            for (std::size_t r = 0; r < n_replicas; ++r) {
                const double c =
                    static_cast<const WormMove&>(*moves_[r][worm_moves_[w]]).worm_constant();
                auto& [mean, low, high] = worm_constants_[w];
                mean += c / static_cast<double>(n_replicas);
                low = r == 0 ? c : std::min(low, c);
                high = r == 0 ? c : std::max(high, c);
            }
        // Exported paths must be closed: drive open replicas closed with
        // worm moves only (after measurement, outside the statistics).
        constexpr std::size_t kMaxClosingAttempts = 10'000'000;
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int s = 0; s < samples; ++s) {
            const auto r = static_cast<std::size_t>(s);
            try {
                for (std::size_t t = 0; paths_[r].is_open(); ++t) {
                    if (t == kMaxClosingAttempts)
                        throw std::runtime_error("the worm did not close; lower C");
                    moves_[r][worm_of_particle_[paths_[r].head().particle]]->attempt(
                        paths_[r], actions_[r], rngs_[r]);
                }
            } catch (...) {
                replica_errors[r] = std::current_exception();
            }
        }
        rethrow_replica_error(replica_errors, "worm closing");
    }

    // Reduce per-replica estimator clones (fixed order → thread-count
    // independent, same as the energy reduction).
    ests_.clear();
    if (!est_templates_.empty()) {
        for (std::size_t e = 0; e < replica_ests[0].size(); ++e) {
            for (int s = 1; s < samples; ++s)
                replica_ests[0][e]->collect(*replica_ests[static_cast<std::size_t>(s)][e]);
            ests_.push_back(std::move(replica_ests[0][e]));
        }
    }

    // ── Reduce ──────────────────────────────────────────────────────────
    // Grand block chains: average the independent replica chains block by
    // block.  The replicas are independent Markov chains of equal length, so
    // their per-block average is a single chain whose temporal correlation
    // structure is intact — one Flyvbjerg–Petersen reblock of it then gives
    // the correlation-corrected error of the grand mean (reblocking across
    // replica boundaries directly would measure correlation that is not
    // there; see tools/reblock.py for the same rule on VMC thread chains).
    std::vector<Block> grand(n_blocks, Block{});
    if (samples > 0 && n_acc > 0) {
        for (std::size_t r = 0; r < n_replicas; ++r)
            for (std::size_t b = 0; b < n_blocks; ++b) {
                const Block& src = sample_blocks[r * n_blocks + b];
                Block& dst = grand[b];
                for (std::size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
            }
        const double inv = 1.0 / static_cast<double>(samples);
        for (Block& g : grand)
            for (std::size_t i = 0; i < g.size(); ++i) g[i] *= inv;
    }

    block_energies_.resize(grand.size());
    block_kinetic_.resize(grand.size());
    block_potential_.resize(grand.size());
    block_winding2_.resize(grand.size());
    block_sign_.resize(grand.size());
    block_closed_.resize(worm_ ? grand.size() : 0);
    if (cfg_.compare_thermodynamic) block_energy_difference_.resize(grand.size());
    for (std::size_t b = 0; b < grand.size(); ++b) {
        block_energies_[b]  = grand[b][0];
        block_kinetic_[b]   = grand[b][1];
        block_potential_[b] = grand[b][2];
        block_winding2_[b]  = grand[b][3];
        block_sign_[b]      = grand[b][5];
        if (worm_) block_closed_[b] = grand[b][6];
        if (cfg_.compare_thermodynamic) block_energy_difference_[b]=grand[b][4];
    }
    if (cfg_.compare_thermodynamic)
        energy_difference_stats_=numerics::reblock(block_energy_difference_);

    // Sampled cycle-length fractions per class (|weight| distribution).
    cycle_fractions_.clear();
    if (quantum_) {
        std::size_t measurements = 0;
        for (const auto count : replica_measurements) measurements += count;
        for (std::size_t c = 0; c < cfg_.exchange_classes.size(); ++c) {
            const std::size_t size = class_offset[c + 1] - class_offset[c];
            std::vector<double> fractions(size, 0.0);
            for (std::size_t r = 0; r < n_replicas; ++r)
                for (std::size_t l = 0; l < size; ++l)
                    fractions[l] += replica_cycles[r][class_offset[c] + l];
            if (measurements > 0)
                for (double& f : fractions)
                    f /= static_cast<double>(measurements * size);
            cycle_fractions_.push_back(std::move(fractions));
        }
    }

    if (weighted_) {
        finalize_signed();
    } else {
        finalize_unsigned();
    }
    finalize_moves();
}

void Pimc::finalize_signed() {
    // <O s>/<s> per observable: plain ratio of means, jackknife error over
    // decorrelated coarse blocks (numerics::ratio_of_means).
    signed_stats_[0] = numerics::ratio_of_means(block_energies_, block_sign_);
    signed_stats_[1] = numerics::ratio_of_means(block_kinetic_, block_sign_);
    signed_stats_[2] = numerics::ratio_of_means(block_potential_, block_sign_);
    signed_stats_[3] = numerics::ratio_of_means(block_winding2_, block_sign_);
    energy_stats_    = signed_stats_[0].numerator;  // block-level diagnostics
    kinetic_stats_   = signed_stats_[1].numerator;
    potential_stats_ = signed_stats_[2].numerator;
    winding_stats_   = signed_stats_[3].numerator;
    energy_          = signed_stats_[0].mean;
    energy_error_    = signed_stats_[0].error;
    naive_error_     = std::numeric_limits<double>::quiet_NaN();  // no ratio analogue
    tau_int_         = std::max(signed_stats_[0].numerator.tau_int,
                                signed_stats_[0].denominator.tau_int);
    plateau_found_   = signed_stats_[0].reliable;
    kinetic_         = signed_stats_[1].mean;
    kinetic_error_   = signed_stats_[1].error;
    potential_       = signed_stats_[2].mean;
    potential_error_ = signed_stats_[2].error;
    winding2_        = signed_stats_[3].mean;
    winding2_error_  = signed_stats_[3].error;
    if (cfg_.compare_thermodynamic) {
        // The paired virial-minus-thermodynamic difference is a signed ratio too.
        signed_stats_[4] = numerics::ratio_of_means(block_energy_difference_, block_sign_);
        energy_difference_stats_ = signed_stats_[4].numerator;
        energy_difference_stats_.mean = signed_stats_[4].mean;
        energy_difference_stats_.error = signed_stats_[4].error;
    }
    sign_stats_ = numerics::reblock(block_sign_);
    average_sign_ = sign_stats_.mean;
    average_sign_error_ = sign_stats_.error;
    if (worm_) {
        // <s> over closed paths: sum s [closed] / sum [closed].
        const auto sign_ratio = numerics::ratio_of_means(block_sign_, block_closed_);
        average_sign_ = sign_ratio.mean;
        average_sign_error_ = sign_ratio.error;
        const auto closed = numerics::reblock(block_closed_);
        closed_fraction_ = closed.mean;
        closed_fraction_error_ = closed.error;
    }
    // |<s>| must be resolved: otherwise the ratio divides by noise.
    sign_unreliable_ = !(std::abs(sign_stats_.mean) >=
                         cfg_.sign_threshold * sign_stats_.error) ||
                       !std::isfinite(energy_);
    if (sign_unreliable_)
        io::Logger::warn("pimc: the average sign " + std::to_string(average_sign_) +
                         " +/- " + std::to_string(average_sign_error_) +
                         " is not resolved; the signed energy is unreliable");
}

void Pimc::finalize_unsigned() {
    const numerics::ReblockResult rb = numerics::reblock(block_energies_);
    energy_stats_ = rb;
    energy_        = rb.mean;
    energy_error_  = rb.error;
    naive_error_   = rb.naive_error;
    tau_int_       = rb.tau_int;
    plateau_found_ = rb.plateau_found;
    if (!rb.plateau_found && block_energies_.size() >= 4)
        io::Logger::warn(
            "pimc: reblocking found no error plateau — the reported error "
            "may be an underestimate; run more accumulation blocks");

    const numerics::ReblockResult rk = numerics::reblock(block_kinetic_);
    kinetic_stats_ = rk;
    kinetic_        = rk.mean;
    kinetic_error_  = rk.error;
    const numerics::ReblockResult rv = numerics::reblock(block_potential_);
    potential_stats_ = rv;
    potential_        = rv.mean;
    potential_error_  = rv.error;
    const numerics::ReblockResult rw = numerics::reblock(block_winding2_);
    winding_stats_ = rw;
    winding2_        = rw.mean;
    winding2_error_  = rw.error;
}

void Pimc::finalize_moves() {
    std::size_t total_acc = 0, total_prop = 0;
    for (auto a : move_accepts_) total_acc += a;
    for (auto p : move_proposals_) total_prop += p;
    acceptance_ = total_prop > 0
                      ? static_cast<double>(total_acc) /
                            static_cast<double>(total_prop)
                      : 0.0;

    if (const auto rejected = infinite_action_rejections(); rejected > 0)
        io::Logger::warn(
            "pimc: " + std::to_string(rejected) + " of " +
            std::to_string(total_prop) + " accumulation proposals had infinite "
            "action (a domain wall truncates the sampled distribution)");

    io::Logger::milestone(
        "pimc: complete  E=" + std::to_string(energy_) + " +/- " +
        std::to_string(energy_error_) + "  acc=" +
        std::to_string(acceptance_) +
        "  tau_int=" + std::to_string(tau_int_));
}

Action::ThermoMeasure Pimc::measure_energy(Action& action,
                                           PathSet& path) const {
    switch (energy_estimator_) {
    case EnergyEstimator::centroid_virial:
        return action.measure_open_centroid(path);
    case EnergyEstimator::periodic_virial:
        return action.measure_periodic_virial(path);
    case EnergyEstimator::thermodynamic:
        break;
    }
    return action.measure_thermo(path);
}

std::size_t Pimc::infinite_action_rejections() const noexcept {
    std::size_t total = 0;
    for (const auto count : move_infinite_rejections_) total += count;
    return total;
}

std::string Pimc::summary() const {
    std::ostringstream ss;
    ss << "pimc: E = " << energy_ << " +/- " << energy_error_
       << "  K = " << kinetic_ << " +/- " << kinetic_error_
       << "  V = " << potential_ << " +/- " << potential_error_
       << "  W2 = " << winding2_ << " +/- " << winding2_error_
       << "  acc = " << acceptance_
       << "  tau_int=" << tau_int_;
    if (fixed_node_) {
        ss << "\npimc: identity connectivity inside one nodal pocket (fixed_node); a fixed-node "
              "ground-state estimate only as beta -> infinity; finite-tau bias is NOT included in error bars";
    } else if (!quantum_) {
        ss << "\npimc: identity connectivity (boltzmannon); finite-tau bias is NOT included in error bars";
    } else {
        ss << "\npimc: exchange classes kept (" << statistics_label(cfg_.exchange_classes) << ")";
        if (cfg_.fixed_permutation_sector)
            ss << "; FIXED permutation sector of the starting paths (validation hook, not the physical ensemble)";
        ss << "; finite-tau bias is NOT included in error bars";
    }
    ss << "; energy estimator=" << cfg_.energy;
    if (signed_)
        ss << "\npimc: signed estimator <O s>/<s>, <s> = " << average_sign_
           << " +/- " << average_sign_error_;
    if (worm_) {
        ss << "\npimc: worm: diagonal observables on closed paths only, closed fraction "
           << closed_fraction_ << " +/- " << closed_fraction_error_;
        for (std::size_t w = 0; w < worm_moves_.size(); ++w)
            ss << "; " << move_specs_[worm_moves_[w]].group << " C = "
               << worm_constants_[w][0] << " [" << worm_constants_[w][1] << ", "
               << worm_constants_[w][2] << "]";
    }
    if (sign_unreliable_)
        ss << "\npimc: WARNING: the average sign is not resolved (|<s>| < "
           << cfg_.sign_threshold << " sigma); the signed results are unreliable";
    const auto potential_notes=actions_.front().potential_approximation_notes();
    if (!potential_notes.empty()) ss << "\npimc: " << potential_notes;
    if (cell_.is_periodic())
        ss << "; minimum-image sigma/L=" << periodic_width_ratio_;
    const auto effective_blocks = [this](const numerics::ReblockResult& stats) {
        return static_cast<double>(block_energies_.size()) /
               (1.0 + 2.0 * stats.tau_int);
    };
    const bool errors_reliable = weighted_ ?
        signed_stats_[0].reliable && signed_stats_[1].reliable &&
        signed_stats_[2].reliable && signed_stats_[3].reliable :
        energy_stats_.plateau_found && kinetic_stats_.plateau_found &&
        potential_stats_.plateau_found && winding_stats_.plateau_found &&
        energy_stats_.selected_n_blocks >= 16 && kinetic_stats_.selected_n_blocks >= 16 &&
        potential_stats_.selected_n_blocks >= 16 && winding_stats_.selected_n_blocks >= 16 &&
        effective_blocks(energy_stats_) >= 16 && effective_blocks(kinetic_stats_) >= 16 &&
        effective_blocks(potential_stats_) >= 16 && effective_blocks(winding_stats_) >= 16;
    if (!errors_reliable)
        ss << "\npimc: WARNING: inadequate reblocking plateau/effective blocks for one or more observables; increase accumulation_blocks";
    if (cfg_.equilibration_blocks == 0)
        ss << "\npimc: WARNING: no equilibration; collapsed initial rings need thermalization";
    if (const auto rejected = infinite_action_rejections(); rejected > 0) {
        std::size_t proposals = 0;
        for (const auto count : move_proposals_) proposals += count;
        if (fixed_node_)
            // The restriction is the intended domain: its rejections are exact.
            // The other components' walls are not told apart from it here.
            ss << "\npimc: " << rejected << " of " << proposals
               << " accumulation proposals were rejected for infinite action: "
                  "crossings of the fixed-node restriction (the sampled domain), "
                  "plus any domain wall of another action component";
        else
            ss << "\npimc: WARNING: " << rejected << " of " << proposals
               << " accumulation proposals were rejected for infinite action "
                  "(e.g. a pair outside the pair_product extent): a domain wall "
                  "truncates the sampled distribution; enlarge the domain or "
                  "check the potential";
    }
    ss << "\npimc: error diagnostics do not establish equilibration; compare independent starts and longer equilibration";
    if (!ests_.empty()) {
        RunResult observables;
        for (const auto& est : ests_) est->finalize(observables);
        for (const auto& [name, values] : observables.series)
            if (name.ends_with("_error_reliable") &&
                std::any_of(values.begin(), values.end(), [](double v){ return v == 0; })) {
                ss << "\npimc: WARNING: some optional-observable error bars have no reliable plateau";
                break;
            }
    }
    return ss.str();
}

RunResult Pimc::run_result() const {
    RunResult r;
    r.method = "pimc";
    const auto add_diagnostics = [&](const std::string& name, const numerics::ReblockResult& stats) {
        const double effective = static_cast<double>(block_energies_.size()) /
                                 (1.0 + 2.0 * stats.tau_int);
        r.add_scalar(name + "_effective_blocks", effective);
        r.add_scalar(name + "_tau_int", stats.tau_int);
        r.add_scalar(name + "_error_naive", stats.naive_error);
        r.add_scalar(name + "_reblock_plateau", stats.plateau_found ? 1 : 0);
        r.add_scalar(name + "_error_reliable", stats.plateau_found &&
            stats.selected_n_blocks >= 16 && effective >= 16 ? 1 : 0);
        r.add_scalar(name + "_reblock_level", static_cast<double>(stats.selected_level));
        r.add_scalar(name + "_reblock_n_blocks", static_cast<double>(stats.selected_n_blocks));
        r.add_scalar(name + "_reblock_onset_level", static_cast<double>(stats.level));
        r.add_scalar(name + "_error_on_error", stats.error_on_error);
    };
    // A signed observable is a ratio: its diagnostics are the jackknife's.
    const auto add_ratio_diagnostics = [&](const std::string& name,
                                           const numerics::RatioResult& stats) {
        r.add_scalar(name + "_error_reliable", stats.reliable ? 1 : 0);
        r.add_scalar(name + "_jackknife_level", static_cast<double>(stats.level));
        r.add_scalar(name + "_jackknife_n_blocks", static_cast<double>(stats.n_coarse_blocks));
        r.add_scalar(name + "_jackknife_bias", stats.jackknife_bias);
        r.add_scalar(name + "_numerator_tau_int", stats.numerator.tau_int);
        r.add_scalar(name + "_sign_tau_int", stats.denominator.tau_int);
    };
    if (weighted_) {
        add_ratio_diagnostics("energy", signed_stats_[0]);
        add_ratio_diagnostics("kinetic", signed_stats_[1]);
        add_ratio_diagnostics("potential", signed_stats_[2]);
        add_ratio_diagnostics("winding2", signed_stats_[3]);
    } else {
        add_diagnostics("energy", energy_stats_);
        add_diagnostics("kinetic", kinetic_stats_);
        add_diagnostics("potential", potential_stats_);
        add_diagnostics("winding2", winding_stats_);
    }
    r.add_metadata("energy_estimator",cfg_.energy);
    r.add_metadata("energy_comparison",cfg_.compare_thermodynamic ? "paired "+cfg_.energy+"-minus-thermodynamic" : "disabled");
    if (energy_estimator_ == EnergyEstimator::periodic_virial)
        r.add_metadata("virial_geometry","smooth periodic fractional centroid field");
    if (cfg_.compare_thermodynamic) {
        if (weighted_)
            add_ratio_diagnostics("energy_difference", signed_stats_[4]);
        else
            add_diagnostics("energy_difference",energy_difference_stats_);
        r.add_scalar("energy_difference",energy_difference_stats_.mean);
        r.add_scalar("energy_difference_error",energy_difference_stats_.error);
        r.add_series("acc_block_energy_difference",block_energy_difference_);
    }
    if (fixed_node_) {
        r.add_metadata("statistics", "fixed_node");
        r.add_metadata("action_approximation",
                       "finite-tau composite; identity connectivity; fixed-node restriction "
                       "(time-independent nodes, checked at beads)");
    } else if (!quantum_) {
        r.add_metadata("statistics", "boltzmannon");
        r.add_metadata("action_approximation", "finite-tau composite; identity connectivity");
    } else {
        r.add_metadata("statistics", statistics_label(cfg_.exchange_classes));
        r.add_metadata("action_approximation",
                       "finite-tau composite; permuted closing links (p, M-1) -> (pi(p), 0)");
        r.add_metadata("permutation_sampling", cfg_.fixed_permutation_sector
            ? "fixed sector of the starting paths (validation hook; Z_P, not the physical ensemble)"
            : "none");
        r.add_scalar("signed_estimator", signed_ ? 1 : 0);
        r.add_scalar("average_sign", average_sign_);
        r.add_scalar("average_sign_error", average_sign_error_);
        r.add_scalar("sign_unreliable", sign_unreliable_ ? 1 : 0);
        r.add_scalar("sign_threshold", cfg_.sign_threshold);
        if (weighted_) {
            r.add_metadata("signed_series",
                           "acc_block_* hold sign-weighted numerators <O s>_b; "
                           "acc_block_sign holds <s>_b");
            r.add_series("acc_block_sign", block_sign_);
        }
        if (worm_) {
            r.add_metadata("worm_measurement",
                           "diagonal observables on closed paths: weights s [closed]; "
                           "acc_block_sign holds <s [closed]>_b, acc_block_closed <[closed]>_b");
            r.add_series("acc_block_closed", block_closed_);
            r.add_scalar("closed_fraction", closed_fraction_);
            r.add_scalar("closed_fraction_error", closed_fraction_error_);
            for (std::size_t w = 0; w < worm_moves_.size(); ++w) {
                const std::string label = "move_" + std::to_string(worm_moves_[w]) + "_constant";
                r.add_scalar(label + "_mean", worm_constants_[w][0]);
                r.add_scalar(label + "_min", worm_constants_[w][1]);
                r.add_scalar(label + "_max", worm_constants_[w][2]);
            }
        }
        for (std::size_t c = 0; c < cycle_fractions_.size(); ++c) {
            const ExchangeClass& ec = cfg_.exchange_classes[c];
            const std::string label = "cycle_fraction_class_" + std::to_string(c);
            r.add_metadata(label + "_identity", ec.group + " flavour " +
                           std::to_string(ec.flavour) + " " + ec.statistics);
            r.add_series(label, cycle_fractions_[c]);
        }
    }
    const auto potential_notes=actions_.front().potential_approximation_notes();
    if (!potential_notes.empty()) r.add_metadata("potential_approximation",potential_notes);
    r.add_metadata("kinetic_kernel", cell_.is_periodic() ? "minimum-image Gaussian approximation" : "open Gaussian");
    r.add_metadata("uncertainty_scope", "statistical only; excludes time-step, finite-size and equilibration biases");
    r.add_scalar("periodic_sigma_over_shortest_length", periodic_width_ratio_);
    r.add_scalar("periodic_free_link_tail_union_bound", periodic_image_tail_bound_);
    r.add_metadata("periodic_free_link_tail_scope",
        "free Gaussian-link coordinate-tail estimate in orthorhombic cells only; "
        "-1 means unavailable; not an interacting-path or image-bias bound");
    r.add_scalar("energy", energy_);
    r.add_scalar("energy_error", energy_error_);
    r.add_scalar("kinetic", kinetic_);
    r.add_scalar("kinetic_error", kinetic_error_);
    r.add_scalar("potential", potential_);
    r.add_scalar("potential_error", potential_error_);
    r.add_scalar("winding2", winding2_);
    r.add_scalar("winding2_error", winding2_error_);
    r.add_scalar("tau_int", tau_int_);
    r.add_scalar("reblock_plateau", plateau_found_ ? 1.0 : 0.0);
    r.add_scalar("acceptance_ratio", acceptance_);
    r.add_scalar("cm_energy_correction", cm_energy_correction_);
    r.add_scalar("internal_energy", energy_ - cm_energy_correction_);
    r.add_scalar("internal_energy_error", energy_error_);
    std::size_t total_proposals = 0;
    for (std::size_t i = 0; i < move_proposals_.size(); ++i) {
        const std::string label = "move_" + std::to_string(i);
        const PimcConfig::MoveSpec& spec = move_specs_[i];
        r.add_metadata(label + "_name", moves_[0][i]->name());
        // The resolved parameters, so a run that used the default move set
        // records exactly what it sampled.
        r.add_scalar(label + "_weight", spec.weight);
        if (spec.type == "bisection")
            r.add_scalar(label + "_levels", static_cast<double>(spec.levels));
        else if (spec.type == "permutation" || spec.type == "joint_permutation") {
            r.add_scalar(label + "_levels", static_cast<double>(spec.levels));
            r.add_scalar(label + "_max_cycle", static_cast<double>(spec.max_cycle));
            if (spec.type == "joint_permutation") {
                r.add_scalar(label + "_pair_radius", spec.pair_radius);
                r.add_metadata(label + "_pair", spec.pair[0] + "," + spec.pair[1]);
            }
        } else
            r.add_scalar(label + "_step", spec.step);
        if (i < move_extra_.size())
            for (const auto& [name, value] : move_extra_[i])
                r.add_scalar(label + "_" + name, value);
        r.add_scalar(label + "_proposals", static_cast<double>(move_proposals_[i]));
        r.add_scalar(label + "_accepts", static_cast<double>(move_accepts_[i]));
        r.add_scalar(label + "_acceptance_ratio", move_proposals_[i] == 0 ? 0.0 :
            static_cast<double>(move_accepts_[i]) / static_cast<double>(move_proposals_[i]));
        r.add_scalar(label + "_infinite_action_rejections",
                     static_cast<double>(move_infinite_rejections_[i]));
        total_proposals += move_proposals_[i];
    }
    const auto rejected = infinite_action_rejections();
    r.add_scalar("infinite_action_rejections", static_cast<double>(rejected));
    r.add_scalar("infinite_action_rejection_fraction",
                 total_proposals == 0 ? 0.0 :
                     static_cast<double>(rejected) /
                         static_cast<double>(total_proposals));
    r.add_scalar("beta", cfg_.beta);
    r.add_scalar("tau", cfg_.time_step);
    r.add_scalar("M", static_cast<double>(M_));
    r.add_scalar("n_samples", static_cast<double>(cfg_.samples));
    r.add_scalar("n_accumulation_blocks",
                 static_cast<double>(cfg_.accumulation_blocks));
    r.add_scalar("n_equilibration_blocks",
                 static_cast<double>(cfg_.equilibration_blocks));
    r.add_metadata("seed", std::to_string(cfg_.seed));
    r.add_metadata("rng_streams",
                   "mt19937_64 per replica s, seeded with "
                   "splitmix64-hashed (seed, replica stream, s)");
    r.add_series("acc_block_energy", block_energies_);
    r.add_series("acc_block_kinetic", block_kinetic_);
    r.add_series("acc_block_potential", block_potential_);
    r.add_series("acc_block_winding2", block_winding2_);
    for (const auto& e : ests_) e->write_results(r);
    r.check_unique_keys();
    return r;
}

}  // namespace alea
