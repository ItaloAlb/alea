// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Pimc — the PIMC driver.  Distinguishable particles (boltzmannons) sample
// identity connectivity.  Exchange classes (Ceperley, Rev. Mod. Phys. 67, 279
// (1995)) are carried by every replica path, with permuted closing links and signed
// estimators <O s>/<s>.  Quantum statistics need a `permutation` move (or the
// fixed-sector validation hook, which samples one permutation sector only).
//
// Parallelized over independent replica paths (samples), one per thread.
// Each replica has its own PathSet, Action clone, Move clones, and RNG.
//
// Lifecycle:
//   1. Construct with config, groups, hamiltonian, cell.
//   2. run() — equilibration + accumulation; measures the selected energy.
//   3. summary() / run_result() — retrieve results with error bars.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "io/parser.h"
#include "io/path_io.h"
#include "pimc/winding.h"
#include "pimc/action.h"
#include "pimc/move.h"
#include "pimc/path_set.h"
#include "estimators/estimator.h"
#include "numerics/random.h"
#include "numerics/statistics.h"
#include "population/exchange_class.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker_ensemble.h"
#include "qmc/driver.h"
#include "qmc/run_result.h"

namespace alea {

// Stream kinds for numerics::stream_seed(seed, kind, index).  Every random
// stream of a PIMC stage is a hash of (stage seed, kind, index), so replica,
// ring-initialization and ensemble-initialization streams never coincide with
// each other, with another stage's, or with VMC/DMC's base + thread streams.
inline constexpr std::uint64_t kPimcReplicaStream = 0x70696d632d726570ULL;  // "pimc-rep"
inline constexpr std::uint64_t kPimcRingInitStream = 0x70696d632d72696eULL; // "pimc-rin"
inline constexpr std::uint64_t kPimcEnsembleInitStream = 0x70696d632d656e73ULL; // "pimc-ens"

struct PimcConfig {
    int           threads = 1;
    int           samples = 64;          // independent replica paths
    double        time_step = 0.125;     // tau
    double        beta = 1.0;            // inverse temperature
    int           equilibration_blocks = 100;
    int           accumulation_blocks = 500;
    int           steps_per_block = 50;
    int           sub_steps = 1;
    // Stage seed: replica s samples with stream_seed(seed, kPimcReplicaStream, s).
    std::uint64_t seed = 42;
    std::string energy = "thermodynamic";
    bool compare_thermodynamic = false; // paired E_virial-E_thermo on the same paths

    // Statistics gate: the largest same-group/same-flavour exchange class.  A
    // class with >1 particle keeps quantum statistics, which need a move that
    // changes the permutation, unless the YAML says `statistics: boltzmannon`;
    // otherwise the driver refuses.  `fixed_node` keeps every path in one
    // nodal pocket of the action's `nodes` restriction, identity permutation.
    std::string statistics;
    int         exchange_class_max = 1;
    // The (group, flavour) classes of fermions/bosons (build_exchange_classes);
    // required when the run keeps quantum statistics.
    std::vector<ExchangeClass> exchange_classes;
    // VALIDATION HOOK, not a YAML key: keep quantum statistics with no move
    // that changes the permutation, so the run samples only the permutation
    // sector of its starting (or imported) paths.  That is the fixed-sector
    // partition function Z_P, not the physical bosonic/fermionic ensemble.
    bool fixed_permutation_sector = false;
    // A signed run flags its result when |<s>| < sign_threshold * sigma_s.
    double sign_threshold = 5.0;

    // Move specs (deserialized from YAML); empty → io::default_pimc_moves(M).
    using MoveSpec = io::PimcMoveSpec;
    std::vector<MoveSpec> moves;
};

class Pimc : public Driver {
public:
    // `ensemble` supplies the initial configuration of each replica: replica
    // s collapses onto walker (s mod nw) — the design's "collapsed ring from
    // the WalkerEnsemble", which makes the VMC -> PIMC carry-over work for
    // free (the caller seeds the ensemble before run()).  Null → each replica
    // starts from its own random ring (used by unit tests).
    // `built_action` is the composite from build_action (the YAML `action:`
    // list); null → the default primitive kinetic + potential action of `ham`.
    // run() throws (after the parallel region, naming the replica) when a
    // replica fails, e.g. a component returning a NaN action change.
    Pimc(const PimcConfig& cfg,
         std::span<const ParticleGroup> groups,
         const Hamiltonian& ham,
         const SimulationCell& cell,
         std::size_t dim,
         std::size_t np,
         WalkerEnsemble* ensemble = nullptr,
         const Action* built_action = nullptr);

    void run() override;

    // Explicit full-path checkpoint (separate from walker configurations).
    void import_paths(const io::PathCheckpoint& checkpoint);
    [[nodiscard]] io::PathCheckpoint export_paths() const;

    [[nodiscard]] std::string summary() const override;
    [[nodiscard]] RunResult   run_result() const override;

    // Run is a no-op until it is called; initial configurations are read from
    // the ensemble INSIDE run() (after any carry-over), so the Driver-level
    // reload is intentionally empty.
    void reload_configurations() override {}

    double energy()       const noexcept { return energy_; }
    double energy_error() const noexcept { return energy_error_; }
    double kinetic()      const noexcept { return kinetic_; }
    double kinetic_error() const noexcept { return kinetic_error_; }
    double potential()    const noexcept { return potential_; }
    double potential_error() const noexcept { return potential_error_; }
    // <W^2>: mean over particles and lattice directions of the squared ring
    // winding number (sum of min-image link vectors / lattice vector).  Zero
    // for open cells.  Exact free-particle value at L=4, beta=2, m=1:
    // 0.03533770482397546 per (particle, direction) — the second, independent
    // physics check that the winding sectors are actually sampled.
    double winding2()      const noexcept { return winding2_; }
    double winding2_error() const noexcept { return winding2_error_; }
    // Cartesian winding tensor per particle, T_ab = <sum_loops W_a W_b>/N
    // (bohr^2), components c = 0..winding_tensor_size(dim)-1 ordered xx, xy,
    // yy, xz, yz, zz (pimc/winding.h).  Valid in any cell shape: rho_s,ab/rho
    // = m T_ab / beta for one species of mass m.  Zero for open cells.
    std::size_t winding_tensor_components() const noexcept { return winding_tensor_size(dim_); }
    double winding_tensor(std::size_t c)       const noexcept { return winding_tensor_[c]; }
    double winding_tensor_error(std::size_t c) const noexcept { return winding_tensor_error_[c]; }
    std::span<const double> block_winding_tensor(std::size_t c) const noexcept {
        return block_winding_tensor_[c];
    }
    double naive_error()  const noexcept { return naive_error_; }
    double tau_int()      const noexcept { return tau_int_; }
    bool   plateau_found() const noexcept { return plateau_found_; }
    double acceptance_ratio() const noexcept { return acceptance_; }
    // Quantum statistics: identical particles keep their exchange classes.
    bool quantum_statistics() const noexcept { return quantum_; }
    // Signed run (a fermionic class of >1 particle): every observable above is
    // the ratio <O s>/<s> with a jackknife error (numerics::ratio_of_means).
    bool signed_run() const noexcept { return signed_; }
    // Worm run (a `worm` move): diagonal observables are measured on closed
    // paths only (weight s [closed]), so every result is a ratio of means and
    // average_sign() is <s> over closed paths.
    bool worm_run() const noexcept { return worm_; }
    // statistics: fixed_node — identity paths restricted to one nodal pocket.
    bool fixed_node_run() const noexcept { return fixed_node_; }
    // Fraction of measurements on closed paths and its reblocked error.
    double closed_fraction() const noexcept { return closed_fraction_; }
    double closed_fraction_error() const noexcept { return closed_fraction_error_; }
    // <s> and its reblocked error; exactly 1 for unsigned runs.
    double average_sign() const noexcept { return average_sign_; }
    double average_sign_error() const noexcept { return average_sign_error_; }
    // |<s>| < sign_threshold * sigma_s: the ratio is unresolved (flagged, not
    // refused — a noisy ratio is still the right estimator).
    bool sign_unreliable() const noexcept { return sign_unreliable_; }
    std::span<const double> block_sign() const noexcept { return block_sign_; }
    // Sampled fraction of class-c particles in cycles of length L = 1..size_c,
    // [class][L-1]; empty unless quantum statistics.  This is the |weight|
    // distribution the chain visits (for fermions, not sign-weighted).
    const std::vector<std::vector<double>>& cycle_fractions() const noexcept {
        return cycle_fractions_;
    }
    // Accumulation-phase proposals rejected for infinite proposed action
    // (Move::infinite_action_rejections summed over replicas and moves).
    std::size_t infinite_action_rejections() const noexcept;
    // The move set actually sampled (YAML moves or the resolved defaults).
    std::span<const PimcConfig::MoveSpec> move_specs() const noexcept {
        return move_specs_;
    }
    std::span<const double> block_energies() const noexcept {
        return block_energies_;
    }
    std::span<const double> block_kinetic() const noexcept {
        return block_kinetic_;
    }
    std::span<const double> block_potential() const noexcept {
        return block_potential_;
    }
    std::span<const double> block_winding2() const noexcept {
        return block_winding2_;
    }

    // Opt-in observable estimators (g(r), density, ...).  Cloned per replica
    // in run() and reduced at the end, mirroring VMC/DMC.
    void set_estimators(std::vector<std::unique_ptr<Estimator>> ests);

private:
    // Energy estimator, resolved once from cfg_.energy (no string dispatch in
    // the measurement loop).
    enum class EnergyEstimator { thermodynamic, centroid_virial, periodic_virial };

    PimcConfig cfg_;
    EnergyEstimator energy_estimator_ = EnergyEstimator::thermodynamic;
    bool quantum_ = false;  // exchange classes kept (statistics not boltzmannon)
    bool signed_ = false;   // some fermionic class has more than one particle
    bool worm_ = false;     // a worm move is in the move set
    bool fixed_node_ = false;  // statistics: fixed_node (identity, restricted)
    bool weighted_ = false; // signed_ || worm_: ratio-of-means measurement
    // Worm moves: indices in the move list, and the worm owning each particle
    // ([np], npos when none) — the move the driver calls on an open path.
    std::vector<std::size_t> worm_moves_, worm_of_particle_;
    // [estimator]: the worm move whose open paths an off-diagonal estimator
    // measures (npos for diagonal estimators).
    std::vector<std::size_t> est_worm_;
    // PathSet exchange labels [np]: class id (singletons keep their own index)
    // and fermion flags; class_index_ [np] maps a particle to its class in
    // cfg_.exchange_classes, or npos for a distinguishable particle.
    std::vector<std::size_t> class_id_, class_index_;
    std::vector<char> fermionic_;
    // Validate a permutation for this run before it reaches any replica.
    void check_permutation(std::span<const std::size_t> pi) const;
    // End-of-run reductions of the block series (run() tail).
    void finalize_signed();
    void finalize_unsigned();
    void finalize_moves();
    std::vector<PimcConfig::MoveSpec> move_specs_; // resolved, [move index]
    std::vector<double> block_energy_difference_; // [accumulation_block], replica means
    numerics::ReblockResult energy_difference_stats_;
    std::size_t M_;           // number of slices = beta / tau
    std::size_t dim_;
    std::size_t np_;
    SimulationCell cell_;
    std::vector<ParticleGroup> groups_;
    WalkerEnsemble* ensemble_ = nullptr;  // initial-config source (not owned)

    // Per-replica state (one PathSet/Action/move set/RNG per sample; OpenMP
    // distributes replicas over threads, so results never depend on how).
    std::vector<PathSet>               paths_;
    std::vector<Action>                actions_;
    std::vector<std::vector<std::unique_ptr<Move>>> moves_;
    std::vector<numerics::Random>      rngs_;
    bool imported_paths_ = false;

    // Observable estimator templates (set via set_estimators) and the final
    // reduced results after run().
    std::vector<std::unique_ptr<Estimator>> est_templates_;
    std::vector<std::unique_ptr<Estimator>> ests_;

    // Results (all errors from Flyvbjerg–Petersen reblocking).
    double energy_        = 0.0;
    double energy_error_  = 0.0;
    double kinetic_       = 0.0;
    double kinetic_error_ = 0.0;
    double potential_     = 0.0;
    double potential_error_ = 0.0;
    double winding2_      = 0.0;   // <W^2> per (particle, direction); 0 when open
    double winding2_error_ = 0.0;
    // Cartesian winding tensor (pimc/winding.h), per component [c]; only the
    // first winding_tensor_size(dim_) entries are used.
    std::array<double, kMaxWindingTensor> winding_tensor_{};
    std::array<double, kMaxWindingTensor> winding_tensor_error_{};
    std::array<std::vector<double>, kMaxWindingTensor> block_winding_tensor_;  // [c][block]
    std::array<numerics::ReblockResult, kMaxWindingTensor> winding_tensor_stats_{};
    std::array<numerics::RatioResult, kMaxWindingTensor> signed_tensor_stats_{};
    double naive_error_   = 0.0;  // SEM assuming independent blocks
    double tau_int_       = 0.0;  // integrated autocorrelation time (blocks)
    bool   plateau_found_ = true;
    double acceptance_    = 0.0;
    double cm_energy_correction_ = 0.0;
    std::vector<std::size_t> move_proposals_, move_accepts_; // [move index]
    std::vector<std::size_t> move_infinite_rejections_;      // [move index]
    // Move-specific counters summed over replicas, [move index][counter].
    std::vector<std::vector<std::pair<std::string, double>>> move_extra_;
    double periodic_width_ratio_ = 0.0;
    double periodic_image_tail_bound_ = -1.0;  // unavailable for skew cells
    numerics::ReblockResult energy_stats_, kinetic_stats_, potential_stats_, winding_stats_;
    std::vector<double> block_energies_;   // grand chain, per block
    std::vector<double> block_kinetic_;
    std::vector<double> block_potential_;
    std::vector<double> block_winding2_;
    // Signed runs: the block series above hold sign-weighted numerators
    // <O s>_b; block_sign_ holds <s>_b (all ones for unsigned runs).
    std::vector<double> block_sign_;
    // Worm runs: fraction of closed-path measurements per block.
    std::vector<double> block_closed_;
    std::array<numerics::RatioResult, 5> signed_stats_{}; // E, K, V, W2, dE
    numerics::ReblockResult sign_stats_;
    double average_sign_ = 1.0;
    double average_sign_error_ = 0.0;
    bool sign_unreliable_ = false;
    double closed_fraction_ = 1.0, closed_fraction_error_ = 0.0;
    // Worm constant C per worm move after equilibration: mean, min, max over
    // replicas, [worm][3].
    std::vector<std::array<double, 3>> worm_constants_;
    std::vector<std::vector<double>> cycle_fractions_; // [class][L-1]

    // Resolve (YAML or default) and validate the move set; fills move_specs_.
    std::vector<std::unique_ptr<Move>> build_moves();
    // The configured energy estimator measured on one replica.
    [[nodiscard]] Action::ThermoMeasure measure_energy(Action& action,
                                                       PathSet& path) const;
};

}  // namespace alea
