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
#include <exception>
#include <iosfwd>
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

namespace detail {
// Rethrow the lowest-index replica failure captured inside a parallel region,
// naming the replica and the phase (pimc.cpp).
void rethrow_replica_error(std::span<const std::exception_ptr> errors, const char* phase);
}  // namespace detail

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

// A grand-canonical stage: a worm move with a `grand_canonical` block
// (Boninsegni, Prokof'ev & Svistunov, Phys. Rev. E 74, 036701 (2006)).  The
// driver is then built for the capacity n_max (groups, Hamiltonian and the
// worm's action of n_max particles, whose bead energies take every
// N-dependent constant at the beads present on a slice), and the caller adds
// the closed-path physics of every sector N in [n_min, n_max]: an Action built
// with the group range [0, N), so that the springs' d N/(2 tau), the
// inverse_power tail and every other N-dependent constant are those of N.
// The N = 0 entry is an empty Action, never used (sector 0 has no particle to
// move or measure).
struct PimcGrandCanonicalSectors {
    std::size_t initial_particles = 0;  // the closed N every replica starts with
    std::vector<Action> actions;        // [N - n_min]
};

// Closed-path results of one sector N of a grand-canonical run.  Every value
// is a ratio estimator over the accumulation blocks, sum_b O_b / sum_b w_b
// with w_b the block's closed measurements in sector N, with a jackknife
// error (numerics::ratio_of_means); a block that never visits N has w_b = 0.
struct PimcSectorResult {
    std::size_t n = 0;
    double fraction = 0.0, fraction_error = 0.0;  // P(N) among closed measurements
    double measurements = 0.0;                    // closed measurements in N, all replicas
    double energy = 0.0, energy_error = 0.0;
    double kinetic = 0.0, kinetic_error = 0.0;
    double potential = 0.0, potential_error = 0.0;
    double winding2 = 0.0, winding2_error = 0.0;  // per (particle, direction)
    std::array<double, kMaxWindingTensor> winding_tensor{}, winding_tensor_error{};
};

struct PimcGrandCanonicalResult {
    double chemical_potential_input = 0.0;
    std::size_t n_min = 0, n_max = 0;
    std::vector<PimcSectorResult> sectors;  // [N - n_min]
    // F(N + 1) - F(N) = mu_in - T ln[P(N + 1)/P(N)], the canonical chemical
    // potential at N + 1/2, [N - n_min] for N in [n_min, n_max - 1]; NaN
    // where a sector was never visited.
    std::vector<double> mu, mu_error;
    // Closed N -> N +- 1 changes between N and N + 1, summed over replicas,
    // [N - n_min]; and all changes of each replica, [replica].
    std::vector<double> pair_transitions, replica_transitions;
    double n_mean = 0.0, n_mean_error = 0.0;    // <N> over closed measurements
    double n2_mean = 0.0, n2_mean_error = 0.0;  // <N^2>
    double n_tau_int = 0.0;                     // of the block series of <N>, in blocks
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
    // `sectors` is required exactly when the move set holds a grand-canonical
    // worm; groups, np, ham and built_action then describe the capacity n_max.
    Pimc(const PimcConfig& cfg,
         std::span<const ParticleGroup> groups,
         const Hamiltonian& ham,
         const SimulationCell& cell,
         std::size_t dim,
         std::size_t np,
         WalkerEnsemble* ensemble = nullptr,
         const Action* built_action = nullptr,
         const PimcGrandCanonicalSectors* sectors = nullptr);

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

    // Grand-canonical run: the estimator set of every sector N, [N - n_min],
    // each built with the group range [0, N) (empty for N = 0).  A sector's
    // estimators bin the closed measurements in N and count every other
    // measurement with weight 0, so each output is a ratio estimator of that
    // sector; its result keys carry the label "N<n>".
    void set_sector_estimators(std::vector<std::vector<std::unique_ptr<Estimator>>> per_sector);
    bool grand_canonical() const noexcept { return gc_; }
    // After run(); empty unless grand_canonical().
    const PimcGrandCanonicalResult& grand_canonical_result() const noexcept { return gc_result_; }

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

    // ── Grand-canonical stage (pimc_grand_canonical.cpp) ──
    bool gc_ = false;
    std::size_t gc_n_min_ = 0, gc_n_max_ = 0, gc_initial_ = 0;
    std::vector<double> gc_start_n_;  // [replica]: N at the start of run()
    double gc_mu_ = 0.0;
    std::vector<std::vector<Action>> sector_actions_;  // [replica][N - n_min]
    // Estimator templates and reduced results, [N - n_min][estimator].
    std::vector<std::vector<std::unique_ptr<Estimator>>> sector_est_templates_, sector_ests_;
    PimcGrandCanonicalResult gc_result_;
    // Grand-block series [N - n_min][block]: closed measurements in N and the
    // sums of E, K, V, W^2 and the winding tensor over them, each divided by
    // the block's measurement count (field kSector*).
    static constexpr std::size_t kSectorWeight = 0, kSectorEnergy = 1, kSectorKinetic = 2,
                                 kSectorPotential = 3, kSectorWinding2 = 4, kSectorTensor = 5,
                                 kSectorFields = 5 + kMaxWindingTensor;
    std::vector<std::array<std::vector<double>, kSectorFields>> gc_sector_blocks_;
    // Aggregate grand-block series: N [closed], N^2 [closed].
    std::vector<double> block_gc_n_, block_gc_n2_;
    // Validate the stage and clone the sector actions (constructor).
    void setup_grand_canonical(const PimcGrandCanonicalSectors& sectors, const Action& base_action);
    // run() of a grand-canonical stage, from the initial rings to finalize.
    void run_grand_canonical();
    void finalize_grand_canonical();
    void write_grand_canonical(RunResult& r) const;
    void summarize_grand_canonical(std::ostringstream& ss) const;

    // Resolve (YAML or default) and validate the move set; fills move_specs_.
    std::vector<std::unique_ptr<Move>> build_moves();
    // The configured energy estimator measured on one replica.
    [[nodiscard]] Action::ThermoMeasure measure_energy(Action& action,
                                                       PathSet& path) const;
};

}  // namespace alea
