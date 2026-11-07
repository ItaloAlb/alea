// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "../estimators/estimator.h"
#include "../numerics/xoshiro_random.h"
#include "../numerics/statistics.h"
#include "../population/walker_set.h"
#include "../wavefunction/trial_wavefunction.h"
#include "../hamiltonian/hamiltonian.h"
#include "driver.h"
#include "population_control.h"

namespace alea {

class WalkerEnsemble;

// Stream kinds for numerics::stream_seed(seed, kind, index): worker t sweeps
// with stream_seed(seed, kDmcThreadStream, t) and branching draws from
// stream_seed(seed, kDmcBranchStream, 0).  Hashing keeps every worker of every
// seed on its own stream; seed + t would hand seed s + 1 the streams of seed s,
// shifted by one worker.
inline constexpr std::uint64_t kDmcThreadStream = 0x646d632d74687264ULL;  // "dmc-thrd"
inline constexpr std::uint64_t kDmcBranchStream = 0x646d632d62726368ULL;  // "dmc-brch"
// The estimators that evaluate ψ_T (Estimator::bind_wavefunction) draw from
// stream_seed(seed, kDmcEstimatorStream, 0).
inline constexpr std::uint64_t kDmcEstimatorStream = 0x646d632d6573746dULL;  // "dmc-estm"

struct DMCConfig {
    int           n_equilibration_blocks = 100;
    int           n_accumulation_blocks  = 100;
    int           n_steps                = 10;
    int           n_substeps             = 1;
    double        tau                    = 0.01;
    int           n_threads              = 1;
    std::uint64_t seed                   = 42;  // stage seed (kDmc*Stream)

    // DMC-specific
    double population_generations = 10.0;  // g: E_T feedback gain 1/g per step
    // Branch-weight cutoff (population_control.h): scheme and α of `universal`.
    BranchCutoff branch_cutoff       = BranchCutoff::universal;
    double       branch_cutoff_alpha = kDefaultBranchCutoffAlpha;
    // Local energy in the branch weight (population_control.h, Reweighting):
    // suppressed needs κ and the correlation time θ of E_L (imaginary time;
    // 0 → measured at the start of run(), at a step that resolves it).
    Reweighting reweighting       = Reweighting::bare;
    double      suppression_kappa = kDefaultSuppressionKappa;
    double      correlation_time  = 0.0;
    // Unweighted DMC-move sampling of θ (Anderson et al., Eq. 15). Discard
    // equilibration first; each chain must span at least ten Sokal windows.
    std::size_t correlation_equilibration_sweeps = 256;
    std::size_t correlation_min_sweeps = 256;
    std::size_t correlation_max_sweeps = 8192;

    // Walkers of a crowd swept together through all particles
    // (sweep_blocking.h); 0 sizes the block from the per-walker state.
    // Never changes results.
    std::size_t sweep_block_walkers = 0;
};

// How DMC keeps the projected state consistent with the trial's sign/phase.
//   None       — ψ_T > 0 everywhere (bosonic/nodeless): nothing to constrain.
//   FixedNode  — real ψ_T with nodes: reject any move whose ratio ψ'/ψ < 0
//                (QMCPACK DMCUpdatePbyPFast's phaseChanged gate).
//   FixedPhase — complex ψ_T = |ψ_T|e^{iφ}: φ is imposed on the solution and
//                DMC projects |Φ| under H + ½Σ|∇φ|²/m, carried by Re E_L
//                (Ortiz, Ceperley & Martin, PRL 71, 2777 (1993)).  No gate:
//                a phase change is not a node crossing (QMCPACK's complex
//                build returns phaseChanged = false).  Drift and acceptance
//                use |ψ_T| only, exactly as for a real trial.
// Fixed-node is fixed-phase with φ ∈ {0, π}, so the choice is dictated by ψ_T
// and is not a user option (forcing FixedPhase on a real ψ would silently let
// walkers cross its nodes).
enum class NodeConstraint { None, FixedNode, FixedPhase };

// The constraint a trial wavefunction implies: complex → FixedPhase, else any
// nodal (sign-changing) component → FixedNode, else None.
[[nodiscard]] NodeConstraint node_constraint(const TrialWavefunction& psi);

[[nodiscard]] const char* to_string(NodeConstraint nc) noexcept;

// Fixed-node gate: does a move whose ratio ψ'/ψ has phase `phase`
// (WalkerSet::phase_diffs, imag log(ψ'/ψ)) cross a node of a real ψ_T?
// A real ratio has phase 0 (same sign) or π (sign flip): cos φ < ε ⇔ crossing.
// QMCPACK's phaseChanged (DMCUpdatePbyPFast.cpp:94).
[[nodiscard]] inline bool crosses_node(double phase) noexcept {
    return std::cos(phase) < std::numeric_limits<double>::epsilon();
}

class DMC : public Driver {
public:
    DMC(const DMCConfig&       cfg,
        WalkerEnsemble&        ensemble,
        const TrialWavefunction& psi,
        const Hamiltonian&       hamiltonian);

    void run() override;

    [[nodiscard]] std::string summary() const override;
    [[nodiscard]] RunResult   run_result() const override;

    // Final branched population (valid after run()); enables walkers_out.
    [[nodiscard]] const WalkerSet* final_population() const override {
        return &pop_;
    }

    // Carry-over: the final branched population (unit weights after the last
    // branch), and a reload of the ensemble's positions into pop_ before run().
    [[nodiscard]] std::vector<double> final_configurations() const override {
        std::vector<double> out;
        pop_.append_configurations(out);
        return out;
    }
    void reload_configurations() override { pop_.load_all(); }

    double energy()           const noexcept { return energy_; }
    double energy_error()     const noexcept { return energy_error_; }
    double energy_variance()  const noexcept { return energy_variance_; }
    double acceptance_ratio() const noexcept { return acceptance_ratio_; }
    // Constraint used by the last run() (decided from the trial at run start).
    NodeConstraint node_constraint() const noexcept { return node_constraint_; }

    std::span<const double> block_energies()  const noexcept { return block_energies_; }
    std::span<const double> block_population() const noexcept { return block_population_; }
    std::span<const double> block_etrial()     const noexcept { return block_etrial_; }

    // Callback invoked after each hamiltonian.evaluate_batch() during
    // accumulation.  Receives the per-thread WalkerSet (with local_energies()
    // populated) and the per-thread wavefunction clone.
    using MeasureHook = std::function<void(const WalkerSet&,
                                           const TrialWavefunction&)>;
    void set_measure_hook(MeasureHook hook) { measure_hook_ = std::move(hook); }

    // Milestone walker checkpoints: invoked with the live population after
    // equilibration (tag "eq", blocks_done 0) and after 25/50/75% of the
    // accumulation blocks ("a25"/"a50"/"a75", blocks_done = completed blocks).
    // Fires between blocks, in the sequential section — safe to read pop_.
    // The final population is NOT a hook milestone: main serialises
    // final_population() after run().  DMC stays I/O-free — the closure owns
    // whatever file write happens (see App::build_drivers).
    using CheckpointHook = std::function<void(const WalkerSet&,
                                              std::string_view tag,
                                              int blocks_done)>;
    void set_checkpoint_hook(CheckpointHook hook) {
        checkpoint_hook_ = std::move(hook);
    }

    // Opt-in observable estimators (e.g. g(r)).  In DMC each is a single
    // instance accumulating the global population, weighted by the per-walker
    // DMC weight (the mixed estimator).  Results append to run_result().
    void set_estimators(std::vector<std::unique_ptr<Estimator>> ests) {
        est_templates_ = std::move(ests);
    }

private:
    struct StepResult {
        double      energy_sum = 0.0;   // Σ_w w·E_L over the range
        double      weight_sum = 0.0;   // Σ_w w over the range
        std::size_t n_accepted = 0;     // accepted single-particle moves
        std::size_t n_cut      = 0;     // sweep weights tapered by the cutoff
        // Suppressed reweighting, summed over walker sweeps: τ_eff/τ and f(R').
        double      tau_eff_ratio_sum = 0.0;
        double      suppression_sum   = 0.0;
    };

    DMCConfig cfg_;

    const TrialWavefunction*  psi_template_ = nullptr;
    const Hamiltonian*        ham_template_ = nullptr;

    // E_T / E_ref feedback and the per-sweep branch weight; started from the
    // initial population energy in run().
    PopulationControl population_;

    // One global population processed as crowds (contiguous walker ranges) by
    // n_threads_ workers.  psi_/ham_ are a single shared instance (no per-thread
    // clones): all per-walker state lives in pop_ / the component caches, and
    // crowds touch disjoint ranges, so concurrent kernels are race-free.
    WalkerSet                            pop_;
    TrialWavefunction                    psi_;        // cloned from template in run()
    Hamiltonian                          ham_;        // cloned from template in run()
    std::vector<numerics::XoshiroRandom> rngs_;       // one per worker (sweeps)
    numerics::XoshiroRandom              branch_rng_; // sequential branching draws
    int                                  n_threads_ = 1;

    std::vector<double> tau_inv_mass_;
    std::vector<double> sqrt_tau_inv_mass_;
    std::vector<double> drift_scale_;  // [nw] Umrigar per-particle drift scale s(|∇logψ|²)
    // Reweighting scratch, [capacity] each, indexed by walker: the
    // dimensionless drift u = Σ (τ/m)(∂ ln|Ψ|)² before and after a sweep, and
    // the sums Σ_i p_i ξ_i² and Σ_i ξ_i² of τ_eff (sweep_walkers). Only
    // suppressed allocates the drift norms; effective needs the last two.
    std::vector<double> drift_u_old_;
    std::vector<double> drift_u_new_;
    std::vector<double> tau_eff_num_;
    std::vector<double> tau_eff_den_;
    // x = suppression_scale_·τ_eff·√u, suppression_scale_ = κ/√(N·τ·θ).
    double suppression_scale_ = 0.0;
    // θ (imaginary time) and how it was obtained: the step τ_θ of the chains,
    // Σρ(k) in sweeps of it, the sweeps measured (all 0 when the config gave
    // θ), and whether the window converged.
    double      correlation_time_     = 0.0;
    double      correlation_step_     = 0.0;
    double      correlation_tau_int_  = 0.0;
    std::size_t correlation_sweeps_   = 0;
    std::size_t correlation_window_   = 0;
    bool        correlation_window_found_ = false;
    bool        correlation_converged_ = false;
    // Accumulation means over walker sweeps: τ_eff/τ and f(R').
    double tau_eff_ratio_mean_ = 0.0;
    double suppression_mean_   = 0.0;
    // One sweep's random numbers for every crowd, [np * (dim + 1) * capacity]
    // with the population capacity as walker stride (sweep_blocking.h).
    std::vector<double> sweep_randoms_;
    std::size_t         sweep_random_stride_ = 0;
    std::size_t         sweep_block_ = 1;   // walkers per cache block

    double energy_           = 0.0;
    double energy_error_     = 0.0;  // reblocked (energy_stats_.error)
    double energy_variance_  = 0.0;
    numerics::ReblockResult energy_stats_;  // reblocking of block_energies_
    double acceptance_ratio_ = 0.0;
    double etrial_final_     = 0.0;
    double walltime_s_       = 0.0;
    NodeConstraint node_constraint_ = NodeConstraint::None;

    // Weighted-mean local energy of each accumulation block (length n_acc):
    // one chain for the whole population, whatever the worker count.
    std::vector<double> block_energies_;

    // Population-control diagnostics, captured between blocks (length n_acc / n_eq).
    // These are the DMC-specific outputs: the trajectory of the population size
    // and of the reference energy E_trial reveal control instabilities.
    std::vector<double> block_population_;     // living walkers per accumulation block
    std::vector<double> block_total_weight_;   // Σ wᵢ (effective population) per acc block
    std::vector<double> block_etrial_;         // E_trial per accumulation block
    std::vector<double> eq_population_;        // living walkers per equilibration block
    std::vector<double> eq_etrial_;            // E_trial per equilibration block
    std::vector<double> eq_block_energies_;    // mixed local energy per equilibration block

    // Branch-cutoff diagnostics: the fraction of walker sweeps whose weight
    // the cutoff tapered, per phase, and the scratch of the E_L spread.
    double              cut_fraction_eq_  = 0.0;
    double              cut_fraction_acc_ = 0.0;
    std::vector<double> spread_scratch_;

    MeasureHook    measure_hook_;
    CheckpointHook checkpoint_hook_;

    // Observable estimators: templates set via set_estimators(), instantiated
    // (single instance each, mixed estimator) into ests_ at the start of run().
    std::vector<std::unique_ptr<Estimator>> est_templates_;
    std::vector<std::unique_ptr<Estimator>> ests_;

    // Mean accumulation population over its target (1 with no block yet).
    [[nodiscard]] double population_mean_fraction() const noexcept;

    // DMC acceptance with drift-diffusion Green's function.
    // Accepts iff  u < |ψ'/ψ|² · exp(log_gb − log_gf).
    static bool dmc_accept(double log_ratio, double log_gf, double log_gb,
                           double u) noexcept;

    // One Langevin (drift) sweep over all particles for the walker range
    // [w0, w_end) of the global population pop_, using DMC acceptance: draw
    // the crowd's random numbers, then move cache blocks of sweep_block_
    // walkers through every particle (sweep_blocking.h).
    // psi_ must already be evaluated on that range.  Returns accepted moves.
    std::size_t sweep(std::span<const double> sigma,
                      std::span<const double> tau_over_mass,
                      numerics::XoshiroRandom& rng,
                      std::size_t w0, std::size_t w_end);

    // Drift-diffusion moves of every particle for walkers [w0, w_end),
    // reading their entries of sweep_randoms_.
    std::size_t sweep_walkers(std::span<const double> sigma,
                              std::span<const double> tau_over_mass,
                              std::size_t w0, std::size_t w_end);

    // One measured step for the crowd [w0, w_end): n_substeps sweeps →
    // local-energy measurement → per-walker branch weight + accumulation.
    // Runs in parallel on disjoint ranges; the global branch happens after join.
    StepResult advance_measure(numerics::XoshiroRandom& rng,
                               std::size_t w0, std::size_t w_end);

    // u = Σ_{i,d} (τ/m_{i,d})·(∂_{i,d} ln|Ψ|)² of walkers [w0, w_end) from the
    // current gradients, into u[w].
    void drift_norms(std::span<double> u, std::size_t w0, std::size_t w_end) const;

    // θ for the suppressed reweighting: drift-diffusion sweeps of step τ_θ ≤ τ
    // without weights or branching of the first kCorrelationChains walkers,
    // until the autocorrelation window of their E_L converges (numerics::
    // integrated_autocorrelation); θ = τ_θ·(½ + Σρ(k)) (numerics::
    // integrated_time), with τ_θ reduced until θ/τ_θ ≥ 10.  Sets
    // correlation_*_.
    void measure_correlation_time();
};

}  // namespace alea
