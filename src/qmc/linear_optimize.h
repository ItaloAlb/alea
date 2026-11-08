// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "cost_function.h"
#include "driver.h"
#include "linear_method.h"
#include "population/simulation_cell.h"

namespace alea {

class WalkerEnsemble;
class TrialWavefunction;
class Hamiltonian;

struct OptimizeConfig {
    // VMC sampling parameters per optimisation iteration.
    int    n_equilibration_blocks = 50;
    int    n_accumulation_blocks  = 50;
    int    n_steps                = 10;
    int    n_substeps             = 1;
    double tau                    = 0.01;
    bool   use_drift              = false;  // true → Langevin moves (random_walk: langevin)

    // Number of walkers.  If 0, uses the ensemble count.
    int    n_walkers              = 128;

    // Which parameters this stage may move: "all", "nodal" (determinants only),
    // or "jastrow" (everything that is not nodal).  Narrowing only ADDS fixing;
    // the flags are restored when the stage ends.
    std::string parameters = "all";

    // ── Optimisation loop ───────────────────────────────────────────────────
    int    max_iterations         = 20;
    // Converged when either
    //   the UNDAMPED linear-method step is small:
    //           ‖δψ‖/‖ψ‖ = √(Δpᵀ S̄ Δp) < psi_step_tol   (scale-invariant),
    //           and its reweighting keeps N_eff ≥ min_effective_fraction, or
    //   the energy has settled: over the energies measured on the last
    //           convergence_window fresh samples, the least-squares slope b
    //           (energy change per iteration) satisfies
    //           b ≥ −max(energy_tol, 2σ_b).
    double psi_step_tol          = 1e-3;
    double energy_tol            = 1e-6;   // Ha per iteration
    int    convergence_window    = 15;     // ≥ 3

    // ── Adaptive shift (QMCPACK three-shift algorithm) ──────────────────────
    // Number of shift values to try per iteration (odd, ≥ 3).
    int    num_shifts             = 3;
    // Maximum relative change allowed for any parameter in a single step.
    // If the update exceeds this, the shift is increased to damp it.
    double max_param_change       = 0.3;
    // Tolerance for cost-function increase when selecting the best shift.
    // A slightly higher cost may be accepted if it brings the shift closer
    // to the target (promoting stability).
    double cost_increase_tol      = 0.0;
    // Target shift_i that the adaptive method should aim for.  Negative
    // disables targeting (any shift can win purely on cost).
    double target_shift_i         = -1.0;
    // Starting identity shift.  Adjusted adaptively during the run.
    double start_shift_i          = 0.1;
    // Starting overlap shift.  Adjusted adaptively during the run.
    double start_shift_s          = 0.0;
    // Maximum number of failed attempts before aborting an iteration.
    int    max_failed_attempts    = 5;
    // Factor by which shift_i grows when a step is rejected.
    double shift_growth_factor    = 4.0;

    // ── Correlated sampling (candidate steps are scored by reweighting) ─────
    // A candidate whose weights leave fewer than this fraction of effective
    // samples, N_eff/N = (Σw)²/(N Σw²), is too far from the sampled ψ to
    // trust and is rejected (QMCPACK's minwalkers, same default).
    double min_effective_fraction = 0.3;
    // A step is rejected only when it raises the reweighted energy by more
    // than this many standard errors of ΔE (and more than cost_increase_tol).
    // The same bar undoes the stage's last step when a fresh sample shows it
    // raised the energy (see run()).
    double reject_sigmas          = 2.0;
    // Memory budget for the stored configurations; above it every k-th
    // measured step is stored (thinning — consecutive steps are correlated
    // anyway, so little information is lost).
    std::size_t sample_store_bytes = std::size_t{512} << 20;

    // Random seed and thread count.
    std::uint64_t seed = 42;
    int n_threads       = 1;   // 0 → all cores

    // Linear method configuration (shift_i / shift_s overridden adaptively).
    LinearMethodConfig lm_config;
};

struct OptimizeResult {
    std::vector<double> final_params;
    // Energy of final_params measured on a fresh sample, with that sample's
    // reblocked error.
    double              final_energy      = 0.0;
    double              final_energy_error = 0.0;
    int                 iterations        = 0;
    bool                converged         = false;
    std::vector<double> energy_history;
    std::vector<double> shift_history;     // best_shift_i per iteration
    // The stage ended on an accepted step, so it was measured on one more
    // fresh sample (check_energy ± check_energy_error) and undone if that
    // showed a significant rise.
    bool                last_step_checked = false;
    bool                last_step_undone  = false;
    double              check_energy       = 0.0;
    double              check_energy_error = 0.0;
};

// Linear-method wavefunction optimiser.
//
// Runs an iterative optimisation loop: VMC sampling → CostFunction → LinearMethod
// → parameter update → repeat.  Each thread owns independent clones of the
// wavefunction, Hamiltonian, RNG, and CostFunction.
class LinearOptimize : public Driver {
public:
    LinearOptimize(OptimizeConfig      cfg,
                   WalkerEnsemble&     ensemble,
                   TrialWavefunction&  psi,
                   const Hamiltonian&  hamiltonian);

    void run() override;

    [[nodiscard]] std::string summary() const override;
    [[nodiscard]] RunResult   run_result() const override;

    [[nodiscard]] const OptimizeResult& result() const noexcept { return result_; }

    // Energy of a candidate step estimated on a stored sample:
    //   w_k = |ψ_new(R_k)/ψ_old(R_k)|²,   E_new = Σ w E_L,new / Σ w,
    //   dE  = E_new − <E_L,old>  on the SAME R_k, so the noise common to both
    //         cancels; its error comes from (thread, block) group means of the
    //         linearised difference (delta method), which respects the serial
    //         correlation of the chain.
    struct Reweighted {
        double dE       = 0.0;
        double dE_error = 0.0;
        double n_eff_fraction = 0.0;   // (Σw)² / (N Σw²)
        bool   finite   = false;
    };

    // Sample the current ψ once (the configured VMC) and score `update` (in the
    // optimiser's parameter order: non-fixed parameters, θ for sign-locked
    // ones) by correlated sampling.  ψ is left unchanged.  The same machinery
    // run() uses; exposed for diagnostics and tests.
    [[nodiscard]] Reweighted correlated_step_energy(const std::vector<double>& update);

    // Ordinary least-squares line E_k ≈ a + b k through `energies` (one per
    // fresh sample, in order).  σ_b comes from the scatter of the residuals,
    //   σ_b² = Σ r_k² / ((n − 2) Σ (k − k̄)²),
    // so the noise of a single measurement is measured, not assumed.  Needs
    // n ≥ 3.
    struct EnergyTrend {
        double slope       = 0.0;   // b, energy change per iteration
        double slope_error = 0.0;   // σ_b
    };
    [[nodiscard]] static EnergyTrend energy_trend(std::span<const double> energies);

    // Has the energy stopped falling?  Its trend over the last `window`
    // fresh energies descends by no more than max(energy_tol, 2σ_b) per
    // iteration.  False while fewer than `window` energies exist.
    [[nodiscard]] static bool energy_has_settled(std::span<const double> fresh_energies,
                                                 int window, double energy_tol);

    // Is the undamped step a genuine small step?  ‖δψ‖/‖ψ‖ < psi_step_tol,
    // confirmed by its reweighting: a step that small leaves N_eff ≈ 1.
    [[nodiscard]] static bool step_is_small(double psi_step, const Reweighted& rw,
                                            double psi_step_tol,
                                            double min_effective_fraction);

    // May a step be scored by reweighting (set on ψ)?  Every |Δp_k| finite and
    // at most max_param_change · max(1, max_k |p_k|).  A nearly singular S̄
    // (parameters that are nearly redundant close to the optimum) gives steps
    // of 1e19; set on ψ they leave the parameters' domain, a sign-locked
    // cutoff L e^{Δθ} = ∞, and the component throws.  NaN fails the test.
    [[nodiscard]] static bool step_within_bounds(std::span<const double> update,
                                                 std::span<const double> params,
                                                 double max_param_change);

    // The energy of one fresh sample: its mean local energy and the reblocked
    // standard error of that mean.
    struct MeasuredEnergy {
        double energy = 0.0;
        double error  = 0.0;
    };

    // Did a step raise the energy?  `before` and `after` are independent fresh
    // samples taken before and after it; the rise counts when
    //   E_after − E_before > reject_sigmas · √(σ_before² + σ_after²).
    [[nodiscard]] static bool step_raised_energy(const MeasuredEnergy& before,
                                                 const MeasuredEnergy& after,
                                                 double reject_sigmas);

private:
    class ScopeGuard;

    OptimizeConfig       cfg_;
    WalkerEnsemble&      ensemble_;
    TrialWavefunction&   psi_;
    const Hamiltonian&   hamiltonian_;
    OptimizeResult       result_;

    // Collect all non-fixed parameter values from the wavefunction into a flat
    // vector, and optionally set them back.
    std::vector<double> current_params() const;
    void apply_update(const std::vector<double>& update);
    // Restore values taken from current_params() (same order) and rebuild ψ.
    void set_params(const std::vector<double>& values);

    // Per-non-fixed-parameter chain-rule scale, in the same flat order as
    // current_params(). For a sign-locked parameter (p = s·exp(θ)) the column
    // factor is dp/dθ = p; for a free parameter it is 1.  Used to convert the
    // wavefunction's ∂logψ/∂p into ∂logψ/∂θ before building the LM matrices.
    std::vector<double> deriv_scales() const;

    // ── Correlated sampling ─────────────────────────────────────────────────
    //
    // The configurations R_k of one VMC run, with what reweighting needs:
    // log|ψ(R_k)| and E_L(R_k) at the SAMPLED parameters, and the potential
    // V(R_k) = E_L − K, which no parameter changes.  One frame = one measured
    // step of one VMC thread (its nw walkers, SoA like WalkerSet).
    struct SampleFrame {
        std::size_t         nw    = 0;
        std::size_t         block = 0;    // (thread, block) = error-bar group
        std::size_t         thread = 0;
        std::vector<double> pos;          // [np * dim * nw]
        std::vector<double> log_psi;      // [nw]
        std::vector<double> e_loc;        // [nw]
        std::vector<double> v_fixed;      // [nw]
    };
    struct SampleStore {
        std::vector<std::vector<SampleFrame>> per_thread;
        std::size_t np = 0, dim = 0;
        SimulationCell cell;               // the sampled walkers' cell
        std::size_t n_samples() const noexcept;
    };

    // Run one full VMC simulation, collecting local energies and parameter
    // derivatives via a measurement hook into a CostFunction, and (when
    // `store` is given) the configurations for correlated sampling.  Returns
    // the VMC's reblocked standard error of the mean energy (the CostFunction's
    // own error is a floor that ignores serial correlation).
    double sample_vmc(CostFunction& cost_fn, SampleStore* store = nullptr) const;

    // Score `update` on `store` without moving any walker.  ψ is left exactly
    // as it was (the update is applied to a copy of the parameters only while
    // the per-thread clones are made).
    Reweighted reweight(const SampleStore& store, const std::vector<double>& update);

    // ── Adaptive shift helpers ──────────────────────────────────────────────

    // Generate geometrically-spaced shift values around a central value.
    // For odd num_shifts: shifts = [central/f^{k}, ..., central, ..., central*f^{k}]
    // where the central value is at the middle index.
    static std::vector<double> prepare_shifts(double central, int num_shifts,
                                               double factor = 4.0);

    // QMCPACK's best-cost selection logic.
    // Returns true if `cost` at `shift_idx` with shift `si` is better than
    // `best_cost` at `best_shift`, given the current best shift `best_si`.
    bool is_best_cost(int shift_idx, const std::vector<double>& shifts,
                      double si, double cost,
                      double best_cost, double best_si) const;

    // Overlap shift switched on by the first failed step when shift_s is 0
    // (QMCPACK's default shift_s).
    static constexpr double kOverlapShiftSeed = 1.0;

    // One attempt of an optimisation iteration: solve the eigenproblem at each
    // of the shifts around best_shift_i, score every admissible candidate by
    // correlated sampling on `store`, and keep the lowest reweighted energy.
    // Accepts (applies the step, returns true) unless that best candidate
    // raises the energy significantly; otherwise grows the shifts and returns
    // false.  `chosen` reports the winning candidate's reweighted statistics.
    bool adaptive_iteration(const Eigen::MatrixXd& Sbar,
                            const Eigen::MatrixXd& Hbar,
                            std::size_t n_opt,
                            const SampleStore& store,
                            double& best_shift_i, double& best_shift_s,
                            std::vector<double>& update,
                            Reweighted& chosen);
};

}  // namespace alea
