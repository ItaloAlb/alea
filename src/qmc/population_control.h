// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <limits>

namespace alea {

// DMC population control: the trial energy E_T that keeps the walker
// population near its target, and the branch weight each walker's step
// carries.  Owned by the DMC driver; the walkers themselves live in its
// WalkerSet, whose branch() turns the weights into copies.
//
// Feedback (Umrigar, Nightingale & Runge, JCP 99, 2865 (1993)):
//   E_T = E_ref − ln(W / N_target) / (g · τ · n_substeps),   W = Σ weights.
// A step multiplies every weight by e^{τ·n_substeps·ΔE_T}, so
// ln(W/N) → (1 − 1/g)·ln(W/N): the loop gain is 1/g whatever τ is.  Dividing
// by a bare constant instead (the old `feedback`) made the gain
// τ·n_substeps/feedback, which overshoots into a period-2 limit cycle at
// τ·n_substeps ≳ 2 (the 2-D HEG at rs = 30/40).  Stable for g > ½; the
// population-control bias is ∝ 1/(g·N).
// E_ref follows the ensemble energy — instantaneous during equilibration, the
// cumulative weighted average during accumulation — so the branching window
// below stays centred and the projection unbiased.
//
// Branch weight (QMCPACK SFNBranch::branchWeight, the UNR-1993 split of the
// branching exponent into a uniform population term and a per-walker term):
//   x  = max(E_ref − E_new, E_ref − E_old)            (low-energy excursion)
//   τ' = τ/2                              x ≤ e_cut
//      = τ/2 · (1 − (x − e_cut)/(e_max − e_cut))   e_cut < x ≤ e_max
//      = 0                                x > e_max
//   w  = exp(τ' · (2·E_T − E_new − E_old)),   e_max = 1.5·e_cut
// (QMCPACK SFNBranch::setBranchCutoff).  The window is centred on E_ref, never
// on E_T: a population-control swing in E_T rescales every walker by one
// factor and the relative weights — the projection itself — survive it.  Only
// the dangerous side (E_L far below E_ref: cusp/node divergences that would
// explode the population) is filtered; high-E_L walkers just die.  Any finite
// e_cut biases the projection; the bias vanishes when e_cut/σ → ∞, σ the
// spread of E_L.
//
// ── The cutoff e_cut (BranchCutoff) ─────────────────────────────────────────
// drv:       e_cut = 2/√τ (DePasquale, Rothstein & Vrbik, JCP 89, 3629 (1988),
//            eq. 3; QMCPACK's "DRV").  τ is an inverse energy, so 2/√τ is an
//            energy only in the unit it was written for, the hartree: it hides
//            an energy scale E_h, e_cut = 2√(E_h/τ).  Rescaling the energy unit
//            by λ (E → λE, τ → τ/λ) moves e_cut/σ by 1/√λ, so whether ordinary
//            walkers get cut depends on the unit, and since σ ∝ √N it also
//            shrinks as N grows (Zen, Sorella, Gillan, Michaelides & Alfè, PRB
//            93, 241118(R) (2016)).  Measured: 2-D dipoles at n r0² = 256 in
//            ħ²/(M r0²) units, τ = 1e-5, N = 16: e_cut = 632 against
//            σ ≈ 6700, the population fell to a third of its target and E/N
//            came out 1.1% high and trial-dependent; with no cutoff both trials
//            agreed.
// universal: e_cut = α·σ/√(σ₁τ),   σ₁ = σ/√N   (= α·√(σ√N/τ)),
//            DRV with E_h replaced by the measured spread per particle σ₁,
//            which is also Zen et al.'s α√(N/τ) with the same replacement.
//            It meets the three conditions a cutoff needs:
//              unit:  σ₁τ is dimensionless, so e_cut scales as an energy;
//              size:  for M independent copies σ → √M σ and N → M N leave σ₁
//                     and e_cut/σ = α/√(σ₁τ) unchanged;
//              τ → 0: e_cut/σ grows as 1/√τ, so the bias vanishes.
//            σ is the robust spread of the walkers' E_L (numerics::
//            robust_spread), set by the driver through set_energy_spread();
//            N counts the particles that diffuse.
// none:      e_cut = ∞ (no taper).  For diagnosis: a trial with E_L
//            divergences can then explode the population.
enum class BranchCutoff { universal, drv, none };

// α of the universal cutoff: a convention, not a per-system optimum.  The
// system already enters e_cut through σ, N and τ, and the τ → 0 limit is
// the same for every α (e_cut/σ → ∞), so α only shapes the finite-τ bias.
// One α for every run keeps compared phases (fluid and crystal, gas and
// solid) under the same rule.  An α tuned to make E(τ) flat would make the
// cutoff's bias cancel the time-step error by accident.  Set it per stage
// with `branch_cutoff: {scheme: universal, alpha: …}` when a study needs it.
//
// The value comes from a calibration on eight reference cases (4 seeds,
// 3 time steps each; α ∈ {0.5, 1, 2, 3, 4, 8}, none, drv):
//   - nodeless trials with their cusps (exciton, two-electron dot singlet
//     and fixed-phase triplet, two dipoles in a trap, 16 dipoles at
//     n r0² = 256): α ≥ 1 gives the same energies as no cutoff at every τ.
//     α = 0.5 already cuts ordinary walkers (+13 per particle for the
//     dipole gas at τ = 2e-5), so α = 2 leaves a margin of 4;
//   - trials with a divergent E_L, where the cutoff works:
//       - fixed-node lithium: drv is flattest in τ there (it acts as α ≈ 1.3);
//       - Gaussian-orbital hydrogen in 2-D and 3-D, and test_dmc's off-cusp
//         exciton: the cusp is missed, so the tail is heavy, the bias falls
//         as 1/α, and no cutoff at all is unstable in 2-D.
//     α = 2 keeps all of them stable.  Its bias is close to drv's in these
//     hartree-scale cases (drv acts as α ≈ 1.3–4.3 across them), so
//     earlier atomic-unit results move only within their time-step bias.
inline constexpr double kDefaultBranchCutoffAlpha = 2.0;

// ── Reweighting: the local energy that enters the branch weight ─────────────
// Anderson, Per & Umrigar, J. Chem. Phys. 160, 104110 (2024), Eqs. 6, 8, 10:
//   Δw = exp(τ_eff·[S(R') + S(R)]/2),
//   S(R) = E_T − E_ref + (E_ref − E_L(R))·f(R),
// E_ref the best energy estimate.  With f = 1 and τ_eff = τ this is the weight
// above.  Near a node at distance d, E_L → ±1/d; f ∝ d cancels that
// divergence, so a walker that starts there gets a finite weight instead of
// one that explodes or vanishes.
//   bare:       f = 1, τ_eff = τ.  The default.  Their "naive" weight, which
//               the text defines by "setting τ_eff = τ and S = E_T − E_L"
//               (before Eq. 7); the time-step error is large, negative and
//               curved wherever E_L diverges (their Fig. 1).
//   effective:  f = 1, τ_eff of Eq. 10.  Not a weight of the paper (their
//               Fig. 1 calls its naive curve "f = 1 in Eq. 8" without naming
//               the step); it isolates the change of time step, so bare →
//               effective → suppressed separates τ_eff from f.
//   suppressed: Eq. 14,  f(x) = (√π/2)·erf(x)/x,
//               with τ_eff of Eq. 10 (see DMC::sweep_walkers) and
//                 x = κ·τ_eff·√(u / (N·τ·θ)),   u = Σ_{i,d} (τ/m_{i,d})·(∂_{i,d} ln|Ψ|)²,
//               N the diffusing particles and θ the integrated
//               autocorrelation time of E_L under the drift-diffusion move
//               without weights (DMC::measure_correlation_time).
//   They write x = c·V·τ_eff/√N with V = |∇ ln Ψ| (electrons, unit mass) and
//   c = k/√(T_corr − 1), k = 15.51, T_corr counted in steps of τ0 = 0.01 Ha⁻¹
//   (their Eq. 15, k fitted to the carbon atom).  That c carries the hartree:
//   T_corr − 1 = 2Σρ(k) = 2θ/τ0 − 1 ≈ 2θ/τ0 (θ ≫ τ0, with θ the trapezoid
//   numerics::integrated_time) gives c = k·√(τ0/2)/√θ = κ/√θ with κ = 15.51·√0.005,
//   and then x is the one above, which holds in any unit and with any masses:
//   u is the dimensionless drift (√u is the drift step over the diffusion
//   step), τ_eff/√(τθ) is a ratio of times, and M independent copies scale u
//   and N together (their requirement 5, approximate size consistency).
//   Independent configurations of disconnected fragments do not generally
//   factorize: exact size consistency requires their separate Sec. IV method.
//   This rewriting is ours; for unit masses at the same θ it is Eq. 14.
//   Properties (their requirements 1–4): f → 1 − O(τ²) as τ → 0 at fixed R,
//   so the τ → 0 limit is the exact projector for every κ and θ; f ∝ 1/x ∝ d
//   as d → 0; f ~ 1/τ_eff at large τ.  κ and θ only shape the finite-τ error.
enum class Reweighting { bare, effective, suppressed };

// κ = 15.51·√(0.01/2): Anderson, Per & Umrigar's k in units of the measured
// correlation time.  Fitted on one pseudopotential atom; set per stage with
// `reweighting: {type: suppressed, kappa: …}` when a study shows a better one.
inline constexpr double kDefaultSuppressionKappa = 1.0967;

// f(x) = (√π/2)·erf(x)/x, x ≥ 0: 1 at x = 0 (series 1 − x²/3 + x⁴/10 − …),
// √π/(2x) for large x, 0 at x = ∞.
[[nodiscard]] double suppression_factor(double x) noexcept;

// The scale of x = scale·τ_eff·√u:  scale = κ/√(N·τ·θ).  0 when τ ≤ 0 (no
// move, no suppression); +∞ when θ ≤ 0, so f = 0 wherever u > 0 (the driver
// never passes it: a given θ must be positive and a measured one is ≥ τ/2).
[[nodiscard]] double suppression_scale(double kappa, double n_particles, double tau,
                                       double correlation_time) noexcept;

class PopulationControl {
public:
    // tau: imaginary time of ONE drift-diffusion sweep (the weight is applied
    // once per sweep); n_substeps: sweeps between two E_T updates, so the
    // imaginary time per population-control step is τ·n_substeps;
    // generations: g, the number of steps over which a population deviation
    // is corrected; target_walkers: N_target.  cutoff, alpha and
    // n_particles (N, ≥ 1) set e_cut; the universal cutoff stays infinite
    // until set_energy_spread() gives it σ.  Throws std::invalid_argument on
    // a universal cutoff with α not finite and positive, or N = 0.
    PopulationControl(double tau, int n_substeps, double generations,
                      std::size_t target_walkers,
                      BranchCutoff cutoff = BranchCutoff::universal,
                      double alpha = kDefaultBranchCutoffAlpha,
                      std::size_t n_particles = 1);

    // Start a run from the initial ensemble energy: E_T = E_ref = energy.
    void start(double energy) noexcept;

    // The spread σ of the walkers' local energies.  Sets e_cut for the
    // universal cutoff (σ ≤ 0 or not finite: no cutoff); recorded but unused
    // by drv and none.  The driver calls it during equilibration only, so
    // e_cut is fixed while it accumulates.
    void set_energy_spread(double sigma) noexcept;

    // Weight of one walker's step from local energy e_old to e_new.  Read-only,
    // so concurrent crowds may call it between two update() calls.
    [[nodiscard]] double branch_weight(double e_new, double e_old) const noexcept;

    // The suppressed weight (Reweighting above):
    //   w = exp(τ'·[2(E_T − E_ref) + (E_ref − E_new)·f_new + (E_ref − E_old)·f_old]),
    // τ' = τ_eff/2 tapered by the cutoff on the larger suppressed excursion
    // (E_ref − E_L)·f.  f = 1 and τ_eff = τ give branch_weight(e_new, e_old)
    // up to rounding.
    [[nodiscard]] double branch_weight(double e_new, double e_old, double f_new,
                                       double f_old, double tau_eff) const noexcept;

    // Whether that step's weight is tapered by the cutoff (x > e_cut).
    [[nodiscard]] bool is_cut(double e_new, double e_old) const noexcept {
        return eref_ - e_new > e_cut_ || eref_ - e_old > e_cut_;
    }
    [[nodiscard]] bool is_cut(double e_new, double e_old, double f_new,
                              double f_old) const noexcept {
        return (eref_ - e_new) * f_new > e_cut_ || (eref_ - e_old) * f_old > e_cut_;
    }

    // After each step: E_ref from the ensemble energy (cumulative weighted
    // average when `accumulate`, else the instantaneous value), then the E_T
    // feedback from the total weight W.  W ≤ 0 leaves E_T unchanged.
    void update(double ensemble_energy, double total_weight, bool accumulate) noexcept;

    // Restart the cumulative E_ref average; once, before accumulation.
    void begin_accumulation() noexcept;

    [[nodiscard]] double etrial() const noexcept { return etrial_; }
    [[nodiscard]] std::size_t target_walkers() const noexcept { return target_walkers_; }
    [[nodiscard]] BranchCutoff cutoff() const noexcept { return cutoff_; }
    [[nodiscard]] double alpha() const noexcept { return alpha_; }
    [[nodiscard]] double n_particles() const noexcept { return n_particles_; }
    [[nodiscard]] double energy_spread() const noexcept { return sigma_; }
    // e_cut (+∞ when nothing is cut).
    [[nodiscard]] double energy_cutoff() const noexcept { return e_cut_; }

private:
    void set_cutoff(double e_cut) noexcept;

    static constexpr double kInf = std::numeric_limits<double>::infinity();

    double       tau_;                // τ of one sweep
    double       feedback_rate_;      // 1/(g·τ·n_substeps); 0 disables the feedback
    BranchCutoff cutoff_;
    double       alpha_;              // universal: e_cut = α·√(σ√N/τ)
    double       n_particles_;        // N of the universal cutoff
    double       sigma_ = 0.0;        // last spread given to set_energy_spread
    double       e_cut_ = kInf;       // taper onset for E_ref − E_L
    double       e_max_ = kInf;       // τ' = 0 beyond (1.5·e_cut)
    std::size_t  target_walkers_;
    double       etrial_    = 0.0;
    double       eref_      = 0.0;
    double       eref_sum_  = 0.0;    // Σ E_ensemble·W over accumulation steps
    double       eref_wsum_ = 0.0;    // Σ W over accumulation steps
};

[[nodiscard]] const char* to_string(BranchCutoff cutoff) noexcept;
[[nodiscard]] const char* to_string(Reweighting reweighting) noexcept;

}  // namespace alea
