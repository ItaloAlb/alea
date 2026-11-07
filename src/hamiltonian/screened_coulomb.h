// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "hamiltonian_component.h"
#include "../numerics/ewald_breakup.h"
#include "../population/particle_group.h"

// Opaque handle from the fast-screened-coulomb C library, forward-declared at
// global scope so the C header does not leak into every translation unit that
// includes this one.  The full definition lives in the library's .c; we only
// ever hold a pointer to it.
struct FSCControlledFit;

namespace alea {

class WalkerSet;

// Screened-Coulomb interaction between two particle groups for an ARBITRARY
// multilayer dielectric stack, consuming the fast-screened-coulomb library
// (ItaloAlb/fast-screened-coulomb; Electrostatic Transfer Matrix method,
// Cavalcante et al., Phys. Rev. B 97, 125427 (2018)).
//
// The library evaluates the real-space potential  V(ρ) = 2 ∫₀^∞ J₀(kρ)/ε(k) dk
// in RYDBERG units; alea works in Hartree atomic units (Coulomb tail 1/εr), so
// the term scales the library output by ½ (½·2/ερ = 1/ερ, matching CoulombTerm).
// The stack ε(k) AND any out-of-plane (layer) separation are baked into the
// pre-fitted potential, so — unlike CoulombTerm/KeldyshTerm — the in-plane
// distance ρ is fed directly (no √(ρ²+d²) and no layer-offset member).
// Rytova–Keldysh (KeldyshTerm) is the single-layer special case of this term.
//
//   Intra-group (a == b):  V = Σ_{i<j} q²    · ½ V_fit(ρ_ij)
//   Inter-group (a != b):  V = Σ_{i,j} q_a q_b · ½ V_fit(ρ_ij)
class ScreenedCoulombTerm : public HamiltonianComponent {
public:
    // epsilons     : per-layer dielectric constants (bottom → top), length N ≥ 1.
    // interfaces   : N-1 interface positions (Bohr); origin at source-layer centre.
    // source_layer : charge layer c (1-based, 1..N).
    // target_layer : observation layer t (1-based); 0 → direct exciton (t = c).
    // z_offset     : out-of-plane offset z_t (Bohr), used only when t ≠ c.
    // fit_tol      : separate sampled relative value/force targets (%).
    // Both must pass refined moment checks (also capped at 5%) or construction fails. These are
    // sampled estimates, excluding clamp error; no global force bound is claimed.
    // Throws std::invalid_argument / std::runtime_error on inconsistent stack
    // parameters or a failed fit (construction is off any hot path).
    ScreenedCoulombTerm(ParticleGroup       group_a,
                        ParticleGroup       group_b,
                        std::vector<double> epsilons,
                        std::vector<double> interfaces,
                        int                 source_layer,
                        int                 target_layer,
                        double              z_offset,
                        double              fit_tol);

    std::string name() const override { return "screened_coulomb"; }
    PairActionRole pair_action_role() const noexcept override { return PairActionRole::radial_pairs; }
    std::vector<RadialPairPotential> radial_pair_potentials() const override;

    // R = lim_{r→0} r·V(r) = q_a q_b · (1/ε_source) for an intralayer pair whose
    // bulk-Coulomb core diverges; exactly 0 for an interlayer pair (the baked layer offset
    // makes V finite at the origin) or a non-matching group pair.  Mirrors
    // CoulombTerm's charge-product convention so the Drummond cusp builder picks up
    // the repulsive e–e/h–h cusp the intralayer screened potential demands.
    double coulomb_cusp_residue(const ParticleGroup& a,
                                const ParticleGroup& b) const override;

    bool is_translation_invariant() const noexcept override { return true; }
    // Ordinary derivative of the sampled fit with its existing validity clamp:
    // G_i = ½ q_a q_b V_fit'(rho) Delta_ij/rho inside the band, zero outside.
    // At the band endpoints use the interior branch. This capability alone
    // does not admit a virial estimator at clamp/minimum-image interfaces.
    bool supports_potential_gradient() const noexcept override;
    bool supports_open_centroid_virial() const noexcept override;
    bool supports_periodic_virial(const SimulationCell&, std::size_t) const noexcept override;
    // Report value/force targets, interpolation and reference refinements.
    // Numerical accuracy alone does not admit a virial estimator at clamps.
    std::string potential_approximation_notes() const override;
    std::unique_ptr<PotentialGradientWorkspace>
        make_potential_gradient_workspace(const ParticleBatch&) const override;
    void evaluate_potential_gradient(const ParticleBatch&, std::span<double>,
        PotentialGradientWorkspace&, std::size_t = 0,
        std::size_t = npos) const override;
    double delta_potential_particle(ParticleBatch&, std::size_t, std::size_t,
        std::span<const double>, std::span<double>) const override;

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<ScreenedCoulombTerm>(*this);
    }

private:
    ParticleGroup group_a_;
    ParticleGroup group_b_;
    double        charge_product_ = 0.0;  // q_a q_b (q² for an intra-group pair)
    double        rho_min_ = 0.0;         // fit validity lower bound (clamp below)
    double        rho_max_ = 0.0;         // fit validity upper bound (clamp above)
    double        cusp_residue_ = 0.0;    // lim r→0 r·½·V_fit(r) (no charge product)
    std::string   approximation_notes_; // constructed once; identical in clones
    bool          is_intragroup_ = false;
    bool          virial_admitted_ = false; // setup checks of the represented kernel

    // Immutable, thread-safe controlled fit of the screened potential.  Held by
    // shared_ptr (custom deleter → fsc_controlled_free) so clone() — one copy per OpenMP
    // thread — duplicates only the pointer; the library documents fsc_controlled_eval as
    // read-only/thread-safe on a shared fit.  This is a single control-plane
    // object never touched in the per-walker hot loop, so it does not run afoul
    // of the no-shared_ptr-in-hot-paths guideline (it mirrors the process-wide
    // static the analytic KeldyshTerm shares, adapted to a per-stack object).
    std::shared_ptr<FSCControlledFit> fit_;
};

// Immutable screened-Coulomb fit plus the radial band over which it is valid.
// Built once (off any hot path) by build_screened_fit and shared read-only across
// OpenMP thread clones.
struct ScreenedFit {
    std::shared_ptr<FSCControlledFit> fit;
    double                  rho_min;       // fit validity lower bound (clamp below)
    double                  rho_max;       // fit validity upper bound (clamp above)
    double                  cusp_residue;  // lim r→0 r·½·V_fit(r) (no charge product)
    double                  requested_error_pct;
    double                  estimated_error_pct; // sampled interpolation value error
    double                  estimated_force_error_pct; // separate sampled force error
    std::string             approximation_notes;
};

// Build the controlled smooth-remainder fit of the screened potential V(ρ) [Rydberg] for a multilayer
// dielectric stack — see ScreenedCoulombTerm for the parameter semantics (epsilons,
// interfaces, source/target layer, z_offset, fit_tol).  Runs the ETM context setup +
// refined value/force moments + C1 Hermite fit once; rejects invalid stacks,
// unresolved targets and exceeded numerical resource limits.  Throws std::invalid_argument /
// std::runtime_error on failure (construction is off any hot path).
[[nodiscard]] ScreenedFit build_screened_fit(const std::vector<double>& epsilons,
                                             const std::vector<double>& interfaces,
                                             int    source_layer,
                                             int    target_layer,
                                             double z_offset,
                                             double fit_tol);

// The periodic (Ewald) kernel of one screened_coulomb pair: an EXACT transform
// pair built from the library's eps(k), for an EwaldTerm on a cell whose
// real-space reach is r_max (the Wigner-Seitz inradius).  Unlike
// build_screened_fit's controlled fit (sampled accuracy, clamped outside its band) it is
// accurate to ~1e-10 at every r in (0, r_max], carries the 1/(eps_c r) core
// analytically, and has exact derivatives -- which the Ewald breakup needs, since
// its real- and reciprocal-space halves must describe the same potential.
// Stack parameters as for ScreenedCoulombTerm.  Throws on a degenerate stack, or
// if eps(k) overflows in the library before it has reached its large-k limit.
[[nodiscard]] numerics::PotentialSpec build_screened_ewald_kernel(
    const std::vector<double>& epsilons, const std::vector<double>& interfaces,
    int source_layer, int target_layer, double z_offset, double r_max);

// Diagnose a multilayer dielectric stack for the screened_coulomb term.  Returns ""
// when the stack is well-conditioned for the ETM fit; otherwise a human-readable
// reason it is DEGENERATE — i.e. the screened response ε(k) would never converge to
// ε_c, so fsc_optimal_params runs its k-cutoff to the cap and the Chebyshev fit
// stalls for minutes.  Pure (no I/O, no library calls): cheap and unit-testable.
//
// Catches the common, clearly-diagnosable causes: non-positive ε, wrong interface
// count, out-of-range / non-monotonic interfaces, a semi-infinite (top/bottom)
// source layer, and a source charge that is not strictly inside its interior layer.
// Subtler cases (e.g. an interior layer too thin to resolve) are caught at
// construction by a k_max backstop instead.
[[nodiscard]] std::string screened_coulomb_stack_issue(
    const std::vector<double>& epsilons,
    const std::vector<double>& interfaces,
    int                        source_layer);

}  // namespace alea
