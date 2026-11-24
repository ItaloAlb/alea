// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "hamiltonian_component.h"
#include "../population/particle_group.h"
#include "../population/simulation_cell.h"

namespace alea {

// Parameters of the HFD-B pair potential (HfdTerm), in the run's units: ε an
// energy, r_m a length, the rest dimensionless.
struct HfdParameters {
    double epsilon = 0.0;  // ε: well depth
    double r_m     = 0.0;  // position of the minimum
    double A       = 0.0;  // repulsive amplitude
    double alpha   = 0.0;  // α
    double beta    = 0.0;  // β (≤ 0)
    double C6      = 0.0;
    double C8      = 0.0;
    double C10     = 0.0;
    double D       = 0.0;  // damping range
};

// Hartree–Fock-dispersion pair potential in the HFD-B form of Aziz, between
// two particle groups:
//
//     V(r) = ε V*(x),   x = r / r_m,
//     V*(x) = A exp(−α x + β x²) − F(x) [C₆/x⁶ + C₈/x⁸ + C₁₀/x¹⁰],
//     F(x) = exp[−(D/x − 1)²]  for x < D,    F(x) = 1  for x ≥ D.
//
// R. A. Aziz, F. R. W. McCourt and C. C. K. Wong, Mol. Phys. 61, 1487 (1987),
// the HFD-B(HE) potential of ⁴He; R. A. Aziz and M. J. Slaman, Chem. Phys.
// 130, 187 (1989), Ne in the same form.  β = 0 is the earlier HFD form of
// HFDHE2 (R. A. Aziz et al., J. Chem. Phys. 70, 4330 (1979)).  The parameter
// sets are in docs/configuration.md.
//
// V is bounded: as x → 0, F(x) vanishes faster than any power of x, so
// V(0) = ε A, and nothing falls to the centre.  F is C¹ at x = D (F' = 0 on
// both sides) but F'' jumps there, so the force is continuous and its slope
// is not.
//
// Forces, with S(x) = Σ_n C_n x^{−n} (n = 6, 8, 10) and y = D/x − 1 > 0 below D:
//     dV*/dx = A (−α + 2β x) exp(−α x + β x²) − F'(x) S(x) − F(x) S'(x),
//     F'(x) = F(x) 2 y D / x²,   S'(x) = −Σ_n n C_n x^{−n−1},
//     v'(r)/r = (ε / r_m²) (dV*/dx) / x.
// Below D and above, y = max(D/x − 1, 0) covers both branches of F without a
// branch: F = exp(−y²) = 1 and F' = 0 exactly when y = 0.
//
// ── Periodic cells: minimum image plus continuum tail ───────────────────────
// As for InversePowerTerm (inverse_power.h, whose header gives the argument):
// the dispersion decays as r⁻⁶, faster than r^{−d} in d ≤ 3, so every pair is
// taken at its minimum image and the images outside the Wigner–Seitz cell are
// a uniform continuum adding the constant
//     E_tail = ½ N_a n_a T  (a == b),   E_tail = N_a n_b T  (a != b),
//     T = ∫_{outside WS} v d^d r = ∮ dΩ f(ρ(Ω)),   f(ρ) = ∫_ρ^∞ v(r) r^{d−1} dr.
// f is a radial quadrature (radial_tail); the angular rule is
// wigner_seitz_exterior_integral's.  APPROXIMATION: g(r) = 1 outside the WS
// cell, an error that vanishes as N → ∞.  This is the "tail correction" of
// liquid-helium simulations, with the WS cell where they usually take the
// sphere of radius L/2.
//
// Supplies the per-configuration gradient (configuration_gradient,
// bead_gradient) for the Takahashi–Imada action, a grand-canonical bead
// energy (intra-group), and an O(N) single-bead potential change.  The
// batched evaluate_potential_gradient is not offered, so PIMC virial
// estimators refuse the term.
class HfdTerm : public HamiltonianComponent {
public:
    // Throws on a non-finite parameter, ε, r_m or D not positive, A < 0, a
    // repulsive exponential that does not decay (β > 0, or β = 0 with α ≤ 0),
    // and C₆, C₈ or C₁₀ negative.
    HfdTerm(ParticleGroup group_a, ParticleGroup group_b, const HfdParameters& parameters,
            const SimulationCell& cell);

    std::string name() const override { return "hfd"; }

    bool is_translation_invariant() const noexcept override { return true; }

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    // O(N): the changed pairs of particle p at configuration k.
    double delta_potential_particle(ParticleBatch& configs, std::size_t p, std::size_t k,
                                    std::span<const double> proposed,
                                    std::span<double> scratch) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<HfdTerm>(*this);
    }

    // The continuum tail E_tail added to every configuration's V (0 when open).
    [[nodiscard]] double tail_energy() const noexcept { return tail_; }

    // v(r) and v'(r)/r of one pair (the tests' handle on the formula).
    [[nodiscard]] double pair_value(double r) const noexcept;
    [[nodiscard]] double slope_over_r(double r) const noexcept;

    // f(ρ) = ∫_ρ^∞ v(r) r^{d−1} dr in dimension d, by Gauss–Legendre after
    // r = ρ/t (t ∈ (0, 1]): f = ρ^d ∫_0^1 v(ρ/t) t^{−d−1} dt, the integrand
    // ∝ t^{5−d} as t → 0.  The t-range is split where x = D, at the jump of
    // F'', and each piece takes `panels` panels of 20 points: for HFD-B(HE) two
    // already agree with 64 to 1e−15 (test_hfd checks 4 against 64).  Setup only.
    [[nodiscard]] double radial_tail(double rho, std::size_t dim, int panels = 4) const;

    // Grand canonical, intra-group terms only: Σ_j v(r) over the counted
    // partners plus E_tail(n + 1) − E_tail(n) = c (n + ½), c = T / A.
    bool supports_bead_energy() const noexcept override { return is_intragroup_; }
    double bead_energy(const ParticleBatch&, std::size_t, std::size_t,
        std::span<const double>, std::size_t, std::size_t) const override;

    // Intra- and inter-group; the tail, a constant of N, has no gradient.
    bool supports_configuration_gradient() const noexcept override { return true; }
    void configuration_gradient(const ParticleBatch&, std::size_t, std::size_t, std::size_t,
                                std::size_t, std::span<double>) const override;
    void bead_gradient(const ParticleBatch&, std::size_t, std::size_t, std::span<const double>,
                       std::size_t, std::size_t, std::span<double>,
                       std::span<double>) const override;

private:
    ParticleGroup group_a_;
    ParticleGroup group_b_;
    HfdParameters q_;
    double        inv_r_m_ = 0.0;
    double        tail_ = 0.0;
    double        tail_pair_density_ = 0.0;  // c = T / A: E_tail = c × pairs
    bool          is_intragroup_;
};

}  // namespace alea
