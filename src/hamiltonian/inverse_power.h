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

// Inverse-power pair interaction between two particle groups
//
//     Intra-group (a == b):  V = Σ_{i<j}      C / r_ij^p
//     Inter-group (a != b):  V = Σ_{i∈a,j∈b}  C / r_ij^p
//
// C = `strength`, p = `exponent`.  p = 3 is the interaction of dipoles
// polarized perpendicular to the plane, V = C_dd / (4π r³) (Astrakharchik,
// Boronat, Kurbakov, Lozovik, Phys. Rev. Lett. 98, 060405 (2007), Eq. 1); in
// units of r0 = M C_dd/(4πħ²) and ħ²/(M r0²) that is mass 1 and C = 1.
//
// ── Periodic cells: minimum image plus continuum tail, no Ewald ────────────
// A lattice sum Σ_L C/|r + L|^p over the images of a d-dimensional cell
// converges absolutely when p > d: the images in a shell [R, R + dR] contribute
// ~ R^{d−1−p} dR.  So, unlike Coulomb, nothing is conditionally convergent and
// no neutralizing background or Ewald split is needed.  Every pair is summed
// at its minimum image — the image inside the Wigner–Seitz (WS) cell centred
// on its partner — and every image outside that cell is replaced by a uniform
// continuum (g → n), which adds the constant
//
//     E_tail = ½ N_a n_a T    (a == b),     E_tail = N_a n_b T    (a != b),
//     T = ∫_{outside WS} C r^{−p} d^d r = C/(p − d) ∮ dΩ ρ(Ω)^{d−p},
//
// with n = N/A the group's number density (A the cell's area, or its volume
// in 3-D) and ρ(Ω) the distance from the cell's centre to the WS boundary in
// direction Ω.  This is Eq. (2) of the
// paper above with the WS cell, rather than a circle of radius L/2, as the
// region summed exactly.  The density counts every particle of the group
// (n, not (N−1)/A): outside the WS cell a particle also meets its own images.
//
// APPROXIMATION: g ≠ n outside the WS cell leaves a finite-size error that
// vanishes as N → ∞ and is removed by extrapolating in N.  Why the WS cell and
// not the circle: for a perfect triangular crystal at p = 3 the relative error
// falls monotonically, ≈ N^{−3/2} (−6.9e−4 at N = 64, −6.4e−5 at N = 256),
// while a circle of radius L/2 lets lattice shells cross its edge and the error
// oscillates in N (−5.7e−3 at N = 36, +8.1e−4 at N = 256).  The minimum-image
// distance is continuous across the WS boundary, so V has no jump there (only
// a kink).  This is not `lattice.truncation: minimum_image` of a Coulomb term,
// whose dropped remainder diverges: here it is integrable and supplied.
//
// evaluate_potential_gradient is not offered, so PIMC virial estimators
// refuse the term.  The per-configuration gradient (configuration_gradient,
// bead_gradient) is, for the Takahashi–Imada action: ∂_i C r^{−p} =
// −p C r^{−p−2} Δ at the minimum image.  That force jumps where a pair
// crosses a WS face (V's kink), so a |∇V|² term jumps there too.
class InversePowerTerm : public HamiltonianComponent {
public:
    // Throws on a non-finite or non-positive exponent, a non-finite strength,
    // an attractive core with p ≥ 2 (the two-body problem falls to the
    // centre), and — in a periodic cell — p ≤ d, whose image sum diverges.
    InversePowerTerm(ParticleGroup group_a, ParticleGroup group_b,
                     double strength, double exponent,
                     const SimulationCell& cell);

    std::string name() const override { return "inverse_power"; }

    bool is_translation_invariant() const noexcept override { return true; }

    void evaluate_potential(const ParticleBatch& batch, std::span<double> v_out,
                            std::size_t w_begin = 0,
                            std::size_t w_end = npos) const override;

    // lim_{r→0} r V = C when p = 1 (the term is then a Coulomb pole), else 0.
    double coulomb_cusp_residue(const ParticleGroup& a,
                                const ParticleGroup& b) const override;

    // {C, p} for this term's own pair when p > 1, else {0, 0}.
    PowerLawCore power_law_core(const ParticleGroup& a,
                                const ParticleGroup& b) const override;

    [[nodiscard]] std::unique_ptr<HamiltonianComponent> clone() const override {
        return std::make_unique<InversePowerTerm>(*this);
    }

    // The continuum tail E_tail added to every configuration's V (0 when open).
    [[nodiscard]] double tail_energy() const noexcept { return tail_; }

    // Grand canonical, intra-group terms only: sum_j C / r^p over the counted
    // partners plus E_tail(n + 1) - E_tail(n) = c (n + 1/2), with
    // E_tail(n) = c n^2 / 2 and c = T / A (the header's T and A).
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
    // v'(r)/r = −p C r^{−p−2}, by repeated multiplication for an integer p.
    [[nodiscard]] double slope_over_r(double r) const noexcept;

    [[nodiscard]] bool is_own_pair(const ParticleGroup& a,
                                   const ParticleGroup& b) const noexcept {
        return (a == group_a_ && b == group_b_) || (a == group_b_ && b == group_a_);
    }

    ParticleGroup group_a_;
    ParticleGroup group_b_;
    double        strength_;   // C
    double        exponent_;   // p
    // p when it is a small positive integer: r^{−p} by repeated multiplication
    // in the walker loop instead of std::pow.  0 otherwise.
    int           integer_exponent_ = 0;
    double        tail_ = 0.0;
    double        tail_pair_density_ = 0.0;  // c = C T_p / A: E_tail = c x pairs
    bool          is_intragroup_;
};

// ∫ over everything outside the Wigner–Seitz cell of r^{−p} d^d r, for a
// periodic cell (1-D, 2-D, or an orthorhombic 3-D box) and p > d:
// (1/(p − d)) ∮ dΩ ρ(Ω)^{d−p}, by wigner_seitz_exterior_integral
// (population/wigner_seitz_exterior.h), whose header gives the angular rules
// and their errors.  Setup only.
[[nodiscard]] double outside_wigner_seitz_integral(const SimulationCell& cell,
                                                   double exponent);

}  // namespace alea
