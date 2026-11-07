// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `- jastrow:` items → PadeJastrow, DrummondJastrow or ZeroEnergyJastrow on a
// group pair.
#include "builders/wavefunction_builders.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hamiltonian/hamiltonian.h"
#include "io/logger.h"
#include "numerics/bessel_j0.h"
#include "numerics/pair_radial.h"
#include "numerics/quadrature.h"
#include "wavefunction/drummond_jastrow.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/zero_energy_jastrow.h"

namespace alea {

namespace {

// Kato cusp Γ = du/dr|₀ for a pair jastrow on (ga,gb): Γ = 2 μ R / (2ℓ + d − 1),
// with R = Σ_terms lim_{r→0} r·V(r) summed over the Hamiltonian's pair
// potentials.  This is potential-driven — each term reports its own 1/r residue,
// so the value is correct for Coulomb (R=q_a q_b/ε), Rytova–Keldysh (R=0,
// cuspless), and any future potential, without this builder ever switching on
// potential type.
//
// ── Why ℓ matters ───────────────────────────────────────────────────────────
// Substituting f = r^ℓ(1 + Γr) into the relative radial equation leaves
// [−(1/2μ)Γ(2ℓ + d − 1) + R] r^{ℓ−1}, so the 1/r divergence of the local energy
// cancels only for Γ = 2μR/(2ℓ + d − 1).  ℓ = 0 is the nodeless (S-wave) case
// that a Jastrow-only wavefunction is in.  But a Slater determinant ALREADY
// vanishes linearly when two same-spin particles coincide — that is a P-wave,
// ℓ = 1 — and then the Jastrow must supply the smaller value or it doubles down
// on a cusp the determinant has partly satisfied and E_L diverges at coalescence.
//
// In 2-D this is the difference between Γ = 1 and Γ = 1/3 for two electrons; in
// 3-D between the familiar 1/2 and 1/4.
double kato_cusp(const Hamiltonian* ham, const ParticleGroup& ga,
                 const ParticleGroup& gb, std::size_t dim, int ell) {
    if (ham == nullptr) {
        io::Logger::warn("jastrow built without a Hamiltonian; "
                         "cusp set to 0 (cuspless).");
        return 0.0;
    }
    if (dim < 2) return 0.0;  // Γ = 2μR/(2ℓ+d−1) undefined for d < 2.
    double R = 0.0;
    for (const auto& c : ham->components())
        R += c->coulomb_cusp_residue(ga, gb);
    const double ma = ga.mass.empty() ? 1.0 : ga.mass[0];
    const double mb = gb.mass.empty() ? 1.0 : gb.mass[0];
    const double mu = ma * mb / (ma + mb);  // reduced mass
    return 2.0 * mu * R / static_cast<double>(2 * ell + dim - 1);
}

// ℓ for the pair (a,b): 1 when a determinant makes the trial wavefunction
// vanish linearly as the two coalesce, 0 otherwise.
//
// That happens only for same-group pairs of IDENTICAL fermions, so the group must
// carry a determinant and be fully polarised.  A Slater determinant over the
// group does it, and so does a pairing determinant whose rows or columns are
// the group: exchanging two of them swaps two rows (or two columns), although
// the pair orbital only ever couples a row to a column.  A group with both
// spins has same-spin pairs (ℓ=1) and opposite-spin pairs (ℓ=0) at once, and a
// single u(r) cannot carry two different cusps — that case keeps the ℓ=0 value
// and says so, since silently picking either one leaves half the pairs with a
// divergent E_L.
int coalescence_ell(const io::Config& cfg, const std::string& group_a,
                    const std::string& group_b) {
    if (group_a != group_b) return 0;
    bool has_det = false;
    for (const io::SlaterSpec* s : io::items_of<io::SlaterSpec>(cfg.wavefunction))
        if (s->group == group_a) has_det = true;
    for (const io::PairingSpec* s : io::items_of<io::PairingSpec>(cfg.wavefunction))
        for (const io::PairingBlockSpec& blk : s->blocks)
            if (blk.rows == group_a || blk.columns == group_a) has_det = true;
    if (!has_det) return 0;
    for (const io::ParticleSpec& p : cfg.particles) {
        if (p.group != group_a) continue;
        if (p.up == 0 || p.down == 0) return 1;
        io::Logger::warn(
            "jastrow on group '" + group_a + "', which carries a "
            "determinant and both spin species: same-spin pairs need the P-wave "
            "cusp and opposite-spin pairs the S-wave one, and one u(r) cannot "
            "hold both.  Using the S-wave value.");
        return 0;
    }
    return 0;
}

std::string pair_label(const io::JastrowSpec& j) { return io::jastrow_pairs_label(j); }

// Drummond–Towler–Needs u(r) = (r − L)^C Θ(L − r) Σ α_l r^l, over every group
// pair of the spec (`pair:` or `pairs:`).
// params: [α0, α2..αN] followed by the cutoff length L; α1 is fixed by the cusp.
std::unique_ptr<WavefunctionComponent> build_drummond(
    const io::JastrowSpec& j, const WavefunctionContext& ctx) {
    std::vector<Parameter> params =
        make_parameters(j.params, j.is_fixed, j.can_flip_sign, "drummond");

    // Cutoff length L: kept strictly positive via sign-locking (L = e^θ) so the
    // nonlinear optimiser never drives it through zero.
    bool L_fixed = j.cutoff_fixed;
    bool L_sign_locked = true;
    if (j.cutoff <= 0.0) {
        io::Logger::warn("drummond jastrow cutoff must be > 0; got " +
            std::to_string(j.cutoff) + " — pinning it fixed.");
        L_fixed = true;
        L_sign_locked = false;
    }
    params.push_back(Parameter{"L", j.cutoff, L_fixed, L_sign_locked});

    // One u(r) has one slope at contact, so every group pair sharing it must
    // need the same Kato cusp; otherwise some pairs would get a wrong cusp and a
    // divergent E_L.  Pairs that differ (masses, charges, a determinant on one
    // group only) belong in separate items.
    std::vector<JastrowGroupPair> pairs;
    double gamma = 0.0;
    std::string first_label;
    for (const auto& [name_a, name_b] : io::jastrow_pairs(j)) {
        const ParticleGroup& ga = find_group(ctx.groups, name_a);
        const ParticleGroup& gb = find_group(ctx.groups, name_b);
        const double g = kato_cusp(ctx.ham, ga, gb, ctx.dim,
                                   coalescence_ell(ctx.cfg, name_a, name_b));
        const std::string label = "[" + name_a + ", " + name_b + "]";
        if (pairs.empty()) {
            gamma = g;
            first_label = label;
        } else if (std::abs(g - gamma) > 1e-12 * std::max(1.0, std::abs(gamma))) {
            throw std::runtime_error(
                "drummond jastrow on " + pair_label(j) + ": " + first_label +
                " needs the cusp " + std::to_string(gamma) + " but " + label +
                " needs " + std::to_string(g) + ".  One u(r) carries one cusp; put "
                "pairs with different cusps in separate items.");
        }
        pairs.push_back({ga, gb});
    }
    try {
        return std::make_unique<DrummondJastrow>(j.truncation, gamma, std::move(pairs),
                                                 std::move(params));
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error("drummond jastrow on " + pair_label(j) + ": " + e.what());
    }
}

// Padé u(r) = (Σ_{k=1}^N a_k r^k) / (1 + Σ_{k=1}^M b_k r^k).  a₁ = du/dr|₀ is
// the Kato cusp: derived here and pinned, never read from the config.  The
// component's parameter vector is [a₁, a₂..a_N, b₁..b_M]; the spec's flags
// cover only the free coefficients after a₁.
std::unique_ptr<WavefunctionComponent> build_pade(const io::JastrowSpec& j,
                                                  const ParticleGroup& ga,
                                                  const ParticleGroup& gb,
                                                  const WavefunctionContext& ctx) {
    const double a1 = kato_cusp(ctx.ham, ga, gb, ctx.dim,
                                coalescence_ell(ctx.cfg, j.pair[0], j.pair[1]));

    std::vector<double> free_values = j.numerator;
    free_values.insert(free_values.end(), j.denominator.begin(), j.denominator.end());
    std::vector<Parameter> params{Parameter{"a1", a1, /*is_fixed=*/true}};
    for (Parameter& p : make_parameters(free_values, j.is_fixed, j.can_flip_sign, "pade"))
        params.push_back(std::move(p));

    return std::make_unique<PadeJastrow>(
        1 + static_cast<int>(j.numerator.size()),
        static_cast<int>(j.denominator.size()), ga, gb, std::move(params));
}

// a₁..a_N of a Padé spec, a₁ the derived cusp.
std::vector<double> pade_numerator(const io::JastrowSpec& j, const ParticleGroup& ga,
                                   const ParticleGroup& gb, const WavefunctionContext& ctx) {
    std::vector<double> a{kato_cusp(ctx.ham, ga, gb, ctx.dim,
                                    coalescence_ell(ctx.cfg, j.pair[0], j.pair[1]))};
    a.insert(a.end(), j.numerator.begin(), j.numerator.end());
    return a;
}

// The strongest short-range core V ≈ C/r^p of the pair (ga, gb): the largest
// exponent any Hamiltonian term reports (HamiltonianComponent::power_law_core),
// with the coefficients of every term at that exponent summed.  Read from the
// potential, never from the config, like the Kato cusp above.
PowerLawCore dominant_core(const Hamiltonian& ham, const ParticleGroup& ga,
                           const ParticleGroup& gb) {
    PowerLawCore core;
    for (const auto& c : ham.components()) {
        const PowerLawCore term = c->power_law_core(ga, gb);
        if (term.coefficient == 0.0) continue;
        if (term.exponent > core.exponent) core = term;
        else if (term.exponent == core.exponent) core.coefficient += term.coefficient;
    }
    return core;
}

// What a zero_energy spec's u is built from: b = 2μC and p of the pair's core,
// and the cutoff r_c.  Under periodic boundaries r_c defaults to the
// Wigner–Seitz inradius R_WS and may not exceed it: u is flat beyond r_c, and
// only within R_WS is the minimum image the exact sum over periodic images.
// An open cell has no natural range, so `cutoff` is required there.
//
// A weaker 1/r residue on the same pair (Coulomb next to the core) needs no
// cusp: ψ vanishes as exp(−κ r^{−s}), faster than any power, so the leftover
// q_a q_b/r in E_L is integrable and the core dominates the contact.
struct ZeroEnergyInputs {
    double b;
    double exponent;
    double cutoff;
};

ZeroEnergyInputs zero_energy_inputs(const io::JastrowSpec& j, const ParticleGroup& ga,
                                    const ParticleGroup& gb,
                                    const WavefunctionContext& ctx) {
    const std::string label = "zero_energy jastrow on " + pair_label(j);
    if (ctx.dim != 2)
        throw std::runtime_error(label + ": K0(kappa r^-s) is the 2-D zero-energy "
                                 "solution; this cell has dimension " +
                                 std::to_string(ctx.dim));
    if (ctx.ham == nullptr)
        throw std::runtime_error(label + ": needs the Hamiltonian, whose C/r^p core "
                                 "sets kappa and s");
    const PowerLawCore core = dominant_core(*ctx.ham, ga, gb);
    if (core.coefficient == 0.0 || core.exponent <= 2.0)
        throw std::runtime_error(
            label + ": the Hamiltonian has no C/r^p core with p > 2 on this pair (an "
            "inverse_power term); a 1/r or softer pair takes a padé or drummond jastrow");
    if (core.coefficient < 0.0)
        throw std::runtime_error(label + ": the core C/r^p is attractive (C = " +
                                 std::to_string(core.coefficient) + "); the pair "
                                 "collapses and has no zero-energy scattering solution");
    const double ma = ga.mass.empty() ? 1.0 : ga.mass[0];
    const double mb = gb.mass.empty() ? 1.0 : gb.mass[0];
    const double mu = ma * mb / (ma + mb);  // reduced mass

    double cutoff = j.cutoff;
    if (ctx.cell.is_periodic()) {
        const double rws = ctx.cell.ws_radius();
        if (cutoff <= 0.0) cutoff = rws;
        if (cutoff > rws)
            throw std::runtime_error(
                label + ": cutoff " + std::to_string(cutoff) + " exceeds the "
                "Wigner-Seitz inradius " + std::to_string(rws) + " of the cell; beyond "
                "it the minimum image no longer sums every periodic image.");
    } else if (!(cutoff > 0.0)) {
        throw std::runtime_error(label + ": an open cell needs `cutoff:` (the range "
                                 "r_c where u reaches 0)");
    }
    if (!(j.matching_radius > 0.0 && j.matching_radius < cutoff))
        throw std::runtime_error(label + ": matching_radius " +
                                 std::to_string(j.matching_radius) +
                                 " must lie inside (0, cutoff = " +
                                 std::to_string(cutoff) + ")");
    return {2.0 * mu * core.coefficient, core.exponent, cutoff};
}

// u(r) = ln[C₁ K₀(κ r^{−s})] inside R_m, ln C₂ − C₃/r − C₃/(2r_c − r) out to
// r_c (wavefunction/zero_energy_jastrow.h).  params: [R_m], sign-locked so the
// optimiser keeps it positive; rebuild() refuses it past r_c.
std::unique_ptr<WavefunctionComponent> build_zero_energy(
    const io::JastrowSpec& j, const ParticleGroup& ga, const ParticleGroup& gb,
    const WavefunctionContext& ctx) {
    const ZeroEnergyInputs in = zero_energy_inputs(j, ga, gb, ctx);
    io::Logger::info("zero_energy jastrow on " + pair_label(j) + ": core b = 2*mu*C = " +
                     std::to_string(in.b) + ", p = " + std::to_string(in.exponent) +
                     ", cutoff r_c = " + std::to_string(in.cutoff) +
                     ", matching radius R_m = " + std::to_string(j.matching_radius));
    std::vector<Parameter> params{
        Parameter{"R_m", j.matching_radius, j.matching_radius_fixed, /*sign_locked=*/true}};
    return std::make_unique<ZeroEnergyJastrow>(in.b, in.exponent, in.cutoff, ga, gb,
                                               std::move(params));
}

bool has_ewald(const Hamiltonian* ham) {
    if (ham == nullptr) return false;
    for (const auto& c : ham->components())
        if (c->name() == "ewald") return true;
    return false;
}

}  // namespace

// ũ(k) = 2π ∫₀^∞ r J₀(kr) u(r) dr of the pair function this spec builds (the
// constant u∞ of a Padé dropped: it only touches k = 0), with the cusp derived
// exactly as the component's.  Drummond: a finite integral over [0, L].  Padé:
// the inner integral to r_c plus its tail transform (numerics/pair_radial.h).
// With `pairs:` every pair shares u and its cusp (build_drummond checks), so
// the first pair stands for all of them.
double jastrow_pair_fourier(const io::JastrowSpec& j, const WavefunctionContext& ctx,
                            double k) {
    const std::array<std::string, 2> pair = io::jastrow_pairs(j).front();
    const ParticleGroup& ga = find_group(ctx.groups, pair[0]);
    const ParticleGroup& gb = find_group(ctx.groups, pair[1]);
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    // u ~ −κ r^{−s} at contact: ∫ r u dr diverges for p ≥ 6 and is singular at
    // r = 0 for every p > 2, so there is no quadrature here to trust.
    if (j.type == "zero_energy")
        throw std::runtime_error(
            "zero_energy jastrow on " + pair_label(j) + ": its Fourier transform is "
            "not available (u diverges at contact), so a kspace_jastrow on the same "
            "pair cannot be seeded from Gaskell's form; give it explicit `params`.");
    if (j.type == "drummond") {
        const double gamma = kato_cusp(ctx.ham, ga, gb, ctx.dim,
                                       coalescence_ell(ctx.cfg, pair[0], pair[1]));
        const DrummondFunction u(j.truncation, gamma, j.cutoff, j.params);
        return two_pi * numerics::gauss20_panels(
                            [&](double r) {
                                double v, d1, d2;
                                u.evaluate(r, v, d1, d2);
                                return r * numerics::bessel_j0(k * r) * v;
                            },
                            0.0, j.cutoff, 16 + static_cast<int>(k * j.cutoff));
    }
    const double rc = ctx.cell.is_periodic() ? ctx.cell.ws_radius() : 1.0;
    const numerics::RadialFunction f =
        numerics::pade_radial(pade_numerator(j, ga, gb, ctx), j.denominator, rc);
    const double inner = numerics::gauss20_panels(
        [&](double r) { return r * numerics::bessel_j0(k * r) * f.value(r); }, 0.0, rc,
        32 + static_cast<int>(k * rc));
    return two_pi * inner + f.tail(k);
}

// Under periodic boundaries every pair Jastrow is the sum over ALL periodic
// images of u (Ceperley, "Introduction to QMC methods applied to the electron
// gas", §2.5), never the minimum image of a long-ranged u, whose gradient
// jumps on the Wigner–Seitz boundary:
//   • Padé, with an Ewald term: summed by the optimized breakup
//     (wavefunction/periodic_pair_sum.h), like the potential it pairs with.
//   • Padé, no Ewald: refused — unless `truncation: minimum_image` makes the
//     whole run an explicit minimum-image model (geometry tests), which the
//     Jastrow then follows, with a warning about the kink.
//   • Drummond: its cutoff must stay within the inradius R_WS, where the
//     minimum image IS the exact image sum; bound_cutoff keeps the optimiser
//     from moving it out.
//   • zero_energy: the same rule for its fixed cutoff (zero_energy_inputs).
std::unique_ptr<WavefunctionComponent> build_component(
    const io::JastrowSpec& spec, const WavefunctionContext& ctx) {
    const bool periodic = ctx.cell.is_periodic();
    const double rws = ctx.cell.ws_radius();

    if (spec.type == "drummond") {
        if (periodic && spec.cutoff > rws)
            throw std::runtime_error(
                "drummond jastrow on " + pair_label(spec) + ": cutoff " +
                std::to_string(spec.cutoff) + " exceeds the Wigner-Seitz inradius " +
                std::to_string(rws) + " of the cell; beyond it the minimum image no "
                "longer sums every periodic image.");
        auto c = build_drummond(spec, ctx);
        if (periodic) static_cast<DrummondJastrow&>(*c).bound_cutoff(rws);
        return c;
    }

    // Padé and zero_energy take one `pair:` (the parser refuses `pairs:`).
    const ParticleGroup& ga = find_group(ctx.groups, spec.pair[0]);
    const ParticleGroup& gb = find_group(ctx.groups, spec.pair[1]);
    if (spec.type == "zero_energy") return build_zero_energy(spec, ga, gb, ctx);

    auto c = build_pade(spec, ga, gb, ctx);
    if (!periodic) return c;
    auto& pade = static_cast<PadeJastrow&>(*c);
    if (has_ewald(ctx.ham)) {
        try {
            pade.sum_over_images(ctx.cell);
        } catch (const std::invalid_argument& e) {
            throw std::runtime_error("jastrow on " + pair_label(spec) +
                                     " cannot be summed over periodic images: " + e.what());
        }
    } else if (ctx.cfg.lattice.truncation == "minimum_image") {
        io::Logger::warn("padé jastrow on " + pair_label(spec) +
                         " follows `truncation: minimum_image`: it is read at the "
                         "nearest image, so its gradient jumps on the Wigner-Seitz "
                         "boundary.  Geometry testing only.");
    } else {
        throw std::runtime_error(
            "padé jastrow on " + pair_label(spec) + " in a periodic cell without an "
            "Ewald term: a long-ranged u must be summed over periodic images like the "
            "potential, and there is no long-range term to pair it with.  Use a "
            "drummond jastrow (cutoff within the Wigner-Seitz inradius), or set "
            "`truncation: minimum_image` for a minimum-image model.");
    }
    return c;
}

}  // namespace alea
