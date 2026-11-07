// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `- kspace_jastrow:` items → a KSpaceJastrow on a group pair: one coefficient
// per |k| shell, seeded from the Gaskell RPA form when the config gives none.
#include "builders/wavefunction_builders.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "hamiltonian/hamiltonian.h"
#include "io/logger.h"
#include "numerics/gaskell.h"
#include "wavefunction/kspace_jastrow.h"

namespace alea {

namespace {

// Σ over the Hamiltonian's terms of q_a q_b Ṽ(k) for this pair.
double pair_fourier(const Hamiltonian* ham, const ParticleGroup& a, const ParticleGroup& b,
                    double k) {
    if (ham == nullptr) return 0.0;
    double v = 0.0;
    for (const auto& c : ham->components()) v += c->pair_fourier(a, b, k);
    return v;
}

// S₀(k) of the group without correlation: free fermions (by spin) when the
// group carries a Slater determinant, 1 (no structure) otherwise.
double uncorrelated_sofk(const io::Config& cfg, const ParticleGroup& g, double k,
                         double area) {
    bool has_det = false;
    for (const io::SlaterSpec* s : io::items_of<io::SlaterSpec>(cfg.wavefunction))
        if (s->group == g.name) has_det = true;
    if (!has_det) return 1.0;
    int up = 0, down = 0;
    for (const io::ParticleSpec& p : cfg.particles)
        if (p.group == g.name) { up += p.up; down += p.down; }
    return numerics::free_fermion_sofk_2d(k, up, down, area);
}

}  // namespace

// Gaskell seed, per shell s at |k_s| (numerics/gaskell.h):
//     p_s = [−ũ_C(k_s) − ũ_short(k_s)] / A,
// since ln Ψ ⊃ Σ_{i<j} u(r_ij) = (1/A) Σ_{k>0} ũ(k)|ρ_k|² + const for a periodic
// pair function u, and the Gaskell form is written for Ψ = exp(−Σ u_C).  ρ is
// the group's density, ṽ the Hamiltonian's own pair potential, λ = 1/(2m).
//
// ũ_short is the transform of every `jastrow` already on this pair.  In RPA
// the pseudopotentials add, so the long-range term supplies only the
// remainder; seeding the full Gaskell form on top of a short-range Jastrow
// counts the correlation twice.  Measured on HEG2D rs = 30 (N = 61, Drummond
// L = 105): S(k) fell to 0.6–0.75 of the plasmon value and E_VMC rose by
// 0.0067 Ha (3 shells) and 0.067 Ha (8 shells).
//
// A two-group pair has no one-component RPA form: it starts at zero.
std::unique_ptr<WavefunctionComponent> build_component(const io::KSpaceJastrowSpec& spec,
                                                       const WavefunctionContext& ctx) {
    const ParticleGroup& ga = find_group(ctx.groups, spec.pair[0]);
    const ParticleGroup& gb = find_group(ctx.groups, spec.pair[1]);
    if (!ctx.cell.is_periodic())
        throw std::runtime_error("kspace_jastrow on [" + spec.pair[0] + ", " + spec.pair[1] +
                                 "]: needs a periodic cell (its terms are reciprocal-"
                                 "lattice plane waves)");
    if (spec.shells < 1)
        throw std::runtime_error("kspace_jastrow: shells must be >= 1");
    const std::size_t n = static_cast<std::size_t>(spec.shells);

    // Build once with zero coefficients to learn the shells' |k|.
    std::vector<Parameter> zero;
    for (std::size_t s = 0; s < n; ++s)
        zero.push_back({"p" + std::to_string(s + 1), 0.0, true});
    auto probe = std::make_unique<KSpaceJastrow>(ctx.cell, ga, gb, n, zero);

    std::vector<double> values = spec.params;
    if (values.empty()) {
        values.assign(n, 0.0);
        if (ga == gb) {
            const double area = ctx.cell.volume();
            const double rho  = static_cast<double>(ga.size()) / area;
            const double inv_m = ga.inv_mass.empty() ? 1.0 : ga.inv_mass[0];
            for (std::size_t s = 0; s < n; ++s) {
                const double k  = probe->shell_k()[s];
                const double s0 = uncorrelated_sofk(ctx.cfg, ga, k, area);
                const double v  = pair_fourier(ctx.ham, ga, ga, k);
                double u_short = 0.0;
                for (const io::JastrowSpec* j : io::items_of<io::JastrowSpec>(ctx.cfg.wavefunction))
                    for (const auto& jp : io::jastrow_pairs(*j))
                        if ((jp[0] == spec.pair[0] && jp[1] == spec.pair[1]) ||
                            (jp[0] == spec.pair[1] && jp[1] == spec.pair[0]))
                            u_short += jastrow_pair_fourier(*j, ctx, k);
                values[s] = (-numerics::gaskell_u(k, rho, s0, v, inv_m) - u_short) / area;
            }
        } else {
            io::Logger::info("kspace_jastrow on [" + spec.pair[0] + ", " + spec.pair[1] +
                             "]: two-group pair, coefficients start at zero");
        }
    } else if (values.size() != n) {
        throw std::runtime_error("kspace_jastrow: " + std::to_string(values.size()) +
                                 " params for " + std::to_string(n) + " shells");
    }

    std::vector<Parameter> params =
        make_parameters(values, spec.is_fixed, spec.can_flip_sign, "kspace_jastrow");
    for (std::size_t s = 0; s < n; ++s) params[s].name = "p" + std::to_string(s + 1);
    return std::make_unique<KSpaceJastrow>(ctx.cell, ga, gb, n, std::move(params));
}

}  // namespace alea
