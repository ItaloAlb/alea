// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Config → trial-wavefunction components, one builder per `wavefunction:` item
// kind.  build_wavefunction (qmc/app.cpp) visits the ordered item list and
// calls the build_component overload for each item; every overload returns
// exactly one component, which is what makes item i become component i.
//
// Adding a kind: add its spec to io::WavefunctionItem (io/parser.h) and a
// build_component overload here, implemented in its own file in this
// directory.  A kind without an overload does not compile.

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../io/parser.h"
#include "common.h"
#include "../population/particle_group.h"
#include "../population/simulation_cell.h"
#include "../wavefunction/wavefunction_component.h"

namespace alea {

class Hamiltonian;

// Everything a builder may read besides its own spec.
struct WavefunctionContext {
    const io::Config&                 cfg;
    const std::vector<ParticleGroup>& groups;
    const Hamiltonian*                ham;    // nullptr → no cusp derivation
    const SimulationCell&             cell;
    std::size_t                       dim;
};

[[nodiscard]] std::unique_ptr<WavefunctionComponent> build_component(
    const io::JastrowSpec& spec, const WavefunctionContext& ctx);
[[nodiscard]] std::unique_ptr<WavefunctionComponent> build_component(
    const io::SlaterSpec& spec, const WavefunctionContext& ctx);
[[nodiscard]] std::unique_ptr<WavefunctionComponent> build_component(
    const io::MoireJastrowSpec& spec, const WavefunctionContext& ctx);
[[nodiscard]] std::unique_ptr<WavefunctionComponent> build_component(
    const io::CenterOfMassMomentumSpec& spec, const WavefunctionContext& ctx);
[[nodiscard]] std::unique_ptr<WavefunctionComponent> build_component(
    const io::KSpaceJastrowSpec& spec, const WavefunctionContext& ctx);
[[nodiscard]] std::unique_ptr<WavefunctionComponent> build_component(
    const io::NosanowSpec& spec, const WavefunctionContext& ctx);
[[nodiscard]] std::unique_ptr<WavefunctionComponent> build_component(
    const io::PairingSpec& spec, const WavefunctionContext& ctx);

// ũ(k) = 2π ∫ r J₀(kr) u(r) dr of the pair function a `jastrow` spec builds
// (cusp derived as for the component; a Padé's constant u∞ dropped).  Lets a
// long-range term seed only what the short-range one does not carry.  Throws
// for a zero_energy spec, whose u diverges at contact.
[[nodiscard]] double jastrow_pair_fourier(const io::JastrowSpec& spec,
                                          const WavefunctionContext& ctx, double k);

// ── Shared with App's cell/walker setup ─────────────────────────────────────

// Parameter vector with the project's defaults: a parameter is FIXED unless
// is_fixed says otherwise, can_flip_sign defaults to true, and a sign lock
// requested on a value of exactly 0 is downgraded with a warning (sign(0) is
// undefined).  `label` prefixes that warning.
[[nodiscard]] std::vector<Parameter> make_parameters(
    const std::vector<double>& values, const std::vector<bool>& is_fixed,
    const std::vector<bool>& can_flip_sign, const std::string& label);

// A twist t (fractional reciprocal coordinates) and t + integer are the same
// boundary condition, so Γ is every integer t — and "no twist" (empty).
[[nodiscard]] bool is_gamma_twist(const std::vector<double>& t);

// A determinant in CRYSTAL mode holds one electron per site: generated sites
// (`sites: triangular`), or explicit `centers` with ONE shared width `alpha`.
// Explicit centers with per-orbital `alphas` keep the molecular meaning — a
// shared orbital pool that both spin blocks draw from.
[[nodiscard]] bool is_crystal_slater(const io::SlaterSpec& s);

// One site per particle of group g, [g.size() * dim]: generated (`sites:
// triangular`, commensurate with a hexagonal periodic cell) or the explicit
// `centers` list.  `label` prefixes the errors.
[[nodiscard]] std::vector<double> lattice_sites(
    const std::string& sites, const std::vector<std::vector<double>>& centers,
    const ParticleGroup& g, const SimulationCell& cell, std::size_t dim,
    const std::string& label);

// The site of every particle of a crystal-mode determinant, [g.size() * dim].
[[nodiscard]] std::vector<double> crystal_sites(const io::SlaterSpec& s,
                                                const ParticleGroup& g,
                                                const SimulationCell& cell,
                                                std::size_t dim);

// (a_M, theta0) of a moiré Jastrow: explicit, or inherited from the
// Hamiltonian's `moire` term(s), which must then agree on the superlattice.
[[nodiscard]] std::pair<double, double> moire_jastrow_geometry(
    const io::Config& cfg, const io::MoireJastrowSpec& m);

}  // namespace alea
