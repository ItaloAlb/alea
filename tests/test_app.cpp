// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "hamiltonian/ewald_term.h"
#include "hamiltonian/moire.h"
#include "builders/exchange_classes.h"
#include "io/parser.h"
#include "population/walker_set.h"
#include "qmc/app.h"
#include "numerics/gaskell.h"
#include "numerics/random.h"
#include "wavefunction/drummond_jastrow.h"
#include "wavefunction/kspace_jastrow.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/nosanow_jastrow.h"
#include "wavefunction/zero_energy_jastrow.h"

using alea::App;
using alea::build_groups;
using alea::build_hamiltonian;
using alea::build_vmc_config;
using alea::build_wavefunction;
using alea::WalkerSet;

namespace {

// The first wavefunction item of kind Spec, which the test knows is there.
template <class Spec>
Spec& first_item(alea::io::Config& c) {
    for (alea::io::WavefunctionItem& item : c.wavefunction)
        if (Spec* spec = std::get_if<Spec>(&item)) return *spec;
    throw std::logic_error("test config has no such wavefunction item");
}

// electron/hole exciton, Coulomb ε=1, both masses 1, charges ∓1, with a Padé
// u = a₁r/(1 + b₁r) at b₁ = 0.  a₁ is the Kato cusp derived from the
// Hamiltonian (−1 here), so built WITH the Hamiltonian this ψ = e^{−r} is the
// exact 2-D exciton ground state.
alea::io::Config make_exciton_cfg() {
    alea::io::Config c;
    c.lattice.dimension = 2;
    alea::io::ParticleSpec e;
    e.name = "e"; e.group = "electron_group"; e.mass = {1.0}; e.charge = -1.0;
    e.up = 1; e.down = 0;
    alea::io::ParticleSpec h;
    h.name = "h"; h.group = "hole_group"; h.mass = {1.0}; h.charge = +1.0;
    h.up = 1; h.down = 0;
    c.particles = {e, h};
    alea::io::JastrowSpec j;
    j.type = "pade"; j.pair = {"electron_group", "hole_group"};
    j.denominator = {0.0}; j.is_fixed = {true};
    c.wavefunction = {j};
    alea::io::CoulombSpec cs;
    cs.pair = {"electron_group", "hole_group"}; cs.epsilon = 1.0;
    c.coulombs = {cs};
    return c;
}

}  // namespace

TEST_CASE("a Pade jastrow's a1 is the Kato cusp, derived and fixed",
          "[app][pade][cusp]") {
    // a₁ = du/dr|₀ is the cusp.  It is not a config input: the builder derives
    // it from the Hamiltonian, Γ = 2μR/(2ℓ+d−1), and pins it, whatever else the
    // optimiser is allowed to move.
    auto a1_of = [](const alea::io::Config& cfg) {
        const auto groups = build_groups(cfg);
        const auto ham    = build_hamiltonian(cfg, groups);
        alea::TrialWavefunction psi = build_wavefunction(cfg, groups, &ham);
        REQUIRE(psi.components().size() == 1);
        const auto params = psi.components()[0]->parameters();
        REQUIRE(params.size() == 2);                 // [a₁, b₁]
        CHECK(params[0].is_fixed);
        CHECK_FALSE(params[1].is_fixed);
        return params[0].value;
    };

    // Coulomb e–h, μ = ½, R = −1, 2-D: Γ = 2·½·(−1)/1 = −1.
    alea::io::Config coulomb = make_exciton_cfg();
    first_item<alea::io::JastrowSpec>(coulomb).denominator = {0.3};
    first_item<alea::io::JastrowSpec>(coulomb).is_fixed    = {false};
    CHECK(a1_of(coulomb) == Catch::Approx(-1.0));

    // ε = 2 halves R, and a heavier hole (m_h = 3, μ = ¾) scales Γ by 2μ.
    alea::io::Config scaled = coulomb;
    scaled.coulombs[0].epsilon = 2.0;
    scaled.particles[1].mass   = {3.0};
    CHECK(a1_of(scaled) == Catch::Approx(2.0 * 0.75 * (-0.5)));

    // Rytova–Keldysh has a log singularity, no 1/r pole: R = 0 → cuspless.
    alea::io::Config keldysh = coulomb;
    keldysh.coulombs.clear();
    alea::io::KeldyshSpec ks;
    ks.pair = {"electron_group", "hole_group"}; ks.epsilon = 1.0; ks.r0 = 5.0;
    keldysh.keldysh = {ks};
    CHECK(a1_of(keldysh) == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("the drummond cusp drops to the P-wave value behind a determinant",
          "[app][drummond][cusp]") {
    // Two IDENTICAL fermions with a Slater determinant: ψ already vanishes
    // linearly as they coalesce, so the pair is in a P-wave and the Kato cusp is
    // Γ = 2μR/(2ℓ+d−1) with ℓ = 1, not the ℓ = 0 value.  In 2-D, with μ = ½ and
    // R = q², that is 1/3 rather than 1.  Handing the Jastrow the S-wave value
    // behind a determinant overshoots the cusp and leaves E_L divergent at
    // coalescence — the failure this derivation exists to prevent.
    auto cfg_for = [](int up, int down, bool determinant) {
        alea::io::Config c;
        c.lattice.dimension = 2;
        alea::io::ParticleSpec e;
        e.name = "e"; e.group = "e_grp"; e.type = "fermions";
        e.mass = {1.0}; e.charge = -1.0; e.up = up; e.down = down;
        c.particles = {e};
        alea::io::JastrowSpec j;
        j.type = "drummond"; j.pair = {"e_grp", "e_grp"};
        j.truncation = 3; j.cutoff = 6.0; j.cutoff_fixed = true;
        j.params = {0.0}; j.is_fixed = {true};
        c.wavefunction = {j};
        alea::io::CoulombSpec cs;
        cs.pair = {"e_grp", "e_grp"}; cs.epsilon = 1.0;
        c.coulombs = {cs};
        if (determinant) {
            alea::io::SlaterSpec s;
            s.group = "e_grp";
            s.orbital_type = "gaussian";
            const int n = std::max(up, down);
            for (int i = 0; i < n; ++i) {
                s.centers.push_back({2.0 * i, 0.0});
                s.alphas.push_back(0.5);
            }
            c.wavefunction.push_back(s);
        }
        return c;
    };

    auto cusp_of = [](const alea::io::Config& cfg) {
        const auto groups = build_groups(cfg);
        const auto ham    = build_hamiltonian(cfg, groups);
        alea::TrialWavefunction psi = build_wavefunction(cfg, groups, &ham);
        for (const auto& comp : psi.components())
            if (auto* d = dynamic_cast<const alea::DrummondJastrow*>(comp.get()))
                return d->cusp();
        FAIL("no drummond jastrow in the built wavefunction");
        return 0.0;
    };

    // No determinant → nodeless → S-wave.
    CHECK(cusp_of(cfg_for(2, 0, false)) == Catch::Approx(1.0));
    // Fully polarised determinant → every pair is same-spin → P-wave.
    CHECK(cusp_of(cfg_for(2, 0, true)) == Catch::Approx(1.0 / 3.0));
    // Both spins behind one determinant: same-spin pairs want 1/3 and
    // opposite-spin pairs want 1, and one u(r) cannot hold both.  The builder
    // keeps the S-wave value rather than silently picking for half the pairs.
    CHECK(cusp_of(cfg_for(2, 2, true)) == Catch::Approx(1.0));

    // A Padé on the same pair gets the same derived value as its a₁.
    auto pade_a1_of = [](alea::io::Config cfg) {
        alea::io::JastrowSpec& j = first_item<alea::io::JastrowSpec>(cfg);
        j.type = "pade"; j.params.clear(); j.denominator = {0.0}; j.is_fixed = {true};
        const auto groups = build_groups(cfg);
        const auto ham    = build_hamiltonian(cfg, groups);
        alea::TrialWavefunction psi = build_wavefunction(cfg, groups, &ham);
        return psi.components()[0]->parameters()[0].value;
    };
    CHECK(pade_a1_of(cfg_for(2, 0, false)) == Catch::Approx(1.0));
    CHECK(pade_a1_of(cfg_for(2, 0, true)) == Catch::Approx(1.0 / 3.0));
}

TEST_CASE("a drummond item over a list of group pairs takes their common cusp",
          "[app][drummond][cusp][pairs]") {
    // Four polarized groups of an electron-hole bilayer (holes at z = 0.4).
    // Same-layer pairs coalesce (Coulomb 1/r, Γ = 1 with no determinant); e-h
    // pairs are d apart (no 1/r pole, Γ = 0).  One u(r) holds one cusp, so a
    // list that mixes the two is refused.
    auto cfg_with = [](std::vector<std::array<std::string, 2>> pairs) {
        alea::io::Config c;
        c.lattice.dimension = 2;
        for (const char* name : {"e_up", "e_dn", "h_up", "h_dn"}) {
            alea::io::ParticleSpec p;
            p.name = name; p.group = name; p.type = "fermions"; p.mass = {1.0};
            const bool hole = name[0] == 'h';
            p.charge = hole ? 1.0 : -1.0;
            p.z = hole ? 0.4 : 0.0;
            p.up = 2; p.down = 0;
            c.particles.push_back(p);
        }
        for (const auto& pair : std::vector<std::array<std::string, 2>>{
                 {"e_up", "e_up"}, {"h_dn", "h_dn"}, {"e_up", "h_dn"}, {"e_dn", "h_up"}}) {
            alea::io::CoulombSpec cs;
            cs.pair = pair; cs.epsilon = 1.0;
            c.coulombs.push_back(cs);
        }
        alea::io::JastrowSpec j;
        j.type = "drummond"; j.pairs = std::move(pairs);
        j.truncation = 3; j.cutoff = 6.0; j.cutoff_fixed = true;
        j.params = {0.0}; j.is_fixed = {true};
        c.wavefunction = {j};
        return c;
    };
    auto cusp_of = [](const alea::io::Config& cfg) {
        const auto groups = build_groups(cfg);
        const auto ham    = build_hamiltonian(cfg, groups);
        alea::TrialWavefunction psi = build_wavefunction(cfg, groups, &ham);
        REQUIRE(psi.components().size() == 1);
        return dynamic_cast<const alea::DrummondJastrow&>(*psi.components()[0]).cusp();
    };

    CHECK(cusp_of(cfg_with({{"e_up", "e_up"}, {"h_dn", "h_dn"}})) == Catch::Approx(1.0));
    CHECK(cusp_of(cfg_with({{"e_up", "h_dn"}, {"e_dn", "h_up"}})) ==
          Catch::Approx(0.0).margin(1e-15));

    using Catch::Matchers::ContainsSubstring;
    const auto mixed = cfg_with({{"e_up", "e_up"}, {"e_up", "h_dn"}});
    const auto groups = build_groups(mixed);
    const auto ham    = build_hamiltonian(mixed, groups);
    CHECK_THROWS_WITH(build_wavefunction(mixed, groups, &ham),
                      ContainsSubstring("one cusp"));
    const auto twice = cfg_with({{"e_up", "h_dn"}, {"h_dn", "e_up"}});
    CHECK_THROWS_WITH(build_wavefunction(twice, build_groups(twice), &ham),
                      ContainsSubstring("given twice"));
}

TEST_CASE("a pairing determinant puts its row and column groups in a P-wave",
          "[app][pairing][cusp]") {
    // Exchanging two e_up (two rows) or two h_dn (two columns) flips det D,
    // so same-spin pairs of those groups get the P-wave cusp 2μR/3 = 1/3,
    // although φ only couples e to h; opposite spins stay S-wave.
    const auto cfg = alea::io::parse_config(R"(
system:
  lattice: { type: open, dimension: 2 }
  particles:
    - { name: e_up, group: e_up, type: fermions, mass: 1.0, charge: -1, up: 2 }
    - { name: e_dn, group: e_dn, type: fermions, mass: 1.0, charge: -1, up: 2 }
    - { name: h_up, group: h_up, type: fermions, mass: 1.0, charge: 1, z: 0.4, up: 2 }
    - { name: h_dn, group: h_dn, type: fermions, mass: 1.0, charge: 1, z: 0.4, up: 2 }
hamiltonian:
  terms:
    - { type: coulomb, pair: [e_up, e_up], epsilon: 1.0 }
    - { type: coulomb, pair: [h_dn, h_dn], epsilon: 1.0 }
    - { type: coulomb, pair: [e_up, e_dn], epsilon: 1.0 }
wavefunction:
  - pairing:
      blocks: [{ rows: e_up, columns: h_dn }, { rows: e_dn, columns: h_up }]
      orbital: { poly_order: 2, cutoff: 5.0 }
  - jastrow: { type: drummond, pairs: [[e_up, e_up], [h_dn, h_dn]], params: [0.0], cutoff: 4.0 }
  - jastrow: { type: drummond, pair: [e_up, e_dn], params: [0.0], cutoff: 4.0 }
)");
    const auto groups = build_groups(cfg);
    const auto ham = build_hamiltonian(cfg, groups);
    alea::TrialWavefunction psi = build_wavefunction(cfg, groups, &ham);
    REQUIRE(psi.components().size() == 3);
    CHECK(psi.components()[0]->is_nodal());
    // [c0, c2, L]: c0 fixed (the scale), c2 fixed by default, L free.
    const auto params = psi.components()[0]->parameters();
    REQUIRE(params.size() == 3);
    CHECK(params[0].name == "c0");
    CHECK(params[2].name == "L");
    CHECK_FALSE(params[2].is_fixed);
    CHECK(dynamic_cast<const alea::DrummondJastrow&>(*psi.components()[1]).cusp() ==
          Catch::Approx(1.0 / 3.0));
    CHECK(dynamic_cast<const alea::DrummondJastrow&>(*psi.components()[2]).cusp() ==
          Catch::Approx(1.0));
}

TEST_CASE("a pairing determinant starts each column particle next to its row partner",
          "[app][pairing]") {
    // φ vanishes beyond L = 4 while particles start in a box of side 60: from
    // uniform starts nearly every φ(r_e − r_h) is zero, the determinant starts
    // at zero and VMC accepts nothing.
    const auto cfg = alea::io::parse_config(R"(
project: { seed: 7 }
system:
  lattice: { type: open, dimension: 2, L: 60.0 }
  particles:
    - { name: e, group: e, type: fermions, mass: 1.0, charge: -1, up: 3 }
    - { name: h, group: h, type: fermions, mass: 1.0, charge: 1, z: 0.4, up: 3 }
wavefunction:
  - pairing:
      blocks: [{ rows: e, columns: h }]
      orbital: { poly_order: 2, cutoff: 4.0, cutoff_fixed: true }
hamiltonian:
  terms:
    - { type: coulomb, pair: [e, h], epsilon: 1.0 }
qmc:
  - { method: vmc, threads: 1, samples: 16, equilibration_blocks: 2,
      accumulation_blocks: 4, steps_per_block: 5, time_step: 0.5 }
)");
    const auto groups = build_groups(cfg);
    const alea::PairPartners partners =
        alea::pairing_partners(cfg, groups, alea::build_cell(cfg, 6));
    REQUIRE(partners.partner.size() == 6);
    for (std::size_t j = 0; j < 3; ++j) {
        CHECK(partners.partner[j] == -1);
        CHECK(partners.partner[3 + j] == std::ptrdiff_t(j));
    }
    CHECK(partners.spread == 0.5);

    auto pack = App(cfg).build_drivers();
    for (const auto& w : pack.ensembles[0]->walkers()) {
        const auto r = w.positions();
        for (std::size_t j = 0; j < 3; ++j)
            CHECK(std::hypot(r[(3 + j) * 2] - r[j * 2], r[(3 + j) * 2 + 1] - r[j * 2 + 1]) <
                  4.0);
    }
    auto* vmc = dynamic_cast<alea::VMC*>(pack.drivers[0].get());
    REQUIRE(vmc != nullptr);
    vmc->run();
    CHECK(std::isfinite(vmc->energy()));
    CHECK(vmc->acceptance_ratio() > 0.1);

    // An extra row (the unpaired electron of an N+½ cell) has no partner.
    const auto charged = alea::io::parse_config(R"(
system:
  lattice: { type: periodic, dimension: 2, L: 60.0, shape: square, capacitor: exclude }
  particles:
    - { name: e, group: e, type: fermions, mass: 1.0, charge: -1, up: 4 }
    - { name: h, group: h, type: fermions, mass: 1.0, charge: 1, z: 0.4, up: 3 }
wavefunction:
  - pairing:
      blocks: [{ rows: e, columns: h, unpaired: { type: plane_wave, k: [[0, 0]] } }]
      orbital: { shells: 2, params: [1.0, 0.5] }
)");
    const auto cgroups = build_groups(charged);
    const auto ccell = alea::build_cell(charged, 7);
    const alea::PairPartners p7 = alea::pairing_partners(charged, cgroups, ccell);
    REQUIRE(p7.partner.size() == 7);
    CHECK(p7.partner[3] == -1);
    CHECK(p7.partner[6] == 2);
    // Cosines only: φ has no cutoff, and the inradius sets the scale.
    CHECK(p7.spread == ccell.ws_radius() / 8.0);
}

TEST_CASE("the pairing builder refuses ill-posed determinants", "[app][pairing]") {
    using Catch::Matchers::ContainsSubstring;
    auto build = [](const std::string& particles, const std::string& pairing,
                    bool periodic = true) {
        const std::string lattice = periodic
            ? "  lattice: { type: periodic, dimension: 2, L: 6.0, shape: square }\n"
            : "  lattice: { type: open, dimension: 2 }\n";
        const auto cfg = alea::io::parse_config("system:\n" + lattice + "  particles:\n" +
                                                particles + "wavefunction:\n  - pairing:\n" +
                                                pairing);
        const auto groups = build_groups(cfg);
        const auto cell = alea::build_cell(cfg, 0);
        return build_wavefunction(cfg, groups, nullptr, cell);
    };
    const std::string eh =
        "    - { name: e, group: e, type: fermions, charge: -1, up: 3 }\n"
        "    - { name: h, group: h, type: fermions, charge: 1, up: 3 }\n";
    const std::string blocks = "      blocks: [{ rows: e, columns: h }]\n";
    CHECK_NOTHROW(build(eh, blocks + "      orbital: { shells: 2, poly_order: 2 }\n"));
    CHECK_THROWS_WITH(build("    - { name: e, group: e, type: fermions, charge: -1, up: 2, down: 1 }\n"
                            "    - { name: h, group: h, type: fermions, charge: 1, up: 3 }\n",
                            blocks + "      orbital: { poly_order: 2 }\n"),
                      ContainsSubstring("both spins"));
    CHECK_THROWS_WITH(build("    - { name: e, group: e, type: bosons, charge: -1, up: 3 }\n"
                            "    - { name: h, group: h, type: fermions, charge: 1, up: 3 }\n",
                            blocks + "      orbital: { poly_order: 2 }\n"),
                      ContainsSubstring("bosons"));
    CHECK_THROWS_WITH(build(eh, blocks + "      orbital: { poly_order: 2, params: [1.0, 0.0],\n"
                                         "                 is_fixed: [false, false] }\n"),
                      ContainsSubstring("every coefficient is free"));
    CHECK_THROWS_WITH(build(eh, blocks + "      orbital: { shells: 2 }\n"),
                      ContainsSubstring("explicit"));
    CHECK_THROWS_WITH(build(eh, blocks + "      orbital: { shells: 2, poly_order: 2 }\n", false),
                      ContainsSubstring("periodic 2-D"));
    CHECK_THROWS_WITH(build(eh, blocks + "      orbital: { poly_order: 2, cutoff: 3.5 }\n"),
                      ContainsSubstring("Wigner-Seitz"));
    const std::string four_two =
        "    - { name: e, group: e, type: fermions, charge: -1, up: 4 }\n"
        "    - { name: h, group: h, type: fermions, charge: 1, up: 2 }\n";
    CHECK_THROWS_WITH(build(four_two, blocks + "      orbital: { poly_order: 2 }\n"),
                      ContainsSubstring("unpaired"));
    CHECK_THROWS_WITH(build(four_two,
                            "      blocks: [{ rows: e, columns: h, unpaired: { type: plane_wave,\n"
                            "                 k: [[1, 0], [-1, 0]] } }]\n"
                            "      orbital: { poly_order: 2 }\n"),
                      ContainsSubstring("same function"));
    CHECK_NOTHROW(build(four_two,
                        "      blocks: [{ rows: e, columns: h, unpaired: { type: plane_wave,\n"
                        "                 k: [[1, 0], [-1, 0]], sin: [false, true] } }]\n"
                        "      orbital: { poly_order: 2 }\n"));
}

TEST_CASE("build_vmc_config maps the qmc block onto a VMCConfig", "[app]") {
    alea::io::Config cfg;
    cfg.seed = 7;
    alea::io::QmcSpec q;
    q.method = "vmc";
    q.random_walk = "metropolis";
    q.threads = 2;
    q.samples = 128;
    q.equilibration_blocks = 10;
    q.accumulation_blocks = 20;
    q.steps_per_block = 5;
    q.sub_steps = 3;
    q.time_step = 0.02;
    cfg.qmc.push_back(q);

    auto vc = build_vmc_config(cfg.qmc[0], cfg.seed);

    CHECK(vc.n_equilibration_blocks == 10);
    CHECK(vc.n_accumulation_blocks == 20);
    CHECK(vc.n_steps == 5);
    CHECK(vc.n_substeps == 3);
    CHECK(vc.tau == 0.02);
    CHECK(vc.use_drift == false);  // metropolis
    CHECK(vc.n_threads == 2);
    CHECK(vc.seed == 7);
}

TEST_CASE("build_optimize_config passes random_walk to the optimizer's sampler",
          "[app]") {
    alea::io::OptimizerSpec spec;  // default random_walk: metropolis
    CHECK_FALSE(alea::build_optimize_config(spec, 7).use_drift);
    spec.random_walk = "langevin";
    CHECK(alea::build_optimize_config(spec, 7).use_drift);
}

TEST_CASE("build_groups assigns contiguous ranges, charges and inverse masses",
          "[app]") {
    alea::io::Config cfg;
    cfg.lattice.dimension = 2;
    alea::io::ParticleSpec e;
    e.group = "electron_group";
    e.mass = {1.0};
    e.charge = -1.0;
    e.up = 1;
    e.down = 0;  // 1 electron
    alea::io::ParticleSpec h;
    h.group = "hole_group";
    h.mass = {2.0};
    h.charge = +1.0;
    h.up = 1;
    h.down = 1;  // 2 holes
    cfg.particles = {e, h};

    auto groups = build_groups(cfg);

    REQUIRE(groups.size() == 2);
    CHECK(groups[0].name == "electron_group");
    CHECK(groups[0].first == 0);
    CHECK(groups[0].last == 1);
    CHECK(groups[0].charge == -1.0);
    REQUIRE(groups[0].inv_mass.size() == 2);
    CHECK(groups[0].inv_mass[0] == 1.0);

    CHECK(groups[1].name == "hole_group");
    CHECK(groups[1].first == 1);
    CHECK(groups[1].last == 3);  // up + down = 2
    CHECK(groups[1].charge == 1.0);
    CHECK(groups[1].inv_mass[0] == 0.5);  // 1/2
}

TEST_CASE("build_wavefunction and build_hamiltonian give the right local energy",
          "[app]") {
    auto cfg = make_exciton_cfg();
    auto groups = build_groups(cfg);
    auto ham = build_hamiltonian(cfg, groups);
    auto psi = build_wavefunction(cfg, groups, &ham);

    // a₁ = −1 (derived) and b₁ = 0 make ψ = e^{−r}, the exact 2-D exciton ground
    // state, so E_L = −1 at every separation.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    for (const double sep : {0.3, 2.0, 7.0}) {
        INFO("separation " << sep);
        ws.positions()[2] = sep;
        ws.dist_tables().build_all(ws.positions());
        psi.evaluate_batch(ws);
        ham.evaluate_batch(ws);
        CHECK(ws.local_energies()[0] == Catch::Approx(-1.0));
    }
}

namespace {
constexpr const char* kBoostedExcitonYaml = R"(
system:
  lattice: { type: open, dimension: 2 }
  particles:
    - { name: e, group: electron_group, type: fermions, mass: 1.0, charge: -1, up: 1, down: 0 }
    - { name: h, group: hole_group, type: fermions, mass: 3.0, charge: +1, up: 1, down: 0 }
wavefunction:
  - jastrow: { type: pade, pair: [electron_group, hole_group], denominator: [0.6], is_fixed: [true] }
  - center_of_mass_momentum: { K: [0.8, -0.4] }
hamiltonian:
  terms:
    - { type: coulomb, pair: [electron_group, hole_group], epsilon: 1.0 }
)";
}  // namespace

TEST_CASE("a center_of_mass_momentum block boosts the trial by K", "[app][fixed_phase]") {
    const auto cfg = alea::io::parse_config(kBoostedExcitonYaml);
    const auto groups = build_groups(cfg);
    auto psi = build_wavefunction(cfg, groups);
    REQUIRE(psi.is_complex());

    // ∇_i φ = m_i K / M with M = 1 + 3.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    ws.positions()[2] = 1.0;
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);
    const double k[2][2] = {{0.2, -0.1}, {0.6, -0.3}};
    for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t d = 0; d < 2; ++d)
            CHECK(ws.phase_gradients()[p * 2 + d] == Catch::Approx(k[p][d]));
}

TEST_CASE("center_of_mass_momentum validates K and refuses periodic cells",
          "[app][fixed_phase]") {
    auto cfg = alea::io::parse_config(kBoostedExcitonYaml);
    first_item<alea::io::CenterOfMassMomentumSpec>(cfg).K = {1.0};   // 1 component in 2D
    REQUIRE_THROWS_WITH(build_wavefunction(cfg, build_groups(cfg)),
                        Catch::Matchers::ContainsSubstring("center_of_mass_momentum"));

    // e^{ik_i·r_i} is single-valued on a torus only for lattice k_i; until the
    // twist machinery exists a periodic boost is refused, not silently wrong.
    cfg = alea::io::parse_config(kBoostedExcitonYaml);
    cfg.lattice.type = "periodic";
    cfg.lattice.L = 10.0;
    cfg.lattice.shape = "square";
    REQUIRE_THROWS_WITH(build_wavefunction(cfg, build_groups(cfg)),
                        Catch::Matchers::ContainsSubstring("periodic"));
}

namespace {
std::string twisted_gas_yaml(const std::string& twist) {
    return std::string(R"(
system:
  lattice: { type: periodic, dimension: 2, shape: square, L: 4.0)") +
           (twist.empty() ? "" : ", twist: " + twist) + R"( }
  particles:
    - { name: e, group: electrons, type: fermions, mass: 1.0, charge: -1, up: 4, down: 2 }
wavefunction:
  - slater: { group: electrons, orbitals: { type: plane_wave } }
)";
}
}  // namespace

TEST_CASE("a twisted plane-wave slater builds a complex determinant",
          "[app][twist]") {
    const auto cfg = alea::io::parse_config(twisted_gas_yaml("[0.2, 0.1]"));
    REQUIRE(cfg.lattice.twist == std::vector<double>{0.2, 0.1});
    const auto groups = build_groups(cfg);
    const std::vector<double> cellv = {4.0, 0.0, 0.0, 4.0};
    const alea::SimulationCell cell(2, std::span<const double>(cellv));
    auto psi = build_wavefunction(cfg, groups, nullptr, cell);
    REQUIRE(psi.components().size() == 1);
    CHECK(psi.components()[0]->name() == "complex_slater_determinant");
    CHECK(psi.is_complex());

    // Free gas: E_L = Σ|k+θ|²/2 exactly.  Up fills 4 of |k+θ|²/b² =
    // 0.05, 0.65, 0.85, 1.25; down fills 2 → 0.05, 0.65.
    const double b = 2.0 * 3.14159265358979323846 / 4.0;
    const double exact = 0.5 * b * b * (0.05 + 0.65 + 0.85 + 1.25 + 0.05 + 0.65);
    auto ham = build_hamiltonian(cfg, groups, cell);
    WalkerSet ws(/*nw=*/2, /*np=*/6, /*dim=*/2, cell);
    for (std::size_t k = 0; k < ws.positions().size(); ++k)
        ws.positions()[k] = 2.0 + 1.5 * std::sin(1.9 * static_cast<double>(k));
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);
    ham.evaluate_batch(ws);
    for (std::size_t w = 0; w < 2; ++w)
        CHECK(ws.local_energies()[w] == Catch::Approx(exact).epsilon(1e-10));
}

TEST_CASE("an integer twist is the Gamma point and stays real", "[app][twist]") {
    // t and t + integer are the same boundary condition; at Γ the cos/sin
    // determinant is exact and cheaper, so no complex machinery is built.
    for (const std::string tw : {"", "[0.0, 0.0]", "[1.0, -2.0]"}) {
        auto yaml = twisted_gas_yaml(tw);
        // Γ needs closed shells: 5 up, 1 down on the square lattice.
        yaml.replace(yaml.find("up: 4, down: 2"), 14, "up: 5, down: 1");
        const auto cfg = alea::io::parse_config(yaml);
        const std::vector<double> cellv = {4.0, 0.0, 0.0, 4.0};
        auto psi = build_wavefunction(cfg, build_groups(cfg), nullptr,
                                      alea::SimulationCell(2, std::span<const double>(cellv)));
        REQUIRE(psi.components().size() == 1);
        CHECK(psi.components()[0]->name() == "slater_determinant");
        CHECK_FALSE(psi.is_complex());
    }
}

namespace {
// One electron (m = 0.7) in a parabolic dot ω₀ = 0.5 with B = 1.3, in the l = −1
// Fock–Darwin state at the exact width β = mΩ:  E = 2Ω + ω_c/2, ω_c = qB/m.
std::string fock_darwin_yaml(double beta_value) {
    // Full precision: std::to_string keeps 6 decimals, which is enough to knock
    // the orbital off the exact eigenstate at the 1e-7 level.
    char beta[32];
    std::snprintf(beta, sizeof(beta), "%.17g", beta_value);
    return R"(
system:
  lattice: { type: open, dimension: 2 }
  particles:
    - { name: e, group: electrons, type: fermions, mass: 0.7, charge: -1, up: 1, down: 0 }
wavefunction:
  - slater: { group: electrons, orbitals: { type: fock_darwin, l: [-1], beta: )" +
           std::string(beta) + R"( } }
hamiltonian:
  terms:
    - { type: harmonic_trap, omega: 0.5 }
    - { type: magnetic_field, B: 1.3 }
)";
}
}  // namespace

TEST_CASE("a Fock-Darwin dot in a magnetic field builds from YAML with its exact energy",
          "[app][gauge]") {
    const double m = 0.7, wc = -1.3 / m, Om = std::sqrt(0.25 + 0.25 * wc * wc);
    const auto cfg = alea::io::parse_config(fock_darwin_yaml(m * Om));
    const auto groups = build_groups(cfg);
    auto psi = build_wavefunction(cfg, groups);
    auto ham = build_hamiltonian(cfg, groups);
    REQUIRE(psi.is_complex());

    WalkerSet ws(/*nw=*/3, /*np=*/1, /*dim=*/2);
    for (std::size_t k = 0; k < ws.positions().size(); ++k)
        ws.positions()[k] = 0.8 * std::cos(1.3 * static_cast<double>(k));
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);
    ham.evaluate_batch(ws);
    for (std::size_t w = 0; w < 3; ++w)
        CHECK(ws.local_energies()[w] == Catch::Approx(2.0 * Om + wc / 2.0).epsilon(1e-10));
}

TEST_CASE("a rotation term adds the centrifugal trap by itself", "[app][gauge]") {
    // In a frame rotating at Ω the l-state has E = (|l|+1)ω₀ − Ωl; the −½mΩ²r²
    // term is part of `rotation`, not something the user must remember.
    std::string yaml = fock_darwin_yaml(0.7 * 0.5);
    yaml.replace(yaml.find("{ type: magnetic_field, B: 1.3 }"), 32,
                 "{ type: rotation, omega: 0.2 }");
    const auto cfg = alea::io::parse_config(yaml);
    const auto groups = build_groups(cfg);
    auto psi = build_wavefunction(cfg, groups);
    auto ham = build_hamiltonian(cfg, groups);
    WalkerSet ws(/*nw=*/2, /*np=*/1, /*dim=*/2);
    ws.positions()[0] = 0.9; ws.positions()[1] = -0.4;
    ws.positions()[2] = 0.3; ws.positions()[3] = 1.1;
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);
    ham.evaluate_batch(ws);
    for (std::size_t w = 0; w < 2; ++w)
        CHECK(ws.local_energies()[w] == Catch::Approx(2.0 * 0.5 + 0.2).epsilon(1e-10));
}

TEST_CASE("gauge fields are refused where they are not defined", "[app][gauge]") {
    auto cfg = alea::io::parse_config(fock_darwin_yaml(0.5));
    cfg.lattice.type = "periodic";   // B on a torus needs flux quantization
    cfg.lattice.L = 10.0;
    cfg.lattice.shape = "square";
    REQUIRE_THROWS_WITH(build_hamiltonian(cfg, build_groups(cfg)),
                        Catch::Matchers::ContainsSubstring("open"));
    REQUIRE_THROWS_WITH(alea::io::parse_config(
                            fock_darwin_yaml(0.5) + "    - { type: magnetic_field, B: 2.0 }\n"),
                        Catch::Matchers::ContainsSubstring("magnetic_field"));
}

TEST_CASE("wavefunction items become components in document order",
          "[app][order]") {
    // One YAML item → one component, in the order written.  The optimiser's
    // parameter vector and the config snapshot both follow component order,
    // so this is the contract that lets the snapshot write each optimised
    // value back into the block it came from.
    const auto cfg = alea::io::parse_config(R"(
system:
  lattice: { type: open, dimension: 2, L: 2.0 }
  particles:
    - { name: e, group: e, type: fermions, mass: 1.0, charge: -1, up: 1, down: 1 }
wavefunction:
  - moire_jastrow: { group: e, shells: 1, a_M: 10.0, theta0: 0.0 }
  - slater: { group: e, orbitals: { type: gaussian, centers: [[0.0, 0.0]], alphas: [0.5] } }
  - jastrow: { type: pade, pair: [e, e], denominator: [0.5] }
hamiltonian:
  terms:
    - { type: coulomb, pair: [e, e] }
)");
    const auto groups = build_groups(cfg);
    const auto ham    = build_hamiltonian(cfg, groups);
    const auto psi    = build_wavefunction(cfg, groups, &ham);
    REQUIRE(psi.components().size() == 3);
    CHECK(psi.components()[0]->name() == "moire_jastrow");
    CHECK(psi.components()[1]->name() == "slater_determinant");
    CHECK(psi.components()[2]->name() == "pade_jastrow");
}

TEST_CASE("a slater on a group with no particles is refused", "[app][order]") {
    // It would build no component at all, breaking item i → component i.
    const auto cfg = alea::io::parse_config(R"(
system:
  lattice: { type: open, dimension: 2, L: 2.0 }
  particles:
    - { name: e, group: e, type: fermions, mass: 1.0, charge: -1, up: 1, down: 0 }
    - { name: x, group: x, type: fermions, mass: 1.0, charge: -1, up: 0, down: 0 }
wavefunction:
  - slater: { group: x, orbitals: { type: gaussian, centers: [[0.0, 0.0]], alphas: [0.5] } }
)");
    REQUIRE_THROWS_WITH(build_wavefunction(cfg, build_groups(cfg)),
                        Catch::Matchers::ContainsSubstring("has no particles"));
}

TEST_CASE("build_wavefunction wires a Slater determinant from the config",
          "[app][slater]") {
    // Two same-spin electrons → a 2×2 up determinant (no down block).
    alea::io::Config cfg;
    cfg.lattice.dimension = 2;
    alea::io::ParticleSpec e;
    e.name = "e"; e.group = "electron_group"; e.mass = {1.0}; e.charge = -1.0;
    e.up = 2; e.down = 0;
    cfg.particles = {e};
    alea::io::SlaterSpec s;
    s.group = "electron_group";
    s.orbital_type = "gaussian";
    s.centers = {{-1.0, 0.0}, {1.0, 0.0}};
    s.alphas = {0.7, 0.7};
    cfg.wavefunction.push_back(s);

    auto groups = build_groups(cfg);
    auto psi = build_wavefunction(cfg, groups);

    auto comps = psi.components();
    REQUIRE(comps.size() == 1);
    CHECK(comps[0]->name() == "slater_determinant");

    // Evaluates to a finite log|ψ| with the electrons at the two orbital centers.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[0] = -1.0;  // e0 x at c0
    pos[2] = 1.0;   // e1 x at c1  (index (1*dim+0)*nw = 2)
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);
    CHECK(ws.log_psi()[0] > -1e9);
    CHECK(ws.log_psi()[0] < 1e9);
}

TEST_CASE("build_wavefunction hands the Gaussian width to the optimiser",
          "[app][slater][optimize]") {
    // ψ = D·e^J with J real and positive ⇒ the nodes of ψ ARE the nodes of D, so
    // no Jastrow parameter can move a fixed-node energy and the crystal's α is
    // the only one that can.  `alpha_fixed: false` is what reaches it.
    auto cfg_for = [](bool fixed) {
        alea::io::Config c;
        c.lattice.dimension = 2;
        alea::io::ParticleSpec e;
        e.name = "e"; e.group = "electron_group"; e.mass = {1.0}; e.charge = -1.0;
        e.up = 2; e.down = 0;
        c.particles = {e};
        alea::io::SlaterSpec s;
        s.group = "electron_group";
        s.orbital_type = "gaussian";
        s.centers = {{-1.0, 0.0}, {1.0, 0.0}};
        s.alphas = {0.7, 0.7};
        s.alpha_fixed = fixed;
        c.wavefunction.push_back(s);
        return c;
    };

    // Default: the determinant exposes its width but pins it, so a config that
    // never asked to optimise keeps the guide it had.
    {
        auto cfg = cfg_for(true);
        auto groups = build_groups(cfg);
        auto psi = build_wavefunction(cfg, groups);
        REQUIRE(psi.components().size() == 1);
        CHECK(psi.components()[0]->parameters().size() == 1);
        CHECK(psi.total_optimizable_params() == 0);
    }
    // Opted in: one nodal parameter reaches the linear method.
    {
        auto cfg = cfg_for(false);
        auto groups = build_groups(cfg);
        auto psi = build_wavefunction(cfg, groups);
        REQUIRE(psi.total_optimizable_params() == 1);
        const alea::Parameter& p = psi.components()[0]->parameters()[0];
        CHECK(p.value == Catch::Approx(0.7));
        CHECK_FALSE(p.is_fixed);
        CHECK(p.sign_locked);   // α = e^θ: the width can never reach 0
    }
}

TEST_CASE("build_wavefunction refuses to optimise a width that does not exist",
          "[app][slater][optimize]") {
    // Per-orbital widths are not ONE width: there is no shared knob to hand the
    // optimiser, and silently tying them would change the wavefunction the
    // config asked for.
    alea::io::Config cfg;
    cfg.lattice.dimension = 2;
    alea::io::ParticleSpec e;
    e.name = "e"; e.group = "electron_group"; e.mass = {1.0}; e.charge = -1.0;
    e.up = 2; e.down = 0;
    cfg.particles = {e};
    alea::io::SlaterSpec s;
    s.group = "electron_group";
    s.orbital_type = "gaussian";
    s.centers = {{-1.0, 0.0}, {1.0, 0.0}};
    s.alphas = {0.7, 0.4};          // different widths
    s.alpha_fixed = false;
    cfg.wavefunction.push_back(s);

    auto groups = build_groups(cfg);
    REQUIRE_THROWS_WITH(build_wavefunction(cfg, groups),
                        Catch::Matchers::ContainsSubstring("alpha_fixed"));
}

TEST_CASE("build_hamiltonian wires a moire term onto the listed groups",
          "[app][moire]") {
    auto cfg = make_exciton_cfg();
    alea::io::MoireSpec ms;
    ms.groups = {"electron_group", "hole_group"};
    ms.V0 = 0.1; ms.Vtilde = 0.02; ms.phi = -1.5708; ms.a_M = 10.0; ms.theta0 = 0.0;
    cfg.moires = {ms};

    auto groups = build_groups(cfg);
    auto ham = build_hamiltonian(cfg, groups);
    auto psi = build_wavefunction(cfg, groups, &ham);

    // p0=(0,0) electron, p1=(2,0) hole.  Kinetic + Coulomb give the exact −1 of
    // ψ = e^{−r} (as above); the moiré term adds V(r) for each particle of both
    // groups.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    ws.positions()[2] = 2.0;  // p1.x
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);
    ham.evaluate_batch(ws);

    const auto g = alea::moire_reciprocal_vectors(ms.a_M, ms.theta0, /*dim=*/2);
    auto Vmoire = [&](double x, double y) {
        double s = 0.0;
        for (const auto& gi : g) s += std::cos(gi[0] * x + gi[1] * y + ms.phi);
        return ms.V0 + 2.0 * ms.Vtilde * s;
    };
    const double expected = -1.0 + Vmoire(0.0, 0.0) + Vmoire(2.0, 0.0);
    CHECK(ws.local_energies()[0] == Catch::Approx(expected));
}

TEST_CASE("build_wavefunction maps can_flip_sign onto Parameter::sign_locked",
          "[app]") {
    auto cfg = make_exciton_cfg();
    // Two optimisable coefficients after the derived a₁: a₂ sign-locked
    // (can_flip_sign=false), b₁ free (can_flip_sign=true).
    first_item<alea::io::JastrowSpec>(cfg).numerator     = {-0.5};
    first_item<alea::io::JastrowSpec>(cfg).denominator   = {0.3};
    first_item<alea::io::JastrowSpec>(cfg).is_fixed      = {false, false};
    first_item<alea::io::JastrowSpec>(cfg).can_flip_sign = {false, true};

    auto groups = build_groups(cfg);
    auto psi = build_wavefunction(cfg, groups);

    REQUIRE(psi.components().size() == 1);
    auto params = psi.components()[0]->parameters();
    REQUIRE(params.size() == 3);   // [a₁, a₂, b₁]
    CHECK(params[0].is_fixed);
    CHECK(params[1].sign_locked == true);
    CHECK(params[2].sign_locked == false);
}

TEST_CASE("build_wavefunction treats a zero-valued sign-locked param as free",
          "[app]") {
    auto cfg = make_exciton_cfg();
    // Both locked, but the second starts at exactly 0 → sign undefined, so it
    // must be demoted to unconstrained.
    first_item<alea::io::JastrowSpec>(cfg).numerator     = {-0.5};
    first_item<alea::io::JastrowSpec>(cfg).denominator   = {0.0};
    first_item<alea::io::JastrowSpec>(cfg).is_fixed      = {false, false};
    first_item<alea::io::JastrowSpec>(cfg).can_flip_sign = {false, false};

    auto groups = build_groups(cfg);
    auto psi = build_wavefunction(cfg, groups);

    auto params = psi.components()[0]->parameters();
    REQUIRE(params.size() == 3);            // [a₁, a₂, b₁]
    CHECK(params[1].sign_locked == true);   // nonzero, stays locked
    CHECK(params[2].sign_locked == false);  // zero value → treated as free
    CHECK(params[2].value == 0.0);          // value left untouched
}

TEST_CASE("build_wavefunction derives a Drummond cusp from the Coulomb term",
          "[app][drummond]") {
    // Replace the Padé jastrow with a Drummond one (α0=0, C=3, L=5). The cusp Γ
    // must be derived from the Coulomb term: Γ = 2μ q_e q_h/(ε(d−1)) = −1
    // (μ=1/2, q_e q_h=−1, ε=1, d=2). A reference DrummondFunction with Γ=−1 then
    // fixes the expected u value.
    auto cfg = make_exciton_cfg();
    first_item<alea::io::JastrowSpec>(cfg).type = "drummond";
    first_item<alea::io::JastrowSpec>(cfg).params = {0.0};       // α0
    first_item<alea::io::JastrowSpec>(cfg).is_fixed = {true};
    first_item<alea::io::JastrowSpec>(cfg).truncation = 3;
    first_item<alea::io::JastrowSpec>(cfg).cutoff = 5.0;
    first_item<alea::io::JastrowSpec>(cfg).cutoff_fixed = true;

    auto groups = build_groups(cfg);
    auto ham = build_hamiltonian(cfg, groups);
    auto psi = build_wavefunction(cfg, groups, &ham);

    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    ws.positions()[2] = 2.0;  // r = 2
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);

    alea::DrummondFunction ref(3, /*gamma=*/-1.0, /*L=*/5.0, {0.0});
    double u = 0, du = 0, d2u = 0;
    ref.evaluate(2.0, u, du, d2u);
    CHECK(ws.log_psi()[0] == Catch::Approx(u));   // −0.432, only if Γ was −1
}

TEST_CASE("build_wavefunction leaves a Rytova-Keldysh Drummond cusp-free",
          "[app][drummond]") {
    // Same pair, but coupled by Rytova–Keldysh instead of Coulomb. RK has no 1/r
    // pole (residue 0) → Γ = 0 → α1 = 0, so with α0 = 0 the jastrow is identically
    // zero (cuspless), confirming the cusp is potential-driven, not assumed Coulomb.
    auto cfg = make_exciton_cfg();
    cfg.coulombs.clear();
    alea::io::KeldyshSpec ks;
    ks.pair = {"electron_group", "hole_group"};
    ks.epsilon = 1.0;
    ks.r0 = 5.0;
    cfg.keldysh = {ks};

    first_item<alea::io::JastrowSpec>(cfg).type = "drummond";
    first_item<alea::io::JastrowSpec>(cfg).params = {0.0};
    first_item<alea::io::JastrowSpec>(cfg).is_fixed = {true};
    first_item<alea::io::JastrowSpec>(cfg).truncation = 3;
    first_item<alea::io::JastrowSpec>(cfg).cutoff = 5.0;
    first_item<alea::io::JastrowSpec>(cfg).cutoff_fixed = true;

    auto groups = build_groups(cfg);
    auto ham = build_hamiltonian(cfg, groups);
    auto psi = build_wavefunction(cfg, groups, &ham);

    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    ws.positions()[2] = 2.0;
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Catch::Approx(0.0));  // Γ=0 → cuspless, α0=0 → u≡0
}

TEST_CASE("App builds VMC driver and runs sane estimates", "[app]") {
    auto cfg = make_exciton_cfg();
    cfg.seed = 42;
    cfg.lattice.L = 1.0;
    alea::io::QmcSpec q;
    q.method = "vmc"; q.random_walk = "metropolis"; q.threads = 1; q.samples = 16;
    q.equilibration_blocks = 3; q.accumulation_blocks = 8;
    q.steps_per_block = 4; q.sub_steps = 2; q.time_step = 0.3;
    cfg.qmc = {q};

    App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 1);

    auto* vmc = dynamic_cast<alea::VMC*>(pack.drivers[0].get());
    REQUIRE(vmc != nullptr);
    vmc->run();

    CHECK(std::isfinite(vmc->energy()));
    CHECK(std::isfinite(vmc->energy_error()));
    CHECK(vmc->energy_variance() >= 0.0);
    CHECK(vmc->acceptance_ratio() > 0.0);
    CHECK(vmc->acceptance_ratio() <= 1.0);
}

namespace {

// Two electrons, two holes in a periodic 10×10 cell with one e–h jastrow.
struct PeriodicBuild {
    alea::io::Config cfg;
    std::vector<alea::ParticleGroup> groups;
    alea::SimulationCell cell;
};
PeriodicBuild periodic_eh(const alea::io::JastrowSpec& j, bool coulomb,
                          const std::string& truncation = "") {
    PeriodicBuild b{make_exciton_cfg(), {}, {}};
    b.cfg.lattice.type       = "periodic";
    b.cfg.lattice.truncation = truncation;
    b.cfg.particles[0].up = 2;
    b.cfg.particles[1].up = 2;
    b.cfg.wavefunction = {j};
    if (!coulomb) b.cfg.coulombs.clear();
    b.groups = build_groups(b.cfg);
    const std::vector<double> a{10.0, 0.0, 0.0, 10.0};
    b.cell = alea::SimulationCell(2, std::span<const double>(a));
    return b;
}

alea::io::JastrowSpec eh_jastrow(const std::string& type) {
    alea::io::JastrowSpec j;
    j.type = type;
    j.pair = {"electron_group", "hole_group"};
    if (type == "drummond") {
        j.params = {0.0}; j.is_fixed = {true}; j.truncation = 3; j.cutoff = 4.0;
    } else {
        j.denominator = {0.5}; j.is_fixed = {true};
    }
    return j;
}

}  // namespace

TEST_CASE("a periodic Pade with an Ewald Hamiltonian is summed over images",
          "[app][image_sum]") {
    // PBC + Ewald: the Jastrow must be summed over every periodic image too,
    // not read at the minimum image (which has a kink on the WS boundary).
    const PeriodicBuild b = periodic_eh(eh_jastrow("pade"), /*coulomb=*/true);
    const auto ham = build_hamiltonian(b.cfg, b.groups, b.cell);
    const auto psi = build_wavefunction(b.cfg, b.groups, &ham, b.cell);
    const auto* pade = dynamic_cast<const alea::PadeJastrow*>(psi.components()[0].get());
    REQUIRE(pade != nullptr);
    CHECK(pade->sums_over_images());
}

TEST_CASE("a periodic Pade without Ewald is refused unless minimum image is explicit",
          "[app][image_sum]") {
    // No long-range term: nothing to tie the Jastrow's images to, and a
    // minimum-image Padé has a kink on the Wigner-Seitz boundary.
    {
        const PeriodicBuild b = periodic_eh(eh_jastrow("pade"), /*coulomb=*/false);
        const auto ham = build_hamiltonian(b.cfg, b.groups, b.cell);
        CHECK_THROWS_WITH(build_wavefunction(b.cfg, b.groups, &ham, b.cell),
                          Catch::Matchers::ContainsSubstring("Ewald"));
    }
    // `truncation: minimum_image` is an explicit minimum-image model (geometry
    // tests): the Padé follows it, unsummed.
    {
        const PeriodicBuild b =
            periodic_eh(eh_jastrow("pade"), /*coulomb=*/true, "minimum_image");
        const auto ham = build_hamiltonian(b.cfg, b.groups, b.cell);
        const auto psi = build_wavefunction(b.cfg, b.groups, &ham, b.cell);
        const auto* pade = dynamic_cast<const alea::PadeJastrow*>(psi.components()[0].get());
        REQUIRE(pade != nullptr);
        CHECK_FALSE(pade->sums_over_images());
    }
}

TEST_CASE("a periodic Pade whose image sum does not exist is refused",
          "[app][image_sum]") {
    // u = a₁r grows without bound (numerator order 1 over denominator 0).
    alea::io::JastrowSpec j = eh_jastrow("pade");
    j.denominator.clear();
    j.is_fixed.clear();
    const PeriodicBuild b = periodic_eh(j, /*coulomb=*/true);
    const auto ham = build_hamiltonian(b.cfg, b.groups, b.cell);
    CHECK_THROWS_WITH(build_wavefunction(b.cfg, b.groups, &ham, b.cell),
                      Catch::Matchers::ContainsSubstring("grows without bound"));
}

TEST_CASE("a periodic Drummond cutoff must stay within the Wigner-Seitz inradius",
          "[app][image_sum]") {
    // L <= R_WS = 5: the minimum image is then the exact image sum.
    alea::io::JastrowSpec j = eh_jastrow("drummond");
    j.cutoff = 6.0;
    const PeriodicBuild far = periodic_eh(j, /*coulomb=*/true);
    const auto ham = build_hamiltonian(far.cfg, far.groups, far.cell);
    CHECK_THROWS_WITH(build_wavefunction(far.cfg, far.groups, &ham, far.cell),
                      Catch::Matchers::ContainsSubstring("Wigner"));

    // Inside it builds, and the optimiser may not push L out.
    const PeriodicBuild near = periodic_eh(eh_jastrow("drummond"), /*coulomb=*/true);
    const auto ham2 = build_hamiltonian(near.cfg, near.groups, near.cell);
    auto psi = build_wavefunction(near.cfg, near.groups, &ham2, near.cell);
    psi.components()[0]->parameters().back().value = 5.5;
    CHECK_THROWS_WITH(psi.components()[0]->rebuild(),
                      Catch::Matchers::ContainsSubstring("Wigner"));
}

namespace {

// A polarized 2-D electron gas: N = 7 in a periodic hexagonal cell at rs = 5,
// Coulomb (one Ewald term), plane-wave determinant.
const char* kHegYaml = R"(
system:
  lattice: { type: periodic, dimension: 2, shape: hexagonal, rs: 5.0 }
  particles:
    - { name: e, group: e, type: fermions, mass: 1.0, charge: -1, up: 7, down: 0 }
wavefunction:
  - slater: { group: e, orbitals: { type: plane_wave } }
  - kspace_jastrow: { pair: [e, e], shells: 3 }
hamiltonian:
  terms:
    - { type: coulomb, pair: [e, e], epsilon: 1.0 }
)";

}  // namespace

TEST_CASE("a periodic inverse_power term is built per spec, outside the Ewald sum",
          "[app][inverse_power]") {
    // 2-D dipolar bosons at n r0² = 256 (rs = 1/√(256π)), 16 on a hexagonal
    // cell.  No Coulomb term → no Ewald term; the dipolar pair carries its own
    // continuum tail, ½ N n · 6/h for the hexagonal WS cell (h = L/2).
    const auto cfg = alea::io::parse_config(R"(
system:
  lattice: { type: periodic, dimension: 2, shape: hexagonal, rs: 0.03526184897173477 }
  particles:
    - { name: d, group: dipoles, type: bosons, mass: 1.0, up: 16, down: 0 }
hamiltonian:
  terms:
    - { type: inverse_power, pair: [dipoles, dipoles], strength: 1.0, exponent: 3 }
)");
    const auto groups = build_groups(cfg);
    const alea::SimulationCell cell = alea::build_cell(cfg, 16);
    const auto ham = build_hamiltonian(cfg, groups, cell);
    std::vector<std::string> names;
    for (const auto& c : ham.components()) names.push_back(c->name());
    CHECK(std::count(names.begin(), names.end(), "inverse_power") == 1);
    CHECK(std::count(names.begin(), names.end(), "ewald") == 0);

    const double n = 16.0 / cell.volume();
    CHECK(n == Catch::Approx(256.0).epsilon(1e-10));
    const double h = cell.ws_radius();
    REQUIRE(0.075 < h);
    WalkerSet ws(1, 16, 2, cell);
    // A row of spacing 0.005: every separation (≤ 0.075) is below R_WS ≈ 0.134,
    // so the direct distance IS the minimum image.
    for (std::size_t i = 0; i < 16; ++i) ws.positions()[i * 2] = 0.005 * static_cast<double>(i);
    ws.dist_tables().build_all(ws.positions());
    std::vector<double> pair_sum(1, 0.0);
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = i + 1; j < 16; ++j)
            pair_sum[0] += std::pow(0.005 * static_cast<double>(j - i), -3.0);
    std::vector<double> v(1, 0.0);
    for (const auto& c : ham.components())
        if (c->is_potential()) c->evaluate_potential(ws, v);
    CHECK(v[0] == Catch::Approx(pair_sum[0] + 0.5 * 16.0 * n * 6.0 / h).epsilon(1e-9));
}

TEST_CASE("a zero_energy jastrow reads its core from the Hamiltonian and its cutoff "
          "from the cell",
          "[app][zero_energy]") {
    // Mass 2 → μ = 1, so b = 2μC = 2·1.5 = 3: read from the inverse_power term,
    // never from the config.  Periodic: r_c defaults to the Wigner-Seitz
    // inradius.
    const char* periodic = R"(
system:
  lattice: { type: periodic, dimension: 2, shape: hexagonal, rs: 0.5 }
  particles:
    - { name: d, group: dipoles, type: bosons, mass: 2.0, up: 7, down: 0 }
wavefunction:
  - jastrow: { type: zero_energy, pair: [dipoles, dipoles], matching_radius: 0.3 }
hamiltonian:
  terms:
    - { type: inverse_power, pair: [dipoles, dipoles], strength: 1.5, exponent: 3 }
)";
    {
        const auto cfg = alea::io::parse_config(periodic);
        const auto groups = build_groups(cfg);
        const auto cell = alea::build_cell(cfg, 7);
        const auto ham = build_hamiltonian(cfg, groups, cell);
        const auto psi = build_wavefunction(cfg, groups, &ham, cell);
        const auto* z = dynamic_cast<const alea::ZeroEnergyJastrow*>(psi.components()[0].get());
        REQUIRE(z != nullptr);
        CHECK(z->core_strength() == Catch::Approx(3.0));
        CHECK(z->exponent() == 3.0);
        CHECK(z->cutoff() == cell.ws_radius());
        REQUIRE(z->parameters().size() == 1);
        CHECK(z->parameters()[0].value == 0.3);
        CHECK(z->parameters()[0].is_fixed);
    }

    using Catch::Matchers::ContainsSubstring;
    auto build = [](std::string yaml) {
        const auto cfg = alea::io::parse_config(yaml);
        const auto groups = build_groups(cfg);
        const auto cell = alea::build_cell(cfg, 7);
        const auto ham = build_hamiltonian(cfg, groups, cell);
        return build_wavefunction(cfg, groups, &ham, cell);
    };
    auto replaced = [&](std::string from, std::string to) {
        std::string yaml = periodic;
        yaml.replace(yaml.find(from), from.size(), to);
        return yaml;
    };
    // A cutoff past the inradius would drop periodic images.
    CHECK_THROWS_WITH(build(replaced("matching_radius: 0.3", "matching_radius: 0.3, cutoff: 5.0")),
                      ContainsSubstring("Wigner-Seitz"));
    // R_m must lie inside the cutoff.
    CHECK_THROWS_WITH(build(replaced("matching_radius: 0.3", "matching_radius: 50.0")),
                      ContainsSubstring("matching_radius"));
    // An open cell has no inradius: the cutoff must be given.
    const std::string open = replaced(
        "{ type: periodic, dimension: 2, shape: hexagonal, rs: 0.5 }",
        "{ type: open, dimension: 2, L: 1.0 }");
    CHECK_THROWS_WITH(build(open), ContainsSubstring("cutoff"));
    std::string open_ok = open;
    open_ok.replace(open_ok.find("matching_radius: 0.3"), 20, "matching_radius: 0.3, cutoff: 2.5");
    const auto psi = build(open_ok);
    const auto* z = dynamic_cast<const alea::ZeroEnergyJastrow*>(psi.components()[0].get());
    REQUIRE(z != nullptr);
    CHECK(z->cutoff() == 2.5);
    // No core with p > 2 on the pair (here 1/r², allowed by the open-cell
    // Hamiltonian): nothing to derive κ and s from.
    std::string soft = open_ok;
    soft.replace(soft.find("exponent: 3"), 11, "exponent: 2");
    CHECK_THROWS_WITH(build(soft), ContainsSubstring("p > 2"));
    // A Gaskell-seeded kspace_jastrow would subtract ũ(k) of this u, which
    // diverges at contact: refused, with the way out (explicit params) named.
    std::string seeded = periodic;
    seeded.replace(seeded.find("hamiltonian:"), 12,
                   "  - kspace_jastrow: { pair: [dipoles, dipoles], shells: 2 }\nhamiltonian:");
    CHECK_THROWS_WITH(build(seeded), ContainsSubstring("params"));
}

TEST_CASE("a nosanow factor sits on a triangular crystal and seeds the walkers there",
          "[app][nosanow]") {
    // 16 dipoles at n r0² = 256 on a hexagonal cell: 16 = 4² is Loeschian, so
    // the sites tile the cell as a triangular lattice of spacing
    // a = √(2/(√3 n)).  R_WS ≈ 0.134 → α must exceed ln(1e6)/R_WS² ≈ 768.
    const char* yaml = R"(
system:
  lattice: { type: periodic, dimension: 2, shape: hexagonal, rs: 0.03526184897173477 }
  particles:
    - { name: d, group: dipoles, type: bosons, mass: 1.0, up: 16, down: 0 }
wavefunction:
  - jastrow: { type: zero_energy, pair: [dipoles, dipoles], matching_radius: 0.04 }
  - nosanow: { group: dipoles, sites: triangular, alpha: 1500.0 }
hamiltonian:
  terms:
    - { type: inverse_power, pair: [dipoles, dipoles], strength: 1.0, exponent: 3 }
)";
    auto build = [](const std::string& text) {
        const auto cfg = alea::io::parse_config(text);
        const auto groups = build_groups(cfg);
        const auto cell = alea::build_cell(cfg, 16);
        const auto ham = build_hamiltonian(cfg, groups, cell);
        return build_wavefunction(cfg, groups, &ham, cell);
    };
    const auto cfg = alea::io::parse_config(yaml);
    const auto groups = build_groups(cfg);
    const auto cell = alea::build_cell(cfg, 16);
    const auto psi = build(yaml);
    const auto* n = dynamic_cast<const alea::NosanowJastrow*>(psi.components()[1].get());
    REQUIRE(n != nullptr);
    CHECK(n->alpha() == 1500.0);
    CHECK(n->parameters()[0].is_fixed);

    // Every site has exactly six neighbours at the lattice spacing.
    const auto sites = n->sites();
    REQUIRE(sites.size() == 32);
    const double a = std::sqrt(2.0 / (std::sqrt(3.0) * 256.0));
    for (std::size_t i = 0; i < 16; ++i) {
        int nearest = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            if (i == j) continue;
            double del[2] = {sites[2 * i] - sites[2 * j], sites[2 * i + 1] - sites[2 * j + 1]};
            const double r = std::sqrt(cell.min_image_dist2(del));
            CHECK(r > a * (1 - 1e-9));
            if (r < a * (1 + 1e-9)) ++nearest;
        }
        CHECK(nearest == 6);
    }

    // The walkers start on those sites, scattered by the factor's own spread.
    const alea::CrystalAnchors anchors = alea::crystal_site_anchors(cfg, groups, cell);
    REQUIRE(anchors.pos.size() == 32);
    for (std::size_t k = 0; k < 32; ++k) CHECK(anchors.pos[k] == sites[k]);
    CHECK(anchors.jitter == Catch::Approx(1.0 / std::sqrt(4.0 * 1500.0)));

    using Catch::Matchers::ContainsSubstring;
    auto replaced = [&](std::string from, std::string to) {
        std::string text = yaml;
        text.replace(text.find(from), from.size(), to);
        return text;
    };
    // Too wide for the cell: the minimum-image fold would not be negligible.
    CHECK_THROWS_WITH(build(replaced("alpha: 1500.0", "alpha: 300.0")),
                      ContainsSubstring("Wigner-Seitz"));
    // Explicit centers: one per particle.
    CHECK_THROWS_WITH(build(replaced("sites: triangular", "centers: [[0, 0], [0.05, 0]]")),
                      ContainsSubstring("one site per particle"));
    // A determinant is the wrong symmetry for bosons: the Gaussian crystal
    // that fermions would use is refused on this group.
    CHECK_THROWS_WITH(build(replaced("  - nosanow: { group: dipoles, sites: triangular, alpha: 1500.0 }",
                                     "  - slater: { group: dipoles, orbitals: { sites: triangular,"
                                     " alpha: 1500.0 } }")),
                      ContainsSubstring("bosons"));
}

TEST_CASE("the Ewald term reports a pair's Fourier potential", "[app][kspace_jastrow]") {
    // q_a q_b Ṽ(k): for in-plane Coulomb, 2π q²/(ε k).
    const auto cfg = alea::io::parse_config(kHegYaml);
    const auto groups = build_groups(cfg);
    const alea::SimulationCell cell = alea::build_cell(cfg, 7);
    const auto ham = build_hamiltonian(cfg, groups, cell);
    double v = 0.0;
    for (const auto& c : ham.components()) v += c->pair_fourier(groups[0], groups[0], 0.7);
    CHECK(v == Catch::Approx(2.0 * std::acos(-1.0) / 0.7));
}

TEST_CASE("a kspace_jastrow item is seeded from the Gaskell RPA form",
          "[app][kspace_jastrow]") {
    // p_s = −ũ_C(k_s)/A with the Hamiltonian's own Ṽ(k) and the free-fermion S₀
    // of the group (polarized here), so the term works with no optimisation.
    const auto cfg = alea::io::parse_config(kHegYaml);
    REQUIRE(alea::io::items_of<alea::io::KSpaceJastrowSpec>(cfg.wavefunction).size() == 1);
    const auto groups = build_groups(cfg);
    const alea::SimulationCell cell = alea::build_cell(cfg, 7);
    const auto ham = build_hamiltonian(cfg, groups, cell);
    const auto psi = build_wavefunction(cfg, groups, &ham, cell);
    const auto* kj = dynamic_cast<const alea::KSpaceJastrow*>(psi.components()[1].get());
    REQUIRE(kj != nullptr);
    REQUIRE(kj->parameters().size() == 3);

    const double area = cell.volume(), rho = 7.0 / area;
    for (std::size_t s = 0; s < 3; ++s) {
        const double k  = kj->shell_k()[s];
        const double s0 = alea::numerics::free_fermion_sofk_2d(k, rho);
        const double u  = alea::numerics::gaskell_u(k, rho, s0, 2 * std::acos(-1.0) / k, 1.0);
        INFO("shell " << s << " |k| = " << k);
        CHECK(kj->parameters()[s].value == Catch::Approx(-u / area));
        CHECK(kj->parameters()[s].value < 0.0);   // repulsion suppresses fluctuations
        CHECK(kj->parameters()[s].is_fixed);      // fixed unless is_fixed says otherwise
    }
}

TEST_CASE("a kspace_jastrow seed leaves out what the pair's short-range jastrow carries",
          "[app][kspace_jastrow]") {
    // RPA pair pseudopotentials add, so on top of a short-range term the long-
    // range one supplies only the remainder: ũ_LR = ũ_Gaskell − ũ_short.  Seeding
    // the full Gaskell form instead counted the correlation twice — measured on
    // HEG2D rs = 30: S fell to 0.6–0.75 of the plasmon value and E_VMC rose by
    // 0.0067 Ha (3 shells) and 0.067 Ha (8 shells).
    std::string yaml = kHegYaml;
    const std::string anchor = "  - kspace_jastrow:";
    yaml.insert(yaml.find(anchor),
                "  - jastrow: { type: drummond, pair: [e, e], params: [0.002], "
                "truncation: 3, cutoff: 9.0 }\n");
    const auto cfg = alea::io::parse_config(yaml);
    const auto groups = build_groups(cfg);
    const auto cell = alea::build_cell(cfg, 7);
    const auto ham = build_hamiltonian(cfg, groups, cell);
    const auto psi = build_wavefunction(cfg, groups, &ham, cell);
    const auto* kj = dynamic_cast<const alea::KSpaceJastrow*>(psi.components()[2].get());
    REQUIRE(kj != nullptr);
    const auto* dj = dynamic_cast<const alea::DrummondJastrow*>(psi.components()[1].get());
    REQUIRE(dj != nullptr);

    // ũ_D(k) = 2π ∫₀^L r J₀(kr) u(r) dr, by an independent midpoint rule, with
    // the cusp the builder derived (P-wave behind the polarized determinant).
    const alea::DrummondFunction u(3, dj->cusp(), 9.0, {0.002});
    auto u_tilde = [&](double k) {
        const int n = 200000;
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            const double r = 9.0 * (i + 0.5) / n;
            double v, d1, d2;
            u.evaluate(r, v, d1, d2);
            s += r * std::cyl_bessel_j(0.0, k * r) * v;
        }
        return 2 * std::acos(-1.0) * s * 9.0 / n;
    };
    const double area = cell.volume(), rho = 7.0 / area;
    for (std::size_t s = 0; s < 3; ++s) {
        const double k  = kj->shell_k()[s];
        const double s0 = alea::numerics::free_fermion_sofk_2d(k, rho);
        const double ug = alea::numerics::gaskell_u(k, rho, s0, 2 * std::acos(-1.0) / k, 1.0);
        INFO("shell " << s);
        CHECK(kj->parameters()[s].value == Catch::Approx((-ug - u_tilde(k)) / area).epsilon(1e-6));
    }
}

TEST_CASE("kspace_jastrow coefficients from the config are used verbatim",
          "[app][kspace_jastrow]") {
    std::string yaml = kHegYaml;
    const std::string from = "{ pair: [e, e], shells: 3 }";
    yaml.replace(yaml.find(from), from.size(),
                 "{ pair: [e, e], shells: 2, params: [-0.01, 0.002], is_fixed: [false, true] }");
    const auto cfg = alea::io::parse_config(yaml);
    const auto groups = build_groups(cfg);
    const auto cell = alea::build_cell(cfg, 7);
    const auto ham = build_hamiltonian(cfg, groups, cell);
    const auto psi = build_wavefunction(cfg, groups, &ham, cell);
    const auto p = psi.components()[1]->parameters();
    REQUIRE(p.size() == 2);
    CHECK(p[0].value == -0.01);
    CHECK_FALSE(p[0].is_fixed);
    CHECK(p[1].value == 0.002);
    CHECK(p[1].is_fixed);
}

TEST_CASE("a kspace_jastrow needs a periodic cell", "[app][kspace_jastrow]") {
    std::string yaml = kHegYaml;
    yaml.replace(yaml.find("type: periodic, dimension: 2, shape: hexagonal, rs: 5.0"),
                 std::string("type: periodic, dimension: 2, shape: hexagonal, rs: 5.0").size(),
                 "type: open, dimension: 2");
    yaml.replace(yaml.find("  - slater: { group: e, orbitals: { type: plane_wave } }\n"),
                 std::string("  - slater: { group: e, orbitals: { type: plane_wave } }\n").size(), "");
    const auto cfg = alea::io::parse_config(yaml);
    const auto groups = build_groups(cfg);
    const auto ham = build_hamiltonian(cfg, groups);
    CHECK_THROWS_WITH(build_wavefunction(cfg, groups, &ham),
                      Catch::Matchers::ContainsSubstring("periodic"));
}

TEST_CASE("a periodic Coulomb bilayer gets a layer-offset Ewald kernel",
          "[app][ewald]") {
    // Electrons and holes in planes 1 apart.  The Ewald term must carry that
    // offset (it used to drop it and sum a monolayer): the cross-plane pair
    // gets 1/sqrt(r^2+d^2), which shows up as the capacitor energy of the two
    // charged layers, -(pi/(eps A)) sum Q_g Q_h d_gh = 2 pi d/(eps A) here.
    auto cfg = make_exciton_cfg();
    cfg.lattice.type   = "periodic";
    cfg.particles[1].z = 1.0;  // holes one unit below the electrons
    const auto groups = build_groups(cfg);
    const std::vector<double> a{10.0, 0.0, 0.0, 10.0};
    const alea::SimulationCell cell(2, std::span<const double>(a));

    const auto ham = build_hamiltonian(cfg, groups, cell);
    const alea::EwaldTerm* ewald = nullptr;
    for (const auto& c : ham.components()) {
        CHECK(c->name() != "coulomb");  // superseded by the single Ewald term
        if (c->name() == "ewald") ewald = dynamic_cast<const alea::EwaldTerm*>(c.get());
    }
    REQUIRE(ewald != nullptr);
    CHECK(ewald->capacitor_energy() ==
          Catch::Approx(2.0 * std::acos(-1.0) * 1.0 / (1.0 * 100.0)));
}

TEST_CASE("a periodic Keldysh bilayer gets a layer-offset Ewald kernel",
          "[app][ewald]") {
    // Same as the Coulomb case, with the cross-plane pair on V_RK(sqrt(r^2+d^2))
    // -- the open-boundary KeldyshTerm's definition, not a new one.
    auto cfg = make_exciton_cfg();
    cfg.coulombs.clear();
    alea::io::KeldyshSpec ks;
    ks.pair = {"electron_group", "hole_group"}; ks.epsilon = 1.0; ks.r0 = 0.5;
    cfg.keldysh = {ks};
    cfg.lattice.type   = "periodic";
    cfg.particles[1].z = 1.0;
    const auto groups = build_groups(cfg);
    const std::vector<double> a{10.0, 0.0, 0.0, 10.0};
    const alea::SimulationCell cell(2, std::span<const double>(a));

    const auto ham = build_hamiltonian(cfg, groups, cell);
    const alea::EwaldTerm* ewald = nullptr;
    for (const auto& c : ham.components()) {
        CHECK(c->name() != "rytova_keldysh");  // superseded by the Ewald term
        if (c->name() == "ewald") ewald = dynamic_cast<const alea::EwaldTerm*>(c.get());
    }
    REQUIRE(ewald != nullptr);
    // The offset reached the kernel: the layers' mean-field energy is that of
    // the in-plane and cross-plane RK kernels at d = 1 (its value is checked
    // against a real-space integral in test_ewald.cpp), not 0.
    const auto in  = alea::numerics::PotentialSpec::rytova_keldysh(1.0, 0.5);
    const auto out = alea::numerics::PotentialSpec::rytova_keldysh(1.0, 0.5, 1.0);
    CHECK(ewald->capacitor_energy() ==
          Catch::Approx((2.0 * in.k0_offset - 2.0 * out.k0_offset) / (2.0 * 100.0)));
    const alea::ParticleGroup& e = groups[0];
    const alea::ParticleGroup& h = groups[1];
    CHECK(ewald->coulomb_cusp_residue(e, h) == 0.0);
}

namespace {

// Electron at the centre of the lower eps = 14 slab (z = 0), hole at the centre
// of the upper one (z = 6), vacuum outside; one screened spec per group pair,
// each in its own source-centred frame (the library's convention).
alea::io::Config make_screened_bilayer_cfg() {
    auto cfg = make_exciton_cfg();
    cfg.coulombs.clear();
    cfg.lattice.type   = "periodic";
    cfg.particles[1].z = 6.0;
    auto spec = [](const char* a, const char* b, std::vector<double> iface, int src,
                   int tgt) {
        alea::io::ScreenedCoulombSpec s;
        s.pair = {a, b}; s.epsilons = {1.0, 14.0, 14.0, 1.0};
        s.interfaces = std::move(iface); s.source_layer = src; s.target_layer = tgt;
        return s;
    };
    cfg.screened = {spec("electron_group", "hole_group", {-3, 3, 9}, 2, 3),
                    spec("electron_group", "electron_group", {-3, 3, 9}, 2, 0),
                    spec("hole_group", "hole_group", {-9, -3, 3}, 3, 0)};
    return cfg;
}

}  // namespace

TEST_CASE("a periodic screened_coulomb bilayer gets per-pair ETM Ewald kernels",
          "[app][ewald][screened]") {
    // The capacitor energy of the two charged slabs follows the series law,
    // 2 pi Q^2 d/(eps A) with both charges inside eps = 14: proof that the
    // stacks and the offset reached the kernels (values checked in depth in
    // test_ewald_screened.cpp).
    const auto cfg    = make_screened_bilayer_cfg();
    const auto groups = build_groups(cfg);
    const std::vector<double> a{10.0, 0.0, 0.0, 10.0};
    const alea::SimulationCell cell(2, std::span<const double>(a));

    const auto ham = build_hamiltonian(cfg, groups, cell);
    const alea::EwaldTerm* ewald = nullptr;
    for (const auto& c : ham.components()) {
        CHECK(c->name() != "screened_coulomb");  // superseded by the Ewald term
        if (c->name() == "ewald") ewald = dynamic_cast<const alea::EwaldTerm*>(c.get());
    }
    REQUIRE(ewald != nullptr);
    CHECK(ewald->capacitor_energy() ==
          Catch::Approx(2.0 * std::acos(-1.0) * 6.0 / (14.0 * 100.0)).epsilon(1e-6));
}

TEST_CASE("a periodic screened_coulomb run needs a stack for every charged pair",
          "[app][ewald][screened]") {
    // Under PBC every charge meets every other one's images -- a group's own
    // included -- and a screened stack cannot be guessed for a pair.
    auto cfg = make_screened_bilayer_cfg();
    cfg.screened.pop_back();  // drop hole-hole
    const auto groups = build_groups(cfg);
    const std::vector<double> a{10.0, 0.0, 0.0, 10.0};
    const alea::SimulationCell cell(2, std::span<const double>(a));
    CHECK_THROWS_WITH(build_hamiltonian(cfg, groups, cell),
                      Catch::Matchers::ContainsSubstring("hole_group"));
}

TEST_CASE("build_hamiltonian refuses a periodic run mixing potential families",
          "[app][ewald]") {
    // One Ewald term carries every pair of a periodic run.  Mapping specs of two
    // families onto it is not wired, and building it anyway would give every
    // pair the first family's kernel and drop the other specs without a word.
    auto cfg = make_exciton_cfg();   // coulomb e-h
    alea::io::KeldyshSpec ks;
    ks.pair = {"electron_group", "electron_group"}; ks.epsilon = 1.0; ks.r0 = 0.5;
    cfg.keldysh = {ks};
    cfg.lattice.type = "periodic";
    const auto groups = build_groups(cfg);
    const std::vector<double> a{10.0, 0.0, 0.0, 10.0};
    const alea::SimulationCell cell(2, std::span<const double>(a));
    CHECK_THROWS_WITH(build_hamiltonian(cfg, groups, cell),
                      Catch::Matchers::ContainsSubstring("mixing"));

    // Minimum-image truncation builds per-pair terms, so mixing is fine there.
    cfg.lattice.truncation = "minimum_image";
    CHECK_NOTHROW(build_hamiltonian(cfg, groups, cell));
}

TEST_CASE("build_hamiltonian refuses periodic coulomb terms of differing epsilon",
          "[app][ewald]") {
    // The reciprocal sum couples every charge through one structure factor, so
    // a cell carries one epsilon; the Ewald term would otherwise use the first.
    auto cfg = make_exciton_cfg();
    alea::io::CoulombSpec ee;
    ee.pair = {"electron_group", "electron_group"}; ee.epsilon = 2.0;
    cfg.coulombs.push_back(ee);
    cfg.lattice.type = "periodic";
    const auto groups = build_groups(cfg);
    const std::vector<double> a{10.0, 0.0, 0.0, 10.0};
    const alea::SimulationCell cell(2, std::span<const double>(a));
    CHECK_THROWS_WITH(build_hamiltonian(cfg, groups, cell),
                      Catch::Matchers::ContainsSubstring("epsilon"));
}

// ── One-body moiré Jastrow and explicit crystal sites under PBC ─────────────

namespace {

// 4 spin-polarized electrons in a 2x2 supercell of a triangular moiré lattice
// (a_M = 10, theta0 = 0), one Gaussian per moiré site, Coulomb via Ewald.
alea::io::Config make_moire_crystal_cfg() {
    const double a_M = 10.0, theta0 = 0.0;
    const auto   g   = alea::moire_reciprocal_vectors(a_M, theta0, 2);
    const double det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    const double k   = 2.0 * std::acos(-1.0) / det;
    const double A1[2] = {k * g[1][1], -k * g[1][0]};
    const double A2[2] = {-k * g[0][1], k * g[0][0]};

    alea::io::Config c;
    c.seed = 7;
    c.lattice.type      = "periodic";
    c.lattice.dimension = 2;
    c.lattice.vectors   = {2 * A1[0], 2 * A1[1], 2 * A2[0], 2 * A2[1]};
    alea::io::ParticleSpec e;
    e.name = "e"; e.group = "electron_group"; e.mass = {1.0}; e.charge = -1.0;
    e.up = 4; e.down = 0;
    c.particles = {e};

    alea::io::SlaterSpec s;
    s.group = "electron_group";
    s.orbital_type = "gaussian";
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            s.centers.push_back({i * A1[0] + j * A2[0], i * A1[1] + j * A2[1]});
    s.alpha = 0.05;
    c.wavefunction.push_back(s);

    alea::io::MoireJastrowSpec mj;
    mj.group = "electron_group";
    mj.shells = 2;
    mj.params = {-0.1, 0.05, 0.02, 0.0};
    mj.is_fixed = {false, false, true, true};
    c.wavefunction.push_back(mj);

    alea::io::CoulombSpec cs;
    cs.pair = {"electron_group", "electron_group"};
    c.coulombs = {cs};
    alea::io::MoireSpec ms;
    ms.groups = {"electron_group"};
    ms.Vtilde = 0.01; ms.phi = std::acos(-1.0) / 3.0; ms.a_M = a_M; ms.theta0 = theta0;
    c.moires = {ms};
    return c;
}

alea::io::QmcSpec short_vmc() {
    alea::io::QmcSpec q;
    q.method = "vmc"; q.random_walk = "metropolis"; q.threads = 1; q.samples = 8;
    q.equilibration_blocks = 2; q.accumulation_blocks = 4;
    q.steps_per_block = 3; q.sub_steps = 1; q.time_step = 2.0;
    return q;
}

}  // namespace

TEST_CASE("moire_jastrow inherits the superlattice and wires its parameters",
          "[app][moire_jastrow]") {
    auto cfg = make_moire_crystal_cfg();
    const auto groups = build_groups(cfg);
    auto psi = build_wavefunction(cfg, groups);

    // Determinant first, then the moiré Jastrow (the order config_writer assumes).
    REQUIRE(psi.components().size() == 2);
    CHECK(psi.components()[0]->name() == "slater_determinant");
    CHECK(psi.components()[1]->name() == "moire_jastrow");
    const auto par = psi.components()[1]->parameters();
    REQUIRE(par.size() == 4);
    CHECK(par[0].name == "a1");
    CHECK(par[3].name == "b2");
    CHECK(par[0].value == Catch::Approx(-0.1));
    CHECK(psi.total_optimizable_params() == 2);   // a1, b1 free; alpha pinned
}

TEST_CASE("moire_jastrow needs a superlattice to live on", "[app][moire_jastrow]") {
    auto cfg = make_moire_crystal_cfg();
    cfg.moires.clear();
    const auto groups = build_groups(cfg);
    CHECK_THROWS_WITH(build_wavefunction(cfg, groups),
                      Catch::Matchers::ContainsSubstring("a_M"));
}

TEST_CASE("explicit crystal centers under PBC run VMC with a moire jastrow",
          "[app][moire_jastrow][periodic]") {
    auto cfg = make_moire_crystal_cfg();
    cfg.qmc = {short_vmc()};
    App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 1);
    auto* vmc = dynamic_cast<alea::VMC*>(pack.drivers[0].get());
    REQUIRE(vmc != nullptr);
    vmc->run();
    CHECK(std::isfinite(vmc->energy()));
    CHECK(vmc->acceptance_ratio() > 0.0);
}

TEST_CASE("crystal centers on the same periodic image are refused",
          "[app][periodic]") {
    auto cfg = make_moire_crystal_cfg();
    // Shift the last center by a full cell vector: same site modulo the cell.
    auto& s = first_item<alea::io::SlaterSpec>(cfg);
    s.centers[3] = {s.centers[0][0] + cfg.lattice.vectors[0],
                    s.centers[0][1] + cfg.lattice.vectors[1]};
    cfg.qmc = {short_vmc()};
    App app(cfg);
    CHECK_THROWS_WITH(app.build_drivers(),
                      Catch::Matchers::ContainsSubstring("same site modulo the cell"));
}

TEST_CASE("a Gaussian orbital pool is still refused under PBC", "[app][periodic]") {
    auto cfg = make_moire_crystal_cfg();
    first_item<alea::io::SlaterSpec>(cfg).alpha.reset();
    first_item<alea::io::SlaterSpec>(cfg).alphas = {0.05, 0.05, 0.05, 0.05};
    cfg.qmc = {short_vmc()};
    App app(cfg);
    CHECK_THROWS_WITH(app.build_drivers(),
                      Catch::Matchers::ContainsSubstring("orbital pool"));
}

TEST_CASE("a moire jastrow off the cell's superlattice is refused",
          "[app][moire_jastrow][periodic]") {
    auto cfg = make_moire_crystal_cfg();
    first_item<alea::io::MoireJastrowSpec>(cfg).a_M    = 7.0;   // not commensurate with the cell
    first_item<alea::io::MoireJastrowSpec>(cfg).theta0 = 0.0;
    cfg.qmc = {short_vmc()};
    App app(cfg);
    CHECK_THROWS_WITH(app.build_drivers(),
                      Catch::Matchers::ContainsSubstring("not commensurate"));
}

namespace {
const std::string integration_system = R"(project: {seed: 713, log: false, output: false}
system:
  lattice: {type: open, dimension: 1, L: 4.0}
  particles:
    - {name: p, group: g, mass: 1.0, up: 1, down: 0}
hamiltonian: {terms: []}
)";
const std::string integration_pimc = R"(  - {method: pimc, beta: 1.0, time_step: 0.125, samples: 2, threads: 1, equilibration_blocks: 0, accumulation_blocks: 4, steps_per_block: 2, sub_steps: 1}
)";
const std::string integration_vmc = R"(  - {method: vmc, random_walk: metropolis, time_step: 0.1, samples: 2, threads: 1, equilibration_blocks: 0, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1}
)";
const std::string integration_dmc = R"(  - {method: dmc, random_walk: drift_diffusion, time_step: 0.01, samples: 2, threads: 1, equilibration_blocks: 0, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1}
)";
// One electron and one hole in a periodic 4x4 cell, no interaction and no
// trial factors: every sampler runs, so estimator wiring can be exercised.
const std::string eh_pair_system = R"(project: {seed: 719, log: false, output: false}
system:
  lattice: {type: periodic, dimension: 2, L: 4.0}
  particles:
    - {name: electron, group: e, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: hole, group: h, type: boltzmannon, mass: 1, charge: 1, up: 1, down: 0}
hamiltonian: {terms: []}
)";
}

TEST_CASE("legacy no-PIMC initialization and sampler seeds are unchanged", "[app][integration_fix]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n  - {method: optimizer, samples: 2, threads: 1}\n" + integration_vmc + integration_dmc);
    auto pack = App(cfg).build_drivers();
    REQUIRE(pack.drivers.size() == 3);
    for (std::size_t i = 1; i < 3; ++i) {
        alea::numerics::Random expected(cfg.seed + i + 1);
        for (const auto& walker : pack.ensembles[i]->walkers())
            CHECK(walker.positions()[0] == expected.uniform(-2.0, 2.0));
    }
    CHECK(alea::build_vmc_config(cfg.qmc[0], cfg.seed).seed == cfg.seed);
    CHECK(alea::build_dmc_config(cfg.qmc[1], cfg.seed).seed == cfg.seed);
}

TEST_CASE("build_dmc_config carries the branch cutoff scheme and alpha", "[app]") {
    alea::io::QmcSpec q;
    q.method = "dmc";
    CHECK(alea::build_dmc_config(q, 1).branch_cutoff == alea::BranchCutoff::universal);
    CHECK(alea::build_dmc_config(q, 1).branch_cutoff_alpha == alea::kDefaultBranchCutoffAlpha);
    q.branch_cutoff = "drv";
    CHECK(alea::build_dmc_config(q, 1).branch_cutoff == alea::BranchCutoff::drv);
    q.branch_cutoff = "none";
    CHECK(alea::build_dmc_config(q, 1).branch_cutoff == alea::BranchCutoff::none);
    q.branch_cutoff = "universal";
    q.branch_cutoff_alpha = 1.25;
    CHECK(alea::build_dmc_config(q, 1).branch_cutoff_alpha == 1.25);
}

TEST_CASE("build_dmc_config carries the reweighting, its correlation time and kappa", "[app]") {
    alea::io::QmcSpec q;
    q.method = "dmc";
    const alea::DMCConfig bare = alea::build_dmc_config(q, 1);
    CHECK(bare.reweighting == alea::Reweighting::bare);
    CHECK(bare.correlation_time == 0.0);
    CHECK(bare.suppression_kappa == alea::kDefaultSuppressionKappa);
    q.reweighting = "effective";
    CHECK(alea::build_dmc_config(q, 1).reweighting == alea::Reweighting::effective);
    q.reweighting = "suppressed";
    CHECK(alea::build_dmc_config(q, 1).reweighting == alea::Reweighting::suppressed);
    CHECK(alea::build_dmc_config(q, 1).correlation_time == 0.0);   // measured
    q.correlation_time = 0.3;
    q.suppression_kappa = 0.9;
    CHECK(alea::build_dmc_config(q, 1).correlation_time == 0.3);
    CHECK(alea::build_dmc_config(q, 1).suppression_kappa == 0.9);
    q.correlation_equilibration_sweeps = 512;
    q.correlation_min_sweeps = 2048;
    q.correlation_max_sweeps = 32768;
    CHECK(alea::build_dmc_config(q, 1).correlation_equilibration_sweeps == 512);
    CHECK(alea::build_dmc_config(q, 1).correlation_min_sweeps == 2048);
    CHECK(alea::build_dmc_config(q, 1).correlation_max_sweeps == 32768);
}

TEST_CASE("PIMC sampling seeds are stable when optimizer is removed", "[app][integration_fix]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n  - {method: optimizer, samples: 2, threads: 1}\n" + integration_vmc + integration_pimc + integration_dmc);
    auto original = App(cfg).build_drivers();
    cfg.optimizers.clear();
    auto replay = App(cfg).build_drivers();
    REQUIRE(original.drivers.size() == 4);
    REQUIRE(replay.drivers.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        const auto a = original.ensembles[i + 1]->walkers();
        const auto b = replay.ensembles[i]->walkers();
        REQUIRE(a.size() == b.size());
        for (std::size_t w = 0; w < a.size(); ++w)
            CHECK(a[w].positions()[0] == b[w].positions()[0]);
    }
    auto* a = dynamic_cast<alea::Pimc*>(original.drivers[2].get());
    auto* b = dynamic_cast<alea::Pimc*>(replay.drivers[1].get());
    REQUIRE(a); REQUIRE(b);
    a->run(); b->run();
    CHECK(a->summary() == b->summary());
}

TEST_CASE("PIMC checkpoint paths are retained by parser", "[app][integration_fix]") {
    const auto cfg = alea::io::parse_config(integration_system + "qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, paths_in: ring.h5, paths_out: final.h5}\n");
    REQUIRE(cfg.pimc.size() == 1);
    CHECK(cfg.pimc[0].paths_in == "ring.h5");
    CHECK(cfg.pimc[0].paths_out == "final.h5");
    auto pack = App(cfg).build_drivers();
    REQUIRE(pack.paths_in.size() == 1);
    REQUIRE(pack.paths_out.size() == 1);
    CHECK(pack.paths_in[0] == "ring.h5");
    CHECK(pack.paths_out[0] == "final.h5");
    const auto with_dir = App(cfg).build_drivers(std::filesystem::path("checkpoints") / "sample");
    CHECK(std::filesystem::path(with_dir.paths_out[0]) ==
          std::filesystem::path("checkpoints") / "sample" / "final.h5");
    CHECK(pack.walkers_in_set[0]);
    const auto mixed = alea::io::parse_config(integration_system + "qmc:\n" + integration_vmc +
        "  - {method: pimc, beta: 1, time_step: 0.125, paths_in: ring.h5, paths_out: final.h5}\n" + integration_dmc);
    auto ordered = App(mixed).build_drivers();
    REQUIRE(ordered.paths_in.size() == 3);
    CHECK(ordered.paths_in[0].empty());
    CHECK(ordered.paths_in[1] == "ring.h5");
    CHECK(ordered.paths_out[1] == "final.h5");
    CHECK(ordered.paths_in[2].empty());
    CHECK(alea::carry_over_walkers(ordered, 0, 1) == 0);
}

TEST_CASE("PIMC rejects unsupported twists and periodic collective density", "[app][integration_fix]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
    cfg.lattice.type = "periodic";
    SECTION("non-Gamma twist") {
        cfg.lattice.twist = {0.25};
        CHECK_THROWS_WITH(App(cfg).build_drivers(), Catch::Matchers::ContainsSubstring("twist"));
    }
    SECTION("periodic collective density") {
        cfg.pimc[0].estimators.push_back(
            alea::io::CollectiveDensitySpec{"center", 2.0, 8, {{"g", 1.0}}});
        CHECK_THROWS_WITH(App(cfg).build_drivers(), Catch::Matchers::ContainsSubstring("collective_density"));
    }
    SECTION("Gamma twist remains supported") {
        cfg.lattice.twist = {0.0};
        CHECK_NOTHROW(App(cfg).build_drivers());
    }
}

TEST_CASE("PIMC validates mass shape and positive finite values", "[app][integration_fix]") {
    for (const auto& mass : std::vector<std::vector<double>>{{1.0, 2.0}, {0.0}, {-1.0}, {INFINITY}, {NAN}}) {
        auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
        cfg.particles[0].mass = mass;
        CHECK_THROWS_WITH(App(cfg).build_drivers(), Catch::Matchers::ContainsSubstring("mass"));
    }
}

TEST_CASE("mixed sampling document order and adjacent carryover are preserved", "[app][integration_fix]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_vmc + integration_pimc + integration_dmc);
    auto pack = App(cfg).build_drivers();
    REQUIRE(pack.drivers.size() == 3);
    REQUIRE(dynamic_cast<alea::VMC*>(pack.drivers[0].get()));
    REQUIRE(dynamic_cast<alea::Pimc*>(pack.drivers[1].get()));
    REQUIRE(dynamic_cast<alea::DMC*>(pack.drivers[2].get()));
    pack.drivers[0]->run();
    const auto final = pack.drivers[0]->final_configurations();
    REQUIRE_FALSE(final.empty());
    REQUIRE(alea::carry_over_walkers(pack, 0, 1) == final.size());
    const auto next = pack.ensembles[1]->walkers();
    for (std::size_t w = 0; w < next.size(); ++w)
        CHECK(next[w].positions()[0] == final[w % final.size()]);

    // Design section 10: paths carry in from VMC, but a bead is not a DMC
    // population. PIMC exports full paths only via its separate checkpoint seam.
    std::vector<double> dmc_initial;
    for (const auto& walker : pack.ensembles[2]->walkers())
        dmc_initial.push_back(walker.positions()[0]);
    pack.drivers[1]->run();
    CHECK(pack.drivers[1]->final_configurations().empty());
    CHECK(pack.drivers[1]->final_population() == nullptr);
    CHECK(alea::carry_over_walkers(pack, 1, 2) == 0);
    for (std::size_t w = 0; w < dmc_initial.size(); ++w)
        CHECK(pack.ensembles[2]->walkers()[w].positions()[0] == dmc_initial[w]);
}

TEST_CASE("optimizer remains before sampling even when interleaved in YAML", "[app][integration_fix]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc +
        "  - {method: optimizer, samples: 2, threads: 1}\n" + integration_vmc);
    auto pack = App(cfg).build_drivers();
    REQUIRE(pack.drivers.size() == 3);
    CHECK(dynamic_cast<alea::LinearOptimize*>(pack.drivers[0].get()));
    CHECK(dynamic_cast<alea::Pimc*>(pack.drivers[1].get()));
    CHECK(dynamic_cast<alea::VMC*>(pack.drivers[2].get()));
}

TEST_CASE("PIMC action builder preserves every YAML component and refuses duplicate springs", "[app][pimc_action_wiring]") {
    auto cfg = alea::io::parse_config(integration_system +
        "action: [{kinetic: primitive}]\nqmc:\n" + integration_pimc);
    // Parser rejects duplicate YAML entries, but programmatic Config callers
    // must not silently collapse the independent action component list either.
    cfg.action.push_back(cfg.action.front());
    CHECK_THROWS_WITH(App(cfg).build_drivers(), Catch::Matchers::ContainsSubstring("primitive free kinetic"));
    cfg = alea::io::parse_config(integration_system +
        "action: [{potential: primitive}, {kinetic: primitive}]\nqmc:\n" + integration_pimc);
    CHECK_NOTHROW(App(cfg).build_drivers());
}

TEST_CASE("PIMC parser rejects unknown keys and unsupported action form maps", "[app][integration_fix]") {
    const std::vector<std::pair<std::string, std::string>> bad = {
        {"qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, betaa: 2}\n", "betaa"},
        {"action: [{primitive: {form: pair_product}}]\nqmc:\n" + integration_pimc, "pair_product"},
        {"action: [{kinetic: {form: pair_product}}]\nqmc:\n" + integration_pimc, "pair_product"},
        {"action: [{primitive: {mystery: 2}}]\nqmc:\n" + integration_pimc, "mystery"},
        {"action: [{primitive: {}, ignored: {}}]\nqmc:\n" + integration_pimc, "action"},
        {"qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, moves: [{centroid: {stepp: 1}}]}\n", "stepp"},
        {"qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, moves: [{bisection: {}, centroid: {}}]}\n", "move"},
        {"qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, moves: [{centroid: {}, mystery: {}}]}\n", "move"},
        {"qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, estimators: [{type: density, nbin: 4}]}\n", "nbin"},
        {"qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, estimators: [{type: collective_density, terms: [{group: g, coefff: 2}]}]}\n", "coefff"}
    };
    for (const auto& [tail, name] : bad) {
        INFO(name);
        CHECK_THROWS_WITH(alea::io::parse_config(integration_system + tail), Catch::Matchers::ContainsSubstring(name));
    }
}

TEST_CASE("PIMC checkpoint outputs cannot replace another stage's checkpoint", "[app][pimc_checkpoint_collision]") {
    for (const std::string first : {"dmc", "pimc"}) {
        INFO(first);
        const std::string stage = first == "dmc"
            ? "  - {method: dmc, random_walk: drift_diffusion, time_step: 0.01, samples: 2, threads: 1, walkers_out: shared.h5}\n"
            : "  - {method: pimc, beta: 1, time_step: 0.125, samples: 2, threads: 1, paths_out: shared.h5}\n";
        auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + stage +
            "  - {method: pimc, beta: 1, time_step: 0.125, samples: 2, threads: 1, paths_out: shared.h5}\n");
        CHECK_THROWS_WITH(App(cfg).build_drivers("checkpoints"),
                          Catch::Matchers::ContainsSubstring("checkpoint"));
    }
}

TEST_CASE("PIMC paths cannot overwrite DMC milestone checkpoints", "[app][pimc_checkpoint_collision]") {
    // DMC's shared.h5 additionally writes shared.eq.h5 and shared.a{25,50,75}.h5.
    // This must fail before either stage runs, in both document orders.
    for (const std::string first : {"dmc", "pimc"}) {
        INFO(first);
        const std::string dmc =
            "  - {method: dmc, random_walk: drift_diffusion, time_step: 0.01, samples: 2, threads: 1, walkers_out: shared.h5}\n";
        const std::string pimc =
            "  - {method: pimc, beta: 1, time_step: 0.125, samples: 2, threads: 1, paths_out: shared.eq.h5}\n";
        auto cfg = alea::io::parse_config(integration_system + "qmc:\n" +
            (first == "dmc" ? dmc + pimc : pimc + dmc));
        CHECK_THROWS_WITH(App(cfg).build_drivers("checkpoints"),
                          Catch::Matchers::ContainsSubstring("checkpoint"));
    }
}

TEST_CASE("repeated estimators of one kind get numbered result keys in every driver",
          "[app][estimator_output_keys]") {
    // A fine and a coarse g(r), and a density twice: the first instance of a
    // kind keeps the plain keys and the repeat writes under <namespace>_2.
    // Nothing is refused, and nothing is lost: this used to be refused in
    // PIMC and silently dropped from the VMC/DMC results file.
    const std::string estimators =
        "estimators: [{type: gofr, rmax: 0.5, n_bins: 4}, {type: gofr, rmax: 1.5, n_bins: 6}, "
        "{type: density, rmax: 2, n_bins: 4}, {type: density, rmax: 2, n_bins: 4}]}\n";
    const std::string blocks = "threads: 1, equilibration_blocks: 0, accumulation_blocks: 2, "
                               "steps_per_block: 1, sub_steps: 1, ";
    for (const std::string stage :
         {"{method: vmc, random_walk: metropolis, time_step: 0.1, samples: 2, ",
          "{method: dmc, random_walk: drift_diffusion, time_step: 0.01, samples: 2, ",
          "{method: pimc, beta: 1, time_step: 0.125, samples: 2, "}) {
        INFO(stage);
        auto pack = App(alea::io::parse_config(eh_pair_system + "qmc:\n  - " + stage + blocks +
                                               estimators)).build_drivers();
        REQUIRE(pack.drivers.size() == 1);
        pack.drivers[0]->run();
        const alea::RunResult r = pack.drivers[0]->run_result();  // unique keys checked
        const auto series = [&](const std::string& key) -> const std::vector<double>* {
            for (const auto& [name, values] : r.series)
                if (name == key) return &values;
            return nullptr;
        };
        for (const std::string key : {"gofr_r", "gofr_2_r", "density_edges", "density_2_edges"}) {
            INFO(key);
            CHECK(series(key) != nullptr);
        }
        // Each g(r) keeps its own grid.
        REQUIRE(series("gofr_r") != nullptr);
        REQUIRE(series("gofr_2_r") != nullptr);
        CHECK(series("gofr_r")->size() == 4);
        CHECK(series("gofr_2_r")->size() == 6);
    }
}

TEST_CASE("momentum_distribution is refused in VMC and DMC stages", "[app][estimator_output_keys]") {
    for (const std::string method : {"vmc", "dmc"}) {
        INFO(method);
        auto cfg = alea::io::parse_config(eh_pair_system + "qmc:\n  - {method: " + method +
            ", samples: 2, estimators: [{type: momentum_distribution, group: e, rmax: 1, "
            "n_bins: 4, kmax: 2, samples: 1}]}\n");
        CHECK_THROWS_WITH(App(cfg).build_drivers(),
                          Catch::Matchers::ContainsSubstring("PIMC-only") &&
                              Catch::Matchers::ContainsSubstring("worm"));
    }
}

TEST_CASE("PIMC rejects overlapping named estimator result keys", "[app][pimc_output_keys]") {
    // Different collective names may still collide: cdensity_x_error is both
    // x's error series and the mean series for a separately named x_error.
    auto cfg = alea::io::parse_config(integration_system +
        "qmc:\n  - {method: pimc, beta: 1, time_step: 0.125, samples: 2, threads: 1, equilibration_blocks: 0, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1, estimators: [{type: collective_density, name: x, rmax: 2, n_bins: 4, terms: [{group: g, coeff: 1}]}, {type: collective_density, name: x_error, rmax: 2, n_bins: 4, terms: [{group: g, coeff: 1}]}]}\n");
    auto pack = App(cfg).build_drivers();
    REQUIRE(pack.drivers.size() == 1);
    pack.drivers[0]->run();
    CHECK_THROWS_WITH(pack.drivers[0]->run_result(),
                      Catch::Matchers::ContainsSubstring("duplicate"));
}

TEST_CASE("Electron-hole proximity is an opt-in named cutoff diagnostic in every sampler", "[app][pimc_proximity]") {
    const std::string eh_system = R"(project: {seed: 718, log: false, output: false}
system:
  lattice: {type: periodic, dimension: 2, L: 4.0}
  particles:
    - {name: electron, group: e, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: hole, group: h, type: boltzmannon, mass: 1, charge: 1, up: 1, down: 0}
hamiltonian: {terms: []}
)";
    const std::string run = "qmc:\n  - {method: pimc, beta: 1, time_step: 0.0625, samples: 2, threads: 1, equilibration_blocks: 0, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1, estimators: [{type: electron_hole_proximity, electron_group: e, hole_group: h, r_cut: 0.5}]}\n";
    auto cfg = alea::io::parse_config(eh_system + run);
    REQUIRE(cfg.pimc.size() == 1);
    REQUIRE(cfg.pimc[0].estimators.size() == 1);
    const auto* spec =
        std::get_if<alea::io::ElectronHoleProximitySpec>(&cfg.pimc[0].estimators[0]);
    REQUIRE(spec != nullptr);
    CHECK(spec->electron_group == "e");
    CHECK(spec->hole_group == "h");
    CHECK(spec->r_cut == Catch::Approx(0.5));
    auto built = App(cfg).build_drivers();
    REQUIRE(built.drivers.size() == 1);
    built.drivers[0]->run();
    const auto result = built.drivers[0]->run_result();
    CHECK(std::any_of(result.series.begin(), result.series.end(), [](const auto& entry) {
        return entry.first == "electron_hole_proximity_e_h_U_e";
    }));
    CHECK(std::any_of(result.series.begin(), result.series.end(), [](const auto& entry) {
        return entry.first == "electron_hole_proximity_e_h_U_e_error";
    }));
    const auto has_series = [](const alea::RunResult& r, const std::string& key) {
        return std::any_of(r.series.begin(), r.series.end(),
                           [&](const auto& entry) { return entry.first == key; });
    };
    const auto metadata = [](const alea::RunResult& r, const std::string& key) {
        for (const auto& [name, value] : r.metadata)
            if (name == key) return value;
        return std::string("<missing>");
    };
    // A repeated instance is numbered, not refused.
    cfg.pimc[0].estimators.push_back(cfg.pimc[0].estimators.front());
    {
        auto twice = App(cfg).build_drivers();
        twice.drivers[0]->run();
        const auto r = twice.drivers[0]->run_result();
        CHECK(has_series(r, "electron_hole_proximity_e_h_U_e"));
        CHECK(has_series(r, "electron_hole_proximity_e_h_2_U_e"));
    }
    cfg.pimc[0].estimators.pop_back();
    std::get<alea::io::ElectronHoleProximitySpec>(cfg.pimc[0].estimators[0]).r_cut = 2.01;
    CHECK_THROWS_WITH(App(cfg).build_drivers(), Catch::Matchers::ContainsSubstring("r_cut"));

    CHECK_THROWS_WITH(alea::io::parse_config(eh_system +
        "qmc:\n  - {method: pimc, beta: 1, time_step: 0.0625, samples: 2, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1, estimators: [{type: electron_hole_proximity, electron_group: e, hole_group: h, r_cut: 0.5, typo: 2}]}\n"),
        Catch::Matchers::ContainsSubstring("typo"));
    // A diagonal observable of R: VMC (|psi_T|^2) and DMC (mixed estimate)
    // measure it too, with U_e + C_e = 1 in [0, 1].
    for (const std::string stage :
         {"{method: vmc, random_walk: metropolis, time_step: 0.1, ",
          "{method: dmc, random_walk: drift_diffusion, time_step: 0.01, "}) {
        INFO(stage);
        auto pack = App(alea::io::parse_config(eh_system + "qmc:\n  - " + stage +
            "samples: 4, threads: 1, equilibration_blocks: 0, accumulation_blocks: 4, "
            "steps_per_block: 2, sub_steps: 1, estimators: [{type: electron_hole_proximity, "
            "electron_group: e, hole_group: h, r_cut: 0.5}]}\n")).build_drivers();
        REQUIRE(pack.drivers.size() == 1);
        pack.drivers[0]->run();
        const auto r = pack.drivers[0]->run_result();
        double u = -1.0, c = -1.0;
        for (const auto& [name, values] : r.series) {
            if (name == "electron_hole_proximity_e_h_U_e") u = values.at(0);
            if (name == "electron_hole_proximity_e_h_C_e") c = values.at(0);
        }
        CHECK(u >= 0.0);
        CHECK(u <= 1.0);
        CHECK(u + c == Catch::Approx(1.0));
        CHECK(has_series(r, "electron_hole_proximity_e_h_U_e_error"));
        CHECK(metadata(r, "electron_hole_proximity_e_h_measure").find("walkers") != std::string::npos);
    }

    // Underscores in distinct group pairs can produce the same HDF5 prefix:
    // e_a/h and e/a_h both encode as electron_hole_proximity_e_a_h.  The
    // second instance is numbered, and each records its own groups.
    const std::string ambiguous_groups = R"(project: {seed: 718, log: false, output: false}
system:
  lattice: {type: periodic, dimension: 2, L: 4.0}
  particles:
    - {name: ea, group: e_a, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: h, group: h, type: boltzmannon, mass: 1, charge: 1, up: 1, down: 0}
    - {name: e, group: e, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: ah, group: a_h, type: boltzmannon, mass: 1, charge: 1, up: 1, down: 0}
hamiltonian: {terms: []}
qmc:
  - {method: pimc, beta: 1, time_step: 0.0625, samples: 2, threads: 1, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1, estimators: [{type: electron_hole_proximity, electron_group: e_a, hole_group: h, r_cut: 0.5}, {type: electron_hole_proximity, electron_group: e, hole_group: a_h, r_cut: 0.5}]}
)";
    {
        auto pack = App(alea::io::parse_config(ambiguous_groups)).build_drivers();
        pack.drivers[0]->run();
        const auto r = pack.drivers[0]->run_result();
        CHECK(metadata(r, "electron_hole_proximity_e_a_h_electron_group") == "e_a");
        CHECK(metadata(r, "electron_hole_proximity_e_a_h_hole_group") == "h");
        CHECK(metadata(r, "electron_hole_proximity_e_a_h_2_electron_group") == "e");
        CHECK(metadata(r, "electron_hole_proximity_e_a_h_2_hole_group") == "a_h");
    }

    // Exact group names, not display prefixes, select the counted particles.
    const std::string overlapping_prefixes = R"(project: {seed: 718, log: false, output: false}
system:
  lattice: {type: periodic, dimension: 2, L: 4.0}
  particles:
    - {name: ea, group: e_a, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: h, group: h, type: boltzmannon, mass: 1, charge: 1, up: 1, down: 0}
    - {name: e, group: e, type: boltzmannon, mass: 1, charge: -1, up: 1, down: 0}
    - {name: ah, group: a_h, type: boltzmannon, mass: 1, charge: 1, up: 1, down: 0}
hamiltonian: {terms: []}
qmc:
  - {method: pimc, beta: 1, time_step: 0.0625, samples: 2, threads: 1, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1, estimators: [{type: electron_hole_proximity, electron_group: e_a, hole_group: h, r_cut: 0.5}]}
)";
    CHECK_NOTHROW(App(alea::io::parse_config(overlapping_prefixes)).build_drivers());
}

TEST_CASE("PIMC parser refuses nonfinite or nonpositive beta and tau", "[app][integration_fix]") {
    for (const std::string key : {"beta", "time_step"})
        for (const std::string value : {".nan", ".inf", "-.inf", "0", "-1"}) {
            const std::string fields = key == "beta" ? "beta: " + value + ", time_step: 0.125" : "beta: 1, time_step: " + value;
            CHECK_THROWS_WITH(alea::io::parse_config(integration_system + "qmc:\n  - {method: pimc, " + fields + "}\n"), Catch::Matchers::ContainsSubstring(key));
        }
}

TEST_CASE("PIMC exchange classes aggregate repeated group and spin entries", "[app][integration_fix]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
    cfg.particles.push_back(cfg.particles[0]);
    cfg.particles.back().name = "p2";
    CHECK_THROWS_WITH(App(cfg).build_drivers(), Catch::Matchers::ContainsSubstring("statistics"));
    cfg.pimc[0].statistics = "boltzmannon";
    CHECK_NOTHROW(App(cfg).build_drivers());
}

TEST_CASE("PIMC accepts declared boltzmannons without an additional statistics opt-in",
          "[app][pimc_statistics_validation]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
    cfg.particles[0].type = "boltzmannon";
    cfg.particles[0].up = 2;
    SECTION("one entry with several particles") {}
    SECTION("repeated entries in the same group") {
        cfg.particles.push_back(cfg.particles[0]);
        cfg.particles.back().name = "p2";
    }
    CHECK_NOTHROW(App(cfg).build_drivers());
}

TEST_CASE("PIMC quantum exchange classes respect particle group and spin",
          "[app][pimc_statistics_validation]") {
    for (const std::string type : {"fermions", "bosons"}) {
        INFO(type);
        auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
        cfg.particles[0].type = type;
        cfg.particles[0].up = 2;
        CHECK_THROWS_WITH(App(cfg).build_drivers(),
                          Catch::Matchers::ContainsSubstring("statistics"));
        cfg.pimc[0].statistics = "boltzmannon";
        CHECK_NOTHROW(App(cfg).build_drivers());

        cfg.pimc[0].statistics.clear();
        cfg.particles[0].up = 1;
        cfg.particles[0].down = 1;  // different spin classes of size one
        CHECK_NOTHROW(App(cfg).build_drivers());

        cfg.particles[0].down = 0;
        cfg.particles.push_back(cfg.particles[0]);
        cfg.particles.back().name = "other";
        cfg.particles.back().group = "other_group";
        CHECK_NOTHROW(App(cfg).build_drivers());

        // A large distinguishable group must neither trigger exchange itself
        // nor exempt a quantum class in another group from the statistics gate.
        cfg.particles.back().type = "boltzmannon";
        cfg.particles.back().up = 3;
        CHECK_NOTHROW(App(cfg).build_drivers());
        cfg.particles[0].up = 2;
        CHECK_THROWS_WITH(App(cfg).build_drivers(),
                          Catch::Matchers::ContainsSubstring("statistics"));
    }
}

TEST_CASE("Flavour counts form one exchange class per group and flavour",
          "[app][pimc_statistics_validation][flavours]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
    cfg.particles[0].up = 0;
    cfg.particles[0].flavours = {2, 0, 1, 3};
    const auto classes = alea::build_exchange_classes(cfg.particles);
    REQUIRE(classes.size() == 3);  // the empty flavour 1 forms no class
    CHECK(classes[0].flavour == 0);
    CHECK(classes[0].particles == std::vector<std::size_t>{0, 1});
    CHECK(classes[1].flavour == 2);
    CHECK(classes[1].particles == std::vector<std::size_t>{2});
    CHECK(classes[2].flavour == 3);
    CHECK(classes[2].particles == std::vector<std::size_t>{3, 4, 5});
    CHECK(classes[2].statistics == "fermions");  // missing type counts as fermions
    CHECK(alea::largest_exchange_class(classes) == 3);
    CHECK(build_groups(cfg)[0].size() == 6);

    CHECK_THROWS_WITH(App(cfg).build_drivers(), Catch::Matchers::ContainsSubstring("statistics"));
    cfg.pimc[0].statistics = "boltzmannon";
    CHECK_NOTHROW(App(cfg).build_drivers());
    cfg.pimc[0].statistics.clear();
    cfg.particles[0].flavours = {1, 1, 1, 1};  // four classes of one: nothing exchanges
    CHECK_NOTHROW(App(cfg).build_drivers());
}

TEST_CASE("up/down give the same classes and indices as two flavours",
          "[app][pimc_statistics_validation][flavours]") {
    auto two_spin = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
    two_spin.particles[0].up = 2;
    two_spin.particles[0].down = 1;
    auto two_flavour = two_spin;
    two_flavour.particles[0].up = 0;
    two_flavour.particles[0].down = 0;
    two_flavour.particles[0].flavours = {2, 1};
    const auto a = alea::build_exchange_classes(two_spin.particles);
    const auto b = alea::build_exchange_classes(two_flavour.particles);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].flavour == b[i].flavour);
        CHECK(a[i].particles == b[i].particles);
    }
    CHECK(a[0].particles == std::vector<std::size_t>{0, 1});  // spin-up first
    CHECK(build_groups(two_spin)[0].size() == build_groups(two_flavour)[0].size());

    two_spin.particles[0].down = -1;  // the parser does not range-check up/down
    CHECK_THROWS_WITH(alea::build_exchange_classes(two_spin.particles),
                      Catch::Matchers::ContainsSubstring("negative"));
}

TEST_CASE("Repeated group entries add to classes by flavour index; boltzmannons form none",
          "[app][pimc_statistics_validation][flavours]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
    cfg.particles[0].up = 0;
    cfg.particles[0].flavours = {2, 1};
    cfg.particles.push_back(cfg.particles[0]);
    cfg.particles.back().name = "p2";
    cfg.particles.back().flavours = {1};
    cfg.particles.push_back(cfg.particles[0]);
    cfg.particles.back().name = "x";
    cfg.particles.back().group = "distinguishable";
    cfg.particles.back().type = "boltzmannon";
    cfg.particles.back().flavours = {3, 3};
    const auto classes = alea::build_exchange_classes(cfg.particles);
    REQUIRE(classes.size() == 2);
    CHECK(classes[0].particles == std::vector<std::size_t>{0, 1, 3});  // not contiguous
    CHECK(classes[1].particles == std::vector<std::size_t>{2});
    CHECK(alea::largest_exchange_class(classes) == 3);
}

TEST_CASE("flavours is refused outside PIMC-only runs",
          "[app][flavours]") {
    using Catch::Matchers::ContainsSubstring;
    const std::string flavoured_system = R"(project: {seed: 713, log: false, output: false}
system:
  lattice: {type: open, dimension: 1, L: 4.0}
  particles:
    - {name: p, group: g, mass: 1.0, flavours: [1, 1]}
hamiltonian: {terms: []}
)";
    CHECK_NOTHROW(App(alea::io::parse_config(flavoured_system + "qmc:\n" + integration_pimc)).build_drivers());
    CHECK_THROWS_WITH(App(alea::io::parse_config(flavoured_system + "qmc:\n" + integration_vmc)).build_drivers(),
                      ContainsSubstring("PIMC-only"));
    CHECK_THROWS_WITH(App(alea::io::parse_config(flavoured_system + "qmc:\n" + integration_dmc)).build_drivers(),
                      ContainsSubstring("PIMC-only"));
    CHECK_THROWS_WITH(App(alea::io::parse_config(flavoured_system +
                          "qmc:\n  - {method: optimizer, samples: 2, threads: 1}\n" + integration_pimc)).build_drivers(),
                      ContainsSubstring("PIMC-only"));
    CHECK_THROWS_WITH(App(alea::io::parse_config(flavoured_system +
                          "wavefunction:\n  - jastrow: {type: pade, pair: [g, g], denominator: [0.5]}\nqmc:\n" + integration_pimc)).build_drivers(),
                      ContainsSubstring("PIMC-only"));
}

namespace {
// Runs one stage of `system` with `estimators` in VMC, DMC and PIMC and hands
// each sampler's results to `check`.
template <class Check>
void run_in_every_sampler(const std::string& system, const std::string& estimators,
                          Check check) {
    const std::string blocks = "threads: 1, equilibration_blocks: 0, accumulation_blocks: 4, "
                               "steps_per_block: 2, sub_steps: 1, ";
    for (const std::string stage :
         {"{method: vmc, random_walk: metropolis, time_step: 0.1, samples: 2, ",
          "{method: dmc, random_walk: drift_diffusion, time_step: 0.01, samples: 2, ",
          "{method: pimc, beta: 1, time_step: 0.125, samples: 2, "}) {
        INFO(stage);
        auto pack = App(alea::io::parse_config(system + "qmc:\n  - " + stage + blocks +
                                               "estimators: " + estimators + "}\n"))
                        .build_drivers();
        REQUIRE(pack.drivers.size() == 1);
        pack.drivers[0]->run();
        check(pack.drivers[0]->run_result());  // unique keys checked
    }
}

const std::vector<double>* find_series(const alea::RunResult& r, const std::string& key) {
    for (const auto& [name, values] : r.series)
        if (name == key) return &values;
    return nullptr;
}
}  // namespace

TEST_CASE("Bragg order is measured by every sampler and a repeat is numbered", "[app][bragg]") {
    // A diagonal observable of R (not PIMC-only).  One particle: |rho(G)|^2/N^2
    // = 1 for every configuration, so B = 1 exactly in every sampler.  The
    // repeated instance writes under bragg_g_2 instead of being refused.
    const std::string system = R"(project: {seed: 731, log: false, output: false}
system:
  lattice: {type: periodic, dimension: 2, L: 4.0}
  particles:
    - {name: p, group: g, type: boltzmannon, mass: 1, up: 1, down: 0}
hamiltonian: {terms: []}
)";
    const std::string bragg = "{type: bragg, group: g, wavevectors: [[1.5707963267948966, 0]]}";
    run_in_every_sampler(system, "[" + bragg + ", " + bragg + "]", [](const alea::RunResult& r) {
        for (const std::string key : {"bragg_g", "bragg_g_2"}) {
            INFO(key);
            const auto* b = find_series(r, key);
            REQUIRE(b != nullptr);
            CHECK(b->at(0) == Catch::Approx(1.0).margin(1e-12));
            CHECK(find_series(r, key + "_error") != nullptr);
            CHECK(find_series(r, key + "_kx") != nullptr);
        }
    });
}

TEST_CASE("Bond order is measured by every sampler and a repeat is numbered", "[app][bond_order]") {
    // Seven particles, no interaction: the sixfold orders stay in [0, 1] and
    // the local one bounds the global one, in every sampler.
    const std::string system = R"(project: {seed: 733, log: false, output: false}
system:
  lattice: {type: periodic, dimension: 2, L: 8.0}
  particles:
    - {name: p, group: g, type: boltzmannon, mass: 1, up: 7, down: 0}
hamiltonian: {terms: []}
)";
    const std::string bond = "{type: bond_order, group: g, reference_spacing: 2.0}";
    run_in_every_sampler(system, "[" + bond + ", " + bond + "]", [](const alea::RunResult& r) {
        for (const std::string ns : {"bond_g", "bond_g_2"}) {
            INFO(ns);
            const auto* global = find_series(r, ns + "_global6");
            const auto* local = find_series(r, ns + "_local6");
            REQUIRE(global != nullptr);
            REQUIRE(local != nullptr);
            CHECK(find_series(r, ns + "_distortion_error") != nullptr);
            CHECK(global->at(0) >= 0.0);
            CHECK(local->at(0) <= 1.0 + 1e-12);
            CHECK(global->at(0) <= local->at(0) + 1e-12);
        }
    });
}

TEST_CASE("PIMC refuses unknown particle types even with a Boltzmann statistics override",
          "[app][pimc_statistics_validation]") {
    auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
    cfg.particles[0].type = "fermion_typo";
    cfg.pimc[0].statistics = "boltzmannon";
    CHECK_THROWS_WITH(App(cfg).build_drivers(),
                      Catch::Matchers::ContainsSubstring("fermion_typo"));
}

TEST_CASE("PIMC refuses incompatible particle types in the same group",
          "[app][pimc_statistics_validation]") {
    for (const std::string type : {"fermions", "bosons", ""}) {
        INFO(type);
        auto cfg = alea::io::parse_config(integration_system + "qmc:\n" + integration_pimc);
        cfg.particles[0].type = type;
        cfg.particles.push_back(cfg.particles[0]);
        cfg.particles.back().type = "boltzmannon";
        cfg.particles.back().name = "p2";
        cfg.pimc[0].statistics = "boltzmannon";
        CHECK_THROWS_WITH(App(cfg).build_drivers(),
                          Catch::Matchers::ContainsSubstring("incompatible particle types"));
    }
}

// ── Backflow on a slater item ───────────────────────────────────────────────

namespace {
std::string backflow_gas_yaml(const std::string& extra_lattice, const std::string& backflow,
                              int up = 5, int down = 0, const std::string& orbitals =
                                  "{ type: plane_wave }") {
    return std::string(R"(
system:
  lattice: { type: periodic, dimension: 2, shape: square, L: 4.0)") + extra_lattice + R"( }
  particles:
    - { name: e, group: electrons, type: fermions, mass: 1.0, charge: -1, up: )" +
           std::to_string(up) + ", down: " + std::to_string(down) + R"( }
    - { name: h, group: holes, type: fermions, mass: 1.0, charge: 1, up: 1, down: 0 }
wavefunction:
  - slater:
      group: electrons
      orbitals: )" + orbitals + R"(
      backflow: )" + backflow + "\n";
}
}  // namespace

TEST_CASE("a slater item with backflow builds a backflow determinant", "[app][backflow]") {
    const auto cfg = alea::io::parse_config(backflow_gas_yaml(
        "", "[{ from: electrons, params: [0.1, 0.02] }, "
            "{ from: holes, cutoff: 1.5, cutoff_fixed: true, params: [-0.1, 0.05], "
            "is_fixed: [false, true] }]"));
    const auto groups = build_groups(cfg);
    const std::vector<double> cellv = {4.0, 0.0, 0.0, 4.0};
    const alea::SimulationCell cell(2, std::span<const double>(cellv));
    auto psi = build_wavefunction(cfg, groups, nullptr, cell);
    REQUIRE(psi.components().size() == 1);
    CHECK(psi.components()[0]->name() == "backflow_slater_determinant");
    CHECK(psi.components()[0]->is_nodal());
    const auto params = psi.components()[0]->parameters();
    REQUIRE(params.size() == 6);   // [c0, c2, L] like-spin, [c0, c1, L] from holes
    CHECK(params[0].name == "bf0_c0");
    CHECK(params[1].name == "bf0_c2");
    CHECK(params[2].name == "bf0_L");
    CHECK(params[2].value == Catch::Approx(2.0));      // default: the inradius of the 4×4 cell
    CHECK_FALSE(params[2].is_fixed);
    CHECK(params[2].upper_bound == Catch::Approx(2.0));
    CHECK(params[4].name == "bf1_c1");
    CHECK(params[4].is_fixed);
    CHECK(params[5].value == 1.5);
    CHECK(params[5].is_fixed);
}

TEST_CASE("backflow is refused where it cannot be represented", "[app][backflow]") {
    const std::vector<double> cellv = {4.0, 0.0, 0.0, 4.0};
    const alea::SimulationCell cell(2, std::span<const double>(cellv));
    auto build = [&](const std::string& yaml) {
        const auto cfg = alea::io::parse_config(yaml);
        return build_wavefunction(cfg, build_groups(cfg), nullptr, cell);
    };
    using Catch::Matchers::ContainsSubstring;
    // Twisted plane waves are complex.
    CHECK_THROWS_WITH(build(backflow_gas_yaml(", twist: [0.2, 0.1]",
                                              "[{ from: holes, params: [0.1, 0.0] }]", 4, 0)),
                      ContainsSubstring("backflow") && ContainsSubstring("twisted"));
    // A same-group channel on a group with both spins.
    CHECK_THROWS_WITH(build(backflow_gas_yaml("", "[{ from: electrons, params: [0.1] }]", 5, 1)),
                      ContainsSubstring("one spin"));
    // A cutoff beyond the inradius.
    CHECK_THROWS_WITH(build(backflow_gas_yaml("", "[{ from: holes, cutoff: 2.5, params: [0.1, 0.0] }]")),
                      ContainsSubstring("inradius"));
}

TEST_CASE("an open-cell backflow channel needs a cutoff", "[app][backflow]") {
    const auto cfg = alea::io::parse_config(R"(
system:
  lattice: { type: open, dimension: 2, L: 1.0 }
  particles:
    - { name: e, group: electrons, type: fermions, mass: 1.0, charge: -1, up: 2, down: 0 }
wavefunction:
  - slater:
      group: electrons
      orbitals: { type: gaussian, centers: [[-1.0, 0.0], [1.0, 0.0]], alphas: [0.3, 0.3] }
      backflow: [{ from: electrons, params: [0.1] }]
)");
    CHECK_THROWS_WITH(build_wavefunction(cfg, build_groups(cfg), nullptr, alea::SimulationCell(2)),
                      Catch::Matchers::ContainsSubstring("cutoff"));
}
