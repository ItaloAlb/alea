// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <utility>
#include <vector>
#include <variant>

#include "io/parser.h"

using alea::io::parse_config;
using alea::io::parse_config_file;
using alea::io::resolve_seed;
using alea::io::JastrowSpec;
using alea::io::MoireJastrowSpec;
using alea::io::SlaterSpec;
using alea::io::CollectiveDensitySpec;
using alea::io::DensitySpec;
using alea::io::ElectronHoleProximitySpec;
using alea::io::GofrSpec;

TEST_CASE("bond order requires an explicit finite spacing and refuses irrelevant keys", "[parser][bond_order]") {
    for (const std::string stage : {"{method: pimc, beta: 1, time_step: .125, ", "{method: vmc, "}) {
        INFO(stage);
        const std::string prefix = "qmc:\n  - " + stage + "estimators: [";
        auto cfg = parse_config(prefix+"{type: bond_order, name: hex, group: e, reference_spacing: 2.5}]}\n");
        const auto& items = cfg.pimc.empty() ? cfg.qmc.at(0).estimators : cfg.pimc.at(0).estimators;
        REQUIRE(items.size() == 1);
        const auto* spec = std::get_if<alea::io::BondOrderSpec>(&items[0]);
        REQUIRE(spec != nullptr);
        CHECK(spec->name == "hex");
        CHECK(spec->group == "e");
        CHECK(spec->reference_spacing == 2.5);
        for (const std::string fields : {"group: e", "reference_spacing: 2", "group: e, reference_spacing: 0",
             "group: e, reference_spacing: .nan", "group: e, reference_spacing: 2, neighbors: 6",
             "group: e, reference_spacing: 2, wavevectors: [[1, 0]]"})
            CHECK_THROWS(parse_config(prefix+"{type: bond_order, "+fields+"}]}\n"));
        CHECK_THROWS(parse_config(prefix+"{type: sofk, kmax: 2, reference_spacing: 2}]}\n"));
    }
}

TEST_CASE("Bragg input preserves Cartesian directions and refuses malformed keys", "[parser][bragg]") {
    for (const std::string stage : {"{method: pimc, beta: 1, time_step: .125, ", "{method: dmc, "}) {
        INFO(stage);
        const std::string prefix = "project: {seed: 42}\nqmc:\n  - " + stage + "estimators: [";
        auto cfg = parse_config(prefix + "{type: bragg, group: e, wavevectors: [[1, 2], [3, 4]]}]}\n");
        const auto& items = cfg.pimc.empty() ? cfg.qmc.at(0).estimators : cfg.pimc.at(0).estimators;
        REQUIRE(items.size() == 1);
        const auto* spec = std::get_if<alea::io::BraggSpec>(&items[0]);
        REQUIRE(spec != nullptr);
        CHECK(spec->name.empty());
        CHECK(spec->group == "e");
        CHECK(spec->wavevectors == std::vector<double>{1., 2., 3., 4.});
        for (const std::string fields : {"group: e", "group: e, wavevectors: []",
                "group: e, wavevectors: [1, 2]", "group: e, wavevectors: [[1, 2, 3]]",
                "group: e, wavevectors: [[.nan, 2]]", "group: e, wavevectors: [[1, 2]], kmax: 3"})
            CHECK_THROWS(parse_config(prefix + "{type: bragg, " + fields + "}]}\n"));
        CHECK_THROWS(parse_config(prefix + "{type: sofk, kmax: 2, wavevectors: [[1, 2]]}]}\n"));
    }
}

TEST_CASE("density_matrix input keeps its groups, pairs and stride, and refuses malformed keys",
          "[parser][density_matrix]") {
    const std::string prefix = "project: {seed: 42}\nqmc:\n  - {method: vmc, estimators: [";
    auto cfg = parse_config(prefix +
        "{type: density_matrix, name: dm, groups: [e_up], pairs: [[e_up, h_dn], [e_dn, h_up]], "
        "rmax: 13.5, n_bins: 60, every: 20}]}\n");
    const auto* spec = std::get_if<alea::io::DensityMatrixSpec>(&cfg.qmc.at(0).estimators.at(0));
    REQUIRE(spec != nullptr);
    CHECK(spec->name == "dm");
    CHECK(spec->groups == std::vector<std::string>{"e_up"});
    REQUIRE(spec->pairs.size() == 2);
    CHECK(spec->pairs[0] == std::pair<std::string, std::string>{"e_up", "h_dn"});
    CHECK(spec->pairs[1] == std::pair<std::string, std::string>{"e_dn", "h_up"});
    CHECK(spec->rmax == 13.5);
    CHECK(spec->n_bins == 60);
    CHECK(spec->every == 20);
    auto defaults = parse_config(prefix + "{type: density_matrix, groups: [e], rmax: 2, n_bins: 4}]}\n");
    CHECK(std::get<alea::io::DensityMatrixSpec>(defaults.qmc.at(0).estimators.at(0)).every == 1);
    for (const std::string fields : {"groups: [e], n_bins: 4", "groups: [e], rmax: 2",
             "rmax: 2, n_bins: 4", "groups: e, rmax: 2, n_bins: 4",
             "pairs: [[e, h, x]], rmax: 2, n_bins: 4", "pairs: [e, h], rmax: 2, n_bins: 4",
             "groups: [e], rmax: 2, n_bins: 4, kmax: 3"})
        CHECK_THROWS(parse_config(prefix + "{type: density_matrix, " + fields + "}]}\n"));
}

TEST_CASE("QMC method and estimator key typos cannot silently change the algorithm", "[parser][qmc_method_keys]") {
    const std::string prefix = "project: {seed: 42}\nqmc:\n  - ";
    CHECK_THROWS_WITH(parse_config(prefix + "{method: dcm, samples: 2}"),
                      Catch::Matchers::ContainsSubstring("method"));
    CHECK_THROWS_WITH(parse_config(prefix + "{samples: 2}"),
                      Catch::Matchers::ContainsSubstring("method"));
    CHECK_THROWS_WITH(parse_config(prefix +
        "{method: vmc, estimators: [{type: density, rmax: 2, n_bins: 10, per_partcle: true}]}"),
        Catch::Matchers::ContainsSubstring("per_partcle"));
    auto valid = parse_config(prefix +
        "{method: vmc, estimators: [{type: density, rmax: 2, n_bins: 10, per_particle: true}]}\n"
        "  - {method: dmc}\n");
    REQUIRE(valid.qmc.size() == 2);
    CHECK(std::get<DensitySpec>(valid.qmc[0].estimators[0]).per_particle);
    CHECK(valid.qmc[1].method == "dmc");
}

TEST_CASE("resolve_seed keeps a non-zero seed unchanged", "[parser][seed]") {
    CHECK(resolve_seed(42) == 42);
    CHECK(resolve_seed(1) == 1);
    CHECK(resolve_seed(0xFFFFFFFFFFFFFFFFULL) == 0xFFFFFFFFFFFFFFFFULL);
}

TEST_CASE("resolve_seed turns a zero seed into a random non-zero seed",
          "[parser][seed]") {
    // A zero seed must never reach the RNG verbatim — it is replaced by a fresh
    // value, and that value is itself non-zero so it round-trips through the
    // snapshot.
    CHECK(resolve_seed(0) != 0);

    // Successive resolutions of 0 draw independent seeds (collisions across two
    // 64-bit draws are astronomically unlikely).
    CHECK(resolve_seed(0) != resolve_seed(0));
}

TEST_CASE("parse_config reads the project name and seed", "[parser]") {
    auto c = parse_config(
        "project:\n"
        "  name: exciton\n"
        "  seed: 42\n");

    CHECK(c.name == "exciton");
    CHECK(c.seed == 42);
}

TEST_CASE("parse_config reads a slater determinant wavefunction block",
          "[parser][slater]") {
    auto c = parse_config(
        "wavefunction:\n"
        "  - slater:\n"
        "      group: electron_group\n"
        "      orbitals:\n"
        "        type: gaussian\n"
        "        centers: [[-1.0, 0.0], [1.0, 0.0]]\n"
        "        alphas: [0.7, 0.5]\n");

    REQUIRE(c.wavefunction.size() == 1);
    const auto& s = std::get<SlaterSpec>(c.wavefunction[0]);
    CHECK(s.group == "electron_group");
    CHECK(s.orbital_type == "gaussian");
    REQUIRE(s.centers.size() == 2);
    CHECK(s.centers[0] == std::vector<double>{-1.0, 0.0});
    CHECK(s.centers[1] == std::vector<double>{1.0, 0.0});
    CHECK(s.alphas == std::vector<double>{0.7, 0.5});
}

TEST_CASE("parse_config reads the slater width's optimisation flag",
          "[parser][slater]") {
    // α sits inside the determinant, so it is the one wavefunction parameter
    // that moves the nodes.  `alpha_fixed` is how a config hands it to the
    // optimiser; it defaults to fixed so existing configs keep their behaviour.
    auto c = parse_config(
        "wavefunction:\n"
        "  - slater:\n"
        "      group: electron_group\n"
        "      orbitals:\n"
        "        type: gaussian\n"
        "        sites: triangular\n"
        "        alpha: 0.0015\n"
        "        alpha_fixed: false\n");

    REQUIRE(c.wavefunction.size() == 1);
    const auto& s = std::get<SlaterSpec>(c.wavefunction[0]);
    CHECK(s.sites == "triangular");
    REQUIRE(s.alpha.has_value());
    CHECK(*s.alpha == 0.0015);
    CHECK(s.alpha_fixed == false);

    auto d = parse_config(
        "wavefunction:\n"
        "  - slater:\n"
        "      group: electron_group\n"
        "      orbitals: { type: gaussian, sites: triangular, alpha: 0.0015 }\n");
    REQUIRE(d.wavefunction.size() == 1);
    CHECK(std::get<SlaterSpec>(d.wavefunction[0]).alpha_fixed == true);
}

TEST_CASE("parse_config keeps wavefunction items in document order",
          "[parser][slater]") {
    auto c = parse_config(
        "wavefunction:\n"
        "  - jastrow:\n"
        "      type: padé\n"
        "      pair: [electron_group, hole_group]\n"
        "      denominator: [0.0]\n"
        "  - slater:\n"
        "      group: electron_group\n"
        "      orbitals: { type: gaussian, centers: [[0.0, 0.0]], alphas: [0.5] }\n");

    REQUIRE(c.wavefunction.size() == 2);
    CHECK(std::holds_alternative<JastrowSpec>(c.wavefunction[0]));
    REQUIRE(std::holds_alternative<SlaterSpec>(c.wavefunction[1]));
    CHECK(std::get<SlaterSpec>(c.wavefunction[1]).group == "electron_group");
}

TEST_CASE("parse_config resolves a zero seed to a random non-zero seed",
          "[parser][seed]") {
    // Explicit seed: 0 means "auto" — the parsed config must carry a concrete,
    // non-zero seed so every downstream driver and the snapshot agree on it.
    auto explicit_zero = parse_config(
        "project:\n"
        "  name: exciton\n"
        "  seed: 0\n");
    CHECK(explicit_zero.seed != 0);

    // An absent seed defaults to 0 and is resolved the same way.
    auto absent = parse_config(
        "project:\n"
        "  name: exciton\n");
    CHECK(absent.seed != 0);
}

TEST_CASE("parse_config reads project.log = true", "[parser]") {
    auto c = parse_config(
        "project:\n"
        "  name: proj\n"
        "  log: true\n");
    CHECK(c.log == true);
}

TEST_CASE("parse_config defaults project.log to false when absent", "[parser]") {
    auto c = parse_config(
        "project:\n"
        "  name: proj\n");
    CHECK(c.log == false);
}

TEST_CASE("parse_config reads the lattice block", "[parser]") {
    auto c = parse_config(
        "system:\n"
        "  lattice:\n"
        "    type: open\n"
        "    dimension: 2\n"
        "    L: 1.5\n");

    CHECK(c.lattice.type == "open");
    CHECK(c.lattice.dimension == 2);
    CHECK(c.lattice.L == 1.5);
}

TEST_CASE("parse_config reads the particle list", "[parser]") {
    auto c = parse_config(
        "system:\n"
        "  particles:\n"
        "    - name: e\n"
        "      group: electron_group\n"
        "      type: fermions\n"
        "      mass: 1.0\n"
        "      charge: -1\n"
        "      up: 1\n"
        "      down: 0\n"
        "      initialization: uniform\n"
        "    - name: h\n"
        "      group: hole_group\n"
        "      type: fermions\n"
        "      mass: 2.0\n"
        "      charge: +1\n"
        "      up: 0\n"
        "      down: 1\n"
        "      initialization: uniform\n");

    REQUIRE(c.particles.size() == 2);
    CHECK(c.particles[0].name == "e");
    CHECK(c.particles[0].group == "electron_group");
    CHECK(c.particles[0].type == "fermions");
    CHECK(c.particles[0].mass == std::vector<double>{1.0});
    CHECK(c.particles[0].charge == -1.0);
    CHECK(c.particles[0].up == 1);
    CHECK(c.particles[0].down == 0);
    CHECK(c.particles[0].initialization == "uniform");
    CHECK(c.particles[1].name == "h");
    CHECK(c.particles[1].group == "hole_group");
    CHECK(c.particles[1].mass == std::vector<double>{2.0});
    CHECK(c.particles[1].charge == 1.0);
    CHECK(c.particles[1].down == 1);
}

TEST_CASE("parse_config reads per-flavour counts and treats up/down as two flavours",
          "[parser][flavours]") {
    auto c = parse_config(
        "system:\n"
        "  particles:\n"
        "    - {name: e, group: electrons, type: fermions, mass: 1.0, charge: -1, flavours: [2, 0, 1, 3]}\n"
        "    - {name: h, group: holes, type: fermions, mass: 1.0, charge: 1, up: 2, down: 1}\n");
    REQUIRE(c.particles.size() == 2);
    CHECK(c.particles[0].flavours == std::vector<int>{2, 0, 1, 3});
    CHECK(c.particles[0].flavour_counts() == std::vector<int>{2, 0, 1, 3});
    CHECK(c.particles[0].count() == 6);
    CHECK(c.particles[1].flavours.empty());
    CHECK(c.particles[1].flavour_counts() == std::vector<int>{2, 1});
    CHECK(c.particles[1].count() == 3);
}

TEST_CASE("parse_config refuses malformed flavour counts and unknown particle keys",
          "[parser][flavours]") {
    const auto particle = [](const std::string& fields) {
        return "system:\n  particles:\n    - {name: e, group: g, mass: 1.0, " + fields + "}\n";
    };
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(parse_config(particle("up: 1, flavours: [1, 1]")),
                      ContainsSubstring("one or the other"));
    CHECK_THROWS_WITH(parse_config(particle("down: 1, flavours: [1, 1]")),
                      ContainsSubstring("one or the other"));
    CHECK_THROWS_WITH(parse_config(particle("flavours: []")), ContainsSubstring("non-empty"));
    CHECK_THROWS_WITH(parse_config(particle("flavours: 2")), ContainsSubstring("non-empty"));
    CHECK_THROWS_WITH(parse_config(particle("flavours: [1, -1]")),
                      ContainsSubstring("non-negative"));
    CHECK_THROWS_WITH(parse_config(particle("flavours: [1.5]")), ContainsSubstring("integers"));
    CHECK_THROWS_WITH(parse_config(particle("flavours: [0, 0]")),
                      ContainsSubstring("no particles"));
    // The likeliest typo must not leave the entry silently empty.
    CHECK_THROWS_WITH(parse_config(particle("flavors: [1, 1]")), ContainsSubstring("flavors"));
    CHECK_NOTHROW(parse_config(particle("flavours: [0, 1]")));
}

TEST_CASE("parse_config reads a Pade-Jastrow factor", "[parser]") {
    // a₁ is the cusp, derived from the Hamiltonian, so the YAML carries only the
    // higher numerator coefficients a₂..a_N and the denominator b₁..b_M.
    auto c = parse_config(
        "wavefunction:\n"
        "  - jastrow:\n"
        "      type: pade\n"
        "      pair: [electron_group, hole_group]\n"
        "      numerator: [0.5]\n"
        "      denominator: [0.3, 0.0]\n"
        "      is_fixed: [true, false, true]\n"
        "      can_flip_sign: [false, true, true]\n");

    REQUIRE(c.wavefunction.size() == 1);
    const auto& j = std::get<JastrowSpec>(c.wavefunction[0]);
    CHECK(j.type == "pade");
    CHECK(j.pair[0] == "electron_group");
    CHECK(j.pair[1] == "hole_group");
    CHECK(j.numerator == std::vector<double>{0.5});
    CHECK(j.denominator == std::vector<double>{0.3, 0.0});
    CHECK(j.is_fixed == std::vector<bool>{true, false, true});
    CHECK(j.can_flip_sign == std::vector<bool>{false, true, true});
}

TEST_CASE("parse_config refuses the cusp and the legacy Pade layout",
          "[parser][cusp]") {
    // The cusp is derived from the Hamiltonian and is never a YAML input.  The
    // old Padé layout (`num_orders` + `params`) put the cusp in params[0]; read
    // under the new meaning it would silently shift every coefficient, so it
    // must fail and say what to write instead.
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: pade, pair: [e, h],\n"
                                   "               num_orders: 1, params: [-1.0, 0.3] }\n"),
                      ContainsSubstring("denominator"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: pade, pair: [e, h], params: [0.3] }\n"),
                      ContainsSubstring("denominator"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: drummond, pair: [e, e], params: [0.0],\n"
                                   "               cutoff: 5.0, cusp: 0.5 }\n"),
                      ContainsSubstring("cusp"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: pade, pair: [e, h], denominator: [0.3],\n"
                                   "               cusp: -1.0 }\n"),
                      ContainsSubstring("cusp"));
    // A Drummond block keeps its `params` ([α0, α2..αN]; α1 was never in it).
    CHECK_NOTHROW(parse_config("wavefunction:\n"
                               "  - jastrow: { type: drummond, pair: [e, e], params: [0.0],\n"
                               "               cutoff: 5.0 }\n"));
    // An unknown type no longer falls through to a Padé.
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: drumond, pair: [e, e], params: [0.0] }\n"),
                      ContainsSubstring("drumond"));
}

TEST_CASE("parse_config reads a drummond jastrow over a list of group pairs",
          "[parser][pairs]") {
    using Catch::Matchers::ContainsSubstring;
    const auto c = parse_config("wavefunction:\n"
                                "  - jastrow: { type: drummond, pairs: [[e_up, e_up], [h_dn, h_dn]],\n"
                                "               params: [0.0], cutoff: 5.0 }\n");
    const auto& j = std::get<JastrowSpec>(c.wavefunction.at(0));
    REQUIRE(j.pairs.size() == 2);
    CHECK(j.pairs[1][0] == "h_dn");
    CHECK(alea::io::jastrow_pairs(j).size() == 2);
    CHECK(alea::io::jastrow_pairs_label(j) == "[e_up, e_up], [h_dn, h_dn]");
    // A single `pair:` still reads as a one-element list.
    const auto s = parse_config("wavefunction:\n"
                                "  - jastrow: { type: drummond, pair: [e, h], params: [0.0], cutoff: 5.0 }\n");
    CHECK(alea::io::jastrow_pairs(std::get<JastrowSpec>(s.wavefunction.at(0))).size() == 1);

    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: drummond, pair: [e, e], pairs: [[h, h]],\n"
                                   "               params: [0.0], cutoff: 5.0 }\n"),
                      ContainsSubstring("not both"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: pade, pairs: [[e, e], [h, h]], denominator: [0.3] }\n"),
                      ContainsSubstring("drummond only"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: drummond, pairs: [[e, e, h]], params: [0.0] }\n"),
                      ContainsSubstring("[group_a, group_b]"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: drummond, pairs: [], params: [0.0] }\n"),
                      ContainsSubstring("empty"));
}

TEST_CASE("parse_config reads lattice.capacitor", "[parser][capacitor]") {
    const auto c = parse_config("system:\n  lattice: { type: periodic, dimension: 2, L: 6.0,\n"
                                "             capacitor: exclude }\n");
    CHECK(c.lattice.capacitor == "exclude");
    CHECK(parse_config("system:\n  lattice: { type: periodic, dimension: 2, L: 6.0 }\n")
              .lattice.capacitor == "include");
    CHECK_THROWS(parse_config("system:\n  lattice: { type: periodic, dimension: 2, L: 6.0,\n"
                              "             capacitor: maybe }\n"));
}

TEST_CASE("parse_config reads a pairing determinant", "[parser][pairing]") {
    using Catch::Matchers::ContainsSubstring;
    const auto c = parse_config(
        "wavefunction:\n"
        "  - pairing:\n"
        "      blocks:\n"
        "        - { rows: e_up, columns: h_dn, unpaired: { type: plane_wave, k: [[1, 0]], sin: [true] } }\n"
        "        - { rows: e_dn, columns: h_up }\n"
        "      orbital: { type: cosine_polynomial, shells: 3, poly_order: 4, cutoff: 2.5,\n"
        "                 cutoff_fixed: true, params: [1, 0, 0, 1, 0, 0, 0], is_fixed: [true] }\n");
    const auto& p = std::get<alea::io::PairingSpec>(c.wavefunction.at(0));
    REQUIRE(p.blocks.size() == 2);
    CHECK(p.blocks[0].rows == "e_up");
    CHECK(p.blocks[0].unpaired_type == "plane_wave");
    REQUIRE(p.blocks[0].unpaired_k.size() == 1);
    CHECK(p.blocks[0].unpaired_k[0] == std::vector<int>{1, 0});
    CHECK(p.blocks[0].unpaired_sin == std::vector<bool>{true});
    CHECK(p.blocks[1].unpaired_k.empty());
    CHECK(p.shells == 3);
    CHECK(p.poly_order == 4);
    CHECK(p.cutoff == 2.5);
    CHECK(p.cutoff_fixed);
    CHECK(p.params.size() == 7);

    CHECK_THROWS_WITH(parse_config("wavefunction:\n  - pairing: { blocks: [], orbital: {shells: 1} }\n"),
                      ContainsSubstring("blocks"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n  - pairing: { blocks: [{rows: e, columns: h}],\n"
                                   "      orbital: { type: gaussian } }\n"),
                      ContainsSubstring("cosine_polynomial"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n  - pairing: { blocks: [{rows: e, columns: h,\n"
                                   "      unpaired: {type: gaussian, k: [[1, 0]]}}], orbital: {shells: 1} }\n"),
                      ContainsSubstring("plane_wave"));
    CHECK_THROWS(parse_config("wavefunction:\n  - pairing: { blocks: [{rows: e, columns: h, extra: 1}],\n"
                              "      orbital: {shells: 1} }\n"));
}

TEST_CASE("parse_config reads Coulomb hamiltonian terms", "[parser]") {
    auto c = parse_config(
        "hamiltonian:\n"
        "  terms:\n"
        "    - type: coulomb\n"
        "      pair: [electron_group, hole_group]\n"
        "      epsilon: 2.5\n");

    REQUIRE(c.coulombs.size() == 1);
    CHECK(c.coulombs[0].pair[0] == "electron_group");
    CHECK(c.coulombs[0].pair[1] == "hole_group");
    CHECK(c.coulombs[0].epsilon == 2.5);
}

TEST_CASE("parse_config reads an inverse_power term and requires every key",
          "[parser][inverse_power]") {
    auto c = parse_config(
        "hamiltonian:\n"
        "  terms:\n"
        "    - { type: inverse_power, pair: [dipoles, dipoles], strength: 1.5, exponent: 3 }\n");
    REQUIRE(c.inverse_powers.size() == 1);
    CHECK(c.inverse_powers[0].pair[0] == "dipoles");
    CHECK(c.inverse_powers[0].pair[1] == "dipoles");
    CHECK(c.inverse_powers[0].strength == 1.5);
    CHECK(c.inverse_powers[0].exponent == 3.0);
    CHECK(c.coulombs.empty());

    // No defaults: a missing exponent or strength must not become some other
    // interaction, and a typo'd key must not be dropped.
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(parse_config("hamiltonian:\n  terms:\n"
                                   "    - { type: inverse_power, pair: [a, a], strength: 1 }\n"),
                      ContainsSubstring("exponent"));
    CHECK_THROWS_WITH(parse_config("hamiltonian:\n  terms:\n"
                                   "    - { type: inverse_power, pair: [a, a], exponent: 3 }\n"),
                      ContainsSubstring("strength"));
    CHECK_THROWS(parse_config("hamiltonian:\n  terms:\n"
                              "    - { type: inverse_power, pair: [a, a], strength: 1,"
                              " exponent: 3, epsilon: 2 }\n"));
}

TEST_CASE("parse_config reads a zero_energy jastrow and refuses keys it does not use",
          "[parser][zero_energy]") {
    auto c = parse_config(
        "wavefunction:\n"
        "  - jastrow: { type: zero_energy, pair: [d, d], matching_radius: 0.05,\n"
        "               matching_radius_fixed: false, cutoff: 0.1 }\n");
    REQUIRE(c.wavefunction.size() == 1);
    const auto& j = std::get<alea::io::JastrowSpec>(c.wavefunction[0]);
    CHECK(j.type == "zero_energy");
    CHECK(j.pair[0] == "d");
    CHECK(j.matching_radius == 0.05);
    CHECK_FALSE(j.matching_radius_fixed);
    CHECK(j.cutoff == 0.1);

    // R_m fixed unless asked; the cutoff is optional here (the builder decides).
    auto d = parse_config("wavefunction:\n"
                          "  - jastrow: { type: zero_energy, pair: [d, d], matching_radius: 0.05 }\n");
    CHECK(std::get<alea::io::JastrowSpec>(d.wavefunction[0]).matching_radius_fixed);

    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: zero_energy, pair: [d, d] }\n"),
                      ContainsSubstring("matching_radius"));
    // Drummond keys mean nothing here (b and p come from the Hamiltonian).
    CHECK_THROWS(parse_config("wavefunction:\n"
                              "  - jastrow: { type: zero_energy, pair: [d, d],"
                              " matching_radius: 0.05, params: [1.0] }\n"));
    CHECK_THROWS(parse_config("wavefunction:\n"
                              "  - jastrow: { type: zero_energy, pair: [d, d],"
                              " matching_radius: 0.05, exponent: 3 }\n"));
}

TEST_CASE("parse_config reads a nosanow factor from sites or centers, not both",
          "[parser][nosanow]") {
    auto c = parse_config("wavefunction:\n"
                          "  - nosanow: { group: d, sites: triangular, alpha: 40.0,"
                          " alpha_fixed: false }\n"
                          "  - nosanow: { group: e, centers: [[0, 0], [1, 0.5]], alpha: 2 }\n");
    REQUIRE(c.wavefunction.size() == 2);
    const auto& a = std::get<alea::io::NosanowSpec>(c.wavefunction[0]);
    CHECK(a.group == "d");
    CHECK(a.sites == "triangular");
    CHECK(a.centers.empty());
    CHECK(a.alpha == 40.0);
    CHECK_FALSE(a.alpha_fixed);
    const auto& b = std::get<alea::io::NosanowSpec>(c.wavefunction[1]);
    REQUIRE(b.centers.size() == 2);
    CHECK(b.centers[1] == std::vector<double>{1.0, 0.5});
    CHECK(b.alpha_fixed);  // fixed unless asked

    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(parse_config("wavefunction:\n  - nosanow: { group: d, sites: triangular }\n"),
                      ContainsSubstring("alpha"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n  - nosanow: { group: d, alpha: 1 }\n"),
                      ContainsSubstring("exactly one"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n  - nosanow: { group: d, alpha: 1,"
                                   " sites: triangular, centers: [[0, 0]] }\n"),
                      ContainsSubstring("exactly one"));
    CHECK_THROWS(parse_config("wavefunction:\n  - nosanow: { group: d, alpha: 1,"
                              " sites: triangular, alphas: [1] }\n"));
}

TEST_CASE("parse_config reads a moire hamiltonian term", "[parser]") {
    auto c = parse_config(
        "hamiltonian:\n"
        "  terms:\n"
        "    - type: moire\n"
        "      group: [electron_group, hole_group]\n"
        "      V0: 0.1\n"
        "      Vtilde: 0.02\n"
        "      phi: -1.5708\n"
        "      a_M: 10.0\n"
        "      theta0: 0.5\n");

    REQUIRE(c.moires.size() == 1);
    REQUIRE(c.moires[0].groups.size() == 2);
    CHECK(c.moires[0].groups[0] == "electron_group");
    CHECK(c.moires[0].groups[1] == "hole_group");
    CHECK(c.moires[0].V0 == 0.1);
    CHECK(c.moires[0].Vtilde == 0.02);
    CHECK(c.moires[0].phi == -1.5708);
    CHECK(c.moires[0].a_M == 10.0);
    CHECK(c.moires[0].theta0 == 0.5);
}

TEST_CASE("parse_config defaults moire theta0 to zero when absent", "[parser]") {
    auto c = parse_config(
        "hamiltonian:\n"
        "  terms:\n"
        "    - type: moire\n"
        "      group: [electron_group]\n"
        "      V0: 0.0\n"
        "      Vtilde: 0.01\n"
        "      phi: 0.0\n"
        "      a_M: 8.0\n");

    REQUIRE(c.moires.size() == 1);
    CHECK(c.moires[0].theta0 == 0.0);
}

TEST_CASE("parse_config rejects an unknown hamiltonian term type", "[parser]") {
    // A silently dropped term would read as a converged physical result — the
    // worst failure mode for a restraint, so a typo must fail loudly.
    CHECK_THROWS(parse_config("hamiltonian:\n"
                              "  terms:\n"
                              "    - { type: sprint, pair: [a, b], k: 1.0 }\n"));
    CHECK_THROWS(parse_config("hamiltonian:\n"
                              "  terms:\n"
                              "    - { pair: [a, b] }\n"));  // missing type
}

TEST_CASE("parse_config rejects an unknown wavefunction item", "[parser]") {
    // A misspelt factor silently dropped builds a different trial than the
    // config describes, and the run still "works" — so it must fail by name.
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrw: { type: pade, pair: [e, h], params: [1.0] }\n"),
                      ContainsSubstring("jastrw"));
    CHECK_THROWS(parse_config("wavefunction:\n"
                              "  - { group: e }\n"));   // no kind at all
}

TEST_CASE("parse_config refuses the removed fixed-R features by name",
          "[parser]") {
    // The fixed-R / spring-restraint route was deleted (tag archive/fixed-r).
    // Each of its keywords must fail loudly: `type: umbrella` would otherwise
    // fall through to a Padé jastrow, and `coordinates`/`pairing` would be
    // dropped, building a different trial wavefunction without a word.
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(parse_config("system:\n"
                                   "  coordinates: { type: relative_fixed_r, separation: 4.0 }\n"),
                      ContainsSubstring("archive/fixed-r"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: padé, pair: [e, h], pairing: true,\n"
                                   "               denominator: [0.0] }\n"),
                      ContainsSubstring("archive/fixed-r"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: umbrella, pair: [h, h], params: [1.0, 2.0] }\n"),
                      ContainsSubstring("archive/fixed-r"));
    CHECK_THROWS_WITH(parse_config("hamiltonian:\n"
                                   "  terms:\n"
                                   "    - { type: spring, pair: [h, h], k: 1.0, R: 2.0 }\n"),
                      ContainsSubstring("archive/fixed-r"));
    // Keys only the fixed-R builder read: ignoring them now would be silent.
    CHECK_THROWS_WITH(parse_config("wavefunction:\n"
                                   "  - jastrow: { type: padé, pair: [e, h], role: intra,\n"
                                   "               denominator: [0.0] }\n"),
                      ContainsSubstring("archive/fixed-r"));
    CHECK_THROWS_WITH(parse_config("system:\n"
                                   "  particles:\n"
                                   "    - { name: e, group: e, up: 1, r0: 40.0 }\n"),
                      ContainsSubstring("archive/fixed-r"));
    CHECK_THROWS_WITH(parse_config("system:\n"
                                   "  particles:\n"
                                   "    - { name: e, group: e, up: 1, epsilon: 4.0 }\n"),
                      ContainsSubstring("archive/fixed-r"));
}

TEST_CASE("parse_config reads the qmc block", "[parser]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: vmc\n"
        "    random_walk: langevin\n"
        "    device: cpu\n"
        "    threads: 0\n"
        "    samples: 512\n"
        "    equilibration_blocks: 512\n"
        "    accumulation_blocks: 2048\n"
        "    steps_per_block: 100\n"
        "    sub_steps: 10\n"
        "    time_step: 0.01\n");

    REQUIRE(c.qmc.size() == 1);
    const auto& q = c.qmc[0];
    CHECK(q.method == "vmc");
    CHECK(q.random_walk == "langevin");
    CHECK(q.device == "cpu");
    CHECK(q.threads == 0);
    CHECK(q.samples == 512);
    CHECK(q.equilibration_blocks == 512);
    CHECK(q.accumulation_blocks == 2048);
    CHECK(q.steps_per_block == 100);
    CHECK(q.sub_steps == 10);
    CHECK(q.time_step == 0.01);
}

TEST_CASE("parse_config reads population_generations and refuses feedback",
          "[parser]") {
    // Default g = 10 when absent.
    auto d = parse_config("qmc:\n  - method: dmc\n    samples: 8\n");
    REQUIRE(d.qmc.size() == 1);
    CHECK(d.qmc[0].population_generations == 10.0);

    auto c = parse_config(
        "qmc:\n  - method: dmc\n    samples: 8\n    population_generations: 25\n");
    CHECK(c.qmc[0].population_generations == 25.0);

    // Loop gain 1/g must not overshoot.
    CHECK_THROWS(parse_config(
        "qmc:\n  - method: dmc\n    population_generations: 0.5\n"));

    // The removed dimensional key is refused, never silently reinterpreted.
    CHECK_THROWS(parse_config("qmc:\n  - method: dmc\n    feedback: 1.0\n"));
}

TEST_CASE("parse_config reads the DMC branch_cutoff as a name or {scheme, alpha}",
          "[parser]") {
    auto stage = [](const std::string& value) {
        return "qmc:\n  - method: dmc\n    samples: 8\n    branch_cutoff: " + value + "\n";
    };
    // Default: universal with the driver's alpha.
    auto d = parse_config("qmc:\n  - method: dmc\n    samples: 8\n");
    CHECK(d.qmc[0].branch_cutoff == "universal");
    CHECK_FALSE(d.qmc[0].branch_cutoff_alpha.has_value());

    CHECK(parse_config(stage("drv")).qmc[0].branch_cutoff == "drv");
    CHECK(parse_config(stage("none")).qmc[0].branch_cutoff == "none");
    auto u = parse_config(stage("{ scheme: universal, alpha: 1.5 }"));
    CHECK(u.qmc[0].branch_cutoff == "universal");
    REQUIRE(u.qmc[0].branch_cutoff_alpha.has_value());
    CHECK(*u.qmc[0].branch_cutoff_alpha == 1.5);

    // Unknown scheme, alpha on a scheme that does not read it, a bad alpha,
    // a map without scheme, an unknown key, and the key on a VMC stage.
    CHECK_THROWS(parse_config(stage("zsgma")));
    CHECK_THROWS(parse_config(stage("{ scheme: drv, alpha: 2.0 }")));
    CHECK_THROWS(parse_config(stage("{ scheme: universal, alpha: 0.0 }")));
    CHECK_THROWS(parse_config(stage("{ scheme: universal, alpha: .nan }")));
    CHECK_THROWS(parse_config(stage("{ alpha: 2.0 }")));
    CHECK_THROWS(parse_config(stage("{ scheme: universal, sigma: 2.0 }")));
    CHECK_THROWS(parse_config(stage("[universal]")));
    CHECK_THROWS(parse_config(
        "qmc:\n  - method: vmc\n    samples: 8\n    branch_cutoff: none\n"));
}

TEST_CASE("parse_config reads the DMC reweighting as a name or {type, correlation_time, kappa}",
          "[parser]") {
    auto stage = [](const std::string& value) {
        return "qmc:\n  - method: dmc\n    samples: 8\n    reweighting: " + value + "\n";
    };
    // Default: the bare weight.
    auto d = parse_config("qmc:\n  - method: dmc\n    samples: 8\n");
    CHECK(d.qmc[0].reweighting == "bare");
    CHECK_FALSE(d.qmc[0].correlation_time.has_value());
    CHECK_FALSE(d.qmc[0].suppression_kappa.has_value());

    CHECK(parse_config(stage("suppressed")).qmc[0].reweighting == "suppressed");
    CHECK(parse_config(stage("effective")).qmc[0].reweighting == "effective");
    CHECK(parse_config(stage("{ type: bare }")).qmc[0].reweighting == "bare");
    // `auto` (or no key) leaves the correlation time to the driver.
    auto a = parse_config(stage("{ type: suppressed, correlation_time: auto }"));
    CHECK(a.qmc[0].reweighting == "suppressed");
    CHECK_FALSE(a.qmc[0].correlation_time.has_value());
    auto s = parse_config(stage("{ type: suppressed, correlation_time: 0.25, kappa: 1.5 }"));
    REQUIRE(s.qmc[0].correlation_time.has_value());
    CHECK(*s.qmc[0].correlation_time == 0.25);
    REQUIRE(s.qmc[0].suppression_kappa.has_value());
    CHECK(*s.qmc[0].suppression_kappa == 1.5);
    auto sampling = parse_config(stage("{type: suppressed, correlation_sampling: "
        "{equilibration_sweeps: 100, min_sweeps: 1024, max_sweeps: 16384}}"));
    CHECK(sampling.qmc[0].correlation_equilibration_sweeps == 100);
    CHECK(sampling.qmc[0].correlation_min_sweeps == 1024);
    CHECK(sampling.qmc[0].correlation_max_sweeps == 16384);
    for (const auto& invalid : {
        "{type: effective, kappa: 1}",
        "{type: effective, correlation_sampling: {min_sweeps: 512}}",
        "{type: suppressed, correlation_sampling: {min_sweeps: 1}}",
        "{type: suppressed, correlation_sampling: {max_sweeps: 10}}",
        "{type: suppressed, correlation_sampling: {equilibration_sweeps: -1}}",
        "{type: suppressed, correlation_sampling: {unknown: 1}}",
        "{type: suppressed, correlation_time: 1, correlation_sampling: {min_sweeps: 512}}"})
        CHECK_THROWS(parse_config(stage(invalid)));

    // Unknown type, the suppressed keys on the bare weight, bad values, a map
    // without type, an unknown key, a list, and the key on a VMC stage.
    CHECK_THROWS(parse_config(stage("umrigar")));
    CHECK_THROWS(parse_config(stage("{ type: bare, kappa: 1.0 }")));
    CHECK_THROWS(parse_config(stage("{ type: bare, correlation_time: 0.1 }")));
    CHECK_THROWS(parse_config(stage("{ type: suppressed, kappa: 0.0 }")));
    CHECK_THROWS(parse_config(stage("{ type: suppressed, kappa: .inf }")));
    CHECK_THROWS(parse_config(stage("{ type: suppressed, correlation_time: -0.1 }")));
    CHECK_THROWS(parse_config(stage("{ type: suppressed, correlation_time: never }")));
    CHECK_THROWS(parse_config(stage("{ kappa: 1.0 }")));
    CHECK_THROWS(parse_config(stage("{ type: suppressed, c: 3.5 }")));
    CHECK_THROWS(parse_config(stage("[suppressed]")));
    CHECK_THROWS(parse_config(
        "qmc:\n  - method: vmc\n    samples: 8\n    reweighting: suppressed\n"));
}

TEST_CASE("parse_config reads opt-in estimators under a driver", "[parser]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: vmc\n"
        "    samples: 8\n"
        "    estimators:\n"
        "      - { type: gofr, rmax: 5.0, n_bins: 200 }\n");

    REQUIRE(c.qmc.size() == 1);
    REQUIRE(c.qmc[0].estimators.size() == 1);
    const auto* g = std::get_if<GofrSpec>(&c.qmc[0].estimators[0]);
    REQUIRE(g != nullptr);
    CHECK(g->name.empty());
    CHECK(g->rmax == 5.0);
    CHECK(g->n_bins == 200);
}

TEST_CASE("parse_config reads a collective_density estimator with terms",
          "[parser]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: dmc\n"
        "    samples: 8\n"
        "    estimators:\n"
        "      - { type: collective_density, name: s, rmax: 8.3, n_bins: 64,\n"
        "          terms: [ { group: rho1 }, { group: rho2, coeff: -1.0 } ] }\n");

    REQUIRE(c.qmc.size() == 1);
    REQUIRE(c.qmc[0].estimators.size() == 1);
    REQUIRE(std::holds_alternative<CollectiveDensitySpec>(c.qmc[0].estimators[0]));
    const auto& e = std::get<CollectiveDensitySpec>(c.qmc[0].estimators[0]);
    CHECK(e.name == "s");
    CHECK(e.rmax == 8.3);
    CHECK(e.n_bins == 64);
    REQUIRE(e.terms.size() == 2);
    CHECK(e.terms[0].first == "rho1");
    CHECK(e.terms[0].second == 1.0);   // coeff defaults to 1.0
    CHECK(e.terms[1].first == "rho2");
    CHECK(e.terms[1].second == -1.0);
}

TEST_CASE("each estimator kind admits only its own keys, in every stage", "[parser][estimators]") {
    // A key another kind owns is an error, never silently ignored — and the
    // same entry is read the same way under vmc, dmc and pimc.
    const std::vector<std::pair<std::string, std::string>> misplaced{
        {"{type: gofr, rmax: 2, n_bins: 4, kmax: 3}", "kmax"},
        {"{type: sofk, kmax: 3, rmax: 2}", "rmax"},
        {"{type: density, rmax: 2, n_bins: 4, terms: [{group: g}]}", "terms"},
        {"{type: electron_hole_proximity, electron_group: e, hole_group: h, r_cut: 0.5, n_bins: 4}", "n_bins"},
        {"{type: momentum_distribution, group: g, rmax: 1, n_bins: 4, kmax: 2, samples: 1, r_cut: 1}", "r_cut"},
        {"{type: collective_density, name: v, rmax: 2, n_bins: 4, terms: [{group: g, coefff: 2}]}", "coefff"}};
    for (const std::string stage : {"vmc", "dmc", "pimc, beta: 1, time_step: 0.125"})
        for (const auto& [entry, key] : misplaced) {
            INFO(stage << ": " << entry);
            CHECK_THROWS_WITH(
                parse_config("qmc:\n  - {method: " + stage + ", estimators: [" + entry + "]}\n"),
                Catch::Matchers::ContainsSubstring(key));
        }
}

TEST_CASE("an unknown estimator type is refused when the config is read", "[parser][estimators]") {
    CHECK_THROWS_WITH(parse_config("qmc:\n  - {method: vmc, estimators: [{type: bogus}]}\n"),
                      Catch::Matchers::ContainsSubstring("unknown type 'bogus'"));
    CHECK_THROWS_WITH(parse_config("qmc:\n  - {method: vmc, estimators: [{rmax: 2}]}\n"),
                      Catch::Matchers::ContainsSubstring("type"));
}

TEST_CASE("every estimator kind reads an optional name that labels its keys", "[parser][estimators]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: vmc\n"
        "    estimators:\n"
        "      - {type: gofr, name: fine, rmax: 0.5, n_bins: 25}\n"
        "      - {type: gofr, rmax: 5.0, n_bins: 50}\n"
        "      - {type: electron_hole_proximity, name: near, electron_group: e, hole_group: h, r_cut: 0.5}\n");
    REQUIRE(c.qmc[0].estimators.size() == 3);
    CHECK(std::get<GofrSpec>(c.qmc[0].estimators[0]).name == "fine");
    CHECK(std::get<GofrSpec>(c.qmc[0].estimators[1]).name.empty());
    const auto& near = std::get<ElectronHoleProximitySpec>(c.qmc[0].estimators[2]);
    CHECK(near.name == "near");
    CHECK(near.electron_group == "e");
    CHECK(near.hole_group == "h");
    CHECK(near.r_cut == 0.5);
    // A name becomes part of an HDF5 key, where '/' would open a group.
    CHECK_THROWS_WITH(parse_config("qmc:\n  - {method: vmc, estimators: [{type: gofr, name: a/b}]}\n"),
                      Catch::Matchers::ContainsSubstring("name"));
}

TEST_CASE("parse_config defaults to no estimators when absent", "[parser]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: vmc\n"
        "    samples: 8\n");

    REQUIRE(c.qmc.size() == 1);
    CHECK(c.qmc[0].estimators.empty());
}

TEST_CASE("parse_config_file parses the real exciton.yaml end to end",
          "[parser]") {
    auto c = parse_config_file(std::string(ALEA_CONFIG_DIR) + "/tests/exciton.yaml");

    CHECK(c.name == "exciton");
    CHECK(c.seed == 42);
    CHECK(c.lattice.dimension == 2);
    REQUIRE(c.particles.size() == 2);
    CHECK(c.particles[0].group == "electron_group");
    CHECK(c.particles[1].group == "hole_group");
    REQUIRE(c.wavefunction.size() == 1);
    const auto& j = std::get<JastrowSpec>(c.wavefunction[0]);
    CHECK(j.type == "padé");  // UTF-8 round-trips
    CHECK(j.pair[0] == "electron_group");
    REQUIRE(c.coulombs.size() == 1);
    CHECK(c.coulombs[0].epsilon == 1.0);
    REQUIRE(c.qmc.size() == 1);
    CHECK(c.qmc[0].method == "dmc");
    CHECK(c.qmc[0].samples == 2048);
    CHECK(c.qmc[0].random_walk == "langevin");
    CHECK(c.log == true);
}

TEST_CASE("parse_config reads an optimizer block with linear method", "[parser]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: optimizer\n"
        "    type: linear\n"
        "    random_walk: metropolis\n"
        "    device: cpu\n"
        "    threads: 4\n"
        "    samples: 128\n"
        "    equilibration_blocks: 30\n"
        "    accumulation_blocks: 60\n"
        "    steps_per_block: 10\n"
        "    sub_steps: 1\n"
        "    time_step: 0.1\n"
        "    max_iterations: 15\n"
        "    psi_step_tol: 1e-3\n"
        "    energy_tol: 1e-6\n"
        "    convergence_window: 12\n"
        "    num_shifts: 5\n"
        "    max_param_change: 0.5\n"
        "    cost_increase_tol: 0.01\n"
        "    target_shift_i: 0.5\n"
        "    max_failed_attempts: 10\n"
        "    shift_i: 0.1\n"
        "    shift_s: 0.0\n"
        "    rescale: 0.5\n"
        "    xi: 0.25\n");

    REQUIRE(c.qmc.empty());
    REQUIRE(c.optimizers.size() == 1);
    const auto& o = c.optimizers[0];
    CHECK(o.type == "linear");
    CHECK(o.random_walk == "metropolis");
    CHECK(o.device == "cpu");
    CHECK(o.threads == 4);
    CHECK(o.samples == 128);
    CHECK(o.equilibration_blocks == 30);
    CHECK(o.accumulation_blocks == 60);
    CHECK(o.steps_per_block == 10);
    CHECK(o.sub_steps == 1);
    CHECK(o.time_step == 0.1);
    CHECK(o.max_iterations == 15);
    CHECK(o.psi_step_tol == 1e-3);
    CHECK(o.energy_tol == 1e-6);
    CHECK(o.convergence_window == 12);
    CHECK(o.num_shifts == 5);
    CHECK(o.max_param_change == 0.5);
    CHECK(o.cost_increase_tol == 0.01);
    CHECK(o.target_shift_i == 0.5);
    CHECK(o.max_failed_attempts == 10);
    CHECK(o.shift_i == 0.1);
    CHECK(o.shift_s == 0.0);
    CHECK(o.rescale == 0.5);
    CHECK(o.xi == 0.25);
}

TEST_CASE("parse_config uses default values for missing optimizer fields",
          "[parser]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: optimizer\n"
        "    type: linear\n"
        "    samples: 64\n"
        "    equilibration_blocks: 10\n"
        "    accumulation_blocks: 20\n"
        "    steps_per_block: 5\n"
        "    time_step: 0.05\n");

    REQUIRE(c.optimizers.size() == 1);
    const auto& o = c.optimizers[0];
    // Required fields.
    CHECK(o.samples == 64);
    CHECK(o.time_step == 0.05);
    // Defaults.
    CHECK(o.random_walk == "metropolis");
    CHECK(o.device == "cpu");
    CHECK(o.threads == 0);
    CHECK(o.sub_steps == 1);
    CHECK(o.max_iterations == 20);
    CHECK(o.psi_step_tol == 1e-3);
    CHECK(o.convergence_window == 15);
    CHECK(o.num_shifts == 3);
    CHECK(o.max_param_change == 0.3);
    CHECK(o.cost_increase_tol == 0.0);
    CHECK(o.target_shift_i == -1.0);
    CHECK(o.max_failed_attempts == 5);
    CHECK(o.shift_i == 0.1);
    CHECK(o.shift_s == 0.0);
    CHECK(o.rescale == 1.0);   // default: the paper's full step, ξ does the damping
    CHECK(o.xi == 0.5);   // the Toulouse-Umrigar recommendation
}

TEST_CASE("parse_config reads the optimizer's parameter scope",
          "[parser][optimize]") {
    auto c = parse_config(
        "qmc:\n"
        "  - method: optimizer\n"
        "    type: linear\n"
        "    parameters: nodal\n");
    REQUIRE(c.optimizers.size() == 1);
    CHECK(c.optimizers[0].parameters == "nodal");

    auto d = parse_config(
        "qmc:\n"
        "  - method: optimizer\n"
        "    type: linear\n");
    REQUIRE(d.optimizers.size() == 1);
    CHECK(d.optimizers[0].parameters == "all");
}

TEST_CASE("parser reads a moire_jastrow wavefunction block", "[parser][moire_jastrow]") {
    auto c = parse_config(
        "wavefunction:\n"
        "  - moire_jastrow:\n"
        "      group: electron_group\n"
        "      shells: 2\n"
        "      params: [-0.1, 0.05, 0.02, 0.0]\n"
        "      is_fixed: [false, false, true, true]\n"
        "  - moire_jastrow: { group: hole_group, a_M: 12.5, theta0: 0.1 }\n");
    REQUIRE(c.wavefunction.size() == 2);   // not mistaken for pair jastrows
    const auto& m = std::get<MoireJastrowSpec>(c.wavefunction[0]);
    CHECK(m.group == "electron_group");
    CHECK(m.shells == 2);
    REQUIRE(m.params.size() == 4);
    CHECK(m.params[1] == 0.05);
    REQUIRE(m.is_fixed.size() == 4);
    CHECK_FALSE(m.is_fixed[0]);
    CHECK(m.is_fixed[2]);
    CHECK_FALSE(m.a_M.has_value());   // inherited from the moire term later
    const auto& m2 = std::get<MoireJastrowSpec>(c.wavefunction[1]);
    CHECK(m2.a_M.value() == 12.5);
    CHECK(m2.theta0.value() == 0.1);
    CHECK(m2.shells == 1);
}

TEST_CASE("parse_config reads a permutation move and refuses it without levels or max_cycle",
          "[parser][pimc][permutation]") {
    const std::string head =
        "qmc:\n  - method: pimc\n    beta: 1.0\n    time_step: 0.125\n    moves:\n";
    auto c = parse_config(head + "      - permutation: {levels: 2, max_cycle: 3, weight: 0.5}\n");
    REQUIRE(c.pimc.size() == 1);
    REQUIRE(c.pimc[0].moves.size() == 1);
    CHECK(c.pimc[0].moves[0].type == "permutation");
    CHECK(c.pimc[0].moves[0].levels == 2);
    CHECK(c.pimc[0].moves[0].max_cycle == 3);
    CHECK(c.pimc[0].moves[0].weight == 0.5);
    CHECK_THROWS_WITH(parse_config(head + "      - permutation: {levels: 2}\n"),
                      Catch::Matchers::ContainsSubstring("max_cycle"));
    CHECK_THROWS_WITH(parse_config(head + "      - permutation: {max_cycle: 3}\n"),
                      Catch::Matchers::ContainsSubstring("levels"));
    CHECK_THROWS_WITH(parse_config(head + "      - permutation: {levels: 2, max_cycle: 3, step: 1}\n"),
                      Catch::Matchers::ContainsSubstring("step"));
}

TEST_CASE("parse_config reads a positive pimc sign_threshold", "[parser][pimc]") {
    const std::string head = "qmc:\n  - {method: pimc, beta: 1.0, time_step: 0.125";
    CHECK(parse_config(head + "}\n").pimc[0].sign_threshold == 5.0);
    CHECK(parse_config(head + ", sign_threshold: 3}\n").pimc[0].sign_threshold == 3.0);
    CHECK_THROWS_WITH(parse_config(head + ", sign_threshold: 0}\n"),
                      Catch::Matchers::ContainsSubstring("sign_threshold"));
    CHECK_THROWS_WITH(parse_config(head + ", sign_threshold: .nan}\n"),
                      Catch::Matchers::ContainsSubstring("sign_threshold"));
}

TEST_CASE("parse_config reads a joint_permutation move and requires its parameters",
          "[parser][pimc][joint]") {
    const std::string head =
        "qmc:\n  - method: pimc\n    beta: 1.0\n    time_step: 0.125\n    moves:\n";
    auto c = parse_config(head +
        "      - joint_permutation: {pair: [electrons, holes], levels: 2, max_cycle: 3, pair_radius: 1.5}\n");
    REQUIRE(c.pimc[0].moves.size() == 1);
    const auto& m = c.pimc[0].moves[0];
    CHECK(m.type == "joint_permutation");
    CHECK(m.pair[0] == "electrons");
    CHECK(m.pair[1] == "holes");
    CHECK(m.levels == 2);
    CHECK(m.max_cycle == 3);
    CHECK(m.pair_radius == 1.5);
    CHECK_THROWS_WITH(parse_config(head +
        "      - joint_permutation: {pair: [electrons, holes], levels: 2, max_cycle: 3}\n"),
        Catch::Matchers::ContainsSubstring("pair_radius"));
    CHECK_THROWS_WITH(parse_config(head +
        "      - joint_permutation: {pair: [electrons], levels: 2, max_cycle: 3, pair_radius: 1}\n"),
        Catch::Matchers::ContainsSubstring("two groups"));
}

// Every block of the config refuses keys it does not read.  A misspelt key was
// previously dropped, so the run used the default while the file claimed
// otherwise (6 repository configs carried `param_tol` for months after it was
// renamed).  Each snippet below differs from a valid config by one typo, and
// the error must name that key.
TEST_CASE("Unknown keys are refused in every config block", "[parser][unknown_keys]") {
    const std::string sys =
        "system:\n"
        "  lattice: {type: open, dimension: 2, L: 1.0}\n"
        "  particles:\n"
        "    - {name: e, group: e, type: fermions, mass: 1, charge: -1, up: 1, down: 0}\n"
        "    - {name: h, group: h, type: fermions, mass: 1, charge: 1, up: 1, down: 0}\n";
    const std::string ham =
        "hamiltonian:\n  terms:\n    - {type: coulomb, pair: [e, h], epsilon: 1.0}\n";
    const std::string vmc =
        "qmc:\n  - {method: vmc, samples: 4, equilibration_blocks: 1, "
        "accumulation_blocks: 1, steps_per_block: 1, sub_steps: 1, time_step: 0.1}\n";
    const std::string valid = "project: {name: t, seed: 1}\n" + sys + ham + vmc;
    REQUIRE_NOTHROW(parse_config(valid));

    struct Case { const char* where; std::string yaml; const char* typo; };
    const Case cases[] = {
        {"top level", valid + "hamiltonain: {}\n", "hamiltonain"},
        {"project", "project: {name: t, sed: 1}\n" + sys + ham + vmc, "sed"},
        {"system", "project: {seed: 1}\nsystem:\n  particle: []\n", "particle"},
        {"lattice", "project: {seed: 1}\nsystem:\n  lattice: {type: open, dimention: 2}\n",
         "dimention"},
        {"padé jastrow", valid + "wavefunction:\n  - jastrow: {type: pade, pair: [e, h], "
         "denominator: [0.5], is_fixd: [false]}\n", "is_fixd"},
        {"padé jastrow (drummond-only key)", valid + "wavefunction:\n  - jastrow: {type: pade, "
         "pair: [e, h], cutoff: 3.0}\n", "cutoff"},
        {"drummond jastrow", valid + "wavefunction:\n  - jastrow: {type: drummond, pair: [e, h], "
         "params: [0.0], cutof: 3.0}\n", "cutof"},
        {"slater", valid + "wavefunction:\n  - slater: {group: e, orbital: {}}\n", "orbital"},
        {"slater orbitals", valid + "wavefunction:\n  - slater: {group: e, orbitals: "
         "{type: gaussian, centres: [[0, 0]], alphas: [1.0]}}\n", "centres"},
        {"moire_jastrow", valid + "wavefunction:\n  - moire_jastrow: {group: e, shell: 1}\n",
         "shell"},
        {"kspace_jastrow", valid + "wavefunction:\n  - kspace_jastrow: {pair: [e, e], shells: 1, "
         "param: [0.1]}\n", "param"},
        {"center_of_mass_momentum", valid + "wavefunction:\n  - center_of_mass_momentum: "
         "{K: [0, 0], Kvec: [1, 0]}\n", "Kvec"},
        {"hamiltonian", "project: {seed: 1}\nhamiltonian:\n  term: []\n", "term"},
        {"coulomb term", "project: {seed: 1}\nhamiltonian:\n  terms:\n    - {type: coulomb, "
         "pair: [e, h], epsilom: 2.0}\n", "epsilom"},
        {"rytova_keldysh term", "project: {seed: 1}\nhamiltonian:\n  terms:\n    - {type: "
         "rytova_keldysh, pair: [e, h], ro: 10.0}\n", "ro"},
        {"harmonic_trap term", "project: {seed: 1}\nhamiltonian:\n  terms:\n    - {type: "
         "harmonic_trap, omega: 1.0, group: [e]}\n", "group"},
        {"vmc stage", "project: {seed: 1}\nqmc:\n  - {method: vmc, sampels: 4}\n", "sampels"},
        {"vmc stage (dmc-only key)", "project: {seed: 1}\nqmc:\n  - {method: vmc, "
         "population_generations: 10}\n", "population_generations"},
        {"dmc stage", "project: {seed: 1}\nqmc:\n  - {method: dmc, time_stp: 0.01}\n",
         "time_stp"},
        {"optimizer stage", "project: {seed: 1}\nqmc:\n  - {method: optimizer, max_iteration: 5}\n",
         "max_iteration"},
    };
    for (const Case& c : cases) {
        INFO(c.where);
        CHECK_THROWS_WITH(parse_config(c.yaml), Catch::Matchers::ContainsSubstring(
            std::string("unknown key '") + c.typo + "'"));
    }
}

TEST_CASE("Each wavefunction item is a single-key map", "[parser][unknown_keys]") {
    CHECK_THROWS_WITH(
        parse_config("project: {seed: 1}\nwavefunction:\n"
                     "  - jastrow: {type: pade, pair: [e, h]}\n"
                     "    slater: {group: e}\n"),
        Catch::Matchers::ContainsSubstring("single-key"));
}

// Keys whose meaning was removed are refused by name, with what to do instead,
// rather than reported as plain typos.  None of them ever reached the driver:
// the DMC branching never read branch_interval, reconfiguration or max_age, and
// param_tol was replaced by the different criterion psi_step_tol.
TEST_CASE("Removed DMC and optimizer keys are refused with a replacement hint",
          "[parser][removed_keys]") {
    const std::string prefix = "project: {seed: 1}\nqmc:\n  - ";
    for (const char* key : {"max_age: 5", "branch_interval: 1", "reconfiguration: true"}) {
        INFO(key);
        CHECK_THROWS_WITH(parse_config(prefix + "{method: dmc, " + key + "}"),
                          Catch::Matchers::ContainsSubstring("never used"));
    }
    CHECK_THROWS_WITH(parse_config(prefix + "{method: optimizer, param_tol: 1e-3}"),
                      Catch::Matchers::ContainsSubstring("psi_step_tol"));
}

// A line through the fresh energies needs at least 3 of them.
TEST_CASE("the optimizer refuses a convergence window below 3", "[parser]") {
    CHECK_THROWS_WITH(
        parse_config("project: {seed: 1}\nqmc:\n  - {method: optimizer, convergence_window: 2}"),
        Catch::Matchers::ContainsSubstring("convergence_window"));
}

// random_walk names the move each method actually makes, so a value the method
// cannot honour is refused instead of silently replaced.
TEST_CASE("random_walk accepts only what each method does", "[parser][random_walk]") {
    const std::string prefix = "project: {seed: 1}\nqmc:\n  - ";
    auto walk = [&](const std::string& method, const std::string& value) {
        return parse_config(prefix + "{method: " + method + ", random_walk: " + value + "}");
    };
    CHECK_NOTHROW(walk("vmc", "metropolis"));
    CHECK_NOTHROW(walk("vmc", "langevin"));
    CHECK_THROWS_WITH(walk("vmc", "drift_diffusion"), Catch::Matchers::ContainsSubstring("vmc"));
    CHECK_NOTHROW(walk("dmc", "drift_diffusion"));
    CHECK_NOTHROW(walk("dmc", "langevin"));
    CHECK_THROWS_WITH(walk("dmc", "metropolis"), Catch::Matchers::ContainsSubstring("dmc"));
    CHECK_NOTHROW(walk("optimizer", "metropolis"));
    CHECK(walk("optimizer", "langevin").optimizers[0].random_walk == "langevin");
    CHECK_THROWS_WITH(walk("optimizer", "drift_diffusion"),
                      Catch::Matchers::ContainsSubstring("optimizer"));
}

TEST_CASE("Only the cpu device and the linear optimizer exist", "[parser]") {
    const std::string prefix = "project: {seed: 1}\nqmc:\n  - ";
    CHECK_NOTHROW(parse_config(prefix + "{method: vmc, device: cpu}"));
    CHECK_THROWS_WITH(parse_config(prefix + "{method: dmc, device: gpu}"),
                      Catch::Matchers::ContainsSubstring("device"));
    CHECK_THROWS_WITH(parse_config(prefix + "{method: optimizer, type: sr}"),
                      Catch::Matchers::ContainsSubstring("linear"));
}

TEST_CASE("forward_walking is a DMC map of increasing lags and an interval", "[parser][forward_walking]") {
    const std::string prefix = "project: {seed: 42}\nqmc:\n  - ";
    auto c = parse_config(prefix +
        "{method: dmc, forward_walking: {lags: [0, 0.5, 2.0], interval: 5}}\n");
    REQUIRE(c.qmc.size() == 1);
    REQUIRE(c.qmc[0].forward_walking.has_value());
    CHECK(c.qmc[0].forward_walking->lags == std::vector<double>{0.0, 0.5, 2.0});
    CHECK(c.qmc[0].forward_walking->interval == 5);

    auto defaults = parse_config(prefix + "{method: dmc, forward_walking: {lags: [1.0]}}\n");
    CHECK(defaults.qmc[0].forward_walking->interval == 1);
    CHECK_FALSE(parse_config(prefix + "{method: dmc}\n").qmc[0].forward_walking.has_value());
}

TEST_CASE("forward_walking is refused outside DMC and with malformed lags", "[parser][forward_walking]") {
    const std::string prefix = "project: {seed: 42}\nqmc:\n  - ";
    using Catch::Matchers::ContainsSubstring;
    // VMC samples |psi_T|^2: there is no mixed distribution to walk forward.
    CHECK_THROWS_WITH(parse_config(prefix + "{method: vmc, forward_walking: {lags: [1.0]}}\n"),
                      ContainsSubstring("forward_walking"));
    for (const std::string fw : {"{lags: []}", "{interval: 2}", "{lags: [1.0, 1.0]}",
                                 "{lags: [2.0, 1.0]}", "{lags: [-1.0]}", "{lags: [1.0], interval: 0}",
                                 "{lags: [1.0], steps: 3}", "[1.0, 2.0]"}) {
        INFO(fw);
        CHECK_THROWS_WITH(parse_config(prefix + "{method: dmc, forward_walking: " + fw + "}\n"),
                          ContainsSubstring("forward_walking"));
    }
}

TEST_CASE("a slater item reads its backflow channels", "[parser][backflow]") {
    const std::string head = "wavefunction:\n  - slater:\n      group: e\n"
                             "      orbitals: { type: plane_wave }\n";
    auto c = parse_config(head +
        "      backflow:\n"
        "        - { from: e, truncation: 4, cutoff: 3.5, cutoff_fixed: true, params: [0.1, 0.2],"
        " is_fixed: [false, true] }\n"
        "        - { from: h, params: [0.3, 0.0, 0.01] }\n");
    const auto& s = std::get<SlaterSpec>(c.wavefunction[0]);
    REQUIRE(s.backflow.size() == 2);
    CHECK(s.backflow[0].from == "e");
    CHECK(s.backflow[0].truncation == 4);
    CHECK(s.backflow[0].cutoff == 3.5);
    CHECK(s.backflow[0].cutoff_fixed);
    CHECK(s.backflow[0].params == std::vector<double>{0.1, 0.2});
    CHECK(s.backflow[0].is_fixed == std::vector<bool>{false, true});
    CHECK(s.backflow[1].truncation == 3);
    CHECK_FALSE(s.backflow[1].cutoff.has_value());
    CHECK_FALSE(s.backflow[1].cutoff_fixed);
    CHECK(s.backflow[1].is_fixed.empty());

    using Catch::Matchers::ContainsSubstring;
    for (const std::string bad : {"[]", "{ from: e, params: [0.1] }", "[{ params: [0.1] }]",
                                  "[{ from: e }]", "[{ from: e, params: [0.1], is_fixed: [true, false] }]",
                                  "[{ from: e, params: [0.1], order: 3 }]"}) {
        INFO(bad);
        CHECK_THROWS_WITH(parse_config(head + "      backflow: " + bad + "\n"),
                          ContainsSubstring("backflow"));
    }
}

TEST_CASE("a pairing item reads backflow channels that name their group", "[parser][backflow]") {
    const std::string head = "wavefunction:\n  - pairing:\n      blocks: [{rows: e, columns: h}]\n"
                             "      orbital: {poly_order: 2, cutoff: 3.0}\n";
    auto c = parse_config(head + "      backflow: [{group: e, from: h, params: [0.1, 0.0]}]\n");
    const auto& ps = std::get<alea::io::PairingSpec>(c.wavefunction[0]);
    REQUIRE(ps.backflow.size() == 1);
    CHECK(ps.backflow[0].group == "e");
    CHECK(ps.backflow[0].from == "h");
    using Catch::Matchers::ContainsSubstring;
    // A pairing channel must name its group; a slater channel must not.
    CHECK_THROWS_WITH(parse_config(head + "      backflow: [{from: h, params: [0.1]}]\n"),
                      ContainsSubstring("group"));
    CHECK_THROWS_WITH(parse_config("wavefunction:\n  - slater:\n      group: e\n"
                                   "      orbitals: { type: plane_wave }\n"
                                   "      backflow: [{group: e, from: e, params: [0.1]}]\n"),
                      ContainsSubstring("group"));
}
