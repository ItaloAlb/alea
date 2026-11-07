// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "io/config_writer.h"
#include "io/parser.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

TEST_CASE("Pair action snapshots retain their explicit domain and resolved grid defaults", "[config_writer][pair_product]") {
    const auto dir=fs::temp_directory_path()/"alea_pair_snapshot";
    fs::create_directories(dir);
    const auto source=dir/"source.yaml",output=dir/"output";
    fs::create_directories(output);
    { std::ofstream file(source); file << "project: {seed: 3}\naction:\n  - kinetic: primitive\n  - pair_product: {extent: 3.25}\nqmc:\n  - {method: pimc, beta: 1, time_step: 0.125}\n"; }
    alea::io::write_config_snapshot(source.string(),{},output,3);
    const auto config=alea::io::parse_config_file((output/"source.yaml").string());
    CHECK(config.action[1].form=="finite_grid"); CHECK(config.action[1].extent==3.25); CHECK(config.action[1].grid_points==17);
    const auto root=YAML::LoadFile((output/"source.yaml").string());
    CHECK(root["action"][1]["pair_product"]["grid_points"].as<int>()==17);
    fs::remove(source); fs::remove(output/"source.yaml"); fs::remove(output); fs::remove(dir);
}

namespace {

// A config with one optimisable jastrow, an optimizer in the qmc list, and a
// downstream vmc run.  This is the shape the snapshot must rewrite.
const char* kSourceYaml = R"(project:
  name: snap_test
  seed: 7

system:
  lattice: { type: open, dimension: 2, L: 1.0 }
  particles:
    - { name: e, group: electron_group, type: fermions, mass: 1.0, charge: -1, up: 1, down: 0 }
    - { name: h, group: hole_group,     type: fermions, mass: 1.0, charge: +1, up: 1, down: 0 }

wavefunction:
  - jastrow:
      type: pade
      pair: [electron_group, hole_group]
      numerator: [0.2]
      denominator: [0.3]
      is_fixed: [false, false]

hamiltonian:
  terms:
    - { type: coulomb, pair: [electron_group, hole_group], epsilon: 1.0 }

qmc:
  - method: optimizer
    type: linear
    samples: 64
    max_iterations: 5
  - method: vmc
    samples: 128
    accumulation_blocks: 64
)";

// Fixture: a unique scratch dir that wipes itself on destruction.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& suffix)
        : path(fs::temp_directory_path() / ("alea_cfgsnap_" + suffix)) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

fs::path write_source(const fs::path& dir) {
    const fs::path src = dir / "exciton.yaml";
    std::ofstream(src) << kSourceYaml;
    return src;
}

}  // namespace

TEST_CASE("snapshot rewrites params and strips the optimizer", "[config_writer]") {
    TempDir tmp("rewrite");
    const fs::path src     = write_source(tmp.path);
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);

    // The component's vector is [a₁, a₂, b₁]; a₁ is the derived cusp.
    const std::vector<std::vector<double>> final_params = {{-1.0, 0.5, 0.123456789012345}};
    alea::io::write_config_snapshot(src.string(), final_params, run_dir, 7);

    // The snapshot keeps the original file name, sitting next to the log.
    const fs::path out = run_dir / "exciton.yaml";
    REQUIRE(fs::exists(out));

    const YAML::Node root = YAML::LoadFile(out.string());

    SECTION("padé coefficients hold the final values, losslessly, without the cusp") {
        const YAML::Node j = root["wavefunction"][0]["jastrow"];
        REQUIRE(j["numerator"].size() == 1);
        CHECK(j["numerator"][0].as<double>() == 0.5);
        REQUIRE(j["denominator"].size() == 1);
        // Full double precision must survive the round-trip.
        CHECK(j["denominator"][0].as<double>() == 0.123456789012345);
        // a₁ is not a config input, so it must not be written back anywhere.
        CHECK_FALSE(j["params"]);
        CHECK_FALSE(j["cusp"]);
    }

    SECTION("the optimizer is gone but the vmc run remains") {
        const YAML::Node qmc = root["qmc"];
        REQUIRE(qmc.size() == 1);
        CHECK(qmc[0]["method"].as<std::string>() == "vmc");
        CHECK(qmc[0]["samples"].as<int>() == 128);
    }

    SECTION("untouched sections are preserved") {
        CHECK(root["project"]["name"].as<std::string>() == "snap_test");
        CHECK(root["system"]["lattice"]["dimension"].as<int>() == 2);
        CHECK(root["hamiltonian"]["terms"][0]["epsilon"].as<double>() == 1.0);
    }
}

TEST_CASE("snapshot preserves effective weights and correlation sampling controls",
          "[config_writer][dmc]") {
    TempDir tmp("reweighting");
    const fs::path src = tmp.path / "reweighting.yaml";
    const fs::path output = tmp.path / "run";
    fs::create_directories(output);
    std::ofstream(src) << "project: {seed: 9}\nqmc:\n"
        "  - {method: dmc, samples: 8, reweighting: effective}\n"
        "  - method: dmc\n    samples: 8\n    reweighting:\n"
        "      type: suppressed\n      correlation_sampling:\n"
        "        equilibration_sweeps: 512\n        min_sweeps: 1024\n        max_sweeps: 16384\n";
    alea::io::write_config_snapshot(src.string(), {}, output, 9);
    const auto cfg = alea::io::parse_config_file((output / src.filename()).string());
    REQUIRE(cfg.qmc.size() == 2);
    CHECK(cfg.qmc[0].reweighting == "effective");
    CHECK(cfg.qmc[1].correlation_equilibration_sweeps == 512);
    CHECK(cfg.qmc[1].correlation_min_sweeps == 1024);
    CHECK(cfg.qmc[1].correlation_max_sweeps == 16384);
}

TEST_CASE("snapshot records the resolved seed, overwriting an auto seed",
          "[config_writer][seed]") {
    TempDir tmp("seed");
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);

    // Source as the user wrote it: seed 0 means "pick one for me".
    const char* kAutoSeedYaml = R"(project:
  name: seed_test
  seed: 0

wavefunction:
  - jastrow:
      type: pade
      pair: [electron_group, hole_group]
      denominator: [0.3]
)";
    const fs::path src = tmp.path / "exciton.yaml";
    std::ofstream(src) << kAutoSeedYaml;

    // The run resolved 0 → this concrete seed; the snapshot must record it so a
    // replay reproduces the same stream rather than drawing a new one.
    const std::uint64_t resolved = 0x0123456789ABCDEFULL;
    alea::io::write_config_snapshot(src.string(), {{-1.0, 0.3}}, run_dir, resolved);

    const YAML::Node root = YAML::LoadFile((run_dir / "exciton.yaml").string());
    CHECK(root["project"]["seed"].as<std::uint64_t>() == resolved);
}

TEST_CASE("snapshot is a no-op when run_dir is empty", "[config_writer]") {
    TempDir tmp("noop");
    const fs::path src = write_source(tmp.path);

    // Empty run_dir → nothing written, no throw.
    REQUIRE_NOTHROW(
        alea::io::write_config_snapshot(src.string(), {{-1.0, 0.2, 0.3}}, fs::path{}, 7));

    // No stray file landed in the source directory.
    CHECK_FALSE(fs::exists(tmp.path / "run"));
}

TEST_CASE("snapshot leaves extra jastrows untouched when params are missing",
          "[config_writer]") {
    TempDir tmp("partial");
    const fs::path src     = write_source(tmp.path);
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);

    // Pass no parameter groups: the existing params must be carried through
    // unchanged rather than cleared.
    alea::io::write_config_snapshot(src.string(), {}, run_dir, 7);

    const YAML::Node root = YAML::LoadFile((run_dir / "exciton.yaml").string());
    const YAML::Node j = root["wavefunction"][0]["jastrow"];
    CHECK(j["numerator"][0].as<double>() == 0.2);
    CHECK(j["denominator"][0].as<double>() == 0.3);
}

// ── The optimised nodal width has to come back out ──────────────────────────
// α lives inside the determinant, so it is the one optimised parameter that
// changes a fixed-node DMC energy.  If the snapshot dropped it, the production
// flow (optimise, then replay the snapshot under DMC) would silently project
// the STARTING nodes and the optimisation would have bought nothing.
namespace {

const char* kSlaterYaml = R"(project:
  name: crystal_snap
  seed: 11

system:
  lattice: { type: periodic, dimension: 2, shape: hexagonal, rs: 30.0 }
  particles:
    - { name: e, group: electron_group, type: fermions, mass: 1.0, charge: -1, up: 7, down: 0 }

wavefunction:
  - jastrow:
      type: pade
      pair: [electron_group, electron_group]
      denominator: [0.3]
      is_fixed: [false]
  - slater:
      group: electron_group
      orbitals:
        type: gaussian
        sites: triangular
        alpha: 0.00146
        alpha_fixed: false

qmc:
  - method: optimizer
    type: linear
)";

}  // namespace

TEST_CASE("snapshot rewrites the slater width with its optimised value",
          "[config_writer][slater]") {
    TempDir tmp("slater");
    const fs::path src = tmp.path / "crystal.yaml";
    std::ofstream(src) << kSlaterYaml;
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);

    // One entry per wavefunction item, in document order.
    const std::vector<std::vector<double>> final_params = {
        {1.0 / 3.0, 0.25},     // the jastrow: [a₁, b₁]
        {0.0021234567890123}}; // the determinant's α
    alea::io::write_config_snapshot(src.string(), final_params, run_dir, 11);

    const YAML::Node root = YAML::LoadFile((run_dir / "crystal.yaml").string());
    CHECK(root["wavefunction"][0]["jastrow"]["denominator"][0].as<double>() == 0.25);
    const YAML::Node orb = root["wavefunction"][1]["slater"]["orbitals"];
    CHECK(orb["alpha"].as<double>() == 0.0021234567890123);
    // The generated sites and the group are untouched: only the width moved.
    CHECK(orb["sites"].as<std::string>() == "triangular");
}

TEST_CASE("snapshot rewrites an explicit uniform alphas list",
          "[config_writer][slater]") {
    TempDir tmp("slater_list");
    const fs::path src = tmp.path / "mol.yaml";
    std::ofstream(src) << R"(wavefunction:
  - slater:
      group: electron_group
      orbitals:
        type: gaussian
        centers: [[-1.0, 0.0], [1.0, 0.0]]
        alphas: [0.7, 0.7]
        alpha_fixed: false
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);

    alea::io::write_config_snapshot(src.string(), {{0.93}}, run_dir, 3);

    const YAML::Node root = YAML::LoadFile((run_dir / "mol.yaml").string());
    const YAML::Node alphas = root["wavefunction"][0]["slater"]["orbitals"]["alphas"];
    // One shared width: every orbital carries the optimised value, which is what
    // makes the snapshot round-trip back into the same uniform-width set.
    REQUIRE(alphas.size() == 2);
    CHECK(alphas[0].as<double>() == 0.93);
    CHECK(alphas[1].as<double>() == 0.93);
}

TEST_CASE("snapshot leaves a parameterless determinant alone",
          "[config_writer][slater]") {
    // Plane waves take their k from the cell: there is no width, the component
    // reports no parameters, and the block must survive untouched.
    TempDir tmp("slater_pw");
    const fs::path src = tmp.path / "fluid.yaml";
    std::ofstream(src) << R"(wavefunction:
  - slater:
      group: electron_group
      orbitals: { type: plane_wave }
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);

    alea::io::write_config_snapshot(src.string(), {{}}, run_dir, 3);

    const YAML::Node root = YAML::LoadFile((run_dir / "fluid.yaml").string());
    const YAML::Node orb = root["wavefunction"][0]["slater"]["orbitals"];
    CHECK(orb["type"].as<std::string>() == "plane_wave");
    CHECK_FALSE(orb["alpha"]);
    CHECK_FALSE(orb["alphas"]);
}

TEST_CASE("snapshot writes a kspace_jastrow's coefficients, making a seed explicit",
          "[config_writer][kspace_jastrow]") {
    // A Gaskell-seeded item has no `params`; after optimisation the snapshot
    // must carry the final ones, or a replay would re-seed and lose them.
    TempDir tmp("kspace");
    const fs::path src = tmp.path / "heg.yaml";
    std::ofstream(src) << R"(wavefunction:
  - kspace_jastrow: { pair: [e, e], shells: 2 }
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    alea::io::write_config_snapshot(src.string(), {{-0.0123456789, 0.000987}}, run_dir, 5);
    const YAML::Node kj = YAML::LoadFile((run_dir / "heg.yaml").string())["wavefunction"][0]["kspace_jastrow"];
    REQUIRE(kj["params"].size() == 2);
    CHECK(kj["params"][0].as<double>() == -0.0123456789);
    CHECK(kj["params"][1].as<double>() == 0.000987);
    CHECK(kj["shells"].as<int>() == 2);
}

TEST_CASE("snapshot writes an optimised zero_energy matching radius back",
          "[config_writer][zero_energy]") {
    // [R_m] is the component's only parameter; the cutoff and the core are
    // not parameters and stay as written.
    TempDir tmp("zero_energy");
    const fs::path src = tmp.path / "dipoles.yaml";
    std::ofstream(src) << R"(wavefunction:
  - jastrow: { type: zero_energy, pair: [d, d], matching_radius: 0.05,
               matching_radius_fixed: false, cutoff: 0.1 }
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    alea::io::write_config_snapshot(src.string(), {{0.0612345678901}}, run_dir, 5);
    const auto cfg = alea::io::parse_config_file((run_dir / "dipoles.yaml").string());
    const auto& j = std::get<alea::io::JastrowSpec>(cfg.wavefunction[0]);
    CHECK(j.matching_radius == 0.0612345678901);
    CHECK(j.cutoff == 0.1);
    CHECK_FALSE(j.matching_radius_fixed);
}

TEST_CASE("snapshot writes the optimised pairing orbital back", "[config_writer][pairing]") {
    // [p_0..p_{S-1}, c_0, c_2..c_M, L]: the linear coefficients go to
    // orbital.params and the last value to orbital.cutoff.  They set the
    // nodes, so a replay must see them.
    TempDir tmp("pairing");
    const fs::path src = tmp.path / "eh.yaml";
    std::ofstream(src) << R"(wavefunction:
  - pairing:
      blocks: [{ rows: e, columns: h }]
      orbital: { shells: 2, poly_order: 2, cutoff: 3.0, params: [1.0, 0.5, 1.0, 0.0] }
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    alea::io::write_config_snapshot(src.string(), {{1.0, 0.25, 1.0, -0.125, 2.75}}, run_dir, 5);
    const auto cfg = alea::io::parse_config_file((run_dir / "eh.yaml").string());
    const auto& ps = std::get<alea::io::PairingSpec>(cfg.wavefunction[0]);
    REQUIRE(ps.params.size() == 4);
    CHECK(ps.params[1] == 0.25);
    CHECK(ps.params[3] == -0.125);
    CHECK(ps.cutoff == 2.75);
    CHECK(ps.shells == 2);
}

TEST_CASE("snapshot writes an optimised nosanow width back", "[config_writer][nosanow]") {
    TempDir tmp("nosanow");
    const fs::path src = tmp.path / "solid.yaml";
    std::ofstream(src) << R"(wavefunction:
  - jastrow: { type: zero_energy, pair: [d, d], matching_radius: 0.05 }
  - nosanow: { group: d, sites: triangular, alpha: 40.0, alpha_fixed: false }
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    alea::io::write_config_snapshot(src.string(), {{0.05}, {52.123456789}}, run_dir, 5);
    const auto cfg = alea::io::parse_config_file((run_dir / "solid.yaml").string());
    const auto& n = std::get<alea::io::NosanowSpec>(cfg.wavefunction[1]);
    CHECK(n.alpha == 52.123456789);
    CHECK(n.sites == "triangular");
    CHECK_FALSE(n.alpha_fixed);
}

TEST_CASE("snapshot maps final_params[i] onto wavefunction item i",
          "[config_writer][order]") {
    // build_wavefunction turns wavefunction item i into component i, so the
    // optimiser's per-component values come back in DOCUMENT order, whatever
    // the kinds are.  A parameterless item (the boost) still owns its slot.
    TempDir tmp("doc_order");
    const fs::path src = tmp.path / "moire.yaml";
    std::ofstream(src) << R"(project: { name: mj, seed: 3 }
wavefunction:
  - moire_jastrow: { group: electron_group, shells: 1, params: [0.0, 0.0] }
  - center_of_mass_momentum: { K: [0.5, 0.0] }
  - slater:
      group: electron_group
      orbitals: { type: gaussian, centers: [[0.0, 0.0]], alpha: 0.5 }
  - jastrow: { type: pade, pair: [electron_group, electron_group], denominator: [0.25] }
qmc: []
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    const std::vector<std::vector<double>> final_params = {
        {-0.12, 0.034}, {}, {0.7}, {1.0, 0.3}};
    alea::io::write_config_snapshot(src.string(), final_params, run_dir, 3);

    const YAML::Node out = YAML::LoadFile((run_dir / "moire.yaml").string());
    const YAML::Node wf  = out["wavefunction"];
    REQUIRE(wf[0]["moire_jastrow"]["params"].size() == 2);
    CHECK(wf[0]["moire_jastrow"]["params"][0].as<double>() == -0.12);
    CHECK(wf[0]["moire_jastrow"]["params"][1].as<double>() == 0.034);
    CHECK(wf[1]["center_of_mass_momentum"]["K"][0].as<double>() == 0.5);
    CHECK(wf[2]["slater"]["orbitals"]["alpha"].as<double>() == 0.7);
    CHECK(wf[3]["jastrow"]["denominator"][0].as<double>() == 0.3);
}

TEST_CASE("a pimc snapshot round-trips action: and the pimc section",
          "[config_writer][pimc]") {
    TempDir tmp("pimc_rt");
    const fs::path src = tmp.path / "pimc.yaml";
    std::ofstream(src) << R"(project: { name: rt, seed: 7 }
system:
  lattice: { type: periodic, dimension: 1, L: 4.0 }
  particles:
    - { name: p, group: g, type: boltzmannon, mass: 1.0, up: 1, down: 0 }
hamiltonian:
  terms: []
qmc:
  - method: pimc
    samples: 32
    time_step: 0.0625
    beta: 2.0
    statistics: boltzmannon
    moves:
      - bisection: { levels: 3, weight: 1.0 }
      - centroid: { step: 0.3 }
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    alea::io::write_config_snapshot(src.string(), {}, run_dir, 7);

    // The snapshot records the DEFAULT action explicitly (`primitive`); the
    // original omits it, so both documents must resolve to the same action.
    const YAML::Node snap_root = YAML::LoadFile((run_dir / "pimc.yaml").string());
    REQUIRE(snap_root["action"]);
    REQUIRE(snap_root["action"].size() == 1);
    CHECK(snap_root["action"][0]["primitive"]);

    const alea::io::Config a = alea::io::parse_config_file(src.string());
    const alea::io::Config b = alea::io::parse_config_file(
        (run_dir / "pimc.yaml").string());

    REQUIRE(a.action.size() == 1);
    REQUIRE(b.action.size() == 1);
    CHECK(b.action[0].component == a.action[0].component);   // "primitive"
    CHECK(b.action[0].form == a.action[0].form);

    REQUIRE(a.pimc.size() == 1);
    REQUIRE(b.pimc.size() == 1);
    const alea::io::PimcSpec& pa = a.pimc[0];
    const alea::io::PimcSpec& pb = b.pimc[0];
    CHECK(pb.beta == pa.beta);
    CHECK(pb.time_step == pa.time_step);
    CHECK(pb.samples == pa.samples);
    CHECK(pb.energy == pa.energy);
    CHECK(pb.statistics == pa.statistics);
    REQUIRE(pb.moves.size() == pa.moves.size());
    for (std::size_t i = 0; i < pa.moves.size(); ++i) {
        CHECK(pb.moves[i].type == pa.moves[i].type);
        CHECK(pb.moves[i].levels == pa.moves[i].levels);
        CHECK(pb.moves[i].step == pa.moves[i].step);
        CHECK(pb.moves[i].weight == pa.moves[i].weight);
    }
    // The recorded seed round-trips, so a replay reproduces the same chain.
    CHECK(b.seed == a.seed);
    CHECK(b.seed == 7u);
}

TEST_CASE("a pimc snapshot round-trips per-flavour particle counts",
          "[config_writer][pimc][flavours]") {
    TempDir tmp("pimc_flavours_rt");
    const fs::path src = tmp.path / "pimc.yaml";
    std::ofstream(src) << R"(project: { name: rt, seed: 7 }
system:
  lattice: { type: open, dimension: 2, L: 4.0 }
  particles:
    - { name: e, group: electrons, type: fermions, mass: 1.0, charge: -1, flavours: [2, 0, 1, 3] }
    - { name: h, group: holes, type: fermions, mass: 1.0, charge: 1, up: 2, down: 1 }
hamiltonian:
  terms: []
qmc:
  - method: pimc
    samples: 2
    time_step: 0.125
    beta: 1.0
    statistics: boltzmannon
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    alea::io::write_config_snapshot(src.string(), {}, run_dir, 7);

    const alea::io::Config a = alea::io::parse_config_file(src.string());
    const alea::io::Config b = alea::io::parse_config_file(
        (run_dir / "pimc.yaml").string());
    REQUIRE(b.particles.size() == 2);
    CHECK(b.particles[0].flavours == std::vector<int>{2, 0, 1, 3});
    CHECK(b.particles[0].flavour_counts() == a.particles[0].flavour_counts());
    CHECK(b.particles[1].flavours.empty());  // up/down stays up/down
    CHECK(b.particles[1].flavour_counts() == std::vector<int>{2, 1});
    // The resolved sign threshold is recorded for a replay.
    const YAML::Node snap = YAML::LoadFile((run_dir / "pimc.yaml").string());
    REQUIRE(snap["qmc"][0]["sign_threshold"]);
    CHECK(snap["qmc"][0]["sign_threshold"].as<double>() == 5.0);
    CHECK(b.pimc[0].sign_threshold == 5.0);
}

TEST_CASE("a pimc snapshot records the resolved default move set",
          "[config_writer][pimc]") {
    // M = beta/time_step selects the default bisection depth; the snapshot
    // must carry it so a replay never re-derives a possibly changed default.
    for (const auto& [time_step, M] :
         std::vector<std::pair<double, std::size_t>>{{1.0, 1}, {0.5, 2}, {0.25, 4}, {0.0625, 16}}) {
        CAPTURE(M);
        TempDir tmp("pimc_default_moves");
        const fs::path src = tmp.path / "pimc.yaml";
        std::ofstream(src) << "project: { name: rt, seed: 7 }\n"
            "system:\n  lattice: { type: open, dimension: 1, L: 1.0 }\n"
            "  particles:\n    - { name: p, group: g, type: boltzmannon, mass: 1.0, up: 1, down: 0 }\n"
            "hamiltonian:\n  terms: []\n"
            "qmc:\n  - { method: pimc, beta: 1.0, time_step: " << time_step << " }\n";
        const fs::path run_dir = tmp.path / "run";
        fs::create_directories(run_dir);
        alea::io::write_config_snapshot(src.string(), {}, run_dir, 7);

        const alea::io::Config original = alea::io::parse_config_file(src.string());
        const alea::io::Config replay =
            alea::io::parse_config_file((run_dir / "pimc.yaml").string());
        REQUIRE(original.pimc.at(0).moves.empty());
        const auto expected = alea::io::default_pimc_moves(M);
        const auto& recorded = replay.pimc.at(0).moves;
        REQUIRE(recorded.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            CHECK(recorded[i].type == expected[i].type);
            CHECK(recorded[i].weight == expected[i].weight);
            if (expected[i].type == "bisection")
                CHECK(recorded[i].levels == expected[i].levels);
            else
                CHECK(recorded[i].step == expected[i].step);
        }
    }
}

TEST_CASE("a VMC-only snapshot gains no action section", "[config_writer]") {
    TempDir tmp("vmc_no_action");
    const fs::path src = tmp.path / "exciton.yaml";
    std::ofstream(src) << kSourceYaml;
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    alea::io::write_config_snapshot(src.string(), {{-1.0, 0.5, 0.1}}, run_dir, 7);
    const YAML::Node root = YAML::LoadFile((run_dir / "exciton.yaml").string());
    CHECK_FALSE(root["action"]);
}

TEST_CASE("mixed sampling snapshot preserves document order after optimizer removal", "[config_writer][integration_fix]") {
    TempDir tmp("mixed_seed");
    const fs::path src = tmp.path / "mixed.yaml";
    std::ofstream(src) << R"(project: {seed: 713, log: false, output: false}
system:
  lattice: {type: open, dimension: 1, L: 4.0}
  particles: [{name: p, group: g, mass: 1.0, up: 1, down: 0}]
hamiltonian: {terms: []}
qmc:
  - {method: optimizer, samples: 2, threads: 1}
  - {method: vmc, samples: 2, threads: 1, time_step: 0.1, random_walk: metropolis, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1}
  - {method: pimc, samples: 2, threads: 1, beta: 1.0, time_step: 0.125, equilibration_blocks: 0, accumulation_blocks: 4, steps_per_block: 2, sub_steps: 1}
  - {method: dmc, samples: 2, threads: 1, time_step: 0.01, accumulation_blocks: 2, steps_per_block: 1, sub_steps: 1}
)";
    const fs::path run_dir = tmp.path / "run";
    fs::create_directories(run_dir);
    const auto original = alea::io::parse_config_file(src.string());
    alea::io::write_config_snapshot(src.string(), {}, run_dir, 713);
    const auto replay = alea::io::parse_config_file((run_dir / "mixed.yaml").string());
    REQUIRE(original.optimizers.size() == 1);
    CHECK(replay.optimizers.empty());
    CHECK(original.seed == replay.seed);
    const auto root = YAML::LoadFile((run_dir / "mixed.yaml").string());
    REQUIRE(root["qmc"].size() == 3);
    CHECK(root["qmc"][0]["method"].as<std::string>() == "vmc");
    CHECK(root["qmc"][1]["method"].as<std::string>() == "pimc");
    CHECK(root["qmc"][2]["method"].as<std::string>() == "dmc");
    CHECK(root["project"]["seed"].as<unsigned long long>() == 713);
    CHECK(original.qmc.size() == replay.qmc.size());
    CHECK(original.pimc.size() == replay.pimc.size());
    REQUIRE(original.sampling_order.size() == 3);
    REQUIRE(replay.sampling_order.size() == 3);
    CHECK(std::get<alea::io::QmcStageIndex>(replay.sampling_order[0]).index == 0);
    CHECK(std::get<alea::io::PimcStageIndex>(replay.sampling_order[1]).index == 0);
    CHECK(std::get<alea::io::QmcStageIndex>(replay.sampling_order[2]).index == 1);
}
