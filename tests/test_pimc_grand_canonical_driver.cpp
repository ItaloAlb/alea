// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Grand-canonical PIMC through the driver: YAML -> App -> Pimc.  The App
// builds the capacity n_max and one closed-path action and estimator set per
// sector N of the window; the driver samples P(N) ∝ e^{beta mu_in N} Z_N and
// reports F(N + 1) - F(N) = mu_in - T ln[P(N + 1)/P(N)] (Boninsegni,
// Prokof'ev & Svistunov, Phys. Rev. E 74, 036701 (2006)).  The move-level
// tests (exact populations, detailed balance of S_V) are in
// test_pimc_grand_canonical.cpp; these check what only the driver does: the
// per-sector physics, the bookkeeping of the sector weights and the outputs.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

#include "io/parser.h"
#include "io/path_io.h"
#include "path_checkpoint_v1.h"
#include "pimc/pimc.h"
#include "qmc/app.h"

namespace {

using Catch::Matchers::ContainsSubstring;

alea::io::Config load(const std::string& file) {
    return alea::io::parse_config_file(std::string(ALEA_CONFIG_DIR) + "/" + file);
}

alea::io::PimcMoveSpec& worm_of(alea::io::Config& cfg) {
    auto& moves = cfg.pimc.front().moves;
    const auto it = std::find_if(moves.begin(), moves.end(),
                                 [](const auto& m) { return m.type == "worm"; });
    REQUIRE(it != moves.end());
    return *it;
}

// Build and run the config's single PIMC stage; `inspect` reads the driver.
void run_stage(const alea::io::Config& cfg, const std::function<void(const alea::Pimc&)>& inspect) {
    alea::App app(cfg);
    alea::DriverPack pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 1);
    auto* pimc = dynamic_cast<alea::Pimc*>(pack.drivers.front().get());
    REQUIRE(pimc != nullptr);
    pimc->run();
    inspect(*pimc);
}

const std::vector<double>& series(const alea::RunResult& r, const std::string& name) {
    const auto it = std::find_if(r.series.begin(), r.series.end(),
                                 [&](const auto& s) { return s.first == name; });
    REQUIRE(it != r.series.end());
    return it->second;
}

double scalar(const alea::RunResult& r, const std::string& name) {
    const auto it = std::find_if(r.scalars.begin(), r.scalars.end(),
                                 [&](const auto& s) { return s.first == name; });
    REQUIRE(it != r.scalars.end());
    return it->second;
}

bool within(double a, double sa, double b, double sb, double sigmas = 4.0) {
    return std::abs(a - b) <= sigmas * std::hypot(sa, sb);
}

}  // namespace

// ── Exact: two dipoles in a trap, window {1, 2} ─────────────────────────────
// The finite-M references of tests/references/inverse_power_trap_primitive.py
// (beta = 2, M = 16, omega = 1, C = 1, p = 3), as in
// test_pimc_grand_canonical.cpp, now reached through the YAML keys, the App's
// sector actions (sector 1 has no pair term, sector 2 has it) and the
// driver's sector weights and mu output.
TEST_CASE("A grand-canonical stage of two trapped dipoles gives the exact F(2) - F(1) and sector energies",
          "[pimc][grand_canonical][physics]") {
    constexpr double kDeltaF = 1.847461147235;       // F(2) - F(1)
    constexpr double kIdealDeltaF = 0.971717715022;   // C = 0
    constexpr double kNoExchangeDeltaF = 1.899507970692;
    constexpr double kEnergy1 = 1.310948307761;       // E(N = 1), thermodynamic
    constexpr double kEnergy2 = 3.442269884856;       // E(N = 2)
    run_stage(load("pimc_gc_trapped_dipoles.yaml"), [&](const alea::Pimc& pimc) {
        REQUIRE(pimc.grand_canonical());
        const alea::PimcGrandCanonicalResult& gc = pimc.grand_canonical_result();
        INFO(pimc.summary());
        REQUIRE(gc.sectors.size() == 2);
        REQUIRE(gc.mu.size() == 1);
        CHECK(gc.pair_transitions[0] >= 100.0);
        const double mu = gc.mu[0], mu_error = gc.mu_error[0];
        CHECK(std::abs(mu - kDeltaF) <= 4.0 * mu_error);
        // Power: dropping the interaction or the exchange cycles moves the
        // reference by more than 5 sigma.
        CHECK(std::abs(kIdealDeltaF - kDeltaF) > 5.0 * mu_error);
        CHECK(std::abs(kNoExchangeDeltaF - kDeltaF) > 5.0 * mu_error);
        const alea::PimcSectorResult& one = gc.sectors[0];
        const alea::PimcSectorResult& two = gc.sectors[1];
        CHECK(one.n == 1);
        CHECK(two.n == 2);
        CHECK(std::abs(one.energy - kEnergy1) <= 4.0 * one.energy_error);
        CHECK(std::abs(two.energy - kEnergy2) <= 4.0 * two.energy_error);
        // Bookkeeping identities of the ratio estimators.
        CHECK(std::abs(one.fraction + two.fraction - 1.0) < 1e-12);
        CHECK(std::abs(gc.n_mean - (one.fraction + 2.0 * two.fraction)) < 1e-12);
        CHECK(std::abs(gc.n2_mean - (one.fraction + 4.0 * two.fraction)) < 1e-12);
        CHECK(one.measurements > 0.0);
        CHECK(two.measurements > 0.0);
        // The aggregate energy is the sector average weighted by P(N).
        CHECK(std::abs(pimc.energy() - (one.fraction * one.energy + two.fraction * two.energy)) <
              1e-9 * std::abs(pimc.energy()));

        // The results file carries the same numbers under the documented keys.
        const alea::RunResult r = pimc.run_result();
        CHECK(series(r, "gc_mu")[0] == mu);
        CHECK(series(r, "gc_mu_N")[0] == 1.5);
        CHECK(series(r, "gc_mu_reliable")[0] == 1.0);
        CHECK(series(r, "gc_sector_N") == std::vector<double>{1.0, 2.0});
        CHECK(scalar(r, "gc_N2_energy") == two.energy);
        CHECK(scalar(r, "gc_chemical_potential_input") == 1.85);
        CHECK(series(r, "acc_block_gc_N1_weight").size() == 400);
    });
}

// ── Moire dipoles: sectors against canonical runs; mu_in independence ───────
// Window {3, 4} on a 3 x 3 moire cell at beta = 1, tau = 0.02 (M = 50).
//  (5b) Each sector's E, W^2 and Bragg intensity must equal a canonical run
//       at that N (same beta, tau, moves).  Power: a sector measured with the
//       action of the other N would be off by d/(2 tau) = 50 per particle in
//       the springs' normalization, and by tail(4) - tail(3) ≈ 9.7 in the
//       inverse_power tail of this cell; 4 sigma of the comparison is ≈ 1.
//  (6)  F(4) - F(3) = mu_in - T ln[P(4)/P(3)] is an identity in mu_in: two
//       runs a few T apart (mu_in = 22.6 and 24.6, T = 1) must agree.  It
//       catches a factor that depends on mu_in, e.g. a wrong beta in
//       e^{beta mu N}; the absolute value is checked by the exact test above.
TEST_CASE("Moire dipoles: grand-canonical sectors match canonical runs and F(4) - F(3) does not depend on mu_in",
          "[pimc][grand_canonical][physics]") {
    struct Sector {
        double e, e_err, w2, w2_err, bragg, bragg_err;
    };
    // Canonical reference at N: the same stage with a canonical worm.
    const auto canonical = [](std::size_t n) {
        alea::io::Config cfg = load("pimc_gc_moire_dipoles.yaml");
        worm_of(cfg).grand_canonical.reset();
        cfg.particles.front().up = static_cast<int>(n);
        Sector out{};
        run_stage(cfg, [&](const alea::Pimc& pimc) {
            REQUIRE(!pimc.grand_canonical());
            const alea::RunResult r = pimc.run_result();
            out = {pimc.energy(), pimc.energy_error(), pimc.winding2(), pimc.winding2_error(),
                   series(r, "bragg_bosons")[0], series(r, "bragg_bosons_error")[0]};
        });
        return out;
    };
    const Sector reference[2] = {canonical(3), canonical(4)};

    std::vector<double> mu, mu_error;
    for (const double mu_in : {22.6, 24.6}) {
        CAPTURE(mu_in);
        alea::io::Config cfg = load("pimc_gc_moire_dipoles.yaml");
        worm_of(cfg).grand_canonical->chemical_potential = mu_in;
        run_stage(cfg, [&](const alea::Pimc& pimc) {
            const alea::PimcGrandCanonicalResult& gc = pimc.grand_canonical_result();
            INFO(pimc.summary());
            REQUIRE(gc.sectors.size() == 2);
            CHECK(gc.pair_transitions[0] >= 100.0);
            const alea::RunResult r = pimc.run_result();
            for (std::size_t i = 0; i < 2; ++i) {
                const alea::PimcSectorResult& s = gc.sectors[i];
                const Sector& c = reference[i];
                const std::string bragg = "bragg_bosons_N" + std::to_string(s.n);
                const double b = series(r, bragg)[0], b_err = series(r, bragg + "_error")[0];
                INFO("N = " << s.n << ": E " << s.energy << " +/- " << s.energy_error
                     << " canonical " << c.e << " +/- " << c.e_err << "; W2 " << s.winding2
                     << " +/- " << s.winding2_error << " canonical " << c.w2 << " +/- "
                     << c.w2_err << "; Bragg " << b << " +/- " << b_err << " canonical "
                     << c.bragg << " +/- " << c.bragg_err);
                CHECK(within(s.energy, s.energy_error, c.e, c.e_err));
                CHECK(within(s.winding2, s.winding2_error, c.w2, c.w2_err));
                CHECK(within(b, b_err, c.bragg, c.bragg_err));
                CHECK(std::hypot(s.energy_error, c.e_err) < 0.5);
            }
            mu.push_back(gc.mu[0]);
            mu_error.push_back(gc.mu_error[0]);
        });
    }
    INFO("F(4) - F(3): " << mu[0] << " +/- " << mu_error[0] << " at mu_in = 22.6, " << mu[1]
         << " +/- " << mu_error[1] << " at mu_in = 24.6; canonical E(4) - E(3) = "
         << reference[1].e - reference[0].e << " (not an identity at T > 0)");
    CHECK(within(mu[0], mu_error[0], mu[1], mu_error[1]));
    CHECK(std::max(mu_error[0], mu_error[1]) < 0.3);
}

// ── What a grand-canonical stage refuses ────────────────────────────────────
TEST_CASE("A grand-canonical stage refuses what it cannot sample", "[pimc][grand_canonical]") {
    const auto refuses = [](const std::function<void(alea::io::Config&)>& edit,
                            const std::string& message) {
        alea::io::Config cfg = load("pimc_gc_trapped_dipoles.yaml");
        edit(cfg);
        alea::App app(cfg);
        CHECK_THROWS_WITH(app.build_drivers(), ContainsSubstring(message));
    };
    const auto window = [](alea::io::Config& cfg) -> alea::io::PimcMoveSpec::GrandCanonical& {
        return *worm_of(cfg).grand_canonical;
    };
    // The insertion density must match the cell.
    refuses([&](auto& cfg) { window(cfg).insertion_width = 0.0; }, "needs `insertion_width`");
    refuses([&](auto& cfg) { window(cfg).insertion_center = {0.0, 0.0, 0.0}; },
            "`insertion_center` needs 2 coordinates");
    refuses([&](auto& cfg) { cfg.particles.front().up = 3; }, "lies outside the window [1, 2]");
    refuses([&](auto& cfg) { cfg.particles.front().type = "fermions"; }, "must be `type: bosons`");
    refuses([&](auto& cfg) { cfg.particles.front().down = 1; }, "put every particle in `up`");
    refuses([&](auto& cfg) { cfg.particles.push_back(cfg.particles.front()); },
            "must hold one particle group");
    refuses([&](auto& cfg) { cfg.pimc.front().energy = "centroid_virial"; },
            "thermodynamic energy only");
    refuses([&](auto& cfg) { cfg.pimc.front().statistics = "boltzmannon"; },
            "samples bosons");
    refuses([&](auto& cfg) {
        alea::io::PimcMoveSpec permutation;
        permutation.type = "permutation";
        permutation.levels = 2;
        permutation.max_cycle = 2;
        cfg.pimc.front().moves.push_back(permutation);
    }, "remove the `permutation` move");
    refuses([&](auto& cfg) {
        alea::io::PimcMoveSpec second = worm_of(cfg);
        second.grand_canonical.reset();
        cfg.pimc.front().moves.push_back(second);
    }, "remove the canonical worm");
    // A potential term without a grand-canonical bead energy.
    refuses([&](auto& cfg) {
        cfg.particles.front().charge = 1.0;
        alea::io::KeldyshSpec keldysh;
        keldysh.pair = {"bosons", "bosons"};
        cfg.keldysh.push_back(keldysh);
    }, "no bead energy for");
    // A periodic cell draws inserted worms uniformly: a width is refused.
    alea::io::Config cfg = load("pimc_gc_moire_dipoles.yaml");
    worm_of(cfg).grand_canonical->insertion_width = 0.5;
    alea::App app(cfg);
    CHECK_THROWS_WITH(app.build_drivers(), ContainsSubstring("remove `insertion_width`"));
}

// -- Path checkpoints of a grand-canonical run (version 2) -------------------
namespace {

// A scratch directory owned by one test.
struct ScratchDirectory {
    std::filesystem::path path;
    ScratchDirectory() {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("alea_gc_paths_" + std::to_string(tick));
        std::filesystem::create_directories(path);
    }
    ~ScratchDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

// The trapped-dipole stage, short: these tests check the bookkeeping of the
// paths, not the physics.
alea::io::Config short_dipole_stage() {
    alea::io::Config cfg = load("pimc_gc_trapped_dipoles.yaml");
    auto& stage = cfg.pimc.front();
    stage.equilibration_blocks = 2;
    stage.accumulation_blocks = 4;
    stage.steps_per_block = 20;
    stage.sub_steps = 10;
    return cfg;
}

// Closed rings of the 4 replicas of that stage (M = 16, d = 2), with N_r
// particles each: particle p circles (+-0.6, 0.1 r) with radius 0.05.  A
// replica of two particles listed in `exchanged` closes as one 2-cycle.
alea::io::PathCheckpoint dipole_rings(const std::vector<std::size_t>& active,
                                      const std::vector<bool>& exchanged) {
    alea::io::PathCheckpoint cp;
    cp.replicas = 4;
    cp.m = 16;
    cp.np = 2;
    cp.dim = 2;
    cp.beta = 2.0;
    cp.tau = 0.125;
    cp.active = active;
    for (std::size_t r = 0; r < cp.replicas; ++r)
        for (std::size_t p = 0; p < active[r]; ++p) {
            cp.permutation.push_back(exchanged[r] ? 1 - p : p);
            for (std::size_t d = 0; d < cp.dim; ++d)
                for (std::size_t k = 0; k < cp.m; ++k) {
                    const double phase = 2.0 * std::numbers::pi * static_cast<double>(k) / 16.0;
                    cp.positions.push_back(d == 0 ? (p == 0 ? 0.6 : -0.6) + 0.05 * std::cos(phase)
                                                  : 0.1 * static_cast<double>(r) +
                                                        0.05 * std::sin(phase));
                }
        }
    return cp;
}

alea::Pimc& only_pimc(alea::DriverPack& pack) {
    REQUIRE(pack.drivers.size() == 1);
    auto* pimc = dynamic_cast<alea::Pimc*>(pack.drivers.front().get());
    REQUIRE(pimc != nullptr);
    return *pimc;
}

void check_same_paths(const alea::io::PathCheckpoint& a, const alea::io::PathCheckpoint& b) {
    CHECK(a.replicas == b.replicas);
    CHECK(a.m == b.m);
    CHECK(a.np == b.np);
    CHECK(a.dim == b.dim);
    CHECK(a.active == b.active);
    CHECK(a.permutation == b.permutation);
    CHECK(a.positions == b.positions);  // bit for bit (no NaN)
}

std::vector<double> as_doubles(const std::vector<std::size_t>& counts) {
    return {counts.begin(), counts.end()};
}

}  // namespace

// Request test 7.  Replicas at N = 1, 2, 1, 2 (one of them an exchange
// cycle) enter through import_paths, leave through export_paths and a
// version-2 file unchanged, start the run at their own N, and a run
// continued from the written file starts where the first one ended.
TEST_CASE("A grand-canonical run imports, continues and re-exports replicas at different N",
          "[pimc][grand_canonical][checkpoint]") {
    const alea::io::Config cfg = short_dipole_stage();
    const auto start = dipole_rings({1, 2, 1, 2}, {false, false, false, true});
    ScratchDirectory scratch;

    alea::App app(cfg);
    alea::DriverPack pack = app.build_drivers();
    alea::Pimc& first = only_pimc(pack);
    first.import_paths(start);
    check_same_paths(first.export_paths(), start);
    first.run();
    const alea::RunResult first_result = first.run_result();
    CHECK(series(first_result, "gc_initial_N") == as_doubles(start.active));

    // The paths after the run: closed, in the window, through a file.
    const auto end = first.export_paths();
    REQUIRE_NOTHROW(alea::io::validate_path_checkpoint(end));
    for (const std::size_t n : end.active) CHECK((n == 1 || n == 2));
    const auto file = scratch.path / "paths.h5";
    alea::io::write_paths(file, end);
    const auto read = alea::io::read_paths(file);
    check_same_paths(read, end);

    // A second run continues from the file.
    alea::App app2(cfg);
    alea::DriverPack pack2 = app2.build_drivers();
    alea::Pimc& second = only_pimc(pack2);
    second.import_paths(read);
    check_same_paths(second.export_paths(), end);
    second.run();
    CHECK(series(second.run_result(), "gc_initial_N") == as_doubles(end.active));
    const double n_mean = scalar(second.run_result(), "gc_N_mean");
    CHECK((n_mean >= 1.0 && n_mean <= 2.0));
}

TEST_CASE("A version-1 path file starts a grand-canonical run at its n_particles",
          "[pimc][grand_canonical][checkpoint]") {
    const alea::io::Config cfg = short_dipole_stage();  // up: 1, window {1, 2}
    const auto canonical = dipole_rings({2, 2, 2, 2}, {false, true, false, false});
    ScratchDirectory scratch;
    const auto file = scratch.path / "paths_v1.h5";
    alea::test::write_version1_paths(file, canonical);

    alea::App app(cfg);
    alea::DriverPack pack = app.build_drivers();
    alea::Pimc& pimc = only_pimc(pack);
    pimc.import_paths(alea::io::read_paths(file));
    check_same_paths(pimc.export_paths(), canonical);
    pimc.run();
    CHECK(series(pimc.run_result(), "gc_initial_N") == std::vector<double>(4, 2.0));
}

TEST_CASE("Path imports refuse particle counts outside the run and nonfinite sector actions",
          "[pimc][grand_canonical][checkpoint]") {
    const auto mixed = dipole_rings({1, 2, 1, 2}, {false, false, false, true});

    // A canonical stage of two particles admits N = 2 only: it takes the
    // paths of a grand-canonical run whose replicas all hold two.
    alea::io::Config canonical_cfg = short_dipole_stage();
    canonical_cfg.particles.front().up = 2;
    worm_of(canonical_cfg).grand_canonical.reset();
    alea::App canonical_app(canonical_cfg);
    alea::DriverPack canonical_pack = canonical_app.build_drivers();
    CHECK_THROWS_WITH(only_pimc(canonical_pack).import_paths(mixed),
                      ContainsSubstring("replica 0 holds 1 particles") &&
                          ContainsSubstring("admits 2"));
    const auto pairs = dipole_rings({2, 2, 2, 2}, {true, false, false, false});
    only_pimc(canonical_pack).import_paths(pairs);
    check_same_paths(only_pimc(canonical_pack).export_paths(), pairs);

    // A grand-canonical window {2, 2} refuses the replicas at N = 1.
    alea::io::Config narrow_cfg = short_dipole_stage();
    narrow_cfg.particles.front().up = 2;
    worm_of(narrow_cfg).grand_canonical->n_min = 2;
    alea::App narrow_app(narrow_cfg);
    alea::DriverPack narrow_pack = narrow_app.build_drivers();
    CHECK_THROWS_WITH(only_pimc(narrow_pack).import_paths(mixed),
                      ContainsSubstring("replica 0 holds 1 particles"));

    // Two dipoles on the same beads: S is infinite in sector 2 (C / r^3),
    // which only the action of that sector sees.  Nothing is imported.
    auto collided = mixed;
    const std::size_t row = collided.m * collided.dim;
    const std::size_t second = alea::io::first_particle(collided, 1);  // replica 1, particle 0
    std::copy_n(collided.positions.begin() + static_cast<std::ptrdiff_t>(second * row), row,
                collided.positions.begin() + static_cast<std::ptrdiff_t>((second + 1) * row));
    alea::App app(short_dipole_stage());
    alea::DriverPack pack = app.build_drivers();
    alea::Pimc& pimc = only_pimc(pack);
    const auto before = pimc.export_paths();
    CHECK_THROWS_WITH(pimc.import_paths(collided),
                      ContainsSubstring("action is nonfinite for replica 1"));
    check_same_paths(pimc.export_paths(), before);
}

TEST_CASE("A grand-canonical stage accepts paths_in and paths_out", "[pimc][grand_canonical]") {
    alea::io::Config cfg = short_dipole_stage();
    cfg.pimc.front().paths_in = "in.h5";
    cfg.pimc.front().paths_out = "out.h5";
    alea::App app(cfg);
    alea::DriverPack pack = app.build_drivers();
    REQUIRE(pack.paths_in.size() == 1);
    CHECK(pack.paths_in.front() == "in.h5");
    CHECK(pack.paths_out.front() == "out.h5");
}
