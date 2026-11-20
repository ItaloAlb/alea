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
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "io/parser.h"
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
    refuses([&](auto& cfg) { cfg.pimc.front().paths_out = "paths.h5"; },
            "need checkpoint version 2");
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
