// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// PIMC physics regression tests: full YAML -> App -> Pimc runs against
// analytic results.
//
// (1) Discretized harmonic oscillator.  For the primitive (first-order
//     Trotter) action the M-slice path integral is a Gaussian, and its exact
//     free energy follows from the normal modes of the ring:
//         E_M = d * tau*omega^2/M * sum_k 1/(4 sin^2(pi k/M) + tau^2 omega^2)
//     The thermodynamic estimator must reproduce E_M itself — NOT the
//     continuum value — so no tau -> 0 extrapolation is involved.  For a
//     quadratic potential the split is exact too: <K> = <V> = E/2.
//
// (2) Free particle in a periodic box.  The exact partition function is the
//     momentum sum Z = sum_n exp(-beta eps_n); at tau = 0.0625 the code's
//     min-image spring agrees with it to ~2e-8 (checked against the
//     transfer-matrix free energy of the discretized action).  V = 0
//     identically, so the potential estimator must be exactly zero.
//     Winding: local sector changes are exponentially suppressed at small
//     tau; relative lattice-frame bisection samples exp(-L² W²/(2 beta))
//     without an artificial cap on W. The equilibrium acceptance follows
//     neighboring-sector flux, not just the W=0 initial acceptance. Both E
//     and <W²> are tested, including broad winding at beta=16 (W²~1, not 2).
//
// Every comparison uses the driver's own reblocked (Flyvbjerg-Petersen) error
// bar at a fixed seed, plus a floor so a collapsed error cannot excuse a
// wrong mean — and the error bar itself must be small enough for the test to
// mean anything.

#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "io/parser.h"
#include "pimc/pimc.h"
#include "qmc/app.h"

namespace {

struct PimcRun {
    double e, e_err;
    double k, k_err;
    double v, v_err;
    double w2, w2_err;      // <W^2> per (particle, lattice direction)
    double naive_err;
    double tau_int;
    bool   plateau;
    double acceptance;
    std::vector<double> blocks;
    std::string summary;
};

PimcRun run_config(const alea::io::Config& cfg) {
    alea::App app(cfg);
    alea::DriverPack pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 1);
    auto* pimc = dynamic_cast<alea::Pimc*>(pack.drivers[0].get());
    REQUIRE(pimc != nullptr);
    pimc->run();

    PimcRun r;
    r.e = pimc->energy();            r.e_err = pimc->energy_error();
    r.k = pimc->kinetic();           r.k_err = pimc->kinetic_error();
    r.v = pimc->potential();         r.v_err = pimc->potential_error();
    r.w2 = pimc->winding2();         r.w2_err = pimc->winding2_error();
    r.naive_err = pimc->naive_error();
    r.tau_int = pimc->tau_int();
    r.plateau = pimc->plateau_found();
    r.acceptance = pimc->acceptance_ratio();
    r.blocks.assign(pimc->block_energies().begin(),
                    pimc->block_energies().end());
    r.summary = pimc->summary();
    return r;
}

PimcRun run_config(const std::string& file) {
    const std::string path = std::string(ALEA_CONFIG_DIR) + "/" + file;
    return run_config(alea::io::parse_config_file(path));
}

// The reblocked error bar is the authority, but a floor keeps a wrongly
// collapsed error (reblock bug, ignored correlations) from waving a wrong
// mean through: the mean itself must be near the analytic value.
double tolerance(double err) { return 4.0 * err + 1e-3; }

}  // namespace

TEST_CASE("PIMC reproduces the discretized harmonic oscillator",
          "[pimc][physics]") {
    // Exact M-slice value (normal modes), d = 1, beta = 2, M = 8, w = 1.
    const double e_exact = 0.6523800866955113;

    const PimcRun r = run_config("pimc_ho_1d.yaml");
    INFO("summary: " << r.summary);
    INFO("E = " << r.e << " +/- " << r.e_err << "  exact = " << e_exact);

    CHECK(std::abs(r.e - e_exact) <= tolerance(r.e_err));
    // The error bar must be informative (not vacuous): below 5% of E.
    CHECK(r.e_err < std::abs(e_exact) / 20.0);
    // Physically sensible diagnostics.
    CHECK(r.plateau);
    CHECK(r.acceptance > 0.0);
    CHECK(r.acceptance < 1.0);
    CHECK(r.blocks.size() == 400);

    // Virial split: <K> = <V> = E/2 exactly for a quadratic potential.
    const double half = e_exact / 2.0;
    INFO("K = " << r.k << " +/- " << r.k_err);
    INFO("V = " << r.v << " +/- " << r.v_err);
    CHECK(std::abs(r.k - half) <= tolerance(r.k_err));
    CHECK(std::abs(r.v - half) <= tolerance(r.v_err));
    CHECK(r.k > 0.0);
    CHECK(r.v > 0.0);
    // E = K + V must hold on the reduced means too.
    CHECK(r.e == Catch::Approx(r.k + r.v).margin(1e-9));
}

TEST_CASE("PIMC free particle under PBC matches the momentum sum",
          "[pimc][physics]") {
    // Exact: E = sum_n eps_n e^{-beta eps_n} / sum_n e^{-beta eps_n},
    // eps_n = (2 pi n / L)^2 / 2, L = 4, beta = 2, m = 1, d = 1.
    const double e_exact = 0.1793245903520491;
    // <W^2> = sum_W W^2 exp(-L^2 W^2 / 2 beta) / Z_W, per direction:
    // = 0.07067540964795092 / 2 (d = 1).
    const double w2_exact = 0.03533770482397546;

    const PimcRun r = run_config("pimc_free_1d_pbc.yaml");
    INFO("summary: " << r.summary);
    INFO("E = " << r.e << " +/- " << r.e_err << "  exact = " << e_exact);
    INFO("W2 = " << r.w2 << " +/- " << r.w2_err
         << "  exact = " << w2_exact);

    // Energy: the winding sectors carry the difference between the quantum
    // 0.179 and the frozen-W=0 classical 0.25; the frame-jump bisection must
    // recover the quantum value within the reblocked error.
    CHECK(std::abs(r.e - e_exact) <= tolerance(r.e_err));
    CHECK(r.e_err < std::abs(e_exact) / 20.0);
    CHECK(r.plateau);

    // Winding populations: an independent, sharper probe — a winding-frozen
    // chain gives W^2 = 0, failing by the full 0.03533 target even if the
    // energy error bar were loose.
    CHECK(std::abs(r.w2 - w2_exact) <= tolerance(r.w2_err));
    CHECK(r.w2_err < w2_exact / 20.0);
    CHECK(r.w2 > 0.0);

    // V = 0 identically for this system: the potential estimator must be
    // exactly zero (it is a sum of evaluate_potential over empty H).
    CHECK(std::abs(r.v) < 1e-15);
    // With V = 0, K carries the whole energy.
    CHECK(r.k == Catch::Approx(r.e).margin(1e-12));
    // Heat-bath relative frames: a full-ring move from sector W to W+s is
    // made with P = w(W+s)/max(Z(W), Z(W+s)), w(W) = exp(-L^2 W^2/2 beta) =
    // exp(-4 W^2), Z(W) = w(W-1)+w(W)+w(W+1); the equilibrium bisection
    // acceptance A = sum_W pi(W) sum_s P, and with equal centroid weight
    // (always accepted for V = 0) the reference is (1+A)/2.
    auto w8 = [](int w) { return std::exp(-4.0 * w * w); };
    auto z8 = [&](int w) { return w8(w - 1) + w8(w) + w8(w + 1); };
    double z = 0.0;
    double flux = 0.0;
    for (int w = -8; w <= 8; ++w) {
        z += w8(w);
        for (int s = -1; s <= 1; ++s)
            flux += w8(w) * w8(w + s) / std::max(z8(w), z8(w + s));
    }
    const double expected = (1.0 + flux / z) / 2.0;
    CHECK(std::abs(r.acceptance - expected) < 0.02);
}

TEST_CASE("low-temperature PIMC samples broad winding sectors", "[pimc][physics][frame-regression]") {
    // No config-file edit: override only run parameters in the existing free
    // fixture. tau stays in the controlled minimum-image regime; beta rises
    // so W=+/-2 and beyond carry appreciable weight (old cap gives W²=.548).
    const std::string path = std::string(ALEA_CONFIG_DIR) + "/pimc_free_1d_pbc.yaml";
    auto cfg = alea::io::parse_config_file(path);
    cfg.seed = 831927;
    auto& run = cfg.pimc.at(0);
    run.beta = 16.0;
    run.time_step = 0.0625;
    run.samples = 32;
    run.equilibration_blocks = 100;
    run.accumulation_blocks = 400;
    run.steps_per_block = 20;
    run.sub_steps = 5;
    run.moves.at(0).levels = 8;
    const PimcRun r = run_config(cfg);
    constexpr double w2_exact = 0.999999788768;
    constexpr double e_exact = 6.6010084974e-9;
    INFO("E=" << r.e << "+/-" << r.e_err << " W²=" << r.w2 << "+/-" << r.w2_err);
    CHECK(std::abs(r.w2 - w2_exact) <= tolerance(r.w2_err));
    CHECK(r.w2_err < 0.02);
    CHECK(std::abs(r.e - e_exact) <= tolerance(r.e_err));
    CHECK(r.e_err < 0.003);
}

TEST_CASE("the same config and seed reproduce bit-identical PIMC results",
          "[pimc][physics]") {
    const PimcRun a = run_config("pimc_ho_1d.yaml");
    const PimcRun b = run_config("pimc_ho_1d.yaml");

    REQUIRE(a.blocks.size() == b.blocks.size());
    for (std::size_t i = 0; i < a.blocks.size(); ++i)
        REQUIRE(a.blocks[i] == b.blocks[i]);
    REQUIRE(a.e == b.e);
    REQUIRE(a.e_err == b.e_err);
    REQUIRE(a.k == b.k);
    REQUIRE(a.v == b.v);
    REQUIRE(a.acceptance == b.acceptance);
}

TEST_CASE("PIMC free particle in a hexagonal cell: Cartesian winding tensor",
          "[pimc][physics][winding]") {
    // Exact: P(W) ~ exp(-|W|^2 / (2 beta)) over the lattice vectors of the
    // hexagonal cell a1 = (4, 0), a2 = (2, 2 sqrt 3), beta = 2, m = 1:
    //   T_xx = T_yy = 0.79289320151651, T_xy = 0, and the lattice-basis
    //   winding2 = 0.06607443345970919 = (4/3) <|W|^2>/(2 L^2).
    const double t_exact = 0.79289320151651;
    const double w2_exact = 0.06607443345970919;
    const std::string path = std::string(ALEA_CONFIG_DIR) + "/pimc_free_2d_hex_pbc.yaml";
    alea::App app(alea::io::parse_config_file(path));
    alea::DriverPack pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 1);
    auto* pimc = dynamic_cast<alea::Pimc*>(pack.drivers[0].get());
    REQUIRE(pimc != nullptr);
    pimc->run();
    REQUIRE(pimc->winding_tensor_components() == 3);
    const double txx = pimc->winding_tensor(0), exx = pimc->winding_tensor_error(0);
    const double txy = pimc->winding_tensor(1), exy = pimc->winding_tensor_error(1);
    const double tyy = pimc->winding_tensor(2), eyy = pimc->winding_tensor_error(2);
    INFO("summary: " << pimc->summary());
    INFO("T_xx=" << txx << "+/-" << exx << " T_xy=" << txy << "+/-" << exy
         << " T_yy=" << tyy << "+/-" << eyy << "  exact " << t_exact);
    INFO("W2=" << pimc->winding2() << "+/-" << pimc->winding2_error() << " exact " << w2_exact);
    CHECK(std::abs(txx - t_exact) <= tolerance(exx));
    CHECK(std::abs(tyy - t_exact) <= tolerance(eyy));
    CHECK(std::abs(txy) <= tolerance(exy));
    CHECK(exx < t_exact / 10.0);
    CHECK(eyy < t_exact / 10.0);
    // The lattice-basis W2 keeps its own exact target (4/3 of the Cartesian one).
    CHECK(std::abs(pimc->winding2() - w2_exact) <= tolerance(pimc->winding2_error()));
    // The result file carries every component, its error and its block series.
    const alea::RunResult r = pimc->run_result();
    const auto has = [](const auto& entries, const std::string& key) {
        return std::count_if(entries.begin(), entries.end(),
                             [&](const auto& e) { return e.first == key; }) == 1;
    };
    CHECK(has(r.scalars, "winding_tensor_xx"));
    CHECK(has(r.scalars, "winding_tensor_xy_error"));
    CHECK(has(r.series, "acc_block_winding_tensor_yy"));
}
