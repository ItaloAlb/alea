// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "io/parser.h"
#include "qmc/app.h"
#include "qmc/driver.h"

namespace {

// Minimal mock to verify the interface is callable.
class MockDriver : public alea::Driver {
public:
    explicit MockDriver(std::string name) : name_(std::move(name)) {}

    void run() override { ran_ = true; }

    [[nodiscard]] std::string summary() const override {
        return name_ + (ran_ ? " ran" : " idle");
    }

    [[nodiscard]] alea::RunResult run_result() const override {
        alea::RunResult r;
        r.method = name_;
        return r;
    }

private:
    std::string name_;
    bool        ran_ = false;
};

// e–h Padé Jastrow u = a₁r/(1 + b₁r): a₁ is the derived cusp, b₁ = 0.5 is
// optimisable (exact answer 0).  Fields are set by name: a positional
// brace-init silently depends on JastrowSpec's member order.
alea::io::JastrowSpec eh_pade_jastrow() {
    alea::io::JastrowSpec j;
    j.type          = "padé";
    j.pair          = {"electron_group", "hole_group"};
    j.denominator   = {0.5};
    j.is_fixed      = {false};
    j.can_flip_sign = {true};
    return j;
}

}  // namespace

TEST_CASE("Driver interface is callable through base pointer", "[driver]") {
    MockDriver d("mock");
    alea::Driver& base = d;

    CHECK(base.summary() == "mock idle");
    base.run();
    CHECK(base.summary() == "mock ran");
}

TEST_CASE("App builds drivers from config and runs them serially", "[driver][app]") {
    // Build a minimal config with one VMC block and one optimizer block.
    using alea::io::Config;
    using alea::io::OptimizerSpec;
    using alea::io::QmcSpec;

    Config cfg;
    cfg.name = "test";
    cfg.seed = 42;
    cfg.lattice.dimension = 2;
    cfg.lattice.L = 2.0;

    // Two particles: e and h.
    cfg.particles.push_back({"e", "electron_group", "fermions", {1.0}, -1.0, 1, 0, "uniform"});
    cfg.particles.push_back({"h", "hole_group", "fermions", {1.0}, +1.0, 1, 0, "uniform"});

    // Jastrow with c=-0.5, non-fixed (optimisable).
    cfg.wavefunction.push_back(eh_pade_jastrow());

    // Coulomb.
    cfg.coulombs.push_back({{"electron_group", "hole_group"}, 1.0});

    // First: optimizer block (short run).
    OptimizerSpec opt;
    opt.type                  = "linear";
    opt.samples               = 32;
    opt.equilibration_blocks  = 0;
    opt.accumulation_blocks   = 1;
    opt.steps_per_block       = 0;   // no MCMC — static evaluation
    opt.time_step             = 0.1;
    opt.max_iterations        = 1;
    opt.shift_i               = 0.1;
    cfg.optimizers.push_back(opt);

    // Second: VMC block (also short).
    QmcSpec vmc;
    vmc.method               = "vmc";
    vmc.samples              = 32;
    vmc.equilibration_blocks = 0;
    vmc.accumulation_blocks  = 1;
    vmc.steps_per_block      = 0;   // static evaluation
    vmc.time_step            = 0.1;
    cfg.qmc.push_back(vmc);

    // ── Build App and get drivers ──────────────────────────────────────────
    alea::App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 2);

    // ── Run serially ───────────────────────────────────────────────────────
    std::stringstream out;
    for (auto& d : pack.drivers) {
        d->run();
        out << d->summary() << '\n';
    }

    // The optimizer summary should mention the optimised parameter.
    std::string output = out.str();
    CHECK(output.find("linear") != std::string::npos);
    CHECK(output.find("vmc") != std::string::npos);

    // Verify psi is shared: the VMC driver should see the parameters
    // that the optimizer left behind.
    auto* vmc_driver = dynamic_cast<alea::VMC*>(pack.drivers[1].get());
    REQUIRE(vmc_driver != nullptr);
    // The VMC worker should have been constructed with the shared psi.
    CHECK(vmc_driver->summary().find("vmc") != std::string::npos);
}

TEST_CASE("build_drivers resolves a DMC walkers_out under a given checkpoint "
          "directory, and leaves it untouched otherwise",
          "[driver][app]") {
    using alea::io::Config;
    using alea::io::QmcSpec;

    Config cfg;
    cfg.name = "test";
    cfg.seed = 42;
    cfg.lattice.dimension = 2;
    cfg.lattice.L = 2.0;

    cfg.particles.push_back({"e", "electron_group", "fermions", {1.0}, -1.0, 1, 0, "uniform"});
    cfg.particles.push_back({"h", "hole_group", "fermions", {1.0}, +1.0, 1, 0, "uniform"});
    cfg.wavefunction.push_back(eh_pade_jastrow());
    cfg.coulombs.push_back({{"electron_group", "hole_group"}, 1.0});

    QmcSpec dmc;
    dmc.method               = "dmc";
    dmc.random_walk          = "drift_diffusion";
    dmc.samples              = 8;
    dmc.equilibration_blocks = 0;
    dmc.accumulation_blocks  = 0;
    dmc.steps_per_block      = 0;
    dmc.sub_steps            = 1;
    dmc.time_step            = 0.1;
    dmc.walkers_out          = "wk.h5";  // a bare filename, no directory
    cfg.qmc.push_back(dmc);

    // No checkpoint_dir given (legacy callers / tests): the YAML's bare
    // filename is used exactly as written, relative to the cwd.
    {
        alea::App app(cfg);
        auto pack = app.build_drivers();
        REQUIRE(pack.walkers_out.size() == 1);
        CHECK(pack.walkers_out[0] == "wk.h5");
    }

    // checkpoint_dir given: the filename lands inside it instead.
    {
        const std::filesystem::path dir =
            std::filesystem::path("checkpoints") / "test" / "2026-01-01_00-00-00-000";
        alea::App app(cfg);
        auto pack = app.build_drivers(dir);
        REQUIRE(pack.walkers_out.size() == 1);
        CHECK(pack.walkers_out[0] == (dir / "wk.h5").string());
    }
}

TEST_CASE("carry_over_walkers seeds the next driver from the previous one's "
          "final configurations, cycling round-robin on a count mismatch",
          "[driver][app]") {
    // QMCPACK carry-over: QMCDriverNew::finalize saves the population and
    // MCPopulation::createWalkers seeds the next driver with walker
    // iw ← config[iw % n_existing].  VMC(8) → DMC(12) duplicates 4 configs;
    // DMC(12) → VMC(5) drops 7.
    using alea::io::Config;
    using alea::io::QmcSpec;

    Config cfg;
    cfg.name = "test";
    cfg.seed = 7;
    cfg.lattice.dimension = 2;
    cfg.lattice.L = 2.0;
    cfg.particles.push_back({"e", "electron_group", "fermions", {1.0}, -1.0, 1, 0, "uniform"});
    cfg.particles.push_back({"h", "hole_group", "fermions", {1.0}, +1.0, 1, 0, "uniform"});
    cfg.wavefunction.push_back(eh_pade_jastrow());
    cfg.coulombs.push_back({{"electron_group", "hole_group"}, 1.0});

    QmcSpec vmc;
    vmc.method               = "vmc";
    vmc.random_walk          = "metropolis";
    vmc.threads              = 2;   // two per-thread WalkerSets to gather from
    vmc.samples              = 8;
    vmc.equilibration_blocks = 2;
    vmc.accumulation_blocks  = 2;
    vmc.steps_per_block      = 3;
    vmc.sub_steps            = 1;
    vmc.time_step            = 0.5;
    cfg.qmc.push_back(vmc);

    QmcSpec dmc;
    dmc.method               = "dmc";
    dmc.random_walk          = "drift_diffusion";
    dmc.threads              = 1;
    dmc.samples              = 12;
    dmc.equilibration_blocks = 1;
    dmc.accumulation_blocks  = 1;
    dmc.steps_per_block      = 2;
    dmc.sub_steps            = 1;
    dmc.time_step            = 0.05;
    cfg.qmc.push_back(dmc);

    QmcSpec vmc2 = vmc;
    vmc2.samples = 5;
    vmc2.threads = 1;
    cfg.qmc.push_back(vmc2);

    alea::App app(cfg);
    auto pack = app.build_drivers();
    REQUIRE(pack.drivers.size() == 3);
    const std::size_t row = 2 * 2;  // np · dim

    // VMC(8) → DMC(12): walker w takes config w % 8.
    pack.drivers[0]->run();
    const std::vector<double> vmc_final = pack.drivers[0]->final_configurations();
    REQUIRE(vmc_final.size() == 8 * row);
    CHECK(alea::carry_over_walkers(pack, 0, 1) == 8);

    std::vector<double> seeded;
    REQUIRE(pack.drivers[1]->final_population() != nullptr);
    pack.drivers[1]->final_population()->append_configurations(seeded);
    REQUIRE(seeded.size() == 12 * row);
    for (std::size_t w = 0; w < 12; ++w)
        for (std::size_t k = 0; k < row; ++k)
            CHECK(seeded[w * row + k] == vmc_final[(w % 8) * row + k]);

    // DMC(12) → VMC(5): the first 5 of the DMC's final configurations.
    pack.drivers[1]->run();
    const std::vector<double> dmc_final = pack.drivers[1]->final_configurations();
    const std::size_t n_dmc = dmc_final.size() / row;
    REQUIRE(n_dmc >= 5);
    CHECK(alea::carry_over_walkers(pack, 1, 2) == n_dmc);
    const std::vector<double> vmc2_start = pack.drivers[2]->final_configurations();
    REQUIRE(vmc2_start.size() == 5 * row);
    for (std::size_t k = 0; k < 5 * row; ++k)
        CHECK(vmc2_start[k] == dmc_final[k]);
    pack.drivers[2]->run();  // runs cleanly from the carried state
}
