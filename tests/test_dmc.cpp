// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "wavefunction/trial_wavefunction.h"
#include "wavefunction/pade_jastrow.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "hamiltonian/coulomb.h"
#include "numerics/statistics.h"
#include "qmc/dmc.h"

using namespace alea;

namespace {

ParticleGroup grp(std::size_t first, std::size_t last, double charge,
                  double inv_m) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_m, inv_m};
    return g;
}

TrialWavefunction make_psi() {
    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", 1.0, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0), std::move(params)));
    return psi;
}

Hamiltonian make_ham() {
    Hamiltonian h;
    h.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}));
    h.add_component(std::make_unique<CoulombTerm>(
        grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0), 1.0));
    return h;
}

}  // namespace

TEST_CASE("DMC with tau=0 measures the exact local energy (deterministic)",
          "[dmc]") {
    // One walker, p0=(0,0), p1=(2,0): E_L = K + V = -1.5 + -0.5 = -2.0.
    WalkerEnsemble ens(/*n_walkers=*/1,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    ens.walkers()[0].set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    DMCConfig cfg;
    cfg.tau = 0.0;
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 3;
    cfg.n_steps = 4;
    cfg.n_substeps = 1;
    cfg.seed = 42;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    // tau=0 → no moves → deterministic; E_L = -2.0 for all walkers.
    CHECK(dmc.energy() == Catch::Approx(-2.0));
    CHECK(dmc.acceptance_ratio() == Catch::Approx(1.0));
    CHECK(dmc.energy_error() == Catch::Approx(0.0).margin(1e-12));
    CHECK(dmc.energy_variance() == Catch::Approx(0.0).margin(1e-12));
    CHECK(dmc.block_energies().size() == 3);
    const RunResult result = dmc.run_result();
    const auto eq = std::find_if(result.series.begin(), result.series.end(),
        [](const auto& entry) { return entry.first == "eq_block_energy"; });
    REQUIRE(eq != result.series.end());
    REQUIRE(eq->second.size() == 2);
    for (const double energy : eq->second)
        CHECK(energy == Catch::Approx(-2.0));

    // Branching: for tau=0, branch_weight is always 1.0, population
    // is stable — the DMC run is deterministic regardless of E_trial.
}

TEST_CASE("DMC records the branch cutoff it used and the share it tapered", "[dmc]") {
    // Each scheme's e_cut, from the spread the run itself measured.
    auto run = [](BranchCutoff scheme) {
        WalkerEnsemble ens(/*n_walkers=*/32,
                           {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
        int k = 0;
        for (auto& w : ens.walkers()) {
            const double r = 0.3 + 0.1 * (k++ % 16);
            w.set_positions(std::vector<double>{0.0, 0.0, r, 0.2 * r});
        }
        TrialWavefunction psi = make_psi();
        Hamiltonian ham = make_ham();
        DMCConfig cfg;
        cfg.tau = 0.05;
        cfg.n_equilibration_blocks = 5;
        cfg.n_accumulation_blocks = 10;
        cfg.n_steps = 5;
        cfg.n_substeps = 2;
        cfg.seed = 7;
        cfg.branch_cutoff = scheme;
        DMC dmc(cfg, ens, psi, ham);
        dmc.run();
        return dmc.run_result();
    };
    auto scalar = [](const RunResult& r, const std::string& name) {
        for (const auto& [n, v] : r.scalars)
            if (n == name) return v;
        FAIL("scalar not found: " << name);
        return 0.0;
    };
    auto has_scalar = [](const RunResult& r, const std::string& name) {
        return std::any_of(r.scalars.begin(), r.scalars.end(),
                           [&](const auto& s) { return s.first == name; });
    };
    auto meta = [](const RunResult& r, const std::string& name) {
        for (const auto& [n, v] : r.metadata)
            if (n == name) return v;
        return std::string();
    };

    // universal: N = 2 diffusing particles, the default α.
    const RunResult u = run(BranchCutoff::universal);
    const double sigma = scalar(u, "branch_cutoff_sigma");
    CHECK(sigma > 0.0);
    CHECK(scalar(u, "branch_cutoff_energy") ==
          Catch::Approx(kDefaultBranchCutoffAlpha * std::sqrt(sigma * std::sqrt(2.0) / 0.05)));
    CHECK(scalar(u, "branch_cutoff_alpha") == kDefaultBranchCutoffAlpha);
    CHECK(scalar(u, "branch_cutoff_particles") == 2.0);
    CHECK(scalar(u, "branch_cutoff_fraction") >= 0.0);
    CHECK(scalar(u, "branch_cutoff_fraction") <= 1.0);
    CHECK(meta(u, "branch_cutoff") == "universal");

    const RunResult d = run(BranchCutoff::drv);
    CHECK(scalar(d, "branch_cutoff_energy") == Catch::Approx(2.0 / std::sqrt(0.05)));
    CHECK_FALSE(has_scalar(d, "branch_cutoff_alpha"));
    CHECK(meta(d, "branch_cutoff") == "drv");

    const RunResult n = run(BranchCutoff::none);
    CHECK(std::isinf(scalar(n, "branch_cutoff_energy")));
    CHECK(scalar(n, "branch_cutoff_fraction") == 0.0);
    CHECK(scalar(n, "branch_cutoff_fraction_eq") == 0.0);
}

TEST_CASE("DMC records the suppressed reweighting and the correlation time it measured",
          "[dmc]") {
    auto run = [](Reweighting rw, double theta, double kappa = kDefaultSuppressionKappa,
                  std::size_t min_sweeps = 256, std::size_t max_sweeps = 8192) {
        WalkerEnsemble ens(/*n_walkers=*/64,
                           {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
        int k = 0;
        for (auto& w : ens.walkers()) {
            const double r = 0.3 + 0.1 * (k++ % 16);
            w.set_positions(std::vector<double>{0.0, 0.0, r, 0.2 * r});
        }
        // A correlation measurement requires a stationary relative measure.
        // u=-r-0.05*r² has the correct cusp and a normalizable, imperfect tail;
        // make_psi() deliberately grows as exp(+r) for other driver tests.
        TrialWavefunction psi;
        std::vector<Parameter> params = {{"a1", -1.0, true}, {"a2", -0.05, true}};
        psi.add_component(std::make_unique<PadeJastrow>(
            2, 0, grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0), std::move(params)));
        Hamiltonian ham = make_ham();
        DMCConfig cfg;
        cfg.tau = 0.05;
        cfg.n_equilibration_blocks = 5;
        cfg.n_accumulation_blocks = 10;
        cfg.n_steps = 5;
        cfg.n_substeps = 2;
        cfg.seed = 7;
        cfg.reweighting = rw;
        cfg.correlation_time = theta;
        cfg.suppression_kappa = kappa;
        cfg.correlation_min_sweeps = min_sweeps;
        cfg.correlation_max_sweeps = max_sweeps;
        DMC dmc(cfg, ens, psi, ham);
        dmc.run();
        CHECK(std::isfinite(dmc.energy()));
        return dmc.run_result();
    };
    auto scalar = [](const RunResult& r, const std::string& name) {
        for (const auto& [n, v] : r.scalars)
            if (n == name) return v;
        FAIL("scalar not found: " << name);
        return 0.0;
    };
    auto has_scalar = [](const RunResult& r, const std::string& name) {
        return std::any_of(r.scalars.begin(), r.scalars.end(),
                           [&](const auto& s) { return s.first == name; });
    };
    auto meta = [](const RunResult& r, const std::string& name) {
        for (const auto& [n, v] : r.metadata)
            if (n == name) return v;
        return std::string();
    };

    // Measured θ: τ_θ·(½ + Σρ(k)) at a step τ_θ ≤ τ that resolves it, from
    // at least the first check after 256 sweeps.
    const RunResult m = run(Reweighting::suppressed, 0.0);
    CHECK(meta(m, "reweighting") == "suppressed");
    const double theta = scalar(m, "correlation_time");
    const double step  = scalar(m, "correlation_step");
    CHECK(theta > 0.0);
    CHECK(std::isfinite(theta));
    CHECK(step > 0.0);
    CHECK(step <= 0.05);
    CHECK(theta == Catch::Approx(step * (0.5 + scalar(m, "correlation_tau_int"))));
    CHECK(theta / step >= 10.0);
    CHECK(scalar(m, "correlation_sweeps") >= 256.0);
    CHECK(scalar(m, "correlation_window_found") == 1.0);
    CHECK(scalar(m, "correlation_equilibration_sweeps") == 256.0);
    CHECK(scalar(m, "correlation_converged") ==
          (scalar(m, "correlation_sweeps") >= 10.0 * scalar(m, "correlation_window") ? 1.0 : 0.0));
    CHECK(scalar(m, "suppression_kappa") == kDefaultSuppressionKappa);
    // τ_eff/τ is the acceptance-weighted share of the diffusion; f ∈ (0, 1].
    CHECK(scalar(m, "tau_eff_ratio") > 0.5);
    CHECK(scalar(m, "tau_eff_ratio") <= 1.0);
    CHECK(scalar(m, "suppression_mean") > 0.0);
    CHECK(scalar(m, "suppression_mean") <= 1.0);

    // A given θ is used as is, with no measuring sweeps.
    const RunResult g = run(Reweighting::suppressed, 0.2);
    CHECK(scalar(g, "correlation_time") == 0.2);
    CHECK(scalar(g, "correlation_sweeps") == 0.0);
    CHECK(scalar(g, "correlation_step") == 0.0);

    // The default weight records its name and nothing of the suppression.
    const RunResult b = run(Reweighting::bare, 0.0);
    CHECK(meta(b, "reweighting") == "bare");
    CHECK_FALSE(has_scalar(b, "correlation_time"));
    CHECK_FALSE(has_scalar(b, "tau_eff_ratio"));

    // Exact f=1 control: fixed θ skips preliminary sampling in both paths.
    // Tiny κ rounds f to 1 here; equality checks the whole stochastic path,
    // including acceptance, the per-walker effective step and branching.
    const RunResult e = run(Reweighting::effective, 0.0);
    const RunResult near_zero = run(Reweighting::suppressed, 0.2, 1e-12);
    CHECK(meta(e, "reweighting") == "effective");
    CHECK_FALSE(has_scalar(e, "correlation_time"));
    CHECK(scalar(e, "suppression_mean") == 1.0);
    CHECK(scalar(e, "energy") == scalar(near_zero, "energy"));
    CHECK(scalar(e, "tau_eff_ratio") == scalar(near_zero, "tau_eff_ratio"));

    // A closed window alone is insufficient: record the entire stopping rule.
    const RunResult short_run = run(Reweighting::suppressed, 0.0,
                                    kDefaultSuppressionKappa, 2, 2);
    CHECK(scalar(short_run, "correlation_sweeps") == 2.0);
    CHECK(scalar(short_run, "correlation_converged") == 0.0);
}

TEST_CASE("DMC refuses an unusable suppressed reweighting", "[dmc]") {
    WalkerEnsemble ens(/*n_walkers=*/4, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, 2);
    for (auto& w : ens.walkers()) w.set_positions(std::vector<double>{0.0, 0.0, 1.0, 0.0});
    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();
    DMCConfig cfg;
    cfg.reweighting = Reweighting::suppressed;
    cfg.suppression_kappa = 0.0;
    CHECK_THROWS_AS(DMC(cfg, ens, psi, ham), std::invalid_argument);
    cfg.suppression_kappa = kDefaultSuppressionKappa;
    cfg.correlation_time = -1.0;
    CHECK_THROWS_AS(DMC(cfg, ens, psi, ham), std::invalid_argument);
    cfg.correlation_time = 0.0;
    cfg.correlation_min_sweeps = 1024;
    cfg.correlation_max_sweeps = 512;
    CHECK_THROWS_AS(DMC(cfg, ens, psi, ham), std::invalid_argument);
    // Neither is read by the bare weight.
    cfg.reweighting = Reweighting::bare;
    CHECK_NOTHROW(DMC(cfg, ens, psi, ham));
}

TEST_CASE("DMC with tau>0 runs the stochastic chain and produces sane stats",
          "[dmc]") {
    WalkerEnsemble ens(/*n_walkers=*/4,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    DMCConfig cfg;
    cfg.tau = 0.3;
    cfg.n_equilibration_blocks = 10;
    cfg.n_accumulation_blocks = 20;
    cfg.n_steps = 5;
    cfg.n_substeps = 2;
    cfg.seed = 42;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    CHECK(dmc.block_energies().size() == 20);
    CHECK(std::isfinite(dmc.energy()));
    CHECK(dmc.acceptance_ratio() > 0.0);
    CHECK(dmc.acceptance_ratio() <= 1.0);
    CHECK(dmc.energy_variance() > 0.0);
    CHECK(dmc.energy_error() > 0.0);
}

TEST_CASE("DMC multithread matches singlethread for tau0",
          "[dmc][threads]") {
    // 4 walkers on 4 threads: deterministic tau=0 must give exact E.
    WalkerEnsemble ens(/*n_walkers=*/4,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    DMCConfig cfg;
    cfg.tau = 0.0;
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 6;
    cfg.n_steps = 4;
    cfg.n_substeps = 1;
    cfg.seed = 42;
    cfg.n_threads = 4;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    CHECK(dmc.energy() == Catch::Approx(-2.0).margin(1e-12));
    CHECK(dmc.energy_error() == Catch::Approx(0.0).margin(1e-12));
    CHECK(dmc.energy_variance() == Catch::Approx(0.0).margin(1e-12));
    CHECK(dmc.acceptance_ratio() == Catch::Approx(1.0));
    CHECK(dmc.block_energies().size() == 6);  // one global block-energy per block
}

TEST_CASE("DMC multithread tau>0 produces sane stats",
          "[dmc][threads]") {
    // 8 walkers, 4 threads: stochastic DMC must finish and produce stats.
    WalkerEnsemble ens(/*n_walkers=*/8,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    DMCConfig cfg;
    cfg.tau = 0.3;
    cfg.n_equilibration_blocks = 8;
    cfg.n_accumulation_blocks = 10;
    cfg.n_steps = 5;
    cfg.n_substeps = 2;
    cfg.seed = 42;
    cfg.n_threads = 4;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    CHECK(dmc.block_energies().size() == 10);  // one global block-energy per block
    CHECK(std::isfinite(dmc.energy()));
    CHECK(dmc.acceptance_ratio() > 0.0);
    CHECK(dmc.acceptance_ratio() <= 1.0);
    CHECK(dmc.energy_variance() > 0.0);
    CHECK(dmc.energy_error() > 0.0);
}

TEST_CASE("DMC reports the reblocked error of its single population chain",
          "[dmc][reblock]") {
    // One-step blocks are correlated through the walkers and E_T, so the
    // reported error must be the reblocking of the block energies the run
    // writes out, not the plain standard error.
    WalkerEnsemble ens(/*n_walkers=*/16,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);
    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    DMCConfig cfg;
    cfg.tau = 0.1;
    cfg.n_equilibration_blocks = 20;
    cfg.n_accumulation_blocks = 256;
    cfg.n_steps = 1;
    cfg.n_substeps = 1;
    cfg.seed = 42;
    cfg.n_threads = 2;
    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    const auto expected = numerics::reblock(dmc.block_energies());
    const RunResult r = dmc.run_result();
    auto scalar = [&](std::string_view name) {
        for (const auto& [key, value] : r.scalars)
            if (key == name) return value;
        FAIL("missing result scalar " << name);
        return 0.0;
    };
    CHECK(dmc.energy() == expected.mean);
    CHECK(dmc.energy_error() == expected.error);
    CHECK(scalar("energy_error") == expected.error);
    CHECK(scalar("energy_error_naive") == expected.naive_error);
    CHECK(scalar("energy_tau_int") == expected.tau_int);
    CHECK(expected.error > expected.naive_error);
}

TEST_CASE("DMC converges to exact exciton ground state at cusp",
          "[dmc][physics]") {
    // For a 2D e-h exciton with ψ = exp(-r) (cusp satisfied), the exact
    // ground-state energy is -1.0.  The DMC (with fixed-node on the exact
    // trial wavefunction) must recover this without time-step bias when
    // τ is small and the trial function is the exact ground state.
    //
    // Configuration: q_e=-1, q_h=+1, m=1, ε=1, ψ=exp(-r), E_0 = -1.0.
    WalkerEnsemble ens(/*n_walkers=*/64,
                       {grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)},
                       /*dim=*/2);

    // Initialise walkers near the origin so they sample the bound state.
    numerics::Random rng(42);
    for (auto& w : ens.walkers()) {
        std::vector<double> p(4);  // 2 particles × 2 dims
        p[0] = rng.uniform(-0.5, 0.5);   // e⁻ x
        p[1] = rng.uniform(-0.5, 0.5);   // e⁻ y
        p[2] = rng.uniform(-0.5, 0.5);   // h⁺ x
        p[3] = rng.uniform(-0.5, 0.5);   // h⁺ y
        w.set_positions(p);
    }

    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", -1.0, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), std::move(params)));

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}));
    ham.add_component(std::make_unique<CoulombTerm>(
        grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), 1.0));

    DMCConfig cfg;
    cfg.tau = 0.01;                       // small time step
    cfg.n_equilibration_blocks = 50;
    cfg.n_accumulation_blocks = 200;
    cfg.n_steps = 10;
    cfg.n_substeps = 3;
    cfg.seed = 42;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    // DMC should converge to the exact ground state (-1.0) when the trial
    // function is the exact eigenfunction (zero variance).
    CHECK(dmc.energy() == Catch::Approx(-1.0).margin(0.05));
    CHECK(std::isfinite(dmc.energy_error()));
    CHECK(dmc.acceptance_ratio() > 0.0);
}

TEST_CASE("DMC population stays controlled (no exponential decay)",
          "[dmc][branch]") {
    // Off-cusp e-h trial (c=-0.5) with τ>0: the old death-only branching made
    // the population decay exponentially.  QMCPACK-batched branching keeps it
    // bounded in [n_min, n_max] and near the target.
    constexpr std::size_t target = 64;
    WalkerEnsemble ens(target, {grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}, 2);
    numerics::Random init_rng(7);
    for (auto& w : ens.walkers()) {
        std::vector<double> p(4);
        for (double& x : p) x = init_rng.uniform(-0.5, 0.5);
        w.set_positions(p);
    }

    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", -0.5, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), std::move(params)));

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}));
    ham.add_component(std::make_unique<CoulombTerm>(
        grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), 1.0));

    DMCConfig cfg;
    cfg.tau = 0.02;
    cfg.n_equilibration_blocks = 20;
    cfg.n_accumulation_blocks  = 30;
    cfg.n_steps                = 5;
    cfg.n_substeps             = 1;
    cfg.seed                   = 7;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    auto pop = dmc.block_population();
    REQUIRE(pop.size() == 30);
    const double n_min = static_cast<double>(target) / 5.0 + 1.0;
    const double n_max = 2.0 * static_cast<double>(target) + 1.0;
    double avg = 0.0;
    for (double n : pop) {
        CHECK(n >= n_min);
        CHECK(n <= n_max);
        avg += n;
    }
    avg /= static_cast<double>(pop.size());
    CHECK(avg > static_cast<double>(target) / 4.0);  // did not collapse
    CHECK(std::isfinite(dmc.energy()));
}

TEST_CASE("DMC population control is stable at large tau*sub_steps",
          "[dmc][branch]") {
    // Regression for the 2-D HEG rs=30/40 runs.  The old update
    //   E_T = E_ref − ln(W/N)/feedback
    // had loop gain τ·n_substeps/feedback; at τ·n_substeps = 4 (feedback 1) it
    // overshot into a period-2 limit cycle: population flipping floor↔ceiling
    // every step, E_T ~10 Ha from E, every walker at the branch clamp.  With
    // the dimensionless form E_T = E_ref − ln(W/N)/(g·τ·n_substeps) the gain is
    // 1/g at ANY τ, so the population must fluctuate strictly inside the
    // branch() bounds and E_T must track the energy.
    //
    // The regime must match the HEG's: large τ·n_substeps but SMALL τ·σ(E_L)
    // (there σ ≈ 0.012 Ha), so that only the feedback gain — not branching
    // noise — can drive the population to a bound.  A dilute exciton does it:
    // ε = 10 scales energies by 1/ε² (E₀ = −0.01) and lengths by ε, so τ = 1
    // is a small step physically while τ·n_substeps = 4 (old gain 4 > 2).
    constexpr double eps    = 10.0;
    constexpr std::size_t target = 128;
    WalkerEnsemble ens(target, {grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}, 2);
    numerics::Random init_rng(11);
    for (auto& w : ens.walkers()) {
        std::vector<double> p(4);
        for (double& x : p) x = init_rng.uniform(-eps, eps);
        w.set_positions(p);
    }

    // Half-cusp trial (a = −½·1/ε): nonzero E_L variance, so branching matters.
    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", -0.5 / eps, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), std::move(params)));

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}));
    ham.add_component(std::make_unique<CoulombTerm>(
        grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), eps));

    DMCConfig cfg;
    cfg.tau                    = 1.0;    // τ·n_substeps = 4: old gain 4 (> 2, unstable)
    cfg.n_substeps             = 4;
    cfg.n_equilibration_blocks = 20;
    cfg.n_accumulation_blocks  = 40;
    cfg.n_steps                = 5;
    cfg.seed                   = 11;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    const auto pop = dmc.block_population();
    const auto et  = dmc.block_etrial();
    const auto eb  = dmc.block_energies();
    REQUIRE(pop.size() == 40);
    // Same integer bounds as WalkerSet::branch().
    const double n_min = static_cast<double>(target / 5 + 1);
    const double n_max = static_cast<double>(2 * target + 1);

    // Not pinned: strictly inside the bounds, and it actually fluctuates.
    double p_lo = pop[0], p_hi = pop[0];
    for (double n : pop) {
        CHECK(n > n_min);
        CHECK(n < n_max);
        p_lo = std::min(p_lo, n);
        p_hi = std::max(p_hi, n);
    }
    CHECK(p_hi > p_lo);

    // E_T tracks the energy (the limit cycle put it ~±10 Ha away).
    double et_mean = 0.0, e_mean = 0.0;
    for (std::size_t b = 0; b < et.size(); ++b) {
        et_mean += et[b];
        e_mean  += eb[b];
    }
    et_mean /= static_cast<double>(et.size());
    e_mean  /= static_cast<double>(eb.size());
    CHECK(std::abs(et_mean - e_mean) < 0.01);  // E₀ = −0.01; cycle gave ~±ln(N)/1
    CHECK(std::isfinite(dmc.energy()));
}

TEST_CASE("DMC measurement hook is called during accumulation", "[dmc]") {
    WalkerEnsemble e(1, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, 2);
    std::vector<double> pos = {0.0, 0.0, 1.0, 0.0};
    e.walkers()[0].set_positions(pos);

    DMCConfig c;
    c.tau = 0.0;
    c.n_equilibration_blocks = 0;
    c.n_accumulation_blocks  = 2;
    c.n_steps                = 1;
    c.n_substeps             = 1;
    c.seed                   = 42;

    int hook_calls = 0;

    auto psi_hook = make_psi();
    auto ham_hook = make_ham();
    DMC dmc(c, e, psi_hook, ham_hook);
    dmc.set_measure_hook([&](const WalkerSet&, const TrialWavefunction&) {
        ++hook_calls;
    });
    dmc.run();

    CHECK(hook_calls == 2);
}

// ── Helium atom (Z=2) fixed-node DMC ────────────────────────────────────────
//
// Trial wavefunction with Kato cusp parameters fixed:
//   e-n:  a₁ = -2  (3D cusp: -q_e·q_n·μ/(d-1) = 2/(1) = 2, signed = -2)
//   e-e:  a₁ = +0.5, b₁ = 0.5  (3D cusp: -q_e²/(d-1) = 0.5)
//
// The exact non-relativistic ground-state energy is ≈ -2.904 a.u. [https://arxiv.org/pdf/quant-ph/0203150]
// With the Kato-satisfying trial function the fixed-node DMC should be
// within ~0.02 a.u. of the exact value.

namespace {

ParticleGroup grp3d(std::size_t first, std::size_t last, double charge,
                    double inv_m) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_m, inv_m, inv_m};  // 3D
    return g;
}

}  // namespace

TEST_CASE("DMC converges to helium-3D ground state", "[dmc][physics][helium]") {
    // 3 particles: e₁ [0], e₂ [1], nucleus [2] in 3D.
    constexpr double inv_m_nucleus = 1.0 / 10000.0;
    auto g_elec = grp3d(0, 2, -1.0, 1.0);
    auto g_nucl = grp3d(2, 3, +2.0, inv_m_nucleus);

    WalkerEnsemble ens(/*n_walkers=*/128, {g_elec, g_nucl}, /*dim=*/3);

    // Initialise walkers near the origin (Bohr radius scale ~ 0.5 a.u.).
    numerics::Random rng(42);
    for (auto& w : ens.walkers()) {
        std::vector<double> p(9);  // 3 particles × 3 dims
        for (double& x : p) x = rng.uniform(-0.5, 0.5);
        w.set_positions(p);
    }

    // ψ = exp(u_en(r₁) + u_en(r₂) + u_ee(r₁₂))
    TrialWavefunction psi;
    {
        // e-n: u_en(r) = -2r  (Kato cusp satisfied, no denominator)
        std::vector<Parameter> p_en = {
            {"a1", -2.0, true}, {"b1", 0.0, true}};
        psi.add_component(std::make_unique<PadeJastrow>(
            1, 1, g_elec, g_nucl, std::move(p_en)));
    }
    {
        // e-e: u_ee(r) = 0.5r / (1 + 0.5r)  → u'(0)=0.5 (Kato cusp)
        //       u(∞) = 1 (mild long-range repulsion)
        std::vector<Parameter> p_ee = {
            {"a1", 0.5, true}, {"b1", 0.5, true}};
        psi.add_component(std::make_unique<PadeJastrow>(
            1, 1, g_elec, g_elec, std::move(p_ee)));
    }

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{g_elec, g_nucl}));
    ham.add_component(std::make_unique<CoulombTerm>(
        g_elec, g_nucl, /*epsilon=*/1.0));
    ham.add_component(std::make_unique<CoulombTerm>(
        g_elec, g_elec, /*epsilon=*/1.0));

    DMCConfig cfg;
    cfg.tau = 0.01;
    cfg.n_equilibration_blocks = 64;
    cfg.n_accumulation_blocks = 128;
    cfg.n_steps = 10;
    cfg.n_substeps = 2;
    cfg.seed = 42;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    // DMC energy should be within ~0.02 a.u. of the exact value (-2.904).
    CHECK(dmc.energy() == Catch::Approx(-2.904).margin(0.05));
    CHECK(std::isfinite(dmc.energy_error()));
    CHECK(dmc.acceptance_ratio() > 0.0);
}

TEST_CASE("DMC projects an OFF-CUSP exciton trial to the exact ground state",
          "[dmc][physics]") {
    // c = -0.5 does NOT satisfy the e-h cusp (E_L diverges as 1/r at r->0), yet
    // the exciton ground state is nodeless, so DMC must still project it to the
    // exact E0 = -1.0 — the trial only affects the variance.  This failed
    // (~-0.82, stuck) before two fixes: (1) branching every diffusion sweep
    // (was 1 branch per n_substeps sweeps → under-projection) and (2) clamping
    // the branch-weight energy instead of zeroing it for outliers (the old
    // window killed the branching of the low-E_L walkers that drive projection).
    WalkerEnsemble ens(/*n_walkers=*/256,
                       {grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}, /*dim=*/2);
    numerics::Random rng(42);
    for (auto& w : ens.walkers()) {
        std::vector<double> p(4);
        for (double& x : p) x = rng.uniform(-0.5, 0.5);
        w.set_positions(p);
    }

    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", -0.5, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), std::move(params)));

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}));
    ham.add_component(std::make_unique<CoulombTerm>(
        grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), 1.0));

    DMCConfig cfg;
    cfg.tau = 0.01;
    cfg.n_equilibration_blocks = 150;
    cfg.n_accumulation_blocks  = 200;
    cfg.n_steps = 10;
    cfg.n_substeps = 2;
    cfg.seed = 42;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();
    CHECK(dmc.energy() == Catch::Approx(-1.0).margin(0.05));
}

TEST_CASE("DMC with the effective and suppressed weights projects an off-cusp exciton exactly",
          "[dmc][physics]") {
    // The off-cusp trial above (E_L diverges as 1/r at r -> 0), with the
    // effective step and with the suppressed weight and a measured
    // correlation time.  τ_eff → τ and f → 1 as τ → 0, so at this small τ DMC
    // must reach the exact E0 = -1.0 as the bare weight does.
    for (const Reweighting rw : {Reweighting::effective, Reweighting::suppressed}) {
        WalkerEnsemble ens(/*n_walkers=*/256,
                           {grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}, /*dim=*/2);
        numerics::Random rng(43);
        for (auto& w : ens.walkers()) {
            std::vector<double> p(4);
            for (double& x : p) x = rng.uniform(-0.5, 0.5);
            w.set_positions(p);
        }
        TrialWavefunction psi;
        std::vector<Parameter> params = {{"a", -0.5, true}, {"b", 0.0, true}};
        psi.add_component(std::make_unique<PadeJastrow>(
            1, 1, grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), std::move(params)));
        Hamiltonian ham;
        ham.add_component(std::make_unique<Kinetic>(
            std::vector<ParticleGroup>{grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}));
        ham.add_component(std::make_unique<CoulombTerm>(
            grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0), 1.0));

        DMCConfig cfg;
        cfg.tau = 0.01;
        cfg.n_equilibration_blocks = 150;
        cfg.n_accumulation_blocks  = 200;
        cfg.n_steps = 10;
        cfg.n_substeps = 2;
        cfg.seed = 42;
        cfg.reweighting = rw;

        DMC dmc(cfg, ens, psi, ham);
        dmc.run();
        INFO("reweighting " << to_string(rw));
        CHECK(dmc.energy() == Catch::Approx(-1.0).margin(0.05));
    }
}

TEST_CASE("DMC energies scale exactly with the units under every reweighting",
          "[dmc][reweighting]") {
    // Masses ×λ with the same charges is the same exciton in units λ times
    // smaller: lengths /λ, energies ×λ, τ and θ /λ.  For λ = 4 every rescaled
    // number is exact in floating point, so with the same seed each walker
    // sees the same acceptances, weights and branchings, and E comes out ×4
    // to rounding.  This checks that τ, the masses, θ and the branch cutoff
    // enter only through unit-free combinations: u = Σ(τ/m)(∂ ln Ψ)²,
    // τ_eff/√(τθ), the Umrigar (τ/m)·v², α·√(σ√N/τ)/σ.  Unequal masses
    // (m_h = 2 m_e) and an off-cusp trial (E_L ~ 1/r) exercise the weights.
    auto run = [](Reweighting rw, double lambda) {
        const ParticleGroup e = grp(0, 1, -1.0, 1.0 / lambda);
        const ParticleGroup h = grp(1, 2, +1.0, 0.5 / lambda);
        WalkerEnsemble ens(/*n_walkers=*/64, {e, h}, /*dim=*/2);
        int k = 0;
        for (auto& w : ens.walkers()) {
            const double r = (0.3 + 0.1 * (k++ % 16)) / lambda;
            w.set_positions(std::vector<double>{0.0, 0.0, r, 0.2 * r});
        }
        // u = a1·r + a2·r²: a1 = −1 misses the cusp −4/3 of μ = 2/3.
        TrialWavefunction psi;
        std::vector<Parameter> params = {{"a1", -1.0 * lambda, true},
                                         {"a2", -0.05 * lambda * lambda, true}};
        psi.add_component(std::make_unique<PadeJastrow>(2, 0, e, h, std::move(params)));
        Hamiltonian ham;
        ham.add_component(std::make_unique<Kinetic>(std::vector<ParticleGroup>{e, h}));
        ham.add_component(std::make_unique<CoulombTerm>(e, h, 1.0));
        DMCConfig cfg;
        cfg.tau = 0.08 / lambda;
        cfg.n_equilibration_blocks = 4;
        cfg.n_accumulation_blocks = 8;
        cfg.n_steps = 5;
        cfg.n_substeps = 2;
        cfg.seed = 11;
        cfg.reweighting = rw;
        cfg.correlation_max_sweeps = 1024;
        DMC dmc(cfg, ens, psi, ham);
        dmc.run();
        return dmc.run_result();
    };
    auto scalar = [](const RunResult& r, const std::string& name) {
        for (const auto& [n, v] : r.scalars)
            if (n == name) return v;
        FAIL("scalar not found: " << name);
        return 0.0;
    };
    for (const Reweighting rw :
         {Reweighting::bare, Reweighting::effective, Reweighting::suppressed}) {
        INFO("reweighting " << to_string(rw));
        const RunResult a = run(rw, 1.0);
        const RunResult b = run(rw, 4.0);
        CHECK(scalar(b, "energy") == Catch::Approx(4.0 * scalar(a, "energy")).epsilon(1e-12));
        CHECK(scalar(b, "energy_error") ==
              Catch::Approx(4.0 * scalar(a, "energy_error")).epsilon(1e-10));
        CHECK(scalar(b, "acceptance_ratio") == scalar(a, "acceptance_ratio"));
        CHECK(scalar(b, "branch_cutoff_energy") ==
              Catch::Approx(4.0 * scalar(a, "branch_cutoff_energy")).epsilon(1e-12));
        CHECK(scalar(b, "branch_cutoff_fraction") == scalar(a, "branch_cutoff_fraction"));
        if (rw == Reweighting::bare) continue;
        CHECK(scalar(b, "tau_eff_ratio") == Catch::Approx(scalar(a, "tau_eff_ratio")).epsilon(1e-12));
        if (rw != Reweighting::suppressed) continue;
        CHECK(scalar(b, "correlation_time") ==
              Catch::Approx(0.25 * scalar(a, "correlation_time")).epsilon(1e-12));
        CHECK(scalar(b, "suppression_mean") ==
              Catch::Approx(scalar(a, "suppression_mean")).epsilon(1e-12));
    }
}

// ── Helium 2D (Z=2) fixed-node DMC ────────────────────────────────────────
//
// Kato cusp in 2D (d-1 = 1):
//   e-n: a₁ = 2·μ·q_e·q_n / (ε·1) = 2·1·(-1)(+2) / 1 = -4
//   e-e: a₁ = 2·μ·q_e²    / (ε·1) = 2·0.5·1     / 1 = +1.0
//
// Exact ground state: -11.90 a.u. [https://arxiv.org/pdf/quant-ph/0203150].

namespace {

ParticleGroup grp2d(std::size_t first, std::size_t last, double charge,
                    double inv_m) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_m, inv_m};  // 2D
    return g;
}

}  // namespace

TEST_CASE("DMC converges to helium-2D ground state", "[dmc][physics][helium]") {
    // 3 particles: e₁ [0], e₂ [1], nucleus [2] in 2D.
    constexpr double inv_m_nucleus = 1.0 / 10000.0;
    auto g_elec = grp2d(0, 2, -1.0, 1.0);
    auto g_nucl = grp2d(2, 3, +2.0, inv_m_nucleus);

    WalkerEnsemble ens(/*n_walkers=*/128, {g_elec, g_nucl}, /*dim=*/2);

    // Initialise walkers near the origin (2D Bohr radius scale ~ 0.3 a.u.).
    numerics::Random rng(42);
    for (auto& w : ens.walkers()) {
        std::vector<double> p(6);  // 3 particles × 2 dims
        for (double& x : p) x = rng.uniform(-0.3, 0.3);
        w.set_positions(p);
    }

    // ψ = exp(u_en(r₁) + u_en(r₂) + u_ee(r₁₂))
    TrialWavefunction psi;
    {
        // e-n: u_en(r) = -4r / (1 + 0.145r)  → u'(0)=-4 (Kato cusp)
        std::vector<Parameter> p_en = {
            {"a1", -4.0, true}, {"b1", 0.145, true}};
        psi.add_component(std::make_unique<PadeJastrow>(
            1, 1, g_elec, g_nucl, std::move(p_en)));
    }
    {
        // e-e: u_ee(r) = 1.0r / (1 + 1.27r)  → u'(0)=1.0 (Kato cusp)
        std::vector<Parameter> p_ee = {
            {"a1", 1.0, true}, {"b1", 1.27, true}};
        psi.add_component(std::make_unique<PadeJastrow>(
            1, 1, g_elec, g_elec, std::move(p_ee)));
    }

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{g_elec, g_nucl}));
    ham.add_component(std::make_unique<CoulombTerm>(
        g_elec, g_nucl, /*epsilon=*/1.0));
    ham.add_component(std::make_unique<CoulombTerm>(
        g_elec, g_elec, /*epsilon=*/1.0));

    // Smaller time step than 3D: the deeper binding and stronger forces
    // require finer Langevin integration for stable acceptance.
    DMCConfig cfg;
    cfg.tau = 0.005;
    cfg.n_equilibration_blocks = 64;
    cfg.n_accumulation_blocks = 128;
    cfg.n_steps = 10;
    cfg.n_substeps = 2;
    cfg.seed = 42;

    DMC dmc(cfg, ens, psi, ham);
    dmc.run();

    // DMC should recover a value near the exact ground state (-11.90).
    CHECK(dmc.energy() == Catch::Approx(-11.90).margin(0.05));
    CHECK(std::isfinite(dmc.energy_error()));
    CHECK(dmc.acceptance_ratio() > 0.0);
}

TEST_CASE("DMC checkpoint hook fires at eq and the quarter milestones",
          "[dmc][checkpoint]") {
    WalkerEnsemble ens(/*n_walkers=*/2,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    DMCConfig cfg;
    cfg.tau = 0.0;
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 8;   // milestones after blocks 2, 4, 6
    cfg.n_steps = 2;
    cfg.n_substeps = 1;
    cfg.seed = 42;

    std::vector<std::pair<std::string, int>> fired;
    DMC dmc(cfg, ens, psi, ham);
    dmc.set_checkpoint_hook([&](const WalkerSet& ws, std::string_view tag,
                                int blocks_done) {
        CHECK(ws.num_walkers() == 2);   // tau=0: population is stable
        fired.emplace_back(std::string(tag), blocks_done);
    });
    dmc.run();

    REQUIRE(fired.size() == 4);
    CHECK(fired[0] == std::pair<std::string, int>{"eq", 0});
    CHECK(fired[1] == std::pair<std::string, int>{"a25", 2});
    CHECK(fired[2] == std::pair<std::string, int>{"a50", 4});
    CHECK(fired[3] == std::pair<std::string, int>{"a75", 6});
}

TEST_CASE("DMC checkpoint hook drops coinciding milestones for tiny runs",
          "[dmc][checkpoint]") {
    // n_acc=2: 25% and 75% round onto other milestones -> only eq + a50 fire.
    WalkerEnsemble ens(/*n_walkers=*/1,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    ens.walkers()[0].set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    DMCConfig cfg;
    cfg.tau = 0.0;
    cfg.n_equilibration_blocks = 1;
    cfg.n_accumulation_blocks = 2;
    cfg.n_steps = 1;
    cfg.n_substeps = 1;
    cfg.seed = 42;

    std::vector<std::string> tags;
    DMC dmc(cfg, ens, psi, ham);
    dmc.set_checkpoint_hook([&](const WalkerSet&, std::string_view tag, int) {
        tags.emplace_back(tag);
    });
    dmc.run();

    REQUIRE(tags.size() == 2);
    CHECK(tags[0] == "eq");
    CHECK(tags[1] == "a50");
}
