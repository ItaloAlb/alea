// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "numerics/statistics.h"

#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "wavefunction/trial_wavefunction.h"
#include "wavefunction/pade_jastrow.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "hamiltonian/coulomb.h"
#include "qmc/vmc.h"
#include "qmc/reblock_diagnostics.h"

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

// Anisotropic variant: per-dimension inverse masses.
ParticleGroup grp_aniso(std::size_t first, std::size_t last, double charge,
                        double inv_mx, double inv_my) {
    ParticleGroup g;
    g.name = "g";
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.inv_mass = {inv_mx, inv_my};
    return g;
}

TrialWavefunction make_psi() {
    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", 1.0, true}, {"b", 0.0, true}};  // u = r
    psi.add_component(std::make_unique<PadeJastrow>(
        1, 1, grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0), std::move(params)));
    return psi;
}

TrialWavefunction make_psi(const ParticleGroup& g1, const ParticleGroup& g2) {
    TrialWavefunction psi;
    std::vector<Parameter> params = {{"a", 1.0, true}, {"b", 0.0, true}};
    psi.add_component(std::make_unique<PadeJastrow>(1, 1, g1, g2, std::move(params)));
    return psi;
}

Hamiltonian make_ham() {
    Hamiltonian h;
    h.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}));
    h.add_component(std::make_unique<CoulombTerm>(grp(0, 1, +1.0, 1.0),
                                                  grp(1, 2, -1.0, 1.0), 1.0));
    return h;
}

Hamiltonian make_ham(const ParticleGroup& g1, const ParticleGroup& g2) {
    Hamiltonian h;
    h.add_component(std::make_unique<Kinetic>(std::vector<ParticleGroup>{g1, g2}));
    h.add_component(std::make_unique<CoulombTerm>(g1, g2, 1.0));
    return h;
}

}  // namespace

TEST_CASE("VMC with tau=0 measures the exact local energy of the start config",
          "[vmc]") {
    // One walker, p0=(0,0), p1=(2,0): E_L = K + V = -1.5 + -0.5 = -2.0.
    WalkerEnsemble ens(/*n_walkers=*/1, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};  // AoS [p*dim + d]
    ens.walkers()[0].set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    VMCConfig cfg;
    cfg.tau = 0.0;  // zero step → no moves → deterministic
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 3;
    cfg.n_steps = 4;
    cfg.n_substeps = 1;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    CHECK(vmc.energy() == Catch::Approx(-2.0));
    CHECK(vmc.acceptance_ratio() == Catch::Approx(1.0));
    CHECK(vmc.energy_error() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.energy_variance() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.block_energies().size() == 3);  // n_accumulation_blocks
}

TEST_CASE("VMC with tau>0 runs the stochastic chain and produces sane stats",
          "[vmc]") {
    WalkerEnsemble ens(/*n_walkers=*/8,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    VMCConfig cfg;
    cfg.tau = 0.5;  // real diffusion → walkers move and decorrelate
    cfg.n_equilibration_blocks = 5;
    cfg.n_accumulation_blocks = 10;
    cfg.n_steps = 5;
    cfg.n_substeps = 2;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    CHECK(vmc.block_energies().size() == 10);
    CHECK(std::isfinite(vmc.energy()));
    CHECK(vmc.acceptance_ratio() > 0.0);
    CHECK(vmc.acceptance_ratio() <= 1.0);
    CHECK(vmc.energy_variance() > 0.0);  // sampling spread is real
    CHECK(vmc.energy_error() > 0.0);
}

TEST_CASE("VMC with tau=0 and n_threads>1 matches single-thread result",
          "[vmc][threads]") {
    // 4 walkers, 4 threads: deterministic tau=0 must give exact same energy
    // regardless of how walkers are distributed across threads.
    WalkerEnsemble ens(/*n_walkers=*/4,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    VMCConfig cfg;
    cfg.tau = 0.0;
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 8;
    cfg.n_steps = 4;
    cfg.n_substeps = 1;
    cfg.seed = 42;
    cfg.n_threads = 4;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    // tau=0 → no moves → deterministic; E_L = K + V = -2.0 for all walkers.
    CHECK(vmc.energy() == Catch::Approx(-2.0).margin(1e-12));
    CHECK(vmc.energy_error() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.energy_variance() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.acceptance_ratio() == Catch::Approx(1.0));
    // 4 threads × 8 accumulation blocks = 32 block energies
    CHECK(vmc.block_energies().size() == 32);
}

TEST_CASE("VMC with tau>0 and n_threads>1 produces sane stats",
          "[vmc][threads]") {
    // 8 walkers, 4 threads: stochastic run must finish and produce finite stats.
    WalkerEnsemble ens(/*n_walkers=*/8,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    VMCConfig cfg;
    cfg.tau = 0.5;
    cfg.n_equilibration_blocks = 5;
    cfg.n_accumulation_blocks = 10;
    cfg.n_steps = 5;
    cfg.n_substeps = 2;
    cfg.seed = 42;
    cfg.n_threads = 4;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    // 4 threads × 10 accumulation blocks = 40 block energies
    CHECK(vmc.block_energies().size() == 40);
    CHECK(std::isfinite(vmc.energy()));
    CHECK(vmc.acceptance_ratio() > 0.0);
    CHECK(vmc.acceptance_ratio() <= 1.0);
    CHECK(vmc.energy_variance() > 0.0);
    CHECK(vmc.energy_error() > 0.0);
}

namespace {
double scalar(const RunResult& r, std::string_view name) {
    for (const auto& [key, value] : r.scalars)
        if (key == name) return value;
    FAIL("missing result scalar " << name);
    return 0.0;
}
}  // namespace

TEST_CASE("VMC reports the reblocked error of its per-thread block chains",
          "[vmc][threads][reblock]") {
    // One-step blocks are correlated, so the reblocked error differs from the
    // naive one; the reported error must be the pooled per-thread reblocking
    // of the block energies the run writes out.
    WalkerEnsemble ens(/*n_walkers=*/8,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);
    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    VMCConfig cfg;
    cfg.tau = 0.5;
    cfg.n_equilibration_blocks = 5;
    cfg.n_accumulation_blocks = 256;
    cfg.n_steps = 1;
    cfg.n_substeps = 1;
    cfg.seed = 42;
    cfg.n_threads = 4;
    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    const auto expected = numerics::reblock_chains(vmc.block_energies(), 4);
    const RunResult r = vmc.run_result();
    CHECK(vmc.energy() == expected.mean);
    CHECK(vmc.energy_error() == expected.error);
    CHECK(scalar(r, "energy_error") == expected.error);
    CHECK(scalar(r, "energy_error_naive") == expected.naive_error);
    CHECK(scalar(r, "energy_tau_int") == expected.tau_int);
    CHECK(scalar(r, "energy_error_reliable") ==
          (expected.plateau_found && expected.selected_n_blocks >= 16 &&
           1024.0 / (1.0 + 2.0 * expected.tau_int) >= 50.0 ? 1.0 : 0.0));
    CHECK(expected.error > expected.naive_error);
}

TEST_CASE("A VMC/DMC error bar needs 50 effective blocks to count as reliable",
          "[vmc][dmc][reblock]") {
    // Short runs underestimate their own tau_int; the WS2 exciton DMC runs that
    // estimated 19-49 effective blocks had errors ~30 % too small.
    numerics::ReblockResult stats;
    stats.plateau_found = true;
    stats.selected_n_blocks = 64;
    const std::size_t n = 512;
    stats.tau_int = (512.0 / 49.0 - 1.0) / 2.0;  // n_eff = 49
    CHECK(effective_blocks(stats, n) == Catch::Approx(49.0));
    CHECK_FALSE(error_reliable(stats, n));
    stats.tau_int = (512.0 / 51.0 - 1.0) / 2.0;  // n_eff = 51
    CHECK(error_reliable(stats, n));
    stats.plateau_found = false;
    CHECK_FALSE(error_reliable(stats, n));
    stats.plateau_found = true;
    stats.selected_n_blocks = 15;
    CHECK_FALSE(error_reliable(stats, n));
}

TEST_CASE("VMC threads of neighbouring seeds draw different random numbers",
          "[vmc][threads][seed]") {
    // Identical walkers on two threads.  With seed + thread streams, thread 1
    // of seed 42 and thread 0 of seed 43 would run the same chain and give
    // the same block energies; hashed streams keep them apart.
    auto run = [](std::uint64_t seed) {
        WalkerEnsemble ens(/*n_walkers=*/8,
                           {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
        std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
        for (auto& w : ens.walkers()) w.set_positions(pos);
        TrialWavefunction psi = make_psi();
        Hamiltonian ham = make_ham();
        VMCConfig cfg;
        cfg.tau = 0.5;
        cfg.n_equilibration_blocks = 2;
        cfg.n_accumulation_blocks = 4;
        cfg.n_steps = 2;
        cfg.seed = seed;
        cfg.n_threads = 2;
        VMC vmc(cfg, ens, psi, ham);
        vmc.run();
        const auto b = vmc.block_energies();
        return std::vector<double>(b.begin(), b.end());
    };
    const std::vector<double> a = run(42), b = run(43);
    REQUIRE(a.size() == 8);
    const std::vector<double> thread1_of_42(a.begin() + 4, a.end());
    const std::vector<double> thread0_of_43(b.begin(), b.begin() + 4);
    CHECK(thread1_of_42 != thread0_of_43);
    CHECK(run(42) == a);  // still reproducible
}

TEST_CASE("VMC with Langevin and tau=0 is deterministic and exact",
          "[vmc][langevin]") {
    WalkerEnsemble ens(/*n_walkers=*/1, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)},
                       /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    ens.walkers()[0].set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    VMCConfig cfg;
    cfg.tau = 0.0;
    cfg.use_drift = true;  // Langevin dynamics
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks = 3;
    cfg.n_steps = 4;
    cfg.n_substeps = 1;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    // tau=0 → sigma=0 → no noise, no drift → E_L = -2.0 exactly.
    CHECK(vmc.energy() == Catch::Approx(-2.0));
    CHECK(vmc.acceptance_ratio() == Catch::Approx(1.0));
    CHECK(vmc.energy_error() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.energy_variance() == Catch::Approx(0.0).margin(1e-12));
    CHECK(vmc.block_energies().size() == 3);
}

TEST_CASE("VMC with Langevin and tau>0 runs and produces sane stats",
          "[vmc][langevin]") {
    WalkerEnsemble ens(/*n_walkers=*/8,
                       {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi();
    Hamiltonian ham = make_ham();

    VMCConfig cfg;
    cfg.tau = 0.5;
    cfg.use_drift = true;  // Langevin dynamics
    cfg.n_equilibration_blocks = 5;
    cfg.n_accumulation_blocks = 10;
    cfg.n_steps = 5;
    cfg.n_substeps = 2;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    CHECK(vmc.block_energies().size() == 10);
    CHECK(std::isfinite(vmc.energy()));
    CHECK(vmc.acceptance_ratio() > 0.0);
    CHECK(vmc.acceptance_ratio() <= 1.0);
    CHECK(vmc.energy_variance() > 0.0);
    CHECK(vmc.energy_error() > 0.0);
}

TEST_CASE("VMC Langevin acceptance is higher than symmetric for the same tau",
          "[vmc][langevin]") {
    // Langevin drift pushes walkers toward higher |ψ|² → higher acceptance.
    const std::size_t nw = 64;
    WalkerEnsemble ens(nw, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi_sym = make_psi();
    TrialWavefunction psi_lan = make_psi();
    Hamiltonian ham_sym = make_ham();
    Hamiltonian ham_lan = make_ham();

    VMCConfig cfg_sym;
    cfg_sym.tau = 0.3;
    cfg_sym.use_drift = false;
    cfg_sym.n_equilibration_blocks = 10;
    cfg_sym.n_accumulation_blocks = 16;
    cfg_sym.n_steps = 5;
    cfg_sym.n_substeps = 3;
    cfg_sym.seed = 42;

    VMCConfig cfg_lan = cfg_sym;
    cfg_lan.use_drift = true;

    WalkerEnsemble ens2(nw, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, /*dim=*/2);
    for (auto& w : ens2.walkers()) w.set_positions(pos);

    VMC vmc_sym(cfg_sym, ens, psi_sym, ham_sym);
    VMC vmc_lan(cfg_lan, ens2, psi_lan, ham_lan);

    vmc_sym.run();
    vmc_lan.run();

    CHECK(vmc_sym.acceptance_ratio() > 0.0);
    CHECK(vmc_lan.acceptance_ratio() > 0.0);
    // Langevin drift should improve acceptance over symmetric Metropolis.
    CHECK(vmc_lan.acceptance_ratio() > vmc_sym.acceptance_ratio());
}

TEST_CASE("VMC stores per-dimension tau_inv_mass for anisotropic masses",
          "[vmc][anisotropy]") {
    // inv_mass = {1.0, 0.3}: tau/m_x != tau/m_y.
    auto g1 = grp_aniso(0, 1, +1.0, /*inv_mx=*/1.0, /*inv_my=*/0.3);
    auto g2 = grp_aniso(1, 2, -1.0, /*inv_mx=*/0.5, /*inv_my=*/2.0);
    WalkerEnsemble ens(/*n_walkers=*/1, {g1, g2}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    ens.walkers()[0].set_positions(pos);

    TrialWavefunction psi = make_psi(g1, g2);
    Hamiltonian ham = make_ham(g1, g2);

    VMCConfig cfg;
    cfg.tau = 0.2;

    VMC vmc(cfg, ens, psi, ham);

    // tau_inv_mass_ layout: [p * dim + d]
    auto tm = vmc.tau_inv_mass();
    CHECK(tm.size() == 2 * 2);  // np=2, dim=2

    // Particle 0: inv_mass = {1.0, 0.3}
    CHECK(tm[0 * 2 + 0] == Catch::Approx(0.2 * 1.0));  // x: tau / m_x
    CHECK(tm[0 * 2 + 1] == Catch::Approx(0.2 * 0.3));  // y: tau / m_y

    // Particle 1: inv_mass = {0.5, 2.0}
    CHECK(tm[1 * 2 + 0] == Catch::Approx(0.2 * 0.5));  // x: tau / m_x
    CHECK(tm[1 * 2 + 1] == Catch::Approx(0.2 * 2.0));  // y: tau / m_y
}

TEST_CASE("VMC with anisotropic masses runs and produces sane stats",
          "[vmc][anisotropy]") {
    auto g1 = grp_aniso(0, 1, +1.0, /*inv_mx=*/1.0, /*inv_my=*/0.3);
    auto g2 = grp_aniso(1, 2, -1.0, /*inv_mx=*/1.0, /*inv_my=*/0.3);
    WalkerEnsemble ens(/*n_walkers=*/8, {g1, g2}, /*dim=*/2);
    std::vector<double> pos = {0.0, 0.0, 2.0, 0.0};
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi = make_psi(g1, g2);
    Hamiltonian ham = make_ham(g1, g2);

    VMCConfig cfg;
    cfg.tau = 0.3;
    cfg.use_drift = true;
    cfg.n_equilibration_blocks = 5;
    cfg.n_accumulation_blocks = 10;
    cfg.n_steps = 5;
    cfg.n_substeps = 2;
    cfg.seed = 42;

    VMC vmc(cfg, ens, psi, ham);
    vmc.run();

    CHECK(vmc.block_energies().size() == 10);
    CHECK(std::isfinite(vmc.energy()));
    CHECK(vmc.acceptance_ratio() > 0.0);
    CHECK(vmc.acceptance_ratio() <= 1.0);
    CHECK(vmc.energy_variance() > 0.0);
    CHECK(vmc.energy_error() > 0.0);
}

TEST_CASE("VMC measurement hook is called during accumulation", "[vmc]") {
    WalkerEnsemble e(1, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, 2);
    std::vector<double> pos = {0.0, 0.0, 1.0, 0.0};
    e.walkers()[0].set_positions(pos);

    VMCConfig c;
    c.tau = 0.0;
    c.n_equilibration_blocks = 0;
    c.n_accumulation_blocks  = 2;
    c.n_steps                = 1;
    c.n_substeps             = 1;
    c.seed                   = 42;

    int hook_calls = 0;
    double last_le = 0.0;

    auto psi_hook = make_psi();
    auto ham_hook = make_ham(grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0));
    VMC vmc(c, e, psi_hook, ham_hook);
    vmc.set_measure_hook([&](const WalkerSet& wset, const TrialWavefunction&) {
        ++hook_calls;
        last_le = wset.local_energies()[0];
    });
    vmc.run();

    CHECK(hook_calls == 2);
    CHECK(last_le == Catch::Approx(-3.0));  // u=r: K=-2, V=-1 → E_L=-3
}

TEST_CASE("VMC measure hook skips equilibration steps", "[vmc]") {
    // The optimiser builds H̄/S̄ from every hook call, so equilibration (the
    // transient from the starting configuration) must never reach it.
    WalkerEnsemble e(1, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, 2);
    std::vector<double> pos = {0.0, 0.0, 1.0, 0.0};
    e.walkers()[0].set_positions(pos);

    VMCConfig c;
    c.tau = 0.1;
    c.n_equilibration_blocks = 3;
    c.n_accumulation_blocks  = 2;
    c.n_steps                = 4;
    c.n_substeps             = 1;
    c.seed                   = 42;

    int hook_calls = 0;
    auto psi_hook = make_psi();
    auto ham_hook = make_ham(grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0));
    VMC vmc(c, e, psi_hook, ham_hook);
    vmc.set_measure_hook([&](const WalkerSet&, const TrialWavefunction&) {
        ++hook_calls;
    });
    vmc.run();

    CHECK(hook_calls == 2 * 4);   // accumulation only, not 5 * 4
}

TEST_CASE("VMC runs without crash when no hook is set", "[vmc]") {
    WalkerEnsemble e(1, {grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0)}, 2);
    std::vector<double> pos = {0.0, 0.0, 1.0, 0.0};
    e.walkers()[0].set_positions(pos);

    VMCConfig c;
    c.tau = 0.0;
    c.n_equilibration_blocks = 0;
    c.n_accumulation_blocks  = 1;
    c.n_steps                = 1;
    c.n_substeps             = 1;
    c.seed                   = 42;

    auto psi_nohook = make_psi();
    auto ham_nohook = make_ham(grp(0, 1, +1.0, 1.0), grp(1, 2, -1.0, 1.0));
    VMC vmc(c, e, psi_nohook, ham_nohook);
    vmc.run();
    CHECK(std::isfinite(vmc.energy()));
}

