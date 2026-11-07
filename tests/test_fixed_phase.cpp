// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Fixed-phase infrastructure: complex trial wavefunctions ψ = e^{u + iφ}.
//
// The test vehicle is the Galilean boost ψ = e^{iK·R_cm} ψ₀, R_cm = Σ m_i r_i / M.
// For a translation-invariant H its fixed-phase energy is exactly E₀ + K²/2M, so
// every layer (phase planes, kinetic energy, VMC, DMC) has an exact oracle.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "qmc/dmc.h"
#include "qmc/vmc.h"
#include "population/walker_set.h"
#include "wavefunction/center_of_mass_momentum.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;

namespace {

ParticleGroup grp(std::string name, std::size_t first, std::size_t last,
                  double charge, double mass) {
    ParticleGroup g;
    g.name = std::move(name);
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.count = last - first;
    g.mass = {mass, mass};
    g.inv_mass = {1.0 / mass, 1.0 / mass};
    return g;
}

// Exciton with unequal masses: e (m=1) at index 0, h (m=3) at index 1.  M = 4.
std::vector<ParticleGroup> exciton_groups() {
    return {grp("e", 0, 1, -1.0, 1.0), grp("h", 1, 2, +1.0, 3.0)};
}

}  // namespace

TEST_CASE("A boosted trial is complex and carries grad phi = m_i K / M",
          "[fixed_phase]") {
    const std::size_t nw = 3, np = 2, dim = 2;
    WalkerSet ws(nw, np, dim);
    auto pos = ws.positions();
    for (std::size_t k = 0; k < pos.size(); ++k) pos[k] = 0.1 * static_cast<double>(k);
    ws.dist_tables().build_all(ws.positions());

    const std::vector<double> K = {0.8, -0.4};
    TrialWavefunction psi;
    psi.add_component(std::make_unique<CenterOfMassMomentum>(exciton_groups(), K));
    REQUIRE(psi.is_complex());

    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);
    REQUIRE(ws.has_phase());

    const auto pg = ws.phase_gradients();
    const auto pl = ws.phase_laplacians();
    const double m[2] = {1.0, 3.0};
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w) {
                const std::size_t idx = p * dim * nw + d * nw + w;
                CHECK(pg[idx] == Catch::Approx(m[p] * K[d] / 4.0));
                CHECK(pl[idx] == Catch::Approx(0.0).margin(1e-15));
            }
    // |ψ| is untouched: the boost is a pure phase.
    for (std::size_t w = 0; w < nw; ++w) CHECK(ws.log_psi()[w] == 0.0);
}

TEST_CASE("A single-particle move of a boosted trial shifts the phase by k_p . dr",
          "[fixed_phase]") {
    const std::size_t nw = 3, np = 2, dim = 2;
    WalkerSet ws(nw, np, dim);
    ws.dist_tables().build_all(ws.positions());

    const std::vector<double> K = {0.8, -0.4};
    TrialWavefunction psi;
    psi.add_component(std::make_unique<CenterOfMassMomentum>(exciton_groups(), K));
    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);

    // Move the hole (p = 1, k = 3K/4) by a different Δr in each walker.
    const std::size_t p = 1;
    const double k[2] = {0.75 * K[0], 0.75 * K[1]};
    std::vector<double> new_pos(dim * nw);
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w)
            new_pos[d * nw + w] = 0.3 * static_cast<double>(w + 1) - 0.5 * static_cast<double>(d);
    ws.propose_move_batch(p, new_pos);
    psi.ratio_batch(ws, p);

    for (std::size_t w = 0; w < nw; ++w) {
        const double dphi = k[0] * new_pos[w] + k[1] * new_pos[nw + w];  // old r = 0
        CHECK(ws.phase_diffs()[w] == Catch::Approx(dphi));
        CHECK(ws.log_ratios()[w] == 0.0);                 // |ψ'/ψ| = 1
        for (std::size_t d = 0; d < dim; ++d)
            CHECK(ws.new_phase_grad()[d * nw + w] == Catch::Approx(k[d]));
    }

    // Commit only walkers 0 and 2.  A sentinel in the committed plane shows
    // which walkers took the proposed ∇φ and which kept their old one.
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w)
            ws.phase_gradients()[p * dim * nw + d * nw + w] = 99.0;
    bool accept[3] = {true, false, true};
    const std::span<const bool> mask(accept, nw);
    psi.apply_move_batch(ws, p, mask);
    ws.apply_move(p, new_pos, ws.new_grad(), ws.new_lap(), mask);

    for (std::size_t d = 0; d < dim; ++d) {
        CHECK(ws.phase_gradients()[p * dim * nw + d * nw + 0] == Catch::Approx(k[d]));
        CHECK(ws.phase_gradients()[p * dim * nw + d * nw + 1] == 99.0);
        CHECK(ws.phase_gradients()[p * dim * nw + d * nw + 2] == Catch::Approx(k[d]));
    }
}

TEST_CASE("Boosting adds K^2/2M to Re E_L and -sum (1/m) grad u . grad phi to Im E_L",
          "[fixed_phase]") {
    // ψ₀ = e^{u(r_eh)} · φ_e(r_e) φ_h(r_h), ψ = e^{iK·R_cm} ψ₀.  The one-body
    // Gaussians (1×1 determinants) break translation invariance: with the
    // Padé alone Σ_i ∇u_i·k_i/m_i = (K/M)·Σ_i ∇u_i = 0 and Im E_L would vanish
    // identically, leaving that check vacuous.
    // With ∇u_i and k_i = ∇_i φ:
    //   Re E_L(ψ) = Re E_L(ψ₀) + Σ_i |k_i|²/2m_i = E_L(ψ₀) + K²/2M
    //   Im E_L(ψ) = −Σ_i (1/2m_i)[∇²φ + 2∇u_i·k_i] = −Σ_i ∇u_i·k_i / m_i
    const std::size_t nw = 3, np = 2, dim = 2;
    const auto groups = exciton_groups();
    const std::vector<double> K = {0.8, -0.4};
    const double m[2] = {1.0, 3.0};
    const double K2_over_2M = (K[0] * K[0] + K[1] * K[1]) / (2.0 * 4.0);

    auto make_ws = [&] {
        WalkerSet ws(nw, np, dim);
        auto pos = ws.positions();
        for (std::size_t k = 0; k < pos.size(); ++k)
            pos[k] = 0.37 * std::sin(1.3 * static_cast<double>(k) + 0.2);
        ws.dist_tables().build_all(ws.positions());
        return ws;
    };
    auto pade = [&] {
        std::vector<Parameter> params = {{"a", -1.0, true}, {"b", 0.5, true}};
        return std::make_unique<PadeJastrow>(1, 1, groups[0], groups[1],
                                             std::move(params));
    };
    auto one_body = [&] {
        std::vector<SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 1, std::make_unique<GaussianOrbitalSet>(
                                    dim, std::vector<double>{0.2, -0.1},
                                    std::vector<double>{0.7})});
        blocks.push_back({1, 2, std::make_unique<GaussianOrbitalSet>(
                                    dim, std::vector<double>{-0.3, 0.4},
                                    std::vector<double>{0.4})});
        return std::make_unique<SlaterDeterminant>(std::move(blocks));
    };
    Hamiltonian H;
    H.add_component(std::make_unique<Kinetic>(groups));

    WalkerSet ws0 = make_ws();
    TrialWavefunction psi0;
    psi0.add_component(pade());
    psi0.add_component(one_body());
    psi0.prepare(nw, np, dim);
    psi0.evaluate_batch(ws0);
    H.evaluate_batch(ws0);

    WalkerSet ws = make_ws();
    TrialWavefunction psi;
    psi.add_component(pade());
    psi.add_component(one_body());
    psi.add_component(std::make_unique<CenterOfMassMomentum>(groups, K));
    psi.prepare(nw, np, dim);
    psi.evaluate_batch(ws);
    H.evaluate_batch(ws);
    double im_abs = 0.0;

    for (std::size_t w = 0; w < nw; ++w) {
        CHECK(ws.local_energies()[w] ==
              Catch::Approx(ws0.local_energies()[w] + K2_over_2M));
        double im = 0.0;
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                im -= ws.gradients()[p * dim * nw + d * nw + w] *
                      (m[p] * K[d] / 4.0) / m[p];
        CHECK(ws.local_energies_imag()[w] == Catch::Approx(im));
        im_abs += std::abs(im);
    }
    REQUIRE(im_abs > 1e-3);   // the Im check above is not vacuous
    // The real trial never grows the imaginary column.
    CHECK_FALSE(ws0.has_phase());
}

namespace {

std::unique_ptr<WavefunctionComponent> exciton_pade(const std::vector<ParticleGroup>& g) {
    // e–h Padé correlation.  Not cusp-exact on purpose: these tests compare
    // runs against each other, so any smooth ψ₀ works.
    std::vector<Parameter> params = {{"a", -0.75, true}, {"b", 0.6, true}};
    return std::make_unique<PadeJastrow>(1, 1, g[0], g[1], std::move(params));
}

Hamiltonian exciton_hamiltonian(const std::vector<ParticleGroup>& g) {
    Hamiltonian h;
    h.add_component(std::make_unique<Kinetic>(g));
    h.add_component(std::make_unique<CoulombTerm>(g[0], g[1], 1.0));
    return h;
}

void spread_walkers(WalkerEnsemble& ens) {
    std::size_t k = 0;
    for (Walker& w : ens.walkers()) {
        std::vector<double> r(w.positions().size());
        for (double& x : r) x = 1.1 * std::sin(0.7 * static_cast<double>(k++) + 0.3);
        w.set_positions(r);
    }
}

VMCConfig short_vmc(bool drift) {
    VMCConfig cfg;
    cfg.tau = 0.3;
    cfg.use_drift = drift;
    cfg.n_equilibration_blocks = 5;
    cfg.n_accumulation_blocks = 20;
    cfg.n_steps = 10;
    cfg.n_substeps = 2;
    cfg.n_threads = 1;
    cfg.seed = 7;
    return cfg;
}

}  // namespace

TEST_CASE("VMC of a boosted exciton is the rest-frame VMC shifted by K^2/2M",
          "[fixed_phase][vmc]") {
    // The phase never enters the Metropolis walk (|ψ'/ψ| and ∇u are unchanged),
    // so with the same seed both runs visit IDENTICAL configurations and every
    // sample's Re E_L differs by exactly K²/2M; Im E_L ≡ 0 because ψ₀ depends
    // only on r_eh (see the kinetic test).  Holds for plain and Langevin moves.
    const auto groups = exciton_groups();
    const std::vector<double> K = {0.8, -0.4};
    const double K2_over_2M = (K[0] * K[0] + K[1] * K[1]) / (2.0 * 4.0);
    const Hamiltonian H = exciton_hamiltonian(groups);

    for (bool drift : {false, true}) {
        WalkerEnsemble ens0(16, groups, 2), ensK(16, groups, 2);
        spread_walkers(ens0);
        spread_walkers(ensK);

        TrialWavefunction psi0;
        psi0.add_component(exciton_pade(groups));
        TrialWavefunction psiK;
        psiK.add_component(exciton_pade(groups));
        psiK.add_component(std::make_unique<CenterOfMassMomentum>(groups, K));

        VMC vmc0(short_vmc(drift), ens0, psi0, H);
        VMC vmcK(short_vmc(drift), ensK, psiK, H);
        vmc0.run();
        vmcK.run();

        CHECK(vmcK.energy() == Catch::Approx(vmc0.energy() + K2_over_2M).margin(1e-10));
        CHECK(vmcK.acceptance_ratio() == vmc0.acceptance_ratio());
        CHECK(vmcK.energy_imag() == Catch::Approx(0.0).margin(1e-10));
    }
}

TEST_CASE("VMC reports <Im E_L> consistent with zero", "[fixed_phase][vmc]") {
    // A trial that is not translation invariant (one-body Gaussians) gives a
    // non-zero Im E_L per sample; Hermiticity forces its |ψ|² average to 0.
    const auto groups = exciton_groups();
    const Hamiltonian H = exciton_hamiltonian(groups);
    WalkerEnsemble ens(32, groups, 2);
    spread_walkers(ens);

    TrialWavefunction psi;
    psi.add_component(exciton_pade(groups));
    std::vector<SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 1, std::make_unique<GaussianOrbitalSet>(
                                2, std::vector<double>{0.2, -0.1}, std::vector<double>{0.3})});
    blocks.push_back({1, 2, std::make_unique<GaussianOrbitalSet>(
                                2, std::vector<double>{-0.3, 0.4}, std::vector<double>{0.2})});
    psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
    psi.add_component(std::make_unique<CenterOfMassMomentum>(groups, std::vector<double>{1.5, 0.5}));

    // 400 blocks make the reblocked error honest: over 1000 seeds (2026-10-20)
    // Im E / error has standard deviation 1.00 and never exceeds 4, whereas
    // with 100 blocks the error was 12% small and 0.1-0.3% of seeds exceeded
    // 4, with either random-number generator.
    VMCConfig cfg = short_vmc(true);
    cfg.n_accumulation_blocks = 400;
    VMC vmc(cfg, ens, psi, H);
    vmc.run();

    REQUIRE(vmc.energy_imag_error() > 0.0);
    CHECK(std::abs(vmc.energy_imag()) < 4.0 * vmc.energy_imag_error());
    // The one-line summary alea prints shows the diagnostic for complex runs.
    CHECK(vmc.summary().find("ImE") != std::string::npos);
}

TEST_CASE("The node constraint follows from the trial wavefunction",
          "[fixed_phase][dmc]") {
    const auto groups = exciton_groups();
    auto det = [&] {
        std::vector<SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 1, std::make_unique<GaussianOrbitalSet>(
                                    2, std::vector<double>{0.0, 0.0}, std::vector<double>{0.3})});
        return std::make_unique<SlaterDeterminant>(std::move(blocks));
    };
    auto boost = [&] {
        return std::make_unique<CenterOfMassMomentum>(groups, std::vector<double>{1.0, 0.0});
    };

    TrialWavefunction nodeless;
    nodeless.add_component(exciton_pade(groups));
    CHECK(node_constraint(nodeless) == NodeConstraint::None);

    TrialWavefunction real_det;
    real_det.add_component(exciton_pade(groups));
    real_det.add_component(det());
    CHECK(node_constraint(real_det) == NodeConstraint::FixedNode);

    TrialWavefunction boosted;
    boosted.add_component(exciton_pade(groups));
    boosted.add_component(boost());
    CHECK(node_constraint(boosted) == NodeConstraint::FixedPhase);

    // Complex wins over nodal: a complex determinant's zeros are handled by
    // the phase, and the fixed-node gate would wrongly reject phase changes.
    TrialWavefunction both;
    both.add_component(det());
    both.add_component(boost());
    CHECK(node_constraint(both) == NodeConstraint::FixedPhase);
}

TEST_CASE("Fixed-phase DMC of a boosted exciton is the rest-frame DMC plus K^2/2M",
          "[fixed_phase][dmc]") {
    // The boost phase is exact, so E_FP(K) = E₀ + K²/2M.  A large K makes the
    // per-move phase change k·Δr exceed π/2 often: had the fixed-node gate
    // stayed on it would reject those moves, visibly lowering the acceptance
    // and shrinking the effective time step.
    const auto groups = exciton_groups();
    const Hamiltonian H = exciton_hamiltonian(groups);
    const std::vector<double> K = {8.0, 0.0};
    const double K2_over_2M = (K[0] * K[0] + K[1] * K[1]) / (2.0 * 4.0);

    DMCConfig cfg;
    cfg.tau = 0.05;
    cfg.n_equilibration_blocks = 20;
    cfg.n_accumulation_blocks = 60;
    cfg.n_steps = 10;
    cfg.n_substeps = 1;
    cfg.n_threads = 1;
    cfg.seed = 11;

    auto run = [&](bool boosted, double& e, double& err, double& acc,
                   NodeConstraint& nc) {
        WalkerEnsemble ens(256, groups, 2);
        spread_walkers(ens);
        TrialWavefunction psi;
        psi.add_component(exciton_pade(groups));
        if (boosted) psi.add_component(std::make_unique<CenterOfMassMomentum>(groups, K));
        DMC dmc(cfg, ens, psi, H);
        dmc.run();
        e = dmc.energy();
        err = dmc.energy_error();
        acc = dmc.acceptance_ratio();
        nc = dmc.node_constraint();
    };
    double e0, err0, acc0, eK, errK, accK;
    NodeConstraint nc0, ncK;
    run(false, e0, err0, acc0, nc0);
    run(true, eK, errK, accK, ncK);

    CHECK(nc0 == NodeConstraint::None);
    CHECK(ncK == NodeConstraint::FixedPhase);
    CHECK(accK == Catch::Approx(acc0).margin(2e-3));
    CHECK(std::abs((eK - e0) - K2_over_2M) < 4.0 * std::hypot(err0, errK) + 1e-9);
}

TEST_CASE("Branching carries the phase planes with their walker", "[fixed_phase]") {
    const std::size_t nw = 3, np = 1, dim = 2;
    WalkerSet ws(nw, np, dim);
    Walker a(np, dim), b(np, dim), c(np, dim);
    for (Walker* w : {&a, &b, &c}) ws.add_walker(*w);
    ws.enable_phase();
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = 0; w < nw; ++w) {
            ws.phase_gradients()[d * nw + w]  = 10.0 * static_cast<double>(w) + static_cast<double>(d);
            ws.phase_laplacians()[d * nw + w] = -ws.phase_gradients()[d * nw + w];
        }

    const std::vector<std::size_t> src = {2, 0, 2, 1};   // reorder + duplicate
    ws.apply_permutation(src);

    REQUIRE(ws.num_walkers() == src.size());
    const std::size_t n = src.size();
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t j = 0; j < n; ++j) {
            const double want = 10.0 * static_cast<double>(src[j]) + static_cast<double>(d);
            CHECK(ws.phase_gradients()[d * n + j] == want);
            CHECK(ws.phase_laplacians()[d * n + j] == -want);
        }
}
