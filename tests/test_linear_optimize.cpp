// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "numerics/random.h"
#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_ensemble.h"
#include "qmc/linear_optimize.h"
#include "population/walker_set.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using alea::Parameter;
using alea::ParticleGroup;

namespace {

// ── helpers ──────────────────────────────────────────────────────────────────

ParticleGroup make_group(const char* name, double charge, double mass,
                         std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name    = name;
    g.charge  = charge;
    g.first   = first;
    g.last    = last;
    g.count   = last - first;
    // mass per dimension (2D).
    g.mass     = {mass, mass};
    g.inv_mass = {1.0 / mass, 1.0 / mass};
    return g;
}

}  // namespace

// ── test cases ───────────────────────────────────────────────────────────────

TEST_CASE("LinearOptimize converges a 1-parameter Jastrow to the cusp value",
          "[linear_optimize][integration]") {
    // 2D electron-hole exciton: m_e = m_h = 1, q_e = -1, q_h = +1, ε = 1.
    // Cusp condition: c = m q_e q_h / ε = -1 (in 2D, c = m q_e q_h / ε).
    // Start from c = -0.5 (violated), expect convergence toward c ≈ -1.
    const std::size_t nw = 64;
    const std::size_t dim = 2;

    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const ParticleGroup hole     = make_group("h", +1.0, 1.0, 1, 2);
    const std::vector<ParticleGroup> groups = {electron, hole};

    // Build wavefunction: u(r) = c * r, c = -0.5 (non-fixed for optimisation).
    std::vector<Parameter> params = {{"c", -0.5, false}};
    alea::TrialWavefunction psi;
    psi.add_component(std::make_unique<alea::PadeJastrow>(
        /*num_order=*/1, /*den_order=*/0, electron, hole, std::move(params)));

    // Build Hamiltonian.
    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::Kinetic>(groups));
    ham.add_component(std::make_unique<alea::CoulombTerm>(electron, hole, /*epsilon=*/1.0));

    // Walker ensemble.
    alea::WalkerEnsemble ensemble(nw, groups, dim);
    // Initialise positions with e near (-0.5,0), h near (+0.5,0) so r≈1.
    {
        alea::numerics::Random rng(42 + 1000);
        auto walkers = ensemble.walkers();
        for (std::size_t wi = 0; wi < walkers.size(); ++wi) {
            std::vector<double> pos(2 * dim);
            // Place electron near (-0.5, 0) and hole near (+0.5, 0) with some
            // random spread so the inter-particle distance is always ~1.
            pos[0] = -0.5 + 0.3 * rng.gaussian();  // e, x
            pos[1] =  0.0 + 0.3 * rng.gaussian();  // e, y
            pos[2] =  0.5 + 0.3 * rng.gaussian();  // h, x
            pos[3] =  0.0 + 0.3 * rng.gaussian();  // h, y
            walkers[wi].set_positions(pos);
        }
    }

    // Optimisation with static evaluation (tau=0: no MCMC, deterministic).
    // This tests the linear method machinery end-to-end.
    alea::OptimizeConfig opt_cfg;
    opt_cfg.n_walkers              = static_cast<int>(nw);
    opt_cfg.n_equilibration_blocks = 0;
    opt_cfg.n_accumulation_blocks  = 1;
    opt_cfg.n_steps                = 1;   // at least 1 step for VMC measurement
    opt_cfg.n_substeps             = 1;
    opt_cfg.tau                    = 0.0;   // tau=0 → static positions
    opt_cfg.max_iterations         = 1;
    opt_cfg.psi_step_tol             = 1e-8;
    opt_cfg.seed                  = 42;
    opt_cfg.n_threads             = 1;
    opt_cfg.lm_config.shift_i     = 0.1;
    opt_cfg.lm_config.shift_s     = 0.0;
    opt_cfg.lm_config.rescale      = 0.5;

    alea::LinearOptimize optimizer(opt_cfg, ensemble, psi, ham);
    optimizer.run();
    const auto& result = optimizer.result();

    // With one iteration, verify the cost function and linear method work.
    REQUIRE(result.iterations > 0);
    REQUIRE(std::isfinite(result.final_energy));
    REQUIRE(result.energy_history.size() >= 1);

    // Energy at c=-0.5 and r≈1: E_L = -0.5/r - 0.25 ≈ -0.75.
    CHECK(result.final_energy == Catch::Approx(-0.75).margin(0.5));

    // Parameter update computed and applied.
    REQUIRE(result.final_params.size() == 1);
}

TEST_CASE("LinearOptimize preserves the sign of a sign-locked parameter",
          "[linear_optimize][integration]") {
    // Same exciton setup, but c is optimisable AND sign-locked
    // (can_flip_sign=false → sign_locked=true). Starting negative, c must stay
    // negative across every iteration: the θ-reparameterisation p = s·exp(θ)
    // updates p *= exp(Δθ), which can never flip the sign.
    const std::size_t nw = 64;
    const std::size_t dim = 2;

    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const ParticleGroup hole     = make_group("h", +1.0, 1.0, 1, 2);
    const std::vector<ParticleGroup> groups = {electron, hole};

    // u(r) = c * r, c = -0.5, non-fixed, sign-locked.
    std::vector<Parameter> params = {{"c", -0.5, false, true}};
    alea::TrialWavefunction psi;
    psi.add_component(std::make_unique<alea::PadeJastrow>(
        /*num_order=*/1, /*den_order=*/0, electron, hole, std::move(params)));

    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::Kinetic>(groups));
    ham.add_component(std::make_unique<alea::CoulombTerm>(electron, hole, 1.0));

    alea::WalkerEnsemble ensemble(nw, groups, dim);
    {
        alea::numerics::Random rng(42 + 1000);
        auto walkers = ensemble.walkers();
        for (std::size_t wi = 0; wi < walkers.size(); ++wi) {
            std::vector<double> pos(2 * dim);
            pos[0] = -0.5 + 0.3 * rng.gaussian();  // e, x
            pos[1] =  0.0 + 0.3 * rng.gaussian();  // e, y
            pos[2] =  0.5 + 0.3 * rng.gaussian();  // h, x
            pos[3] =  0.0 + 0.3 * rng.gaussian();  // h, y
            walkers[wi].set_positions(pos);
        }
    }

    alea::OptimizeConfig opt_cfg;
    opt_cfg.n_walkers              = static_cast<int>(nw);
    opt_cfg.n_equilibration_blocks = 0;
    opt_cfg.n_accumulation_blocks  = 1;
    opt_cfg.n_steps                = 1;
    opt_cfg.n_substeps             = 1;
    opt_cfg.tau                    = 0.0;   // static positions → deterministic
    opt_cfg.max_iterations         = 5;     // several exp-updates in a row
    opt_cfg.psi_step_tol             = 1e-8;
    opt_cfg.seed                  = 42;
    opt_cfg.n_threads             = 1;
    opt_cfg.lm_config.shift_i     = 0.1;
    opt_cfg.lm_config.shift_s     = 0.0;
    opt_cfg.lm_config.rescale      = 0.5;

    alea::LinearOptimize optimizer(opt_cfg, ensemble, psi, ham);
    optimizer.run();
    const auto& result = optimizer.result();

    REQUIRE(result.final_params.size() == 1);
    REQUIRE(std::isfinite(result.final_params[0]));
    // The locked sign must be preserved.
    CHECK(result.final_params[0] < 0.0);
}

TEST_CASE("LinearOptimize handles fixed parameters", "[linear_optimize]") {
    // Same setup but with all parameters fixed.
    const std::size_t nw = 32;
    const std::size_t dim = 2;

    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const ParticleGroup hole     = make_group("h", +1.0, 1.0, 1, 2);
    const std::vector<ParticleGroup> groups = {electron, hole};

    // Parameter c = -1 is FIXED.
    std::vector<Parameter> params = {{"c", -1.0, true}};
    alea::TrialWavefunction psi;
    psi.add_component(std::make_unique<alea::PadeJastrow>(
        1, 0, electron, hole, std::move(params)));

    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::Kinetic>(groups));
    ham.add_component(std::make_unique<alea::CoulombTerm>(electron, hole, 1.0));

    alea::WalkerEnsemble ensemble(nw, groups, dim);
    {
        alea::numerics::Random rng(123);
        auto walkers = ensemble.walkers();
        for (std::size_t wi = 0; wi < walkers.size(); ++wi) {
            std::vector<double> pos(2 * dim);
            pos[0] = -0.5 + 0.3 * rng.gaussian();  // e, x
            pos[1] =  0.0 + 0.3 * rng.gaussian();  // e, y
            pos[2] =  0.5 + 0.3 * rng.gaussian();  // h, x
            pos[3] =  0.0 + 0.3 * rng.gaussian();  // h, y
            walkers[wi].set_positions(pos);
        }
    }

    alea::OptimizeConfig opt_cfg;
    opt_cfg.n_walkers              = static_cast<int>(nw);
    opt_cfg.n_equilibration_blocks = 0;
    opt_cfg.n_accumulation_blocks  = 1;
    opt_cfg.n_steps                = 1;   // at least 1 step for VMC measurement   // static evaluation
    opt_cfg.tau                    = 0.1;  // unused
    opt_cfg.max_iterations         = 3;
    opt_cfg.seed                  = 456;
    opt_cfg.n_threads             = 1;

    alea::LinearOptimize optimizer(opt_cfg, ensemble, psi, ham);
    optimizer.run();
    const auto& result = optimizer.result();

    // With all parameters fixed, there are 0 optimisable params.
    REQUIRE(result.final_params.size() == 0);
    CHECK(result.converged);
    // Energy at c=-1 and r≈1: exactly -1 (cusp-satisfying).
    CHECK(result.final_energy == Catch::Approx(-1.0).margin(0.01));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Nodal parameters: the width inside the determinant
//
//  ψ = D·e^J with J real and positive, so the nodes of ψ are exactly the nodes
//  of D: optimising a Jastrow cannot move a fixed-node energy by so much as a
//  rounding error, and for a Gaussian-orbital guide the width α is the only
//  parameter that can.  This is the end-to-end check that it reaches the linear
//  method — anchored on the one system whose answer is known in closed form.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// External harmonic trap V = ½ω²Σ_i|r_i|².  Test-local on purpose: `alea` has
// no harmonic external potential, and this one exists only to give the width a
// variational optimum that can be written down.
class HarmonicTrap final : public alea::HamiltonianComponent {
public:
    HarmonicTrap(double omega, std::size_t np) : omega_(omega), np_(np) {}

    std::string name() const override { return "harmonic_trap"; }

    void evaluate_batch(alea::WalkerSet& batch, std::size_t w_begin,
                        std::size_t w_end) override {
        const std::size_t nw = batch.num_walkers(), dim = batch.dim();
        if (w_end > nw) w_end = nw;
        const std::span<const double> pos = batch.positions();
        const std::span<double> le = batch.local_energies();
        const double half_k = 0.5 * omega_ * omega_;
        for (std::size_t p = 0; p < np_; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                for (std::size_t w = w_begin; w < w_end; ++w) {
                    const double x = pos[(p * dim + d) * nw + w];
                    le[w] += half_k * x * x;
                }
    }

    [[nodiscard]] std::unique_ptr<alea::HamiltonianComponent> clone()
        const override {
        return std::make_unique<HarmonicTrap>(*this);
    }

private:
    double      omega_;
    std::size_t np_;
};

}  // namespace

TEST_CASE("LinearOptimize converges the Gaussian width of a trapped particle",
          "[linear_optimize][slater][optimize][integration]") {
    // One particle in 2-D, H = -½∇² + ½ω²r², guide ψ = exp(-α r²) written as a
    // 1×1 Slater determinant.  With ⟨r²⟩ = d/(4α) under |ψ|²,
    //
    //     E(α) = α d/2 + ω² d/(8α)   ⇒   α* = ω/2,   E(α*) = ω d/2,
    //
    // and at α* the guide IS the ground state, so E_L is constant and the linear
    // method's own approximation is exact there.  ω = 1, d = 2 ⇒ α* = 0.5, E = 1.
    const std::size_t nw = 256, dim = 2;
    const double omega = 1.0;
    const double alpha0 = 0.25;   // twice too broad: σ = 1 instead of 1/√2

    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const std::vector<ParticleGroup> groups = {electron};

    std::vector<alea::SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 1,
                      std::make_unique<alea::GaussianOrbitalSet>(
                          dim, std::vector<double>{0.0, 0.0},
                          std::vector<double>{alpha0})});
    auto det = std::make_unique<alea::SlaterDeterminant>(std::move(blocks));
    REQUIRE(det->parameters().size() == 1);
    det->parameters()[0].is_fixed = false;

    alea::TrialWavefunction psi;
    psi.add_component(std::move(det));
    REQUIRE(psi.total_optimizable_params() == 1);

    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::Kinetic>(groups));
    ham.add_component(std::make_unique<HarmonicTrap>(omega, 1));

    alea::WalkerEnsemble ensemble(nw, groups, dim);
    {
        alea::numerics::Random rng(2026);
        auto walkers = ensemble.walkers();
        const double sigma = 1.0 / std::sqrt(4.0 * alpha0);
        for (std::size_t wi = 0; wi < walkers.size(); ++wi) {
            const std::vector<double> pos = {sigma * rng.gaussian(),
                                             sigma * rng.gaussian()};
            walkers[wi].set_positions(pos);
        }
    }

    alea::OptimizeConfig cfg;
    cfg.n_walkers              = static_cast<int>(nw);
    cfg.n_equilibration_blocks = 20;
    cfg.n_accumulation_blocks  = 20;
    cfg.n_steps                = 10;
    cfg.n_substeps             = 1;
    cfg.tau                    = 0.5;   // Metropolis proposal variance
    cfg.max_iterations         = 15;
    cfg.psi_step_tol           = 1e-3;
    cfg.seed                   = 7;
    cfg.n_threads              = 1;

    alea::LinearOptimize optimizer(cfg, ensemble, psi, ham);
    optimizer.run();

    const double alpha = psi.components()[0]->parameters()[0].value;
    INFO("alpha = " << alpha << ", iterations = " << optimizer.result().iterations
                    << ", converged = " << optimizer.result().converged);
    CHECK(alpha > alpha0);                       // it moved, and in the right direction
    CHECK(alpha == Catch::Approx(0.5).margin(0.002));
    CHECK(optimizer.result().final_energy == Catch::Approx(1.0).margin(1e-4));

    // The optimised value must be the one the orbitals actually evaluate with —
    // a parameter list that drifted out of sync with its kernel would report a
    // converged α while sampling the starting one.
    alea::WalkerSet ws(/*nw=*/1, /*np=*/1, dim);
    ws.positions()[0] = 0.7;   // r = (0.7, 0)
    ws.dist_tables().build_all(ws.positions());
    psi.evaluate_batch(ws);
    CHECK(ws.log_psi()[0] == Catch::Approx(-alpha * 0.49).epsilon(1e-12));
}

// ── ∂E_L/∂p ────────────────────────────────────────────────────────────────
// Only the kinetic term depends on the parameters, so ∂E_L/∂p is assembled from
// the wavefunction's ∂(∇lnψ)/∂p and ∂(∇²lnψ)/∂p.  Checked twice: against finite
// differences of the full local energy, and against a closed form.
TEST_CASE("Kinetic assembles dE_L/dp for a Gaussian width",
          "[kinetic][optimize]") {
    const std::size_t dim = 2, nw = 3, np = 1;
    const double alpha = 0.3, omega = 1.0;
    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const std::vector<ParticleGroup> groups = {electron};
    const std::vector<double> xs = {0.4, -0.9, 1.3}, ys = {-0.2, 0.5, 0.7};

    auto make_psi = [&](double a) {
        std::vector<alea::SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 1,
                          std::make_unique<alea::GaussianOrbitalSet>(
                              dim, std::vector<double>{0.0, 0.0},
                              std::vector<double>{a})});
        auto det = std::make_unique<alea::SlaterDeterminant>(std::move(blocks));
        det->parameters()[0].is_fixed = false;
        alea::TrialWavefunction psi;
        psi.add_component(std::move(det));
        return psi;
    };
    auto make_ham = [&]() {
        alea::Hamiltonian ham;
        ham.add_component(std::make_unique<alea::Kinetic>(groups));
        ham.add_component(std::make_unique<HarmonicTrap>(omega, np));
        return ham;
    };
    auto load = [&](alea::WalkerSet& ws) {
        for (std::size_t w = 0; w < nw; ++w) {
            ws.positions()[0 * nw + w] = xs[w];
            ws.positions()[1 * nw + w] = ys[w];
        }
        ws.dist_tables().build_all(ws.positions());
    };
    auto energies_at = [&](double a) {
        alea::TrialWavefunction psi = make_psi(a);
        alea::Hamiltonian ham = make_ham();
        alea::WalkerSet ws(nw, np, dim);
        load(ws);
        psi.evaluate_batch(ws);
        ham.evaluate_batch(ws);
        return std::vector<double>(ws.local_energies().begin(),
                                   ws.local_energies().end());
    };

    alea::TrialWavefunction psi = make_psi(alpha);
    alea::Hamiltonian ham = make_ham();
    alea::WalkerSet ws(nw, np, dim);
    load(ws);
    psi.evaluate_batch(ws);
    ham.evaluate_batch(ws);

    std::vector<double> dgrad(np * dim * nw), dlap(np * dim * nw);
    std::vector<char> supported(1, 0);
    REQUIRE(psi.compute_parameter_gradient_derivatives(ws, dgrad, dlap,
                                                       supported));
    CHECK(supported[0] == 1);

    std::vector<double> dE(nw, 0.0);
    ham.local_energy_parameter_derivatives(ws, dgrad, dlap, 1, dE);

    const double h = 1e-6;
    const std::vector<double> ep = energies_at(alpha + h);
    const std::vector<double> em = energies_at(alpha - h);
    for (std::size_t w = 0; w < nw; ++w) {
        CHECK(dE[w] == Catch::Approx((ep[w] - em[w]) / (2 * h)).margin(1e-5));
        // ψ = exp(-α r²) ⇒ E_L = 2α - 2α²r² + ½r², so ∂E_L/∂α = 2 - 4α r².
        const double r2 = xs[w] * xs[w] + ys[w] * ys[w];
        CHECK(dE[w] == Catch::Approx(2.0 - 4.0 * alpha * r2).epsilon(1e-10));
    }
}

// Silent degradation is the failure mode this whole exercise was about, so a
// component that cannot supply the curvature term must be visible.
TEST_CASE("components report whether they can supply the curvature term",
          "[optimize]") {
    std::vector<alea::SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 1,
                      std::make_unique<alea::GaussianOrbitalSet>(
                          2, std::vector<double>{0.0, 0.0},
                          std::vector<double>{0.4})});
    alea::SlaterDeterminant det(std::move(blocks));
    CHECK(det.supports_parameter_gradient_derivatives());

    const ParticleGroup e = make_group("e", -1.0, 1.0, 0, 1);
    const ParticleGroup h = make_group("h", +1.0, 1.0, 1, 2);
    alea::PadeJastrow jas(1, 0, e, h, {Parameter{"c", -0.5, false}});
    CHECK(jas.supports_parameter_gradient_derivatives());
}

// A step that raises the energy must be undone.  This mattered little while the
// steps were too short to do damage; with the length fixed it is what keeps a
// badly scaled run from bouncing.  Measured on this seed with rescale 20 (40x the
// then-default 0.5; 20x today's 1): without rejection the energy climbed back at four separate
// iterations (1.0021 -> 1.0140, 0.9998 -> 1.0068, 1.0001 -> 1.0151) and the run
// ended at a = 0.4714; with it the trajectory is monotone and ends at 0.4992.
//
// Candidates are now scored by correlated sampling on the stored sample, so a
// runaway (a = 1e4) no longer reports a spuriously LOW energy from walkers that
// never re-equilibrated: its weights collapse (N_eff/N < min_effective_fraction)
// and it is refused.  max_param_change stays at its default as a second guard.
TEST_CASE("LinearOptimize undoes a step that raises the energy",
          "[linear_optimize][optimize]") {
    const std::size_t nw = 256, dim = 2;
    const double omega = 1.0, alpha0 = 0.35;
    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const std::vector<ParticleGroup> groups = {electron};

    std::vector<alea::SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 1,
                      std::make_unique<alea::GaussianOrbitalSet>(
                          dim, std::vector<double>{0.0, 0.0},
                          std::vector<double>{alpha0})});
    auto det = std::make_unique<alea::SlaterDeterminant>(std::move(blocks));
    det->parameters()[0].is_fixed = false;
    alea::TrialWavefunction psi;
    psi.add_component(std::move(det));

    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::Kinetic>(groups));
    ham.add_component(std::make_unique<HarmonicTrap>(omega, 1));

    alea::WalkerEnsemble ensemble(nw, groups, dim);
    {
        alea::numerics::Random rng(2026);
        auto walkers = ensemble.walkers();
        const double sigma = 1.0 / std::sqrt(4.0 * alpha0);
        for (std::size_t wi = 0; wi < walkers.size(); ++wi) {
            const std::vector<double> pos = {sigma * rng.gaussian(),
                                             sigma * rng.gaussian()};
            walkers[wi].set_positions(pos);
        }
    }

    alea::OptimizeConfig cfg;
    cfg.n_walkers              = static_cast<int>(nw);
    cfg.n_equilibration_blocks = 10;
    cfg.n_accumulation_blocks  = 10;
    cfg.n_steps                = 10;
    cfg.n_substeps             = 1;
    cfg.tau                    = 0.5;
    cfg.max_iterations         = 10;
    cfg.seed                   = 11;
    cfg.n_threads              = 1;
    cfg.lm_config.rescale      = 20.0;   // 20x the default, deliberately

    alea::LinearOptimize optimizer(cfg, ensemble, psi, ham);
    optimizer.run();

    const double alpha = psi.components()[0]->parameters()[0].value;
    const auto& history = optimizer.result().energy_history;
    // Correlated sampling stops the run once no candidate lowers the energy
    // significantly — here the exact ψ is in the family (zero variance at
    // α = 1/2), so that happens after a handful of steps.
    REQUIRE(history.size() >= 2);
    INFO("alpha = " << alpha << ", iterations = " << history.size());

    double worst_rise = 0.0;
    for (std::size_t i = 1; i < history.size(); ++i)
        worst_rise = std::max(worst_rise, history[i] - history[i - 1]);
    CHECK(worst_rise < 0.005);              // 0.015 without rejection
    CHECK(alpha == Catch::Approx(0.5).margin(0.02));
}

// Alternating (nodes, then Jastrow) is the way to avoid an Hbar whose blocks
// have different quality: only the determinant supplies the curvature term.
// Each stage must move only its own parameters and restore the flags it found.
TEST_CASE("LinearOptimize honours a parameter scope", "[linear_optimize][optimize]") {
    const std::size_t nw = 128, dim = 2;
    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const ParticleGroup hole     = make_group("h", +1.0, 1.0, 1, 2);
    const std::vector<ParticleGroup> groups = {electron, hole};

    auto build = [&](alea::TrialWavefunction& psi) {
        std::vector<alea::SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 1,
                          std::make_unique<alea::GaussianOrbitalSet>(
                              dim, std::vector<double>{0.0, 0.0},
                              std::vector<double>{0.4})});
        auto det = std::make_unique<alea::SlaterDeterminant>(std::move(blocks));
        det->parameters()[0].is_fixed = false;
        psi.add_component(std::move(det));
        psi.add_component(std::make_unique<alea::PadeJastrow>(
            1, 0, electron, hole, std::vector<Parameter>{{"c", -0.5, false}}));
    };

    auto run_scope = [&](const std::string& scope, double& alpha, double& c) {
        alea::TrialWavefunction psi;
        build(psi);
        alea::Hamiltonian ham;
        ham.add_component(std::make_unique<alea::Kinetic>(groups));
        ham.add_component(std::make_unique<HarmonicTrap>(1.0, 2));

        alea::WalkerEnsemble ensemble(nw, groups, dim);
        {
            alea::numerics::Random rng(5);
            auto walkers = ensemble.walkers();
            for (std::size_t wi = 0; wi < walkers.size(); ++wi) {
                const std::vector<double> pos = {
                    0.8 * rng.gaussian(), 0.8 * rng.gaussian(),
                    0.8 * rng.gaussian(), 0.8 * rng.gaussian()};
                walkers[wi].set_positions(pos);
            }
        }
        alea::OptimizeConfig cfg;
        cfg.n_walkers              = static_cast<int>(nw);
        cfg.n_equilibration_blocks = 5;
        cfg.n_accumulation_blocks  = 5;
        cfg.n_steps                = 10;
        cfg.tau                    = 0.5;
        cfg.max_iterations         = 3;
        cfg.seed                   = 3;
        cfg.n_threads              = 1;
        cfg.parameters             = scope;

        alea::LinearOptimize(cfg, ensemble, psi, ham).run();
        alpha = psi.components()[0]->parameters()[0].value;
        c     = psi.components()[1]->parameters()[0].value;
        // The scope is a stage property: the flags must come back as found, or
        // a second optimizer driver in the same qmc list would inherit them.
        CHECK_FALSE(psi.components()[0]->parameters()[0].is_fixed);
        CHECK_FALSE(psi.components()[1]->parameters()[0].is_fixed);
    };

    double a_nodal = 0, c_nodal = 0, a_jas = 0, c_jas = 0, a_all = 0, c_all = 0;
    run_scope("nodal",   a_nodal, c_nodal);
    run_scope("jastrow", a_jas,   c_jas);
    run_scope("all",     a_all,   c_all);

    CHECK(a_nodal != Catch::Approx(0.4).epsilon(1e-9));   // the width moved
    CHECK(c_nodal == Catch::Approx(-0.5).epsilon(1e-12)); // the jastrow did not
    CHECK(a_jas   == Catch::Approx(0.4).epsilon(1e-12));  // and the other way
    CHECK(c_jas   != Catch::Approx(-0.5).epsilon(1e-9));
    CHECK(a_all   != Catch::Approx(0.4).epsilon(1e-9));   // both together
    CHECK(c_all   != Catch::Approx(-0.5).epsilon(1e-9));
}

// Giving up needs TWO consecutive failed iterations, as the code intends.  The
// history bitset starts all-zero, which made the very first failure match the
// "both rejected" pattern and end the run on the spot.
TEST_CASE("LinearOptimize does not give up after one failed iteration",
          "[linear_optimize][optimize]") {
    const std::size_t nw = 64, dim = 2;
    const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    const std::vector<ParticleGroup> groups = {electron};

    std::vector<alea::SlaterDeterminant::Block> blocks;
    blocks.push_back({0, 1,
                      std::make_unique<alea::GaussianOrbitalSet>(
                          dim, std::vector<double>{0.0, 0.0},
                          std::vector<double>{0.25})});
    auto det = std::make_unique<alea::SlaterDeterminant>(std::move(blocks));
    det->parameters()[0].is_fixed = false;
    alea::TrialWavefunction psi;
    psi.add_component(std::move(det));

    alea::Hamiltonian ham;
    ham.add_component(std::make_unique<alea::Kinetic>(groups));
    ham.add_component(std::make_unique<HarmonicTrap>(1.0, 1));

    alea::WalkerEnsemble ensemble(nw, groups, dim);
    {
        alea::numerics::Random rng(1);
        auto walkers = ensemble.walkers();
        for (std::size_t wi = 0; wi < walkers.size(); ++wi) {
            const std::vector<double> pos = {rng.gaussian(), rng.gaussian()};
            walkers[wi].set_positions(pos);
        }
    }

    alea::OptimizeConfig cfg;
    cfg.n_walkers              = static_cast<int>(nw);
    cfg.n_equilibration_blocks = 2;
    cfg.n_accumulation_blocks  = 2;
    cfg.n_steps                = 5;
    cfg.tau                    = 0.5;
    cfg.max_iterations         = 5;
    cfg.seed                   = 2;
    cfg.n_threads              = 1;
    cfg.max_param_change       = 1.0e-14;   // every proposed step is refused

    alea::LinearOptimize optimizer(cfg, ensemble, psi, ham);
    optimizer.run();
    // Every iteration fails, so it stops at the second — not the first.
    CHECK(optimizer.result().iterations == 2);
}

// Correlated sampling, checked where the answer is known exactly: one particle
// in a 2-D trap (ω = 1) with ψ = exp(−α r²) has E(α) = α + 1/(4α).
namespace {

struct TrapFixture {
    static constexpr std::size_t nw = 256, dim = 2;
    ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
    alea::TrialWavefunction psi;
    alea::Hamiltonian ham;
    alea::WalkerEnsemble ensemble{nw, std::vector<ParticleGroup>{electron}, dim};
    alea::OptimizeConfig cfg;

    explicit TrapFixture(double alpha0, int n_threads) {
        std::vector<alea::SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 1,
                          std::make_unique<alea::GaussianOrbitalSet>(
                              dim, std::vector<double>{0.0, 0.0},
                              std::vector<double>{alpha0})});
        auto det = std::make_unique<alea::SlaterDeterminant>(std::move(blocks));
        det->parameters()[0].is_fixed = false;
        psi.add_component(std::move(det));
        ham.add_component(std::make_unique<alea::Kinetic>(
            std::vector<ParticleGroup>{electron}));
        ham.add_component(std::make_unique<HarmonicTrap>(1.0, 1));
        alea::numerics::Random rng(7);
        const double sigma = 1.0 / std::sqrt(4.0 * alpha0);
        for (auto& w : ensemble.walkers()) {
            const std::vector<double> pos = {sigma * rng.gaussian(),
                                             sigma * rng.gaussian()};
            w.set_positions(pos);
        }
        cfg.n_equilibration_blocks = 10;
        cfg.n_accumulation_blocks  = 40;
        cfg.n_steps                = 10;
        cfg.tau                    = 0.5;
        cfg.seed                   = 5;
        cfg.n_threads              = n_threads;   // > 1 exercises per-thread frames
    }

    // Optimiser-space step that takes α to `alpha_new` (θ = log α if the
    // parameter is sign-locked, plain α otherwise).
    std::vector<double> step_to(double alpha_new) {
        const alea::Parameter& p = psi.components()[0]->parameters()[0];
        return {p.sign_locked ? std::log(alpha_new / p.value) : alpha_new - p.value};
    }
};

double trap_energy(double alpha) { return alpha + 0.25 / alpha; }

}  // namespace

TEST_CASE("correlated sampling: a zero step changes nothing",
          "[linear_optimize][optimize][reweight]") {
    // w ≡ 1 and E_L,new = V + K must reproduce the stored E_L on every sample,
    // so dE is zero to rounding — this pins the V/K split and the bookkeeping.
    TrapFixture f(0.35, 2);
    alea::LinearOptimize opt(f.cfg, f.ensemble, f.psi, f.ham);
    const auto rw = opt.correlated_step_energy({0.0});
    REQUIRE(rw.finite);
    CHECK(std::abs(rw.dE) < 1e-12);
    CHECK(rw.n_eff_fraction == Catch::Approx(1.0).epsilon(1e-12));
    CHECK(rw.dE_error < 1e-12);
}

TEST_CASE("correlated sampling reproduces the exact energy change",
          "[linear_optimize][optimize][reweight]") {
    // α 0.35 → 0.42: exact ΔE = E(0.42) − E(0.35) ≈ −0.049.  The estimate is
    // taken on configurations drawn at 0.35 only; ψ must come back unchanged.
    TrapFixture f(0.35, 2);
    alea::LinearOptimize opt(f.cfg, f.ensemble, f.psi, f.ham);
    const double before = f.psi.components()[0]->parameters()[0].value;
    const auto rw = opt.correlated_step_energy(f.step_to(0.42));
    REQUIRE(rw.finite);
    const double exact = trap_energy(0.42) - trap_energy(0.35);
    INFO("dE = " << rw.dE << " +- " << rw.dE_error << ", exact " << exact
                 << ", N_eff = " << rw.n_eff_fraction);
    CHECK(rw.dE_error > 0.0);
    CHECK(rw.dE_error < 0.1 * std::abs(exact));        // correlated: precise
    CHECK(std::abs(rw.dE - exact) < 4.0 * rw.dE_error);
    CHECK(rw.n_eff_fraction > 0.8);
    CHECK(f.psi.components()[0]->parameters()[0].value == before);
}

TEST_CASE("correlated sampling flags a runaway step by N_eff",
          "[linear_optimize][optimize][reweight]") {
    // α 0.35 → 50: the new ψ lives where the sample has no walkers, so the
    // weights collapse — this is what now refuses a runaway, instead of a
    // spuriously low energy from walkers that never re-equilibrated.
    TrapFixture f(0.35, 1);
    alea::LinearOptimize opt(f.cfg, f.ensemble, f.psi, f.ham);
    const auto rw = opt.correlated_step_energy(f.step_to(50.0));
    CHECK(rw.n_eff_fraction < f.cfg.min_effective_fraction);
}

// The energy trend is a least-squares line through the energies of fresh
// samples, with its error from the residuals.  On an exact line the slope is
// exact and its error zero.
TEST_CASE("the energy trend of an exact line has the line's slope and no error",
          "[linear_optimize][optimize][convergence]") {
    using alea::LinearOptimize;
    std::vector<double> e(15);
    for (std::size_t k = 0; k < e.size(); ++k) e[k] = -2.0 - 1e-3 * static_cast<double>(k);
    const auto t = LinearOptimize::energy_trend(e);
    CHECK(t.slope == Catch::Approx(-1e-3).epsilon(1e-10));
    CHECK(t.slope_error < 1e-12);
    CHECK_FALSE(LinearOptimize::energy_has_settled(e, 15, 1e-6));   // still falling

    std::fill(e.begin(), e.end(), -2.0);                             // flat: settled
    CHECK(LinearOptimize::energy_has_settled(e, 15, 1e-6));
    // A descent slower than energy_tol per iteration counts as flat.
    for (std::size_t k = 0; k < e.size(); ++k) e[k] = -2.0 - 1e-7 * static_cast<double>(k);
    CHECK(LinearOptimize::energy_has_settled(e, 15, 1e-6));
    // Too few energies for the window: no verdict yet.
    CHECK_FALSE(LinearOptimize::energy_has_settled(std::span(e).first(14), 15, 1e-6));
    // Only the last `window` energies count: an early drop is forgotten.
    std::vector<double> late(20, -2.0);
    for (std::size_t k = 0; k < 5; ++k) late[k] = -1.0 + 0.2 * static_cast<double>(k);
    CHECK(LinearOptimize::energy_has_settled(late, 15, 1e-6));
    CHECK_THROWS(LinearOptimize::energy_trend(std::span(e).first(2)));
}

// The energies measured on the last 15 fresh samples of two real stages
// (e–h bilayer, N = 58, 512 walkers × 40 blocks per iteration, Ha*).
// r_s = 10 was still falling by 4e-4 per iteration, just under each single
// step's 2σ: a rule judging one step at a time stopped such a run 0.006 Ha*
// early.  r_s = 5 had reached its plateau, where each step's own correlated
// ΔE kept predicting −1e-3 to −7e-3 that never materialized.
TEST_CASE("the energy trend keeps a slow real descent going and stops on a plateau",
          "[linear_optimize][optimize][convergence]") {
    using alea::LinearOptimize;
    const std::vector<double> rs10 = {
        -29.760159, -29.760139, -29.761154, -29.761963, -29.764549,
        -29.760910, -29.763711, -29.762351, -29.765537, -29.763789,
        -29.763197, -29.765210, -29.764930, -29.765149, -29.766622};
    const std::vector<double> rs5 = {
        -30.796702, -30.799870, -30.795234, -30.799880, -30.795944,
        -30.797169, -30.802414, -30.795992, -30.797211, -30.797774,
        -30.800899, -30.796211, -30.799063, -30.800917, -30.796245};
    const auto t10 = LinearOptimize::energy_trend(rs10);
    INFO("r_s = 10 slope " << t10.slope << " +- " << t10.slope_error);
    CHECK(t10.slope < -4.0 * t10.slope_error);
    CHECK_FALSE(LinearOptimize::energy_has_settled(rs10, 15, 1e-6));
    CHECK(LinearOptimize::energy_has_settled(rs5, 15, 1e-6));
}

// A genuinely small step barely moves the weights, so its N_eff is ≈ 1.  A
// step whose ‖δψ‖/‖ψ‖ only rounded to zero (a huge Δp along a near-null
// direction of S̄) collapses them: the e–h bilayer at r_s = 2 logged
// |dpsi|/|psi| = 0 with ΔE = −2.251e28 and N_eff = 0, and that must not count
// as converged.
TEST_CASE("a small step counts only when its reweighting agrees",
          "[linear_optimize][optimize][convergence]") {
    using alea::LinearOptimize;
    TrapFixture f(0.35, 2);
    LinearOptimize opt(f.cfg, f.ensemble, f.psi, f.ham);
    const auto genuine = opt.correlated_step_energy(f.step_to(0.3501));
    INFO("N_eff of a 3e-4 relative step = " << genuine.n_eff_fraction);
    CHECK(genuine.n_eff_fraction > 0.999);
    CHECK(LinearOptimize::step_is_small(1e-4, genuine, 1e-3, 0.3));
    CHECK_FALSE(LinearOptimize::step_is_small(2e-3, genuine, 1e-3, 0.3));

    LinearOptimize::Reweighted collapsed;
    collapsed.dE             = -2.251e28;
    collapsed.dE_error       = 8.6e-3;
    collapsed.n_eff_fraction = 0.0;
    collapsed.finite         = true;
    CHECK_FALSE(LinearOptimize::step_is_small(0.0, collapsed, 1e-3, 0.3));

    LinearOptimize::Reweighted non_finite = genuine;
    non_finite.finite = false;
    CHECK_FALSE(LinearOptimize::step_is_small(0.0, non_finite, 1e-3, 0.3));
}

// A nearly singular S̄ near the optimum (the two backflow channels of a bilayer
// exciton enter only through eta_eh + eta_he) gave an undamped step of 1.8e19;
// scoring it set a sign-locked cutoff to L e^{Δθ} = inf and the run died with
// "the cutoff length L must be positive".  Such a step must not be set on ψ.
TEST_CASE("a step is scored only within max_param_change and when finite",
          "[linear_optimize][optimize]") {
    using alea::LinearOptimize;
    const std::vector<double> params = {0.5, -2.0, 3.0};   // max |p| = 3
    CHECK(LinearOptimize::step_within_bounds(std::vector<double>{0.1, -0.89, 0.0}, params, 0.3));
    CHECK_FALSE(LinearOptimize::step_within_bounds(std::vector<double>{0.1, -0.91, 0.0}, params, 0.3));
    CHECK_FALSE(LinearOptimize::step_within_bounds(std::vector<double>{0.0, 1.8e19, 0.0}, params, 0.3));
    CHECK_FALSE(LinearOptimize::step_within_bounds(
        std::vector<double>{0.0, std::numeric_limits<double>::quiet_NaN(), 0.0}, params, 0.3));
    CHECK_FALSE(LinearOptimize::step_within_bounds(
        std::vector<double>{std::numeric_limits<double>::infinity()}, params, 0.3));
    // Small parameters count as size 1: a step of 0.3 is still allowed.
    CHECK(LinearOptimize::step_within_bounds(std::vector<double>{0.3}, std::vector<double>{1e-4}, 0.3));
}

// The last steps of two real stages (e–h bilayer, r_s = 5, N − 1 pairs, Ha*;
// σ = 0.0023 is the error of a 512-walker, 40-block sample there).  One ended
// 0.0274 Ha* (8σ) above the sample it started from and must be undone; the
// other ended 0.002 (0.6σ) above, within noise, and must be kept.
TEST_CASE("the last step is undone only when its fresh energy rose significantly",
          "[linear_optimize][optimize][convergence]") {
    using alea::LinearOptimize;
    using E = LinearOptimize::MeasuredEnergy;
    const double sigma = 0.0023;
    CHECK(LinearOptimize::step_raised_energy(E{-30.249740, sigma}, E{-30.222297, sigma}, 2.0));
    CHECK_FALSE(LinearOptimize::step_raised_energy(E{-30.248846, sigma}, E{-30.246805, sigma}, 2.0));
    // A step that lowered the energy, or left it unchanged, is never undone.
    CHECK_FALSE(LinearOptimize::step_raised_energy(E{-30.2470, sigma}, E{-30.2600, sigma}, 2.0));
    CHECK_FALSE(LinearOptimize::step_raised_energy(E{-30.2470, sigma}, E{-30.2470, sigma}, 2.0));
    // The bar is reject_sigmas combined σ: 2σ√2 = 0.0065 here.
    CHECK(LinearOptimize::step_raised_energy(E{0.0, sigma}, E{0.0066, sigma}, 2.0));
    CHECK_FALSE(LinearOptimize::step_raised_energy(E{0.0, sigma}, E{0.0064, sigma}, 2.0));
}

// A stage that ends on an accepted step measures it once more.  One iteration
// from α = 0.25 (E = 1.25) steps towards α = 1/2 (E = 1), so the check keeps
// it and reports the energy it measured at the new α.
TEST_CASE("LinearOptimize measures its last step and keeps a good one",
          "[linear_optimize][optimize]") {
    TrapFixture f(0.25, 2);
    f.cfg.max_iterations = 1;
    alea::LinearOptimize opt(f.cfg, f.ensemble, f.psi, f.ham);
    opt.run();
    const auto& r = opt.result();
    const double alpha = f.psi.components()[0]->parameters()[0].value;
    INFO("alpha = " << alpha << ", E = " << r.final_energy << " +- " << r.final_energy_error
                    << ", exact " << trap_energy(alpha));
    REQUIRE(r.iterations == 1);
    REQUIRE(r.last_step_checked);
    CHECK_FALSE(r.last_step_undone);
    CHECK(alpha > 0.3);
    CHECK(r.final_params[0] == alpha);
    CHECK(r.final_energy == r.check_energy);
    CHECK(r.final_energy_error > 0.0);
    CHECK(std::abs(r.final_energy - trap_energy(alpha)) < 5.0 * r.final_energy_error);
}

// The undo path, forced: a negative reject_sigmas makes every check count as a
// rise.  ψ must come back exactly to the parameters the step started from (and
// evaluate with them), and the stage must report the energy measured there.
TEST_CASE("LinearOptimize undoes its last step back to the sampled parameters",
          "[linear_optimize][optimize]") {
    TrapFixture f(0.25, 2);
    f.cfg.max_iterations = 1;
    f.cfg.reject_sigmas  = -1e9;
    alea::LinearOptimize opt(f.cfg, f.ensemble, f.psi, f.ham);
    opt.run();
    const auto& r = opt.result();
    REQUIRE(r.last_step_checked);
    REQUIRE(r.last_step_undone);
    const double alpha = f.psi.components()[0]->parameters()[0].value;
    CHECK(alpha == 0.25);
    CHECK(r.final_params[0] == 0.25);
    CHECK(r.final_energy == r.energy_history.front());
    // The check measured the stepped ψ, which lies lower than the start.
    CHECK(r.check_energy < r.final_energy);

    alea::WalkerSet ws(/*nw=*/1, /*np=*/1, TrapFixture::dim);
    ws.positions()[0] = 0.7;   // r = (0.7, 0)
    ws.dist_tables().build_all(ws.positions());
    f.psi.evaluate_batch(ws);
    CHECK(ws.log_psi()[0] == Catch::Approx(-0.25 * 0.49).epsilon(1e-12));
}

// `random_walk: langevin` on an optimizer stage reaches its VMC sampler.  With
// the same seed and start, Langevin (drift) and Metropolis proposals visit
// different configurations, so the first sampled energies differ; before the
// flag was wired, the optimizer sampled with Metropolis whatever it was told.
TEST_CASE("LinearOptimize samples with Langevin moves when use_drift is set",
          "[linear_optimize]") {
    auto first_energy = [](bool use_drift) {
        const std::size_t nw = 16;
        const std::size_t dim = 2;
        const ParticleGroup electron = make_group("e", -1.0, 1.0, 0, 1);
        const ParticleGroup hole     = make_group("h", +1.0, 1.0, 1, 2);
        const std::vector<ParticleGroup> groups = {electron, hole};

        std::vector<Parameter> params = {{"c", -0.5, false}};
        alea::TrialWavefunction psi;
        psi.add_component(std::make_unique<alea::PadeJastrow>(
            /*num_order=*/1, /*den_order=*/0, electron, hole, std::move(params)));
        alea::Hamiltonian ham;
        ham.add_component(std::make_unique<alea::Kinetic>(groups));
        ham.add_component(std::make_unique<alea::CoulombTerm>(electron, hole, /*epsilon=*/1.0));

        alea::WalkerEnsemble ensemble(nw, groups, dim);
        alea::numerics::Random rng(7);
        for (alea::Walker& w : ensemble.walkers()) {
            std::vector<double> pos(2 * dim);
            for (double& x : pos) x = rng.gaussian();
            w.set_positions(pos);
        }

        alea::OptimizeConfig cfg;
        cfg.n_walkers              = static_cast<int>(nw);
        cfg.n_equilibration_blocks = 2;
        cfg.n_accumulation_blocks  = 4;
        cfg.n_steps                = 5;
        cfg.n_substeps             = 2;
        cfg.tau                    = 0.1;
        cfg.use_drift              = use_drift;
        cfg.max_iterations         = 1;
        cfg.seed                   = 42;
        cfg.n_threads              = 1;
        alea::LinearOptimize optimizer(cfg, ensemble, psi, ham);
        optimizer.run();
        REQUIRE_FALSE(optimizer.result().energy_history.empty());
        return optimizer.result().energy_history.front();
    };
    const double metropolis = first_energy(false);
    const double langevin   = first_energy(true);
    CHECK(std::isfinite(metropolis));
    CHECK(std::isfinite(langevin));
    CHECK(metropolis != langevin);
}
