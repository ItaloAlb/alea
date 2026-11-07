// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "population/particle_group.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "population/walker_ensemble.h"
#include "wavefunction/trial_wavefunction.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/wavefunction_component.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "qmc/dmc.h"

using namespace alea;

namespace {

constexpr double kPi = 3.14159265358979323846;

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

// ── Test-only sign-carrying component ────────────────────────────────────────
//
// A minimal wavefunction factor with |ψ_c| ≡ 1 and a node plane at x = 0 of a
// watched particle (default particle 0): ψ_c = sign(x_watched).  Because the
// magnitude is constant it contributes NOTHING to log_psi / gradient /
// laplacian — it is inert for the physics.  Only its SIGN matters, and that
// surfaces only through the phase of the proposed-move ratio:
//   ψ'/ψ = x'_p / x_p ;  imag(log(ψ'/ψ)) = π  ⇔  the move crossed the node.
// This is exactly the signal QMCPACK's fixed-node gate reads from PhaseDiff.
class SignNode : public WavefunctionComponent {
public:
    explicit SignNode(std::size_t watched_particle = 0)
        : watched_(watched_particle) {}

    std::string name() const override { return "sign_node"; }

    void evaluate_batch(WalkerSet& batch) override {
        // |ψ| ≡ 1 → no contribution.  Still size the base caches so branching's
        // permute_caches has (zero) state to carry.
        resize_caches(batch.num_walkers(), batch.n_particles(), batch.dim());
    }

    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin,
                     std::size_t w_end) override {
        if (p != watched_) return;  // only the watched particle has a node
        const std::size_t nw = batch.num_walkers();
        if (w_end > nw) w_end = nw;
        const std::size_t dim = batch.dim();
        const std::span<const double> pos     = batch.positions();   // [p*dim*nw + d*nw + w]
        const std::span<const double> new_pos = batch.new_pos();     // [d*nw + w]
        std::span<double>             phase   = batch.phase_diffs(); // [nw]
        const std::size_t off = p * dim * nw;  // d = 0 (x-coordinate), walker w
        for (std::size_t w = w_begin; w < w_end; ++w) {
            const double old_x = pos[off + w];
            const double new_x = new_pos[w];  // d = 0 → index w
            if (old_x * new_x < 0.0) phase[w] += kPi;  // |ratio| = 1 → log_ratios untouched
        }
    }

    void apply_move_batch(WalkerSet&, std::size_t, std::span<const bool>,
                          std::size_t, std::size_t) override {
        // Gate-only scope: no committed sign is cached, so nothing to update.
    }

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<SignNode>(*this);
    }

    std::span<Parameter>       parameters() override { return {}; }
    std::span<const Parameter> parameters() const override { return {}; }
    void rebuild() override {}
    void compute_parameter_derivatives(const WalkerSet&,
                                       std::span<double>) const override {}

private:
    std::size_t watched_;
};

std::unique_ptr<WavefunctionComponent> make_pade(double a) {
    std::vector<Parameter> params = {{"a", a, true}, {"b", 0.0, true}};
    return std::make_unique<PadeJastrow>(1, 1, grp(0, 1, +1.0, 1.0),
                                         grp(1, 2, -1.0, 1.0), std::move(params));
}

}  // namespace

// ── Test 1: crosses_node characterization ────────────────────────────────────
TEST_CASE("crosses_node flags a sign flip (phase pi) but not phase 0",
          "[fixed_node]") {
    CHECK(crosses_node(0.0) == false);   // cos(0)=1 → no crossing
    CHECK(crosses_node(kPi) == true);    // cos(π)=-1 < ε → crossing
    CHECK(crosses_node(1e-3) == false);  // tiny phase → no crossing
}

// ── Test 2: WalkerSet exposes a zero-initialised phase_diffs scratch ──────────
TEST_CASE("WalkerSet::phase_diffs is sized [nw] and zero-initialised",
          "[fixed_node]") {
    WalkerSet ws(/*nw=*/3, /*np=*/2, /*dim=*/2);
    const std::span<double> pd = ws.phase_diffs();
    REQUIRE(pd.size() == 3);
    for (double x : pd) CHECK(x == 0.0);
}

// ── Test 3: ratio_batch zeroes phase_diffs, then a sign flip writes π ─────────
TEST_CASE("TrialWavefunction::ratio_batch accumulates the node-crossing phase",
          "[fixed_node]") {
    auto setup = [](double new_x0) {
        WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
        auto pos = ws.positions();
        pos[0] = 1.0;  // particle 0 x = +1 (positive nodal pocket)
        pos[2] = 3.0;  // particle 1 x
        ws.dist_tables().build_all(ws.positions());
        std::vector<double> new_pos = {new_x0, 0.0};  // move particle 0 in x
        ws.propose_move_batch(0, new_pos);
        ws.phase_diffs()[0] = 7.0;  // garbage that ratio_batch must zero first
        return ws;
    };

    SECTION("sign-flipping move of the watched particle -> phase pi") {
        WalkerSet ws = setup(/*new_x0=*/-1.0);
        TrialWavefunction psi;
        psi.add_component(std::make_unique<SignNode>(0));
        psi.ratio_batch(ws, 0);
        CHECK(ws.phase_diffs()[0] == Catch::Approx(kPi));
    }

    SECTION("same-sign move of the watched particle -> phase 0") {
        WalkerSet ws = setup(/*new_x0=*/+0.5);
        TrialWavefunction psi;
        psi.add_component(std::make_unique<SignNode>(0));
        psi.ratio_batch(ws, 0);
        CHECK(ws.phase_diffs()[0] == Catch::Approx(0.0));
    }

    SECTION("nodeless (positive) wavefunction never accumulates phase") {
        WalkerSet ws = setup(/*new_x0=*/-1.0);  // a crossing, but no node here
        TrialWavefunction psi;
        psi.add_component(make_pade(1.0));
        psi.ratio_batch(ws, 0);
        CHECK(ws.phase_diffs()[0] == Catch::Approx(0.0));
    }
}

// ── Test 4: the DMC gate confines a walker to its nodal pocket ────────────────
TEST_CASE("DMC fixed-node gate keeps the walker on one side of the node",
          "[fixed_node][dmc]") {
    // Trial = SignNode only (|ψ| ≡ 1) and H = Kinetic only ⇒ E_L = 0, drift = 0:
    // walkers pure-diffuse and every step proposes free Gaussian moves that
    // frequently try to cross x = 0.  With the gate, a sign-flipping move of the
    // watched electron is rejected, so its x-coordinate can never change sign.
    constexpr std::size_t nw = 32;
    WalkerEnsemble ens(nw, {grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}, /*dim=*/2);
    std::vector<double> pos = {1.0, 0.0, 0.0, 0.0};  // electron x = +1
    for (auto& w : ens.walkers()) w.set_positions(pos);

    TrialWavefunction psi;
    psi.add_component(std::make_unique<SignNode>(0));  // node on the electron (p0)

    Hamiltonian ham;
    ham.add_component(std::make_unique<Kinetic>(
        std::vector<ParticleGroup>{grp(0, 1, -1.0, 1.0), grp(1, 2, +1.0, 1.0)}));

    DMCConfig cfg;
    cfg.tau = 0.2;  // large step ⇒ many crossing attempts
    cfg.n_equilibration_blocks = 0;
    cfg.n_accumulation_blocks = 20;
    cfg.n_steps = 10;
    cfg.n_substeps = 5;
    cfg.seed = 123;

    DMC dmc(cfg, ens, psi, ham);

    double min_electron_x = 1.0;
    dmc.set_measure_hook([&](const WalkerSet& p, const TrialWavefunction&) {
        const std::span<const double> q = p.positions();
        const std::size_t n = p.num_walkers();
        // particle 0, dim 0, walker w → index 0*dim*n + 0*n + w = w.
        for (std::size_t w = 0; w < n; ++w) min_electron_x = std::min(min_electron_x, q[w]);
    });
    dmc.run();

    // No walker ever crossed to x < 0: the node confined the population.
    CHECK(min_electron_x > 0.0);
}
