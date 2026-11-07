// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <memory>
#include <vector>

#include "population/particle_group.h"
#include "population/walker_set.h"
#include "wavefunction/trial_wavefunction.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/wavefunction_component.h"

using alea::ParticleGroup;
using alea::Parameter;
using alea::PadeJastrow;
using alea::TrialWavefunction;
using alea::WavefunctionComponent;
using alea::WalkerSet;

namespace {

ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = "g";
    g.charge = 0.0;
    g.first = first;
    g.last = last;
    return g;
}

// An inter-group Padé–Jastrow with u(r) = a * r (b = 0).
std::unique_ptr<WavefunctionComponent> make_pade(double a) {
    std::vector<Parameter> params = {{"a", a, true}, {"b", 0.0, true}};
    return std::make_unique<PadeJastrow>(1, 1, group(0, 1), group(1, 2),
                                         std::move(params));
}

}  // namespace

TEST_CASE("TrialWavefunction evaluate_batch zeroes then sums components",
          "[trial_wf]") {
    // p0=(0,0), p1=(2,0), r=2. Components u_A=r, u_B=2r.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 2.0;  // p1 x
    ws.dist_tables().build_all(ws.positions());

    // Garbage that must be overwritten (zeroed) rather than added to.
    ws.log_psi()[0] = 100.0;
    ws.gradients()[0] = 100.0;
    ws.laplacians()[0] = 100.0;

    TrialWavefunction psi;
    psi.add_component(make_pade(1.0));  // u = r
    psi.add_component(make_pade(2.0));  // u = 2r
    psi.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Catch::Approx(6.0));      // u_A(2)+u_B(2) = 2+4
    CHECK(ws.gradients()[0] == Catch::Approx(-3.0));   // (-1) + (-2), p0 x
    // Laplacian per dimension: ∂²_x = 0, ∂²_y = 0.5+1.0 = 1.5
    CHECK(ws.laplacians()[0] == Catch::Approx(0.0));   // p0, x
    CHECK(ws.laplacians()[1] == Catch::Approx(1.5));   // p0, y
}

TEST_CASE("TrialWavefunction ratio_batch zeroes then sums components",
          "[trial_wf]") {
    // p0=(0,0), p1=(3,4), r=5. Move p0 → (3,0), r'=4. Components u_A=r, u_B=2r.
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2);
    auto pos = ws.positions();
    pos[2] = 3.0;
    pos[3] = 4.0;
    ws.dist_tables().build_all(ws.positions());

    std::vector<double> new_pos = {3.0, 0.0};
    ws.propose_move_batch(0, new_pos);

    // Garbage that must be zeroed first.
    ws.log_ratios()[0] = 7.0;
    ws.new_grad()[1] = 7.0;
    ws.new_lap()[0] = 7.0;

    TrialWavefunction psi;
    psi.add_component(make_pade(1.0));  // u = r
    psi.add_component(make_pade(2.0));  // u = 2r
    psi.ratio_batch(ws, 0);

    // log(ψ'/ψ): [u_A(4)-u_A(5)] + [u_B(4)-u_B(5)] = -1 + -2 = -3.
    CHECK(ws.log_ratios()[0] == Catch::Approx(-3.0));
    // new grad_0 y: u'_A(0,-1) + u'_B(0,-2) → -3.
    CHECK(ws.new_grad()[1] == Catch::Approx(-3.0));
    // new lap_0: 0.25 + 0.5 = 0.75.
    CHECK(ws.new_lap()[0] == Catch::Approx(0.75));
}

TEST_CASE("TrialWavefunction full move cycle matches a fresh eval of the proposal",
          "[trial_wf]") {
    auto make_psi = [] {
        TrialWavefunction p;
        p.add_component(make_pade(1.0));
        p.add_component(make_pade(2.0));
        return p;
    };

    // Current config p0=(0,0), p1=(3,4); evaluate, propose p0→(3,0), ratio, accept.
    WalkerSet cur(1, 2, 2);
    auto cp = cur.positions();
    cp[2] = 3.0;
    cp[3] = 4.0;
    cur.dist_tables().build_all(cur.positions());
    TrialWavefunction psi = make_psi();
    psi.evaluate_batch(cur);

    std::vector<double> new_pos = {3.0, 0.0};
    cur.propose_move_batch(0, new_pos);
    psi.ratio_batch(cur, 0);
    std::array<bool, 1> accept = {true};
    psi.apply_move_batch(cur, 0, accept);

    // Proposed config built fresh.
    WalkerSet prop(1, 2, 2);
    auto pp = prop.positions();
    pp[0] = 3.0;
    pp[2] = 3.0;
    pp[3] = 4.0;
    prop.dist_tables().build_all(prop.positions());
    TrialWavefunction ref = make_psi();
    ref.evaluate_batch(prop);

    CHECK(cur.log_psi()[0] == Catch::Approx(prop.log_psi()[0]));
    for (std::size_t k = 0; k < cur.gradients().size(); ++k)
        CHECK(cur.gradients()[k] == Catch::Approx(prop.gradients()[k]).margin(1e-12));
    for (std::size_t k = 0; k < cur.laplacians().size(); ++k)
        CHECK(cur.laplacians()[k] == Catch::Approx(prop.laplacians()[k]).margin(1e-12));
}

TEST_CASE("TrialWavefunction clone is an independent deep copy", "[trial_wf]") {
    TrialWavefunction psi;
    psi.add_component(make_pade(1.0));
    psi.add_component(make_pade(2.0));
    TrialWavefunction copy = psi.clone();

    auto eval_log_psi = [](TrialWavefunction& wf) {
        WalkerSet ws(1, 2, 2);
        auto pos = ws.positions();
        pos[2] = 2.0;  // r = 2
        ws.dist_tables().build_all(ws.positions());
        wf.evaluate_batch(ws);
        return ws.log_psi()[0];
    };

    // The clone evaluates correctly, and the original still works afterwards
    // (clone is const → it cannot have moved the components out: deep copy).
    CHECK(eval_log_psi(copy) == Catch::Approx(6.0));
    CHECK(eval_log_psi(psi) == Catch::Approx(6.0));
}
