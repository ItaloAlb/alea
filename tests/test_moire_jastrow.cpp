// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "hamiltonian/moire.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "wavefunction/moire_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using alea::MoireJastrow;
using alea::Parameter;
using alea::TrialWavefunction;
using alea::WalkerSet;
using Catch::Approx;

namespace {

constexpr std::size_t kDim = 2;
const double          kPi  = std::acos(-1.0);

std::vector<double> shells(double a_M, double theta0, std::size_t n) {
    return alea::moire_jastrow_shell_vectors(
        alea::moire_reciprocal_vectors(a_M, theta0, kDim), n, kDim);
}

std::vector<Parameter> coeffs(std::vector<double> v, bool fixed = false) {
    std::vector<Parameter> p;
    for (std::size_t i = 0; i < v.size(); ++i)
        p.push_back({(i % 2 ? "b" : "a") + std::to_string(i / 2 + 1), v[i], fixed});
    return p;
}

// u(r) for one particle, written out independently of the implementation.
double u_ref(const std::vector<double>& k, const std::vector<double>& ab,
             double x, double y) {
    double u = 0.0;
    for (std::size_t kk = 0; kk < k.size() / kDim; ++kk) {
        const double ph = k[kk * kDim] * x + k[kk * kDim + 1] * y;
        u += ab[2 * (kk / 3)] * std::cos(ph) + ab[2 * (kk / 3) + 1] * std::sin(ph);
    }
    return u;
}

void set_pos(WalkerSet& ws, std::size_t p, std::size_t w, double x, double y) {
    const std::size_t nw = ws.num_walkers();
    ws.positions()[p * kDim * nw + 0 * nw + w] = x;
    ws.positions()[p * kDim * nw + 1 * nw + w] = y;
}

}  // namespace

TEST_CASE("moire jastrow shells have magnitudes |g|, sqrt3|g|, 2|g|",
          "[moire_jastrow][geometry]") {
    const double a_M = 7.0;
    const double g   = 4.0 * kPi / (std::sqrt(3.0) * a_M);
    const auto   k   = shells(a_M, 0.2, 3);
    REQUIRE(k.size() == 9 * kDim);
    const double expect[3] = {g, std::sqrt(3.0) * g, 2.0 * g};
    for (std::size_t kk = 0; kk < 9; ++kk)
        CHECK(std::hypot(k[kk * kDim], k[kk * kDim + 1]) ==
              Approx(expect[kk / 3]));
}

TEST_CASE("moire jastrow rejects params that do not match the shells",
          "[moire_jastrow]") {
    CHECK_THROWS(MoireJastrow(kDim, shells(5.0, 0.0, 2), {{0, 1}},
                              coeffs({0.1, 0.2})));
    CHECK_THROWS(MoireJastrow(kDim, shells(5.0, 0.0, 1), {{0, 1}},
                              coeffs({0.1, 0.2, 0.3})));
}

TEST_CASE("moire jastrow log psi, gradient and laplacian match the closed form",
          "[moire_jastrow]") {
    const double a_M = 6.0;
    const auto   k   = shells(a_M, 0.3, 2);
    const std::vector<double> ab = {0.21, -0.13, 0.05, 0.08};
    const double x = 1.3, y = -0.7, h = 1e-4;

    WalkerSet ws(/*nw=*/1, /*np=*/1, kDim);
    set_pos(ws, 0, 0, x, y);
    MoireJastrow jas(kDim, k, {{0, 1}}, coeffs(ab));
    jas.evaluate_batch(ws);

    CHECK(ws.log_psi()[0] == Approx(u_ref(k, ab, x, y)));
    // Central finite differences of the reference u.
    const double gx = (u_ref(k, ab, x + h, y) - u_ref(k, ab, x - h, y)) / (2 * h);
    const double gy = (u_ref(k, ab, x, y + h) - u_ref(k, ab, x, y - h)) / (2 * h);
    const double u0 = u_ref(k, ab, x, y);
    const double lx =
        (u_ref(k, ab, x + h, y) - 2 * u0 + u_ref(k, ab, x - h, y)) / (h * h);
    const double ly =
        (u_ref(k, ab, x, y + h) - 2 * u0 + u_ref(k, ab, x, y - h)) / (h * h);
    CHECK(ws.gradients()[0] == Approx(gx).epsilon(1e-6));
    CHECK(ws.gradients()[1] == Approx(gy).epsilon(1e-6));
    CHECK(ws.laplacians()[0] == Approx(lx).epsilon(1e-5));
    CHECK(ws.laplacians()[1] == Approx(ly).epsilon(1e-5));
}

TEST_CASE("moire jastrow is periodic under a moire lattice translation",
          "[moire_jastrow][periodic]") {
    // A1 is a moiré lattice vector (g_i · A1 ∈ 2πZ for every first-shell g_i),
    // so every shell, and hence log psi, is invariant under r → r + A1.
    const double a_M = 5.0, theta0 = 0.4;
    const auto   k   = shells(a_M, theta0, 3);
    const auto   g   = alea::moire_reciprocal_vectors(a_M, theta0, kDim);
    const double det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    const double A1x = 2 * kPi * g[1][1] / det, A1y = -2 * kPi * g[1][0] / det;
    const std::vector<double> ab = {0.3, -0.2, 0.1, 0.05, -0.04, 0.02};
    CHECK(u_ref(k, ab, 0.9 + 3 * A1x, -1.1 + 3 * A1y) ==
          Approx(u_ref(k, ab, 0.9, -1.1)).margin(1e-12));
}

TEST_CASE("moire jastrow acts only on its own particles",
          "[moire_jastrow]") {
    const auto k = shells(4.0, 0.0, 1);
    WalkerSet ws(/*nw=*/1, /*np=*/2, kDim);
    set_pos(ws, 0, 0, 0.5, 0.5);
    set_pos(ws, 1, 0, 1.7, -0.3);
    MoireJastrow jas(kDim, k, {{1, 2}}, coeffs({0.4, 0.1}));
    jas.evaluate_batch(ws);
    CHECK(ws.log_psi()[0] == Approx(u_ref(k, {0.4, 0.1}, 1.7, -0.3)));
    CHECK(ws.gradients()[0] == 0.0);   // particle 0 untouched
    CHECK(ws.gradients()[1] == 0.0);
}

TEST_CASE("moire jastrow ratio and accept/reject agree with full re-evaluation",
          "[moire_jastrow]") {
    const auto k = shells(5.0, 0.1, 2);
    const std::vector<double> ab = {0.25, -0.1, 0.07, 0.03};
    const std::size_t nw = 3, np = 2;

    WalkerSet ws(nw, np, kDim);
    for (std::size_t w = 0; w < nw; ++w) {
        set_pos(ws, 0, w, 0.3 * w, -0.2);
        set_pos(ws, 1, w, 1.0, 0.4 + 0.1 * w);
    }
    ws.dist_tables().build_all(ws.positions());
    TrialWavefunction psi;
    psi.add_component(std::make_unique<MoireJastrow>(kDim, k,
        std::vector<std::pair<std::size_t, std::size_t>>{{0, 2}}, coeffs(ab)));
    psi.evaluate_batch(ws);
    const std::vector<double> lp_old(ws.log_psi().begin(), ws.log_psi().end());

    std::vector<double> new_p1(kDim * nw);
    for (std::size_t w = 0; w < nw; ++w) {
        new_p1[0 * nw + w] = 2.2 - 0.4 * w;
        new_p1[1 * nw + w] = -0.9;
    }
    ws.propose_move_batch(1, new_p1);
    psi.ratio_batch(ws, 1);
    for (std::size_t w = 0; w < nw; ++w) {
        const double du = u_ref(k, ab, 2.2 - 0.4 * w, -0.9) -
                          u_ref(k, ab, 1.0, 0.4 + 0.1 * w);
        CHECK(ws.log_ratios()[w] == Approx(du));
    }

    // Accept walkers 0 and 2, reject 1; then the incrementally updated state
    // must equal a from-scratch evaluation of the resulting configuration.
    const bool accept[3] = {true, false, true};
    const std::span<const bool> mask(accept, 3);
    psi.apply_move_batch(ws, 1, mask);
    ws.apply_move(1, ws.new_pos(), ws.new_grad(), ws.new_lap(), mask);
    const std::vector<double> lp(ws.log_psi().begin(), ws.log_psi().end());
    const std::vector<double> gr(ws.gradients().begin(), ws.gradients().end());
    const std::vector<double> la(ws.laplacians().begin(), ws.laplacians().end());

    psi.evaluate_batch(ws);
    for (std::size_t w = 0; w < nw; ++w) CHECK(lp[w] == Approx(ws.log_psi()[w]));
    for (std::size_t i = 0; i < gr.size(); ++i) {
        CHECK(gr[i] == Approx(ws.gradients()[i]).margin(1e-12));
        CHECK(la[i] == Approx(ws.laplacians()[i]).margin(1e-12));
    }
    CHECK(lp[1] == Approx(lp_old[1]));   // the rejected walker kept its value
}

TEST_CASE("moire jastrow cached shares stay exact across moves, wraps, branching and clones",
          "[moire_jastrow][periodic]") {
    // ratio_batch reads the committed share u(r_p) instead of evaluating the
    // old position, so that cache must follow every accepted and rejected
    // move, the wrap of accepted positions into the cell, a branching
    // permutation (duplicate, drop, resize) and a clone.  After each step the
    // incremental state is compared with a fresh evaluation of the stored
    // (wrapped) positions; the cached u is the value at the unwrapped point,
    // equal up to rounding because every k is on the reciprocal lattice.
    const double a_M = 5.0, theta0 = 0.1;
    const auto   k   = shells(a_M, theta0, 2);
    const auto   g   = alea::moire_reciprocal_vectors(a_M, theta0, kDim);
    const double det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    // Cell = 2x2 moire cells: g_i · A_j = 2π δ_ij for the primitive A_j.
    const std::array<double, 4> cell_vectors{
        2 * 2 * kPi * g[1][1] / det, -2 * 2 * kPi * g[1][0] / det,
        -2 * 2 * kPi * g[0][1] / det, 2 * 2 * kPi * g[0][0] / det};
    const alea::SimulationCell cell(kDim, std::span<const double>(cell_vectors));
    const std::vector<double> ab = {0.25, -0.1, 0.07, 0.03};
    const std::size_t np = 2;
    std::size_t nw = 4;

    std::vector<alea::Walker> walkers;
    walkers.reserve(nw);
    for (std::size_t w = 0; w < nw; ++w) {
        alea::Walker wk(np, kDim);
        wk.set_positions(std::vector<double>{0.3 * w, -0.2, 1.0, 0.4 + 0.1 * w});
        walkers.push_back(std::move(wk));
    }
    WalkerSet ws(nw, np, kDim, cell);
    for (auto& wk : walkers) ws.add_walker(wk);
    ws.load_all();
    TrialWavefunction psi;
    psi.add_component(std::make_unique<MoireJastrow>(kDim, k,
        std::vector<std::pair<std::size_t, std::size_t>>{{0, 2}}, coeffs(ab)));
    psi.evaluate_batch(ws);

    double max_share_dev = 0.0;   // largest |s_p(cached) − s_p(fresh)| seen
    double r_max         = 0.0;   // largest |r| of a stored or proposed position
    auto check_against_fresh = [&](const TrialWavefunction& state, const WalkerSet& set) {
        WalkerSet ref(nw, np, kDim, cell);
        std::copy(set.positions().begin(), set.positions().end(), ref.positions().begin());
        ref.dist_tables().build_all(ref.positions());
        TrialWavefunction fresh = state.clone();
        fresh.evaluate_batch(ref);
        for (std::size_t w = 0; w < nw; ++w)
            CHECK(set.log_psi()[w] == Approx(ref.log_psi()[w]).margin(1e-12));
        for (std::size_t i = 0; i < set.gradients().size(); ++i) {
            CHECK(set.gradients()[i] == Approx(ref.gradients()[i]).margin(1e-12));
            CHECK(set.laplacians()[i] == Approx(ref.laplacians()[i]).margin(1e-12));
        }
        // The shares themselves: for one proposal, the ratio from the
        // incremental state minus the ratio from the fresh evaluation is
        // s_p(fresh) − s_p(cached), since both evaluate u(r_p') the same way.
        TrialWavefunction cached = state.clone();
        std::vector<double> probe(kDim * nw);
        for (std::size_t i = 0; i < probe.size(); ++i) probe[i] = 0.37 * i - 1.1;
        for (std::size_t p = 0; p < np; ++p) {
            ref.propose_move_batch(p, probe);
            fresh.ratio_batch(ref, p);
            const std::vector<double> lr_fresh(ref.log_ratios().begin(), ref.log_ratios().end());
            cached.ratio_batch(ref, p);
            for (std::size_t w = 0; w < nw; ++w)
                max_share_dev = std::max(max_share_dev, std::abs(ref.log_ratios()[w] - lr_fresh[w]));
        }
    };

    auto move = [&](std::size_t step) {
        const std::size_t p = step % np;   // each particle moves every other step
        std::vector<double> stored(kDim * nw), proposal(kDim * nw);
        for (std::size_t w = 0; w < nw; ++w)
            for (std::size_t d = 0; d < kDim; ++d) {
                stored[d * nw + w] = ws.positions()[(p * kDim + d) * nw + w];
                // Up to ~1.5 cell lengths away: most accepted moves are wrapped.
                proposal[d * nw + w] = stored[d * nw + w] + 7.0 * std::sin(1.3 * step + w + 2.1 * d);
            }
        for (std::size_t w = 0; w < nw; ++w)
            r_max = std::max({r_max, std::hypot(stored[w], stored[nw + w]),
                              std::hypot(proposal[w], proposal[nw + w])});
        ws.propose_move_batch(p, proposal);
        psi.ratio_batch(ws, p);
        for (std::size_t w = 0; w < nw; ++w)
            CHECK(ws.log_ratios()[w] ==
                  Approx(u_ref(k, ab, proposal[w], proposal[nw + w]) -
                         u_ref(k, ab, stored[w], stored[nw + w])).margin(1e-12));
        std::array<bool, 8> accept{};   // nw <= 5 here
        for (std::size_t w = 0; w < nw; ++w) accept[w] = (w + step) % 3 != 0;
        const std::span<const bool> mask(accept.data(), nw);
        psi.apply_move_batch(ws, p, mask);
        ws.apply_move(p, ws.new_pos(), ws.new_grad(), ws.new_lap(), mask);
    };

    for (std::size_t step = 0; step < 6; ++step) {
        move(step);
        check_against_fresh(psi, ws);
    }

    // Branching: duplicate walker 0, drop walker 1, grow to 5 walkers.
    const std::vector<std::size_t> src = {3, 0, 0, 2, 3};
    ws.apply_permutation(src);
    psi.permute_caches(src);
    nw = src.size();
    check_against_fresh(psi, ws);
    for (std::size_t step = 6; step < 10; ++step) {
        move(step);
        check_against_fresh(psi, ws);
    }

    // The cached share is u at the unwrapped point r, the fresh one u at the
    // wrapped r_w = r − m.  k·m ∈ 2πZ, so the two differ only by rounding: the
    // phases k·r and k·r_w, and the wrap itself, are each exact to a few
    // ε|k|R (R the largest |r| seen), and cos/sin to an ulp, which bounds
    // |Δu| by Σ_k (|a_k| + |b_k|)·16ε(1 + |k|R).
    double share_tol = 0.0;
    for (std::size_t kk = 0; kk < k.size() / kDim; ++kk)
        share_tol += (std::abs(ab[2 * (kk / 3)]) + std::abs(ab[2 * (kk / 3) + 1])) * 16.0 *
                     std::numeric_limits<double>::epsilon() *
                     (1.0 + std::hypot(k[kk * kDim], k[kk * kDim + 1]) * r_max);
    CHECK(max_share_dev <= share_tol);

    // A clone (one per OpenMP thread) carries the shares: its next ratio must
    // equal the original's.
    TrialWavefunction twin = psi.clone();
    std::vector<double> proposal(kDim * nw);
    for (std::size_t i = 0; i < proposal.size(); ++i) proposal[i] = 0.7 * i - 1.9;
    ws.propose_move_batch(1, proposal);
    psi.ratio_batch(ws, 1);
    const std::vector<double> lr(ws.log_ratios().begin(), ws.log_ratios().end());
    twin.ratio_batch(ws, 1);
    for (std::size_t w = 0; w < nw; ++w) CHECK(ws.log_ratios()[w] == lr[w]);
}

TEST_CASE("moire jastrow refuses a ratio before its shares are evaluated",
          "[moire_jastrow]") {
    WalkerSet ws(/*nw=*/2, /*np=*/1, kDim);
    MoireJastrow jas(kDim, shells(5.0, 0.0, 1), {{0, 1}}, coeffs({0.1, 0.2}));
    CHECK_THROWS(jas.ratio_batch(ws, 0));
}

TEST_CASE("moire jastrow parameter derivatives match finite differences",
          "[moire_jastrow][optimize]") {
    const auto k = shells(6.0, 0.25, 2);
    const std::vector<double> ab = {0.2, -0.15, 0.06, 0.04};
    const double x0 = 0.8, y0 = -1.9, x1 = -2.1, y1 = 0.6, h = 1e-6;

    WalkerSet ws(/*nw=*/1, /*np=*/2, kDim);
    set_pos(ws, 0, 0, x0, y0);
    set_pos(ws, 1, 0, x1, y1);

    // Fix b_1 so the optimisable-slot mapping is exercised: slots are a1, a2, b2.
    std::vector<Parameter> par = coeffs(ab);
    par[1].is_fixed = true;
    MoireJastrow jas(kDim, k, {{0, 2}}, par);
    jas.evaluate_batch(ws);
    REQUIRE(jas.num_optimizable_params() == 3);

    std::vector<double> d(3);
    jas.compute_parameter_derivatives(ws, d);
    const std::size_t live[3] = {0, 2, 3};
    for (std::size_t s = 0; s < 3; ++s) {
        auto up = ab, dn = ab;
        up[live[s]] += h;
        dn[live[s]] -= h;
        const double fd = (u_ref(k, up, x0, y0) + u_ref(k, up, x1, y1) -
                           u_ref(k, dn, x0, y0) - u_ref(k, dn, x1, y1)) / (2 * h);
        CHECK(d[s] == Approx(fd).epsilon(1e-6));
    }

    // ∂(grad)/∂p and ∂(lap)/∂p: the model is linear in p, so evaluating with
    // one coefficient set to 1 and the rest 0 gives the exact derivative.
    const std::size_t block = 2 * kDim * 1;
    std::vector<double> dg(3 * block), dl(3 * block);
    REQUIRE(jas.compute_parameter_gradient_derivatives(ws, dg, dl));
    for (std::size_t s = 0; s < 3; ++s) {
        std::vector<double> unit(4, 0.0);
        unit[live[s]] = 1.0;
        WalkerSet w1(1, 2, kDim);
        set_pos(w1, 0, 0, x0, y0);
        set_pos(w1, 1, 0, x1, y1);
        MoireJastrow basis(kDim, k, {{0, 2}}, coeffs(unit));
        basis.evaluate_batch(w1);
        for (std::size_t i = 0; i < block; ++i) {
            CHECK(dg[s * block + i] == Approx(w1.gradients()[i]).margin(1e-12));
            CHECK(dl[s * block + i] == Approx(w1.laplacians()[i]).margin(1e-12));
        }
    }
}

TEST_CASE("moire jastrow at zero coefficients is the identity factor",
          "[moire_jastrow]") {
    WalkerSet ws(/*nw=*/2, /*np=*/1, kDim);
    set_pos(ws, 0, 1, 3.3, -0.4);
    MoireJastrow jas(kDim, shells(5.0, 0.0, 1), {{0, 1}}, coeffs({0.0, 0.0}));
    jas.evaluate_batch(ws);
    for (const double v : ws.log_psi()) CHECK(v == 0.0);
    for (const double v : ws.gradients()) CHECK(v == 0.0);
    for (const double v : ws.laplacians()) CHECK(v == 0.0);
}
