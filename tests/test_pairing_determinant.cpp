// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "numerics/random.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "wavefunction/orbital_set.h"
#include "wavefunction/pair_orbital.h"
#include "wavefunction/pairing_determinant.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;

// Checks of the pairing determinant ψ = Π_b det φ(r_row − r_column)
// (López Ríos et al., PRL 120, 177701 (2018), Eq. S2–S4) that do not trust
// its own algebra: an exact reduction to products of Slater determinants,
// finite differences, and long accepted-move sequences against fresh
// evaluations.

namespace {

constexpr double kSide = 6.0;   // square cell; Wigner–Seitz inradius 3
constexpr double kTwoPi = 6.283185307179586;

ParticleGroup group(const char* name, std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = name;
    g.charge = 0.0;
    g.first = first;
    g.last = last;
    g.count = last - first;
    return g;
}

SimulationCell square_cell() {
    const std::array<double, 4> v{kSide, 0.0, 0.0, kSide};
    return SimulationCell(2, v);
}

// The 9 closed-shell plane waves of the square cell (|n|² ≤ 2): 1, then cos
// and sin at (1,0), (0,1), (1,1), (1,−1), in units of b = 2π/side; plus extra
// cosines at the given extra vectors.
std::unique_ptr<PlaneWaveOrbitalSet> nine_plane_waves(
    std::vector<std::array<int, 2>> extra = {}) {
    const double b = kTwoPi / kSide;
    std::vector<double> k{0.0, 0.0};
    std::vector<char> is_sin{0};
    for (const std::array<int, 2> n : {std::array<int, 2>{1, 0}, {0, 1}, {1, 1}, {1, -1}})
        for (char s : {char(0), char(1)}) {
            k.push_back(b * n[0]);
            k.push_back(b * n[1]);
            is_sin.push_back(s);
        }
    for (const std::array<int, 2> n : extra) {
        k.push_back(b * n[0]);
        k.push_back(b * n[1]);
        is_sin.push_back(0);
    }
    return std::make_unique<PlaneWaveOrbitalSet>(2, std::move(k), std::move(is_sin));
}

std::vector<Parameter> fixed_params(const std::vector<double>& values) {
    std::vector<Parameter> p;
    for (std::size_t i = 0; i < values.size(); ++i)
        p.push_back({"p" + std::to_string(i), values[i], true, false});
    return p;
}

// Walkers with every particle uniform in the cell, or (grid = true) the
// first 9 of each block of 9 particles near the sites of a 3×3 grid: a
// plane-wave matrix of grid points is a discrete Fourier matrix, so the
// Slater factors start well conditioned (random points make F, and the
// pairing matrix F_e W F_hᵀ squares its condition number, ill conditioned).
struct Walkers {
    std::vector<Walker> walkers;
    WalkerSet set;
    Walkers(std::size_t nw, std::size_t np, const SimulationCell& cell, std::uint64_t seed,
            bool grid = false)
        : set(nw, np, 2, cell) {
        numerics::Random rng(seed);
        walkers.reserve(nw);
        for (std::size_t w = 0; w < nw; ++w) {
            std::vector<double> pos(np * 2);
            for (double& x : pos) x = rng.uniform(0.0, kSide);
            if (grid)
                for (std::size_t q = 0; q + 9 <= np; q += 9)
                    for (std::size_t i = 0; i < 9; ++i) {
                        pos[(q + i) * 2]     = 2.0 * (i % 3) + 0.3 * rng.uniform(-1, 1);
                        pos[(q + i) * 2 + 1] = 2.0 * (i / 3) + 0.3 * rng.uniform(-1, 1);
                    }
            walkers.emplace_back(np, 2);
            walkers.back().set_positions(pos);
        }
        for (auto& w : walkers) set.add_walker(w);
        set.load_all();
    }
};

// A fresh WalkerSet holding the positions of `ws` (all walkers).
WalkerSet copy_positions(const WalkerSet& ws, const SimulationCell& cell) {
    WalkerSet fresh(ws.num_walkers(), ws.n_particles(), ws.dim(), cell);
    std::copy(ws.positions().begin(), ws.positions().end(), fresh.positions().begin());
    fresh.dist_tables().build_all(fresh.positions());
    return fresh;
}

void require_close(std::span<const double> a, std::span<const double> b, double tol) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::isfinite(a[i]));
        REQUIRE(a[i] == Catch::Approx(b[i]).epsilon(tol).margin(tol));
    }
}

// Metropolis acceptance of the proposals in `ws` (log_ratios already
// summed): accept when |ψ'/ψ|² > u, u from the test's own seeded generator.
// Sampling with |ψ|² keeps walkers away from the nodes, the regime in which
// every updated inverse must stay accurate.
void metropolis(WalkerSet& ws, numerics::Random& rng) {
    auto accept = ws.accept_mask();
    for (std::size_t w = 0; w < ws.num_walkers(); ++w)
        accept[w] = 2.0 * ws.log_ratios()[w] > std::log(rng.uniform());
}

// Proposes a move of particle p in every walker, runs ratio on both trials,
// accepts by Metropolis on the first trial's ratios, and applies.
void move_both(TrialWavefunction& a, WalkerSet& wa, TrialWavefunction& b, WalkerSet& wb,
               std::size_t p, std::size_t step, double tol, numerics::Random& rng) {
    const std::size_t nw = wa.num_walkers();
    std::vector<double> proposal(2 * nw);
    for (std::size_t d = 0; d < 2; ++d)
        for (std::size_t w = 0; w < nw; ++w)
            proposal[d * nw + w] = wa.positions()[(p * 2 + d) * nw + w]
                                 + 0.2 * std::sin(0.37 * step + 1.3 * d + 2.1 * w);
    wa.propose_move_batch(p, proposal);
    wb.propose_move_batch(p, proposal);
    a.ratio_batch(wa, p);
    b.ratio_batch(wb, p);
    require_close(wa.log_ratios(), wb.log_ratios(), tol);
    require_close(wa.new_grad(), wb.new_grad(), tol);
    require_close(wa.new_lap(), wb.new_lap(), tol);
    for (std::size_t w = 0; w < nw; ++w)
        REQUIRE(std::cos(wa.phase_diffs()[w]) == Catch::Approx(std::cos(wb.phase_diffs()[w])));
    metropolis(wa, rng);
    auto acc_a = wa.accept_mask();
    auto acc_b = wb.accept_mask();
    for (std::size_t w = 0; w < nw; ++w) acc_b[w] = acc_a[w];
    a.apply_move_batch(wa, p, acc_a);
    b.apply_move_batch(wb, p, acc_b);
    wa.apply_move(p, proposal, wa.new_grad(), wa.new_lap(), acc_a);
    wb.apply_move(p, proposal, wb.new_grad(), wb.new_lap(), acc_b);
}

}  // namespace

TEST_CASE("PairOrbital stars hold every vector of one length and k = 0 first",
          "[pairing][orbital]") {
    const SimulationCell cell = square_cell();
    const PairOrbital orb(cell, 4, -1);   // |n|² = 0, 1, 2, 4
    REQUIRE(orb.n_stars() == 4);
    // Half-space vectors: k = 0, then 2 per star of |n|² = 1, 2, 4.
    REQUIRE(orb.n_k() == 7);
    CHECK(orb.multiplicity(0) == 1.0);
    CHECK(orb.multiplicity(1) == 2.0);
    const double b = kTwoPi / kSide;
    CHECK(orb.star_k(1) == Catch::Approx(b));
    CHECK(orb.star_k(2) == Catch::Approx(b * std::sqrt(2.0)));
    CHECK(orb.star_k(3) == Catch::Approx(2.0 * b));
}

TEST_CASE("PairOrbital polynomial is cuspless and its derivatives match finite differences",
          "[pairing][orbital]") {
    const SimulationCell cell(2);   // open: polynomial only
    PairOrbital orb(cell, 0, 4);
    const std::vector<double> c{0.8, 0.2, -0.05, 0.01};   // c0, c2, c3, c4
    orb.set({}, c, 2.5);
    double g, dg, d2g;
    orb.radial(0.0, g, dg, d2g);
    CHECK(g == Catch::Approx(0.8));
    CHECK(dg == Catch::Approx(0.0).margin(1e-14));   // φ′(0) = 0
    orb.radial(2.6, g, dg, d2g);
    CHECK(g == 0.0);
    const double h = 1e-5;
    for (double r : {0.3, 1.1, 2.2}) {
        double gp, dgp, d2gp, gm, dgm, d2gm;
        orb.radial(r, g, dg, d2g);
        orb.radial(r + h, gp, dgp, d2gp);
        orb.radial(r - h, gm, dgm, d2gm);
        CHECK(dg == Catch::Approx((gp - gm) / (2 * h)).epsilon(1e-7));
        CHECK(d2g == Catch::Approx((dgp - dgm) / (2 * h)).epsilon(1e-7));
        // ∂g/∂θ for c0, c2..c4 and L, against changing θ.
        for (std::size_t j = 0; j <= 4; ++j) {
            double hj, dhj, d2hj;
            orb.radial_param(j, r, hj, dhj, d2hj);
            auto shifted = [&](double eps) {
                PairOrbital o2(cell, 0, 4);
                std::vector<double> c2 = c;
                double L = 2.5;
                if (j < 4) c2[j] += eps; else L += eps;
                o2.set({}, c2, L);
                double a, da, d2a;
                o2.radial(r, a, da, d2a);
                return std::array<double, 3>{a, da, d2a};
            };
            const auto up = shifted(h), dn = shifted(-h);
            CHECK(hj == Catch::Approx((up[0] - dn[0]) / (2 * h)).epsilon(1e-6).margin(1e-9));
            CHECK(dhj == Catch::Approx((up[1] - dn[1]) / (2 * h)).epsilon(1e-6).margin(1e-9));
            CHECK(d2hj == Catch::Approx((up[2] - dn[2]) / (2 * h)).epsilon(1e-6).margin(1e-9));
        }
    }
}

TEST_CASE("A pairing determinant over the occupied plane-wave shells is a product of two "
          "Slater determinants", "[pairing][reduction]") {
    // φ(Δ) = Σ_{|n|²≤2} cos(k·Δ) = Σ_m χ_m(r_e) χ_m(r_h) over the 9 real
    // plane waves (weights 1 at k = 0 and 2 for each cos/sin pair), so
    // D = F_e W F_hᵀ and det D = det F_e · det W · det F_h.  log|ψ| then
    // differs from the Slater pair by a constant, and every ratio, gradient
    // and laplacian is identical, through accepted rows and columns.
    const SimulationCell cell = square_cell();
    const std::size_t n = 9, np = 2 * n, nw = 4;
    auto pairing = [&] {
        TrialWavefunction psi;
        std::vector<PairingDeterminant::Block> blocks;
        blocks.push_back({group("e", 0, n), group("h", n, np), nullptr});
        psi.add_component(std::make_unique<PairingDeterminant>(
            PairOrbital(cell, 3, -1), std::move(blocks), fixed_params({1.0, 1.0, 1.0})));
        return psi;
    };
    auto slater = [&] {
        TrialWavefunction psi;
        std::vector<SlaterDeterminant::Block> blocks;
        blocks.push_back({0, n, nine_plane_waves()});
        blocks.push_back({n, np, nine_plane_waves()});
        psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
        return psi;
    };
    Walkers a(nw, np, cell, 31, true), b(nw, np, cell, 31, true);
    TrialWavefunction pa = pairing(), pb = slater();
    pa.evaluate_batch(a.set);
    pb.evaluate_batch(b.set);
    const double offset = a.set.log_psi()[0] - b.set.log_psi()[0];
    for (std::size_t w = 0; w < nw; ++w)
        REQUIRE(a.set.log_psi()[w] - b.set.log_psi()[w] == Catch::Approx(offset).margin(1e-9));
    require_close(a.set.gradients(), b.set.gradients(), 1e-9);
    require_close(a.set.laplacians(), b.set.laplacians(), 1e-9);

    // After moves both sides carry their own Sherman-Morrison roundoff, so
    // the updated fields agree to 1e-8; fresh evaluations of the final
    // configuration differ only by the constant det W.
    numerics::Random rng(101);
    for (std::size_t step = 0; step < 200; ++step)
        move_both(pa, a.set, pb, b.set, step % np, step, 1e-8, rng);
    for (std::size_t w = 0; w < nw; ++w)
        REQUIRE(a.set.log_psi()[w] - b.set.log_psi()[w] == Catch::Approx(offset).margin(1e-8));
    require_close(a.set.gradients(), b.set.gradients(), 1e-8);
    require_close(a.set.laplacians(), b.set.laplacians(), 1e-8);
    WalkerSet fa = copy_positions(a.set, cell), fb = copy_positions(b.set, cell);
    TrialWavefunction qa = pairing(), qb = slater();
    qa.evaluate_batch(fa);
    qb.evaluate_batch(fb);
    for (std::size_t w = 0; w < nw; ++w)
        REQUIRE(fa.log_psi()[w] - fb.log_psi()[w] == Catch::Approx(offset).margin(1e-9));
}

TEST_CASE("An unpaired plane wave completes an N+1/2 block as a Slater determinant with "
          "one more orbital", "[pairing][reduction][unpaired]") {
    // Ten electrons, nine holes: D = [F_e W F_hᵀ | χ(r_e)] = [F_e | χ_e] ·
    // diag(W F_hᵀ, 1), so det D ∝ det[F_e | χ_e] · det F_h.  The same with
    // the extra particle among the holes (extra rows of χ(r_h)).
    const SimulationCell cell = square_cell();
    const std::size_t nw = 3;
    for (const bool extra_electron : {true, false}) {
        CAPTURE(extra_electron);
        const std::size_t ne = extra_electron ? 10 : 9, nh = extra_electron ? 9 : 10;
        const std::size_t np = ne + nh;
        const double b = kTwoPi / kSide;
        auto pairing = [&] {
            TrialWavefunction psi;
            std::vector<PairingDeterminant::Block> blocks;
            blocks.push_back({group("e", 0, ne), group("h", ne, np),
                              std::make_unique<PlaneWaveOrbitalSet>(
                                  2, std::vector<double>{2 * b, 0.0}, std::vector<char>{0})});
            psi.add_component(std::make_unique<PairingDeterminant>(
                PairOrbital(cell, 3, -1), std::move(blocks), fixed_params({1.0, 1.0, 1.0})));
            return psi;
        };
        auto slater = [&] {
            TrialWavefunction psi;
            std::vector<SlaterDeterminant::Block> blocks;
            blocks.push_back({0, ne, extra_electron ? nine_plane_waves({{2, 0}}) : nine_plane_waves()});
            blocks.push_back({ne, np, extra_electron ? nine_plane_waves() : nine_plane_waves({{2, 0}})});
            psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
            return psi;
        };
        Walkers a(nw, np, cell, 47, true), c(nw, np, cell, 47, true);
        TrialWavefunction pa = pairing(), pc = slater();
        pa.evaluate_batch(a.set);
        pc.evaluate_batch(c.set);
        const double offset = a.set.log_psi()[0] - c.set.log_psi()[0];
        for (std::size_t w = 0; w < nw; ++w)
            REQUIRE(a.set.log_psi()[w] - c.set.log_psi()[w] == Catch::Approx(offset).margin(1e-9));
        require_close(a.set.gradients(), c.set.gradients(), 1e-9);
        numerics::Random rng(202);
        for (std::size_t step = 0; step < 120; ++step)
            move_both(pa, a.set, pc, c.set, (5 * step) % np, step, 1e-8, rng);
        require_close(a.set.gradients(), c.set.gradients(), 1e-8);
        require_close(a.set.laplacians(), c.set.laplacians(), 1e-8);
    }
}

namespace {

// Two blocks (e↑h↓, e↓h↑) of 3 pairs each, cosine shells plus a polynomial,
// in the square cell; c0 fixed, every other coefficient and L free.
const std::vector<double> kParams{0.3, 0.2, 0.1, 1.0, 0.2, -0.05, 0.01, 2.8};
// p0 p1 p2 | c0 c2 c3 c4 | L

TrialWavefunction two_block_psi(const SimulationCell& cell, const std::vector<double>& v) {
    std::vector<Parameter> params;
    const char* names[] = {"p0", "p1", "p2", "c0", "c2", "c3", "c4", "L"};
    for (std::size_t i = 0; i < v.size(); ++i)
        params.push_back({names[i], v[i], /*fixed=*/i == 3, false});
    std::vector<PairingDeterminant::Block> blocks;
    blocks.push_back({group("e_up", 0, 3), group("h_dn", 9, 12), nullptr});
    blocks.push_back({group("e_dn", 3, 6), group("h_up", 6, 9), nullptr});
    TrialWavefunction psi;
    psi.add_component(std::make_unique<PairingDeterminant>(PairOrbital(cell, 3, 4),
                                                           std::move(blocks), std::move(params)));
    return psi;
}

}  // namespace

TEST_CASE("Pairing determinant gradients and laplacians match finite differences of log psi",
          "[pairing][derivatives]") {
    const SimulationCell cell = square_cell();
    const std::size_t np = 12, nw = 3;
    Walkers base(nw, np, cell, 5);
    TrialWavefunction psi = two_block_psi(cell, kParams);
    psi.evaluate_batch(base.set);
    // Central differences extrapolated from h and h/2 (Richardson), so the
    // O(h^2) truncation error near a node does not mask a wrong derivative.
    const double h = 2e-4;
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t d = 0; d < 2; ++d) {
            auto log_at = [&](double eps) {
                WalkerSet s = copy_positions(base.set, cell);
                for (std::size_t w = 0; w < nw; ++w) s.positions()[(p * 2 + d) * nw + w] += eps;
                s.dist_tables().build_all(s.positions());
                TrialWavefunction q = two_block_psi(cell, kParams);
                q.evaluate_batch(s);
                return std::vector<double>(s.log_psi().begin(), s.log_psi().end());
            };
            const auto up = log_at(h), dn = log_at(-h), up2 = log_at(h / 2), dn2 = log_at(-h / 2);
            for (std::size_t w = 0; w < nw; ++w) {
                const double l0 = base.set.log_psi()[w];
                const std::size_t idx = (p * 2 + d) * nw + w;
                const double g1 = (up[w] - dn[w]) / (2 * h), g2 = (up2[w] - dn2[w]) / h;
                const double l1 = (up[w] - 2 * l0 + dn[w]) / (h * h);
                const double l2 = (up2[w] - 2 * l0 + dn2[w]) / (h * h / 4);
                CHECK(base.set.gradients()[idx] ==
                      Catch::Approx((4 * g2 - g1) / 3).epsilon(1e-7).margin(1e-7));
                CHECK(base.set.laplacians()[idx] ==
                      Catch::Approx((4 * l2 - l1) / 3).epsilon(1e-4).margin(1e-4));
            }
        }
}

TEST_CASE("Pairing determinant parameter derivatives match finite differences",
          "[pairing][derivatives]") {
    const SimulationCell cell = square_cell();
    const std::size_t np = 12, nw = 3;
    Walkers base(nw, np, cell, 9);
    TrialWavefunction psi = two_block_psi(cell, kParams);
    psi.evaluate_batch(base.set);
    const WavefunctionComponent& comp = *psi.components()[0];
    const std::size_t n_opt = comp.num_optimizable_params();
    REQUIRE(n_opt == 7);
    std::vector<double> derivs(n_opt * nw);
    comp.compute_parameter_derivatives(base.set, derivs);
    const std::size_t block = np * 2 * nw;
    std::vector<double> dg(n_opt * block), dl(n_opt * block);
    REQUIRE(comp.compute_parameter_gradient_derivatives(base.set, dg, dl));

    const double h = 1e-6;
    std::size_t k = 0;
    for (std::size_t i = 0; i < kParams.size(); ++i) {
        if (i == 3) continue;   // c0 fixed
        auto eval = [&](double eps) {
            std::vector<double> v = kParams;
            v[i] += eps;
            WalkerSet s = copy_positions(base.set, cell);
            TrialWavefunction q = two_block_psi(cell, v);
            q.evaluate_batch(s);
            return std::array<std::vector<double>, 3>{
                std::vector<double>(s.log_psi().begin(), s.log_psi().end()),
                std::vector<double>(q.components()[0]->grad_c().begin(),
                                    q.components()[0]->grad_c().end()),
                std::vector<double>(q.components()[0]->lap_c().begin(),
                                    q.components()[0]->lap_c().end())};
        };
        const auto up = eval(h), dn = eval(-h);
        for (std::size_t w = 0; w < nw; ++w)
            CHECK(derivs[k * nw + w] ==
                  Catch::Approx((up[0][w] - dn[0][w]) / (2 * h)).epsilon(1e-5).margin(1e-7));
        for (std::size_t j = 0; j < block; ++j) {
            CHECK(dg[k * block + j] ==
                  Catch::Approx((up[1][j] - dn[1][j]) / (2 * h)).epsilon(1e-5).margin(1e-6));
            CHECK(dl[k * block + j] ==
                  Catch::Approx((up[2][j] - dn[2][j]) / (2 * h)).epsilon(1e-4).margin(1e-5));
        }
        ++k;
    }
}

TEST_CASE("Pairing determinant accepted moves, clones and permutations agree with fresh "
          "evaluations", "[pairing][sequence]") {
    // Rows and columns of both blocks move in turn under Metropolis
    // acceptance, over disjoint walker crowds, through LU rebuilds, a clone
    // and a population permutation with duplicates.
    const SimulationCell cell = square_cell();
    const std::size_t np = 12;
    Walkers cur(5, np, cell, 13);
    TrialWavefunction psi = two_block_psi(cell, kParams);
    psi.evaluate_batch(cur.set);
    WalkerSet& ws = cur.set;
    numerics::Random rng(303);
    for (std::size_t step = 0; step < 600; ++step) {
        if (step == 250) {
            const std::vector<std::size_t> src{4, 0, 0, 2, 1, 3};
            ws.apply_permutation(src);
            psi.permute_caches(src);
            psi = psi.clone();
        }
        const std::size_t nw = ws.num_walkers(), p = (7 * step) % np;
        std::vector<double> proposal(2 * nw);
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                proposal[d * nw + w] = ws.positions()[(p * 2 + d) * nw + w]
                                     + 0.5 * std::sin(0.71 * step + d + 3 * w);
        ws.propose_move_batch(p, proposal);
        psi.ratio_batch(ws, p, 0, nw / 2);
        psi.ratio_batch(ws, p, nw / 2, nw);
        metropolis(ws, rng);
        auto accept = ws.accept_mask();
        psi.apply_move_batch(ws, p, accept, 0, nw / 2);
        psi.apply_move_batch(ws, p, accept, nw / 2, nw);
        ws.apply_move(p, proposal, ws.new_grad(), ws.new_lap(), accept);
        if (step % 50 == 49) {
            CAPTURE(step);
            WalkerSet fresh = copy_positions(ws, cell);
            TrialWavefunction ref = two_block_psi(cell, kParams);
            ref.evaluate_batch(fresh);
            require_close(ws.log_psi(), fresh.log_psi(), 1e-9);
            require_close(ws.gradients(), fresh.gradients(), 1e-8);
            require_close(ws.laplacians(), fresh.laplacians(), 1e-7);
            const auto& mine = dynamic_cast<const PairingDeterminant&>(*psi.components()[0]);
            const auto& theirs = dynamic_cast<const PairingDeterminant&>(*ref.components()[0]);
            for (std::size_t b = 0; b < 2; ++b)
                for (std::size_t w = 0; w < ws.num_walkers(); ++w)
                    require_close(mine.inverse(b, w), theirs.inverse(b, w), 1e-7);
        }
    }
}

TEST_CASE("The pairing determinant vanishes when two row (or two column) particles meet",
          "[pairing][antisymmetry]") {
    // Two equal rows (or columns): the ratio of moving particle 0 onto
    // particle 1 is zero, the P-wave node of a same-spin pair.
    const SimulationCell cell = square_cell();
    const std::size_t np = 12, nw = 2;
    for (const std::size_t mover : {std::size_t{0}, std::size_t{9}}) {
        Walkers cur(nw, np, cell, 21);
        TrialWavefunction psi = two_block_psi(cell, kParams);
        psi.evaluate_batch(cur.set);
        const std::size_t target = mover + 1;   // same group
        std::vector<double> proposal(2 * nw);
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                proposal[d * nw + w] = cur.set.positions()[(target * 2 + d) * nw + w];
        cur.set.propose_move_batch(mover, proposal);
        psi.ratio_batch(cur.set, mover);
        for (std::size_t w = 0; w < nw; ++w) CHECK(cur.set.log_ratios()[w] < -20.0);
    }
}

TEST_CASE("Pairing determinant construction refuses inconsistent blocks",
          "[pairing][refusal]") {
    const SimulationCell cell = square_cell();
    auto make = [&](std::vector<PairingDeterminant::Block> blocks, std::size_t n_params) {
        return PairingDeterminant(PairOrbital(cell, 3, -1), std::move(blocks),
                                  fixed_params(std::vector<double>(n_params, 1.0)));
    };
    std::vector<PairingDeterminant::Block> unbalanced;
    unbalanced.push_back({group("e", 0, 4), group("h", 4, 7), nullptr});
    CHECK_THROWS_AS(make(std::move(unbalanced), 3), std::invalid_argument);
    std::vector<PairingDeterminant::Block> twice;
    twice.push_back({group("e", 0, 3), group("h", 3, 6), nullptr});
    twice.push_back({group("e", 0, 3), group("x", 6, 9), nullptr});
    CHECK_THROWS_AS(make(std::move(twice), 3), std::invalid_argument);
    std::vector<PairingDeterminant::Block> ok;
    ok.push_back({group("e", 0, 3), group("h", 3, 6), nullptr});
    CHECK_THROWS_AS(make(std::move(ok), 2), std::invalid_argument);   // 3 shells, 2 params
    CHECK_THROWS_AS(PairOrbital(SimulationCell(2), 2, -1), std::invalid_argument);  // open + cosines
    CHECK_THROWS_AS(PairOrbital(cell, 0, 0), std::invalid_argument);   // poly_order 0
}

TEST_CASE("an optimizer step holds the pair-orbital cutoff within the inradius",
          "[pairing]") {
    // L is the last parameter [p, c, L]; the step is clamped at the bound,
    // and rebuild() still refuses an L set past it by any other route.
    const SimulationCell cell = square_cell();
    TrialWavefunction psi = two_block_psi(cell, kParams);
    auto& pd = static_cast<PairingDeterminant&>(*psi.components().front());
    pd.bound_cutoff(3.0);
    const Parameter& L = pd.parameters().back();
    CHECK(L.upper_bound == 3.0);
    CHECK(pd.parameters().front().upper_bound == std::numeric_limits<double>::infinity());
    CHECK(alea::stepped_value(L, 1.0) == 3.0);
    pd.parameters().back().value = 3.2;
    CHECK_THROWS_WITH(pd.rebuild(), Catch::Matchers::ContainsSubstring("Wigner"));
}
