// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Zero-energy pair Jastrow (K₀ core for V = C/r^p, p > 2) and the scaled
// modified Bessel functions it is built on.
//
// Independent references: libstdc++'s std::cyl_bessel_k (Temme's method) and
// the large-x asymptotic series for e^x K_ν; finite differences of log ψ for
// every derivative; a fresh evaluate_batch for every incremental update; and
// the zero-energy equation itself, u″ + u′/r + u′² = b r^{−p}, which says the
// pair's local energy vanishes inside the matching radius.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "numerics/scaled_bessel_k01.h"
#include "numerics/special_functions.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"
#include "wavefunction/trial_wavefunction.h"
#include "wavefunction/zero_energy_jastrow.h"

using alea::Parameter;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::TrialWavefunction;
using alea::WalkerSet;
using alea::ZeroEnergyFunction;
using alea::ZeroEnergyJastrow;
using alea::numerics::ScaledBesselK01;

namespace {

// e^x K_ν(x) from the asymptotic series (DLMF 10.40.2), 8 terms: for x ≥ 1e3
// the first omitted term is below 1e-30.
double asymptotic_scaled_k(double nu, double x) {
    const double mu = 4.0 * nu * nu;
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 8; ++k) {
        term *= (mu - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (k * 8.0 * x);
        sum += term;
    }
    return std::sqrt(std::numbers::pi / (2.0 * x)) * sum;
}

}  // namespace

TEST_CASE("scaled K0 and K1 agree with libstdc++ on both sides of the x = 2 seam",
          "[zero_energy][bessel]") {
    const ScaledBesselK01& k01 = ScaledBesselK01::table();
    for (double x : {1e-3, 0.01, 0.1, 0.5, 1.0, 1.9, 2.0, 2.0000001, 2.5, 5.0, 10.0,
                     30.0, 100.0, 300.0, 600.0}) {
        double k0e = 0.0, k1e = 0.0;
        k01.evaluate(x, k0e, k1e);
        const double e = std::exp(x);
        INFO("x = " << x);
        CHECK(k0e == Catch::Approx(e * std::cyl_bessel_k(0.0, x)).epsilon(1e-13));
        CHECK(k1e == Catch::Approx(e * std::cyl_bessel_k(1.0, x)).epsilon(1e-13));
    }
}

TEST_CASE("scaled K0 and K1 follow the asymptotic series where K itself underflows",
          "[zero_energy][bessel]") {
    const ScaledBesselK01& k01 = ScaledBesselK01::table();
    for (double x : {1e3, 1e4, 1e6, 1e9}) {
        double k0e = 0.0, k1e = 0.0;
        k01.evaluate(x, k0e, k1e);
        INFO("x = " << x);
        CHECK(k0e == Catch::Approx(asymptotic_scaled_k(0.0, x)).epsilon(1e-14));
        CHECK(k1e == Catch::Approx(asymptotic_scaled_k(1.0, x)).epsilon(1e-14));
    }
}

TEST_CASE("the scaled integral reference equals e^x times the unscaled one",
          "[zero_energy][bessel]") {
    for (double nu : {0.0, 1.0})
        for (double x : {0.05, 0.7, 3.0, 20.0}) {
            INFO("nu = " << nu << ", x = " << x);
            CHECK(alea::numerics::bessel_k_scaled(nu, x) ==
                  Catch::Approx(std::exp(x) * alea::numerics::bessel_k(nu, x)).epsilon(1e-13));
        }
}

// ---------------------------------------------------------------------------
//  ZeroEnergyFunction
// ---------------------------------------------------------------------------

TEST_CASE("inside R_m the pair satisfies the zero-energy equation for any p > 2",
          "[zero_energy]") {
    // u″ + u′/r + u′² = b r^{−p}: −(1/2μ)∇²ψ/ψ cancels C/r^p (b = 2μC) exactly.
    for (double p : {3.0, 4.0, 6.0, 12.0})
        for (double b : {0.3, 1.0, 7.0}) {
            const ZeroEnergyFunction f(b, p, 0.8, 2.0);
            for (double r : {0.05, 0.2, 0.5, 0.79}) {
                double u = 0, du = 0, d2u = 0;
                f.evaluate(r, u, du, d2u);
                const double v = b * std::pow(r, -p);
                INFO("p = " << p << ", b = " << b << ", r = " << r);
                CHECK(d2u + du / r + du * du == Catch::Approx(v).epsilon(1e-11));
            }
        }
}

TEST_CASE("for dipoles the core is the paper's K0(2 sqrt(b/r))", "[zero_energy]") {
    // p = 3: κ r^{−s} = 2√(b/r).  Compare differences of u (ln C₁ cancels)
    // with libstdc++'s K₀.
    const double b = 1.7;
    const ZeroEnergyFunction f(b, 3.0, 0.9, 2.0);
    auto lnk0 = [&](double r) { return std::log(std::cyl_bessel_k(0.0, 2.0 * std::sqrt(b / r))); };
    double u1 = 0, u2 = 0, d = 0;
    f.evaluate(0.15, u1, d, d);
    f.evaluate(0.7, u2, d, d);
    CHECK(u1 - u2 == Catch::Approx(lnk0(0.15) - lnk0(0.7)).epsilon(1e-12));
}

TEST_CASE("u and u' are continuous at R_m, and u = u' = 0 from r_c on", "[zero_energy]") {
    const ZeroEnergyFunction f(1.0, 3.0, 0.4, 1.1);
    const double eps = 1e-10;
    double ul, dul, d2ul, ur, dur, d2ur;
    f.evaluate(0.4 * (1 - eps), ul, dul, d2ul);
    f.evaluate(0.4 * (1 + eps), ur, dur, d2ur);
    CHECK(ur == Catch::Approx(ul).epsilon(1e-8));
    CHECK(dur == Catch::Approx(dul).epsilon(1e-8));
    f.evaluate(1.1 * (1 - eps), ul, dul, d2ul);
    CHECK(ul == Catch::Approx(0.0).margin(1e-9));
    CHECK(dul == Catch::Approx(0.0).margin(1e-8));
    for (double r : {1.1, 1.5, 2.2, 3.0, 50.0}) {  // 2r_c − r ≤ 0 must not leak NaN
        f.evaluate(r, ur, dur, d2ur);
        CHECK(ur == 0.0);
        CHECK(dur == 0.0);
        CHECK(d2ur == 0.0);
    }
}

TEST_CASE("u' and u'' match finite differences of u in both regions", "[zero_energy]") {
    const ZeroEnergyFunction f(1.0, 3.0, 0.4, 1.1);
    const double h = 1e-5;
    for (double r : {0.08, 0.2, 0.39, 0.41, 0.7, 1.05}) {
        double u0, du0, d2u0, up, dup, d2up, um, dum, d2um;
        f.evaluate(r, u0, du0, d2u0);
        f.evaluate(r + h, up, dup, d2up);
        f.evaluate(r - h, um, dum, d2um);
        INFO("r = " << r);
        CHECK(du0 == Catch::Approx((up - um) / (2 * h)).epsilon(1e-7));
        CHECK(d2u0 == Catch::Approx((dup - dum) / (2 * h)).epsilon(1e-7));
    }
}

TEST_CASE("the batched evaluation matches the scalar one", "[zero_energy]") {
    const ZeroEnergyFunction f(2.0, 3.0, 0.4, 1.1);
    const std::vector<double> r = {0.05, 0.3, 0.399, 0.4, 0.9, 1.1, 1.7, 2.5};
    std::vector<double> u(r.size()), du(r.size()), d2u(r.size());
    f.evaluate(r, u, du, d2u);
    for (std::size_t w = 0; w < r.size(); ++w) {
        double su, sdu, sd2u;
        f.evaluate(r[w], su, sdu, sd2u);
        INFO("r = " << r[w]);
        CHECK(u[w] == Catch::Approx(su).epsilon(1e-14).margin(1e-14));
        CHECK(du[w] == Catch::Approx(sdu).epsilon(1e-14).margin(1e-14));
        CHECK(d2u[w] == Catch::Approx(sd2u).epsilon(1e-14).margin(1e-14));
    }
}

TEST_CASE("the matching-radius derivative matches finite differences", "[zero_energy]") {
    const double b = 1.3, p = 3.0, r_m = 0.45, r_c = 1.2, h = 1e-6;
    const ZeroEnergyFunction f(b, p, r_m, r_c), fp(b, p, r_m + h, r_c), fm(b, p, r_m - h, r_c);
    for (double r : {0.1, 0.3, 0.6, 1.0, 1.5}) {
        double d, d1, d2;
        f.matching_radius_derivative(r, d, d1, d2);
        double up, dup, d2up, um, dum, d2um;
        fp.evaluate(r, up, dup, d2up);
        fm.evaluate(r, um, dum, d2um);
        INFO("r = " << r);
        CHECK(d == Catch::Approx((up - um) / (2 * h)).epsilon(1e-6).margin(1e-8));
        CHECK(d1 == Catch::Approx((dup - dum) / (2 * h)).epsilon(1e-6).margin(1e-8));
        CHECK(d2 == Catch::Approx((d2up - d2um) / (2 * h)).epsilon(1e-6).margin(1e-6));
    }
}

TEST_CASE("u stays finite down to contact distances", "[zero_energy]") {
    const ZeroEnergyFunction f(1.0, 3.0, 0.4, 1.1);
    for (double r : {1e-3, 1e-6, 1e-9}) {
        double u, du, d2u;
        f.evaluate(r, u, du, d2u);
        CHECK(std::isfinite(u));
        CHECK(std::isfinite(du));
        CHECK(std::isfinite(d2u));
        CHECK(u < -1.0 / std::sqrt(r));  // ψ ~ exp(−2√(b/r))
    }
}

TEST_CASE("ZeroEnergyFunction refuses inputs with no zero-energy solution", "[zero_energy]") {
    CHECK_THROWS_AS(ZeroEnergyFunction(0.0, 3.0, 0.4, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(ZeroEnergyFunction(-1.0, 3.0, 0.4, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(ZeroEnergyFunction(1.0, 2.0, 0.4, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(ZeroEnergyFunction(1.0, 3.0, 1.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(ZeroEnergyFunction(1.0, 3.0, 0.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(ZeroEnergyFunction(1.0, 3.0, 0.4, INFINITY), std::invalid_argument);
}

// ---------------------------------------------------------------------------
//  ZeroEnergyJastrow: the component contract
// ---------------------------------------------------------------------------

namespace {

constexpr double kB = 1.0, kP = 3.0, kRc = 1.1;

ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = "g";
    g.first = first;
    g.last = last;
    return g;
}

ZeroEnergyJastrow make_jastrow(double r_m, bool fixed = true, std::size_t np = 3,
                               double cutoff = kRc) {
    return ZeroEnergyJastrow(kB, kP, cutoff, group(0, np), group(0, np),
                             {Parameter{"R_m", r_m, fixed, true}});
}

// Three particles, three walkers, positions [w][p][d]: walker 0 has a pair
// inside R_m = 0.4, walker 1 a pair beyond r_c, walker 2 everything in the tail.
constexpr std::size_t kNw = 3, kNp = 3, kDim = 2;
constexpr double kXY[kNw][kNp][kDim] = {{{0.0, 0.0}, {0.25, 0.1}, {-0.5, 0.6}},
                                        {{0.1, -0.2}, {1.4, 0.3}, {-0.3, 0.4}},
                                        {{0.0, 0.0}, {0.6, -0.3}, {0.2, 0.75}}};

void place(WalkerSet& ws, const double (&xy)[kNw][kNp][kDim]) {
    auto pos = ws.positions();
    for (std::size_t w = 0; w < kNw; ++w)
        for (std::size_t p = 0; p < kNp; ++p)
            for (std::size_t d = 0; d < kDim; ++d)
                pos[(p * kDim + d) * kNw + w] = xy[w][p][d];
    ws.dist_tables().build_all(ws.positions());
}

double log_psi_at(ZeroEnergyJastrow jas, const double (&xy)[kNw][kNp][kDim],
                  std::size_t w, const SimulationCell& cell = SimulationCell{}) {
    WalkerSet ws(kNw, kNp, kDim, cell);
    place(ws, xy);
    jas.evaluate_batch(ws);
    return jas.log_psi_c()[w];
}

}  // namespace

TEST_CASE("ZeroEnergyJastrow log psi is the sum of u over each pair once",
          "[zero_energy_jastrow]") {
    ZeroEnergyJastrow jas = make_jastrow(0.4);
    WalkerSet ws(kNw, kNp, kDim);
    place(ws, kXY);
    std::fill(ws.log_psi().begin(), ws.log_psi().end(), 2.5);  // must be added to
    jas.evaluate_batch(ws);
    const ZeroEnergyFunction f(kB, kP, 0.4, kRc);
    for (std::size_t w = 0; w < kNw; ++w) {
        double sum = 0.0;
        for (std::size_t i = 0; i < kNp; ++i)
            for (std::size_t j = i + 1; j < kNp; ++j) {
                const double r = std::hypot(kXY[w][i][0] - kXY[w][j][0],
                                            kXY[w][i][1] - kXY[w][j][1]);
                double u, du, d2u;
                f.evaluate(r, u, du, d2u);
                sum += u;
            }
        CHECK(jas.log_psi_c()[w] == Catch::Approx(sum).epsilon(1e-13));
        CHECK(ws.log_psi()[w] == Catch::Approx(sum + 2.5).epsilon(1e-13));
    }
}

TEST_CASE("ZeroEnergyJastrow gradient and laplacian match finite differences of log psi",
          "[zero_energy_jastrow]") {
    ZeroEnergyJastrow jas = make_jastrow(0.4);
    WalkerSet ws(kNw, kNp, kDim);
    place(ws, kXY);
    jas.evaluate_batch(ws);
    const double h = 1e-4;
    for (std::size_t w = 0; w < kNw; ++w)
        for (std::size_t p = 0; p < kNp; ++p)
            for (std::size_t d = 0; d < kDim; ++d) {
                double xy_p[kNw][kNp][kDim], xy_m[kNw][kNp][kDim];
                std::copy(&kXY[0][0][0], &kXY[0][0][0] + kNw * kNp * kDim, &xy_p[0][0][0]);
                std::copy(&kXY[0][0][0], &kXY[0][0][0] + kNw * kNp * kDim, &xy_m[0][0][0]);
                xy_p[w][p][d] += h;
                xy_m[w][p][d] -= h;
                const double lp = log_psi_at(make_jastrow(0.4), xy_p, w);
                const double lm = log_psi_at(make_jastrow(0.4), xy_m, w);
                const double l0 = jas.log_psi_c()[w];
                const std::size_t k = (p * kDim + d) * kNw + w;
                INFO("w = " << w << ", p = " << p << ", d = " << d);
                CHECK(jas.grad_c()[k] == Catch::Approx((lp - lm) / (2 * h)).epsilon(1e-6).margin(1e-7));
                CHECK(jas.lap_c()[k] ==
                      Catch::Approx((lp - 2 * l0 + lm) / (h * h)).epsilon(1e-4).margin(1e-3));
            }
}

TEST_CASE("ZeroEnergyJastrow cancels the C/r^3 core in the two-body local energy",
          "[zero_energy_jastrow]") {
    // Two particles of mass 1 (μ = 1/2, b = 2μC = C) at r < R_m:
    //   E_L = −½ Σ_{i,d} (∂²_d log ψ + (∂_d log ψ)²) + C/r³ = 0.
    const double C = kB;
    ZeroEnergyJastrow jas(kB, kP, kRc, group(0, 1), group(1, 2),
                          {Parameter{"R_m", 0.5, true, true}});
    for (double r : {0.03, 0.1, 0.3, 0.49}) {
        WalkerSet ws(1, 2, 2);
        auto pos = ws.positions();
        pos[2] = 0.6 * r;  // particle 1 at (0.6 r, 0.8 r)
        pos[3] = 0.8 * r;
        ws.dist_tables().build_all(ws.positions());
        jas.evaluate_batch(ws);
        double kinetic = 0.0;
        for (std::size_t k = 0; k < 4; ++k)
            kinetic += -0.5 * (ws.laplacians()[k] + ws.gradients()[k] * ws.gradients()[k]);
        const double v = C / (r * r * r);
        INFO("r = " << r);
        CHECK(kinetic + v == Catch::Approx(0.0).margin(1e-11 * v));
    }
}

TEST_CASE("ZeroEnergyJastrow ratio and commit equal a fresh evaluation, accept and reject",
          "[zero_energy_jastrow]") {
    // Move particle 1 of every walker across R_m and r_c: walker 0 from inside
    // R_m to the tail, walker 1 from beyond r_c to inside R_m, walker 2 stays.
    double moved[kNw][kNp][kDim];
    std::copy(&kXY[0][0][0], &kXY[0][0][0] + kNw * kNp * kDim, &moved[0][0][0]);
    moved[0][1][0] = 0.9;  moved[0][1][1] = 0.2;
    moved[1][1][0] = 0.3;  moved[1][1][1] = -0.1;
    moved[2][1][0] = 0.55; moved[2][1][1] = -0.25;

    for (bool accept_all : {true, false}) {
        ZeroEnergyJastrow jas = make_jastrow(0.4);
        WalkerSet ws(kNw, kNp, kDim);
        place(ws, kXY);
        jas.evaluate_batch(ws);
        const std::vector<double> log0(ws.log_psi().begin(), ws.log_psi().end());
        const std::vector<double> grad0(ws.gradients().begin(), ws.gradients().end());

        std::vector<double> proposal(kDim * kNw);
        for (std::size_t d = 0; d < kDim; ++d)
            for (std::size_t w = 0; w < kNw; ++w) proposal[d * kNw + w] = moved[w][1][d];
        ws.propose_move_batch(1, proposal);
        jas.ratio_batch(ws, 1);

        ZeroEnergyJastrow ref = make_jastrow(0.4);
        WalkerSet wr(kNw, kNp, kDim);
        place(wr, moved);
        ref.evaluate_batch(wr);
        for (std::size_t w = 0; w < kNw; ++w) {
            CHECK(ws.log_ratios()[w] ==
                  Catch::Approx(ref.log_psi_c()[w] - log0[w]).epsilon(1e-12).margin(1e-12));
            for (std::size_t d = 0; d < kDim; ++d) {
                const std::size_t k = (1 * kDim + d) * kNw + w;
                CHECK(ws.new_grad()[d * kNw + w] == Catch::Approx(ref.grad_c()[k]).margin(1e-10));
                CHECK(ws.new_lap()[d * kNw + w] == Catch::Approx(ref.lap_c()[k]).margin(1e-9));
            }
        }

        std::array<bool, kNw> accept{};
        accept.fill(accept_all);
        jas.apply_move_batch(ws, 1, accept);
        for (std::size_t w = 0; w < kNw; ++w) {
            const double expect = accept_all ? ref.log_psi_c()[w] : log0[w] ;
            CHECK(jas.log_psi_c()[w] == Catch::Approx(expect).epsilon(1e-12).margin(1e-12));
        }
        for (std::size_t k = 0; k < grad0.size(); ++k) {
            if (accept_all) {
                CHECK(jas.grad_c()[k] == Catch::Approx(ref.grad_c()[k]).margin(1e-10));
                CHECK(jas.lap_c()[k] == Catch::Approx(ref.lap_c()[k]).margin(1e-9));
            } else {
                CHECK(ws.gradients()[k] == grad0[k]);
            }
        }
    }
}

TEST_CASE("ZeroEnergyJastrow crowd split, permutation and clone reproduce the full batch",
          "[zero_energy_jastrow][crowd]") {
    auto setup = [](WalkerSet& ws, TrialWavefunction& psi) {
        place(ws, kXY);
        psi.add_component(std::make_unique<ZeroEnergyJastrow>(make_jastrow(0.4)));
        psi.evaluate_batch(ws);
        std::vector<double> proposal(kDim * kNw);
        for (std::size_t w = 0; w < kNw; ++w) {
            proposal[0 * kNw + w] = 0.3 + 0.1 * w;
            proposal[1 * kNw + w] = 0.05;
        }
        ws.propose_move_batch(0, proposal);
    };
    WalkerSet wf(kNw, kNp, kDim); TrialWavefunction pf; setup(wf, pf);
    WalkerSet ws(kNw, kNp, kDim); TrialWavefunction ps; setup(ws, ps);
    pf.ratio_batch(wf, 0);
    ps.ratio_batch(ws, 0, 0, 1);
    ps.ratio_batch(ws, 0, 1, 3);
    for (std::size_t w = 0; w < kNw; ++w)
        CHECK(ws.log_ratios()[w] == Catch::Approx(wf.log_ratios()[w]).epsilon(1e-14));
    std::array<bool, kNw> accept = {true, false, true};
    pf.apply_move_batch(wf, 0, accept);
    ps.apply_move_batch(ws, 0, accept, 0, 1);
    ps.apply_move_batch(ws, 0, accept, 1, 3);
    for (std::size_t k = 0; k < kNp * kDim * kNw; ++k)
        CHECK(ws.gradients()[k] == Catch::Approx(wf.gradients()[k]).margin(1e-12));

    // Permutation: column c becomes old column src[c] (reorder, duplicate, drop).
    ZeroEnergyJastrow jas = make_jastrow(0.4);
    WalkerSet wp(kNw, kNp, kDim);
    place(wp, kXY);
    jas.evaluate_batch(wp);
    const std::vector<double> log0(jas.log_psi_c().begin(), jas.log_psi_c().end());
    const std::vector<double> lap0(jas.lap_c().begin(), jas.lap_c().end());
    const std::vector<std::size_t> src = {2, 0, 0, 1};
    jas.permute_caches(src);
    REQUIRE(jas.log_psi_c().size() == src.size());
    for (std::size_t c = 0; c < src.size(); ++c) {
        CHECK(jas.log_psi_c()[c] == log0[src[c]]);
        for (std::size_t k = 0; k < kNp * kDim; ++k)
            CHECK(jas.lap_c()[k * src.size() + c] == lap0[k * kNw + src[c]]);
    }

    // Clone: bit-for-bit, including the rebuilt kernel.
    ZeroEnergyJastrow original = make_jastrow(0.4, /*fixed=*/false);
    original.parameters()[0].value = 0.55;
    original.rebuild();
    auto copy = original.clone();
    WalkerSet wa(kNw, kNp, kDim), wb(kNw, kNp, kDim);
    place(wa, kXY);
    place(wb, kXY);
    original.evaluate_batch(wa);
    copy->evaluate_batch(wb);
    for (std::size_t w = 0; w < kNw; ++w) CHECK(wb.log_psi()[w] == wa.log_psi()[w]);
    for (std::size_t k = 0; k < kNp * kDim * kNw; ++k) {
        CHECK(wb.gradients()[k] == wa.gradients()[k]);
        CHECK(wb.laplacians()[k] == wa.laplacians()[k]);
    }
}

TEST_CASE("ZeroEnergyJastrow rebuild follows R_m and refuses it outside the cutoff",
          "[zero_energy_jastrow][rebuild]") {
    ZeroEnergyJastrow jas = make_jastrow(0.4, /*fixed=*/false);
    const double before = log_psi_at(jas, kXY, 0);
    jas.rebuild();  // unchanged parameter: identical
    CHECK(log_psi_at(jas, kXY, 0) == before);
    jas.parameters()[0].value = 0.6;
    jas.rebuild();
    CHECK(log_psi_at(jas, kXY, 0) == Catch::Approx(log_psi_at(make_jastrow(0.6), kXY, 0)));
    CHECK(log_psi_at(jas, kXY, 0) != Catch::Approx(before));
    jas.parameters()[0].value = kRc;
    CHECK_THROWS_AS(jas.rebuild(), std::runtime_error);
}

TEST_CASE("ZeroEnergyJastrow d log psi / d R_m matches finite differences",
          "[zero_energy_jastrow][derivatives]") {
    ZeroEnergyJastrow jas = make_jastrow(0.4, /*fixed=*/false);
    WalkerSet ws(kNw, kNp, kDim);
    place(ws, kXY);
    jas.evaluate_batch(ws);
    REQUIRE(jas.num_optimizable_params() == 1);
    std::vector<double> derivs(kNw);
    jas.compute_parameter_derivatives(ws, derivs);
    const double h = 1e-6;
    for (std::size_t w = 0; w < kNw; ++w) {
        const double fd = (log_psi_at(make_jastrow(0.4 + h), kXY, w) -
                           log_psi_at(make_jastrow(0.4 - h), kXY, w)) / (2 * h);
        CHECK(derivs[w] == Catch::Approx(fd).epsilon(1e-6).margin(1e-7));
    }

    // A fixed R_m has nothing to report.
    ZeroEnergyJastrow fixed = make_jastrow(0.4, /*fixed=*/true);
    fixed.evaluate_batch(ws);
    CHECK(fixed.num_optimizable_params() == 0);
}

TEST_CASE("ZeroEnergyJastrow parameter gradient derivatives match finite differences",
          "[zero_energy_jastrow][derivatives]") {
    ZeroEnergyJastrow jas = make_jastrow(0.4, /*fixed=*/false);
    WalkerSet ws(kNw, kNp, kDim);
    place(ws, kXY);
    jas.evaluate_batch(ws);
    const std::size_t block = kNp * kDim * kNw;
    std::vector<double> dg(block), dl(block);
    REQUIRE(jas.supports_parameter_gradient_derivatives());
    REQUIRE(jas.compute_parameter_gradient_derivatives(ws, dg, dl));

    const double h = 1e-6;
    ZeroEnergyJastrow up = make_jastrow(0.4 + h), dn = make_jastrow(0.4 - h);
    WalkerSet wu(kNw, kNp, kDim), wd(kNw, kNp, kDim);
    place(wu, kXY);
    place(wd, kXY);
    up.evaluate_batch(wu);
    dn.evaluate_batch(wd);
    for (std::size_t k = 0; k < block; ++k) {
        CHECK(dg[k] == Catch::Approx((up.grad_c()[k] - dn.grad_c()[k]) / (2 * h))
                           .epsilon(1e-5).margin(1e-7));
        CHECK(dl[k] == Catch::Approx((up.lap_c()[k] - dn.lap_c()[k]) / (2 * h))
                           .epsilon(1e-5).margin(1e-6));
    }
}

TEST_CASE("ZeroEnergyJastrow in a periodic cell uses the minimum image and is smooth "
          "across the Wigner-Seitz boundary",
          "[zero_energy_jastrow][periodic]") {
    // Hexagonal cell, side 3: R_WS = L/2 = 1.5, r_c = R_WS.  Particles near
    // opposite edges are close through the boundary.
    const double L = 3.0;
    const std::array<double, 4> vectors{L, 0.0, 0.5 * L, 0.5 * std::sqrt(3.0) * L};
    const SimulationCell cell(2, vectors);
    const double rc = cell.ws_radius();
    auto make = [&] {
        return ZeroEnergyJastrow(kB, kP, rc, group(0, kNp), group(0, kNp),
                                 {Parameter{"R_m", 0.5, true, true}});
    };
    const double xy[kNw][kNp][kDim] = {{{0.1, 0.2}, {2.85, 0.35}, {1.4, 1.3}},
                                       {{1.45, 0.05}, {3.0, 2.55}, {0.2, 1.0}},
                                       {{0.0, 0.0}, {1.386, 0.8}, {0.7, 1.2}}};
    ZeroEnergyJastrow jas = make();
    WalkerSet ws(kNw, kNp, kDim, cell);
    {
        auto pos = ws.positions();
        for (std::size_t w = 0; w < kNw; ++w)
            for (std::size_t p = 0; p < kNp; ++p)
                for (std::size_t d = 0; d < kDim; ++d)
                    pos[(p * kDim + d) * kNw + w] = xy[w][p][d];
        ws.dist_tables().build_all(ws.positions());
    }
    jas.evaluate_batch(ws);
    // Walkers 0 and 1 each have a pair inside R_m only through the boundary;
    // walker 2 has a pair 1.6 apart toward a corner of the hexagon, beyond r_c (u flat).
    CHECK(ws.dist_tables()(0, 0, 1) == Catch::Approx(std::hypot(0.1 - (2.85 - L), 0.2 - 0.35)));
    CHECK(ws.dist_tables()(0, 0, 1) < 0.5);
    CHECK(ws.dist_tables()(1, 0, 1) < 0.5);
    CHECK(ws.dist_tables()(2, 0, 1) == Catch::Approx(std::hypot(1.386, 0.8)));
    CHECK(rc < ws.dist_tables()(2, 0, 1));
    const double h = 1e-4;
    for (std::size_t w = 0; w < kNw; ++w)
        for (std::size_t p = 0; p < kNp; ++p)
            for (std::size_t dd = 0; dd < kDim; ++dd) {
                double xp[kNw][kNp][kDim], xm[kNw][kNp][kDim];
                std::copy(&xy[0][0][0], &xy[0][0][0] + kNw * kNp * kDim, &xp[0][0][0]);
                std::copy(&xy[0][0][0], &xy[0][0][0] + kNw * kNp * kDim, &xm[0][0][0]);
                xp[w][p][dd] += h;
                xm[w][p][dd] -= h;
                const double lp = log_psi_at(make(), xp, w, cell);
                const double lm = log_psi_at(make(), xm, w, cell);
                const std::size_t k = (p * kDim + dd) * kNw + w;
                INFO("w = " << w << ", p = " << p << ", d = " << dd);
                CHECK(jas.grad_c()[k] == Catch::Approx((lp - lm) / (2 * h)).epsilon(1e-6).margin(1e-7));
                CHECK(jas.lap_c()[k] == Catch::Approx((lp - 2 * jas.log_psi_c()[w] + lm) / (h * h))
                                            .epsilon(1e-4).margin(1e-3));
            }
}
