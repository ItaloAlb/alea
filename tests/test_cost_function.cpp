// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <Eigen/Dense>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <span>
#include <vector>

#include "qmc/cost_function.h"
#include "qmc/linear_method.h"

#include <cmath>
#include <random>

using alea::CostFunction;

TEST_CASE("CostFunction accumulates samples and builds matrices", "[cost_function]") {
    // 2 parameters, 3 samples with known values.
    // d0 = [1, 0, -1], d1 = [0, 1, 1], E_L = [2, 3, 4].
    const std::size_t n_params = 2;
    const std::size_t nw = 3;

    std::vector<double> energies = {2.0, 3.0, 4.0};
    // Layout: derivs[k * nw + w]
    std::vector<double> derivs = {
        1.0, 0.0, -1.0,   // d0
        0.0, 1.0,  1.0    // d1
    };

    CostFunction cf;
    cf.add_samples(energies, derivs, n_params, nw);

    REQUIRE(cf.n_samples() == 3);
    REQUIRE(cf.mean_energy() == Catch::Approx(3.0));  // (2+3+4)/3

    Eigen::MatrixXd Sbar, Hbar;
    cf.build_matrices(Sbar, Hbar, n_params);

    // The basis is SEMI-ORTHOGONALISED: Psi_bar_i = Psi_i - <d_i> Psi, i.e. the
    // component of each derivative along Psi itself is projected out.  That
    // component is pure normalisation (see the invariance test below), so
    // everything below is written in the centred deviations
    //     D_i = d_i - <d_i>,      dE = E_L - <E_L>.
    //
    //   Sbar_00 = 1     Sbar_0i = 0     Sbar_ij = <D_i D_j>
    //   Hbar_00 = <E_L> Hbar_0i = Hbar_i0 = <D_i dE>
    //   Hbar_ij = <D_i dE D_j>
    //
    // Hbar_0i equals Hbar_i0 because <dE_L/dp> vanishes identically for a real
    // Psi and a hermitian H — the same identity behind the energy-gradient
    // formula 2<dE D_i>.
    REQUIRE(Sbar.rows() == 3);
    REQUIRE(Sbar.cols() == 3);

    // <d0> = 0, <d1> = 2/3  =>  D0 = [1, 0, -1],  D1 = [-2/3, 1/3, 1/3].
    // <D0 D0> = (1+0+1)/3 = 2/3
    // <D0 D1> = (-2/3 + 0 - 1/3)/3 = -1/3
    // <D1 D1> = (4/9 + 1/9 + 1/9)/3 = 2/9
    CHECK(Sbar(0, 0) == Catch::Approx(1.0));
    CHECK(Sbar(0, 1) == Catch::Approx(0.0).margin(1e-12));
    CHECK(Sbar(0, 2) == Catch::Approx(0.0).margin(1e-12));
    CHECK(Sbar(1, 0) == Catch::Approx(0.0).margin(1e-12));
    CHECK(Sbar(2, 0) == Catch::Approx(0.0).margin(1e-12));
    CHECK(Sbar(1, 1) == Catch::Approx(2.0 / 3.0));
    CHECK(Sbar(1, 2) == Catch::Approx(-1.0 / 3.0));
    CHECK(Sbar(2, 1) == Catch::Approx(-1.0 / 3.0));
    CHECK(Sbar(2, 2) == Catch::Approx(2.0 / 9.0));

    // <E_L> = 3  =>  dE = [-1, 0, 1].
    // <D0 dE> = (1*(-1) + 0 + (-1)*1)/3 = -2/3
    // <D1 dE> = ((-2/3)(-1) + 0 + (1/3)(1))/3 = 1/3
    //
    // Hbar_ij is <D_i D_j E_L>, NOT the energy-shifted <D_i dE D_j>: the
    // identity (H Psi_j)/Psi = ∂_j E_L + E_L d_j leaves no shift, and Hbar_00
    // is <E_L> unshifted, so shifting only the ij block would mix conventions.
    // <D_i D_j E_L> = <D_i dE D_j> + <E_L><D_i D_j>:
    //   (0) + 3*(2/3) = 2        (1/9) + 3*(-1/3) = -8/9
    //   (-1/9) + 3*(2/9) = 5/9
    CHECK(Hbar(0, 0) == Catch::Approx(3.0));
    CHECK(Hbar(0, 1) == Catch::Approx(-2.0 / 3.0));
    CHECK(Hbar(1, 0) == Catch::Approx(-2.0 / 3.0));
    CHECK(Hbar(0, 2) == Catch::Approx(1.0 / 3.0));
    CHECK(Hbar(2, 0) == Catch::Approx(1.0 / 3.0));
    CHECK(Hbar(1, 1) == Catch::Approx(2.0));
    CHECK(Hbar(1, 2) == Catch::Approx(-8.0 / 9.0));
    CHECK(Hbar(2, 1) == Catch::Approx(-8.0 / 9.0));
    CHECK(Hbar(2, 2) == Catch::Approx(5.0 / 9.0));
}

TEST_CASE("CostFunction clear resets all state", "[cost_function]") {
    std::vector<double> energies = {1.0};
    std::vector<double> derivs = {2.0};

    CostFunction cf;
    cf.add_samples(energies, derivs, 1, 1);
    REQUIRE(cf.n_samples() == 1);

    cf.clear();
    REQUIRE(cf.n_samples() == 0);
}

TEST_CASE("CostFunction merge combines two cost functions", "[cost_function]") {
    CostFunction a, b;
    std::vector<double> e_a = {1.0, 2.0};
    std::vector<double> d_a = {0.5, -0.5};  // 1 param
    a.add_samples(e_a, d_a, 1, 2);

    std::vector<double> e_b = {3.0};
    std::vector<double> d_b = {1.0};
    b.add_samples(e_b, d_b, 1, 1);

    a.merge(b);
    REQUIRE(a.n_samples() == 3);
    REQUIRE(a.mean_energy() == Catch::Approx(2.0));  // (1+2+3)/3

    Eigen::MatrixXd Sbar, Hbar;
    a.build_matrices(Sbar, Hbar, 1);

    // <d> = (0.5 - 0.5 + 1.0)/3 = 1/3, so the merged variance is
    // <d²> - <d>² = (0.25 + 0.25 + 1)/3 - 1/9 = 1/2 - 1/9 = 7/18.
    CHECK(Sbar(0, 1) == Catch::Approx(0.0).margin(1e-12));
    CHECK(Sbar(1, 1) == Catch::Approx(7.0 / 18.0));
}

// ═══════════════════════════════════════════════════════════════════════════
//  The normalisation gauge
//
//  Psi(R, p) is defined only up to a p-dependent normalisation N(p): VMC
//  averages over |Psi|²/∫|Psi|², so N(p) cancels out of every expectation
//  value and cannot change the variational problem.  But it DOES change the
//  quantity the optimiser consumes,
//
//      d_i = ∂lnPsi/∂p_i  ->  d_i + ∂lnN/∂p_i = d_i + c,
//
//  so any correct linear method must return the same step under d_i -> d_i + c.
//  Semi-orthogonalising the derivative basis (Psi_i - <d_i>Psi) is exactly the
//  projection that removes that gauge component; the centred moments it leaves
//  behind are invariant by construction.
//
//  This matters far beyond bookkeeping: <d_i> is EXTENSIVE for the parameters
//  that matter here (a Gaussian width gives <d> = -N/2; a Jastrow coefficient
//  sums over pairs), while the informative fluctuation only grows as sqrt(N).
//  Leaving the gauge part in divides the step by ~(1 + <d>²/Var(d)) = ~(1 + N),
//  which is what made a 61-particle crystal's width unoptimisable in practice.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("the linear-method step is invariant under a normalisation shift",
          "[cost_function][gauge]") {
    const std::size_t ns = 4000;
    std::mt19937_64 rng(1);
    std::normal_distribution<double> g(0.0, 1.0);

    std::vector<double> d(ns), e(ns);
    for (std::size_t i = 0; i < ns; ++i) {
        d[i] = g(rng);
        e[i] = -0.2 + 0.05 * d[i] + 0.01 * g(rng);
    }

    auto step_with_offset = [&](double c) {
        std::vector<double> dc(ns);
        for (std::size_t i = 0; i < ns; ++i) dc[i] = d[i] + c;
        CostFunction cf;
        cf.add_samples(e, dc, 1, ns);
        Eigen::MatrixXd S, H;
        cf.build_matrices(S, H, 1);
        return alea::LinearMethod(alea::LinearMethodConfig{}).solve(S, H).update[0];
    };

    const double ref = step_with_offset(0.0);
    REQUIRE(std::abs(ref) > 1e-6);   // a step worth being invariant about
    for (double c : {1.0, -2.0, 3.5, 10.0, -30.5})
        CHECK(step_with_offset(c) == Catch::Approx(ref).epsilon(1e-10));
}

TEST_CASE("the step follows its closed form in particle number",
          "[cost_function][gauge]") {
    // N non-interacting particles in a harmonic trap, one shared width a = 1/4.
    // With r² exponential of mean 1/(2a) = 2 the moments are exact, so is the
    // 2x2 problem:
    //   Var(d) = N/4      <E_L> = 5N/4      cov(d,E_L) = -3N/8
    //   Hbar_11 = <D² E_L> + <D ∂E_L> = 5N²/16 + 5N/8
    //   =>  0.09375 N x² + 0.625 x - 0.375 = 0
    //   =>  x(N) = [-0.625 + sqrt(0.390625 + 0.140625 N)] / (0.1875 N)
    //
    // x ~ 2/sqrt(N) for large N.  That decay is the linear method's own
    // linearisation error for an EXTENSIVE parameter — Psi + x dPsi cannot
    // represent e^{x theta} applied to N particles — and not the gauge defect,
    // which was ~1/(1+N) on top of it (0.645 at N = 1 down to 0.016 at N = 61).
    const std::size_t ns = 400000;
    const double a = 0.25;

    auto step_for = [&](std::size_t N) {
        std::mt19937_64 rng(20260924);
        std::exponential_distribution<double> r2(2.0 * a);
        std::vector<double> d(ns), e(ns), de(ns);
        for (std::size_t i = 0; i < ns; ++i) {
            double sum_r2 = 0.0;
            for (std::size_t k = 0; k < N; ++k) sum_r2 += r2(rng);
            d[i]  = -a * sum_r2;
            e[i]  = 0.5 * N + 0.375 * sum_r2;
            de[i] = 0.5 * N - 0.25 * sum_r2;
        }
        CostFunction cf;
        cf.add_samples(e, d, de, 1, ns);
        Eigen::MatrixXd S, H;
        cf.build_matrices(S, H, 1);
        alea::LinearMethodConfig lc;
        lc.shift_i = 0.0;
        lc.rescale = 1.0;
        lc.xi      = 1.0;   // the raw eigenvector ratio this closed form is for
        return alea::LinearMethod(lc).solve(S, H).update[0];
    };

    for (std::size_t N : {1u, 7u, 19u, 61u}) {
        const double n = static_cast<double>(N);
        const double exact =
            (-0.625 + std::sqrt(0.390625 + 0.140625 * n)) / (0.1875 * n);
        CHECK(step_for(N) == Catch::Approx(exact).epsilon(0.03));
    }
}

// ── The curvature term <D_i ∂_j E_L> ───────────────────────────────────────
// Hbar_ij = <D_i dE D_j> + <D_i ∂_j E_L>.  The second term is what carries
// curvature; without it the step direction is right but its length is not.
// ∂_j E_L is gauge-invariant on its own (E_L = HPsi/Psi is), so adding it keeps
// the invariance above.

TEST_CASE("Hbar picks up the local-energy derivative term",
          "[cost_function][curvature]") {
    // Same 3 samples as the first test, plus ∂E_L/∂p.
    const std::size_t n_params = 2, nw = 3;
    std::vector<double> energies = {2.0, 3.0, 4.0};
    std::vector<double> derivs   = {1.0, 0.0, -1.0,
                                    0.0, 1.0,  1.0};
    std::vector<double> el_derivs = {2.0, -1.0, 0.0,    // ∂E_L/∂p0
                                     1.0,  1.0, 4.0};   // ∂E_L/∂p1

    CostFunction cf;
    cf.add_samples(energies, derivs, el_derivs, n_params, nw);
    Eigen::MatrixXd Sbar, Hbar;
    cf.build_matrices(Sbar, Hbar, n_params);

    // D0 = [1, 0, -1], D1 = [-2/3, 1/3, 1/3]; <∂_0 E_L> = 1/3, <∂_1 E_L> = 2.
    // <D0 ∂_0 E_L> = (1*2 + 0 + (-1)*0)/3 = 2/3
    // <D0 ∂_1 E_L> = (1*1 + 0 + (-1)*4)/3 = -1
    // <D1 ∂_0 E_L> = ((-2/3)*2 + (1/3)(-1) + (1/3)(0))/3 = -5/9
    // <D1 ∂_1 E_L> = ((-2/3)*1 + (1/3)(1) + (1/3)(4))/3 = 1/3
    // Base values (no curvature) are 2, -8/9, -8/9, 5/9 from the first test.
    CHECK(Hbar(1, 1) == Catch::Approx(2.0 + 2.0 / 3.0));
    CHECK(Hbar(1, 2) == Catch::Approx(-8.0 / 9.0 - 1.0));
    CHECK(Hbar(2, 1) == Catch::Approx(-8.0 / 9.0 - 5.0 / 9.0));
    CHECK(Hbar(2, 2) == Catch::Approx(5.0 / 9.0 + 1.0 / 3.0));

    // Sbar is untouched by the new term.
    CHECK(Sbar(1, 1) == Catch::Approx(2.0 / 3.0));
    CHECK(Sbar(2, 2) == Catch::Approx(2.0 / 9.0));
}

TEST_CASE("the curvature term keeps the normalisation invariance",
          "[cost_function][curvature][gauge]") {
    const std::size_t ns = 4000;
    std::mt19937_64 rng(5);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> d(ns), e(ns), de(ns);
    for (std::size_t i = 0; i < ns; ++i) {
        d[i]  = g(rng);
        e[i]  = -0.2 + 0.05 * d[i] + 0.01 * g(rng);
        de[i] = 0.3 - 0.1 * d[i];
    }
    auto step = [&](double c) {
        std::vector<double> dc(ns);
        for (std::size_t i = 0; i < ns; ++i) dc[i] = d[i] + c;
        CostFunction cf;
        cf.add_samples(e, dc, de, 1, ns);
        Eigen::MatrixXd S, H;
        cf.build_matrices(S, H, 1);
        return alea::LinearMethod(alea::LinearMethodConfig{}).solve(S, H).update[0];
    };
    const double ref = step(0.0);
    for (double c : {1.0, -2.0, 10.0, -30.5})
        CHECK(step(c) == Catch::Approx(ref).epsilon(1e-10));
}

TEST_CASE("the linear-method step hits its closed form in a harmonic trap",
          "[cost_function][curvature]") {
    // One particle, 2D, psi = exp(-a r²), H = -½∇² + ½r², at a = 1/4.
    // |psi|² ~ e^{-2a r²} makes r² exponential with mean 1/(2a) = 2, so
    //   <r²> = 2,  Var(r²) = 4,  third central moment = 16,
    // and with theta = ln a:
    //   d     = -a r²            => Var(d)   = 1/4,  <E_L> = 5/4
    //   E_L   = 1/2 + 3r²/8      => cov(d,E) = -3/8,  <D dE D> = 3/8
    //   ∂E_L  = 1/2 - r²/4       => <D ∂E_L> = 1/4
    // Hbar_11 = <D² E_L> + <D ∂E_L> = (3/8 + (5/4)(1/4)) + 1/4 = 15/16, and the
    // 2x2 problem reduces to
    //   3x² + 20x - 12 = 0   =>   x = (2 sqrt(34) - 10)/3 = 0.5539684...
    // Close to, but not equal to, the Newton step -E'/E'' = 0.6: the linear
    // method is exact for a wavefunction linear in the parameters, not for a
    // quadratic energy.
    const std::size_t ns = 400000;
    const double a = 0.25;
    std::mt19937_64 rng(31415);
    std::exponential_distribution<double> r2(2.0 * a);
    std::vector<double> d(ns), e(ns), de(ns);
    for (std::size_t i = 0; i < ns; ++i) {
        const double r = r2(rng);
        d[i]  = -a * r;
        e[i]  = 0.5 + 0.375 * r;
        de[i] = 0.5 - 0.25 * r;
    }
    CostFunction cf;
    cf.add_samples(e, d, de, 1, ns);
    Eigen::MatrixXd S, H;
    cf.build_matrices(S, H, 1);

    alea::LinearMethodConfig lc;
    lc.shift_i = 0.0;
    lc.rescale = 1.0;   // raw eigenvector ratio
    lc.xi      = 1.0;
    const double x = alea::LinearMethod(lc).solve(S, H).update[0];
    const double exact = (2.0 * std::sqrt(34.0) - 10.0) / 3.0;
    CHECK(x == Catch::Approx(exact).epsilon(0.02));
}

TEST_CASE("CostFunction reports a standard error for the mean energy",
          "[cost_function]") {
    std::vector<double> e = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> d(4, 0.0);
    CostFunction cf;
    cf.add_samples(e, d, 1, 4);
    CHECK(cf.mean_energy() == Catch::Approx(2.5));
    // population variance = 1.25, so the error of the mean is sqrt(1.25/4).
    CHECK(cf.mean_energy_error() == Catch::Approx(std::sqrt(1.25 / 4.0)));
}
