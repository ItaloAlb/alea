// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <Eigen/Dense>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <complex>
#include <span>
#include <stdexcept>
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

// ═══════════════════════════════════════════════════════════════════════════
//  Complex samples (a complex trial, real parameters)
// ═══════════════════════════════════════════════════════════════════════════

namespace {

using cplx = std::complex<double>;

// O = d + i dt, E_L = e + i f, dE_L = c + i ct, with correlations between all
// parts so every term of the matrices is exercised.
struct ComplexSamples {
    std::size_t n = 0, ns = 0;
    std::vector<double> e, f, d, dt, c, ct;
};

ComplexSamples random_complex_samples(std::size_t n, std::size_t ns, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    ComplexSamples s;
    s.n = n;
    s.ns = ns;
    s.e.resize(ns);
    s.f.resize(ns);
    for (auto* v : {&s.d, &s.dt, &s.c, &s.ct}) v->resize(n * ns);
    for (std::size_t w = 0; w < ns; ++w) {
        const double x = g(rng), y = g(rng);
        s.e[w] = -0.4 + 0.1 * x + 0.02 * g(rng);
        s.f[w] = 0.07 * y + 0.03 * x;
        for (std::size_t k = 0; k < n; ++k) {
            const double kk = static_cast<double>(k + 1);
            s.d[k * ns + w] = 0.3 * kk + kk * 0.2 * x + 0.1 * g(rng);
            s.dt[k * ns + w] = -0.5 + 0.15 * y - 0.05 * kk * x + 0.1 * g(rng);
            s.c[k * ns + w] = 0.2 * x * y + 0.05 * g(rng);
            s.ct[k * ns + w] = -0.1 * kk * y + 0.05 * g(rng);
        }
    }
    return s;
}

void complex_matrices(const ComplexSamples& s, Eigen::MatrixXd& S, Eigen::MatrixXd& H) {
    CostFunction cf;
    cf.add_complex_samples(s.e, s.f, s.d, s.dt, s.c, s.ct, s.n, s.ns);
    REQUIRE(cf.is_complex());
    cf.build_matrices(S, H, s.n);
}

}  // namespace

TEST_CASE("complex samples with no imaginary parts give the real matrices",
          "[cost_function][complex]") {
    ComplexSamples s = random_complex_samples(3, 500, 11);
    std::fill(s.f.begin(), s.f.end(), 0.0);
    std::fill(s.dt.begin(), s.dt.end(), 0.0);
    std::fill(s.ct.begin(), s.ct.end(), 0.0);
    Eigen::MatrixXd Sc, Hc, Sr, Hr;
    complex_matrices(s, Sc, Hc);
    CostFunction real;
    real.add_samples(s.e, s.d, s.c, s.n, s.ns);
    real.build_matrices(Sr, Hr, s.n);
    for (Eigen::Index i = 0; i < Sr.rows(); ++i)
        for (Eigen::Index j = 0; j < Sr.cols(); ++j) {
            CHECK(Sc(i, j) == Catch::Approx(Sr(i, j)).margin(1e-13));
            CHECK(Hc(i, j) == Catch::Approx(Hr(i, j)).margin(1e-13));
        }
}

TEST_CASE("the complex matrices are the real parts of the centred complex averages",
          "[cost_function][complex]") {
    // Two passes, by definition: means first, then the centred products,
    // against the one-pass expansion onto raw sums in CostFunction.
    const ComplexSamples s = random_complex_samples(3, 400, 12);
    Eigen::MatrixXd S, H;
    complex_matrices(s, S, H);
    const std::size_t n = s.n, ns = s.ns;
    const double nd = static_cast<double>(ns);
    const auto O = [&](std::size_t k, std::size_t w) {
        return cplx(s.d[k * ns + w], s.dt[k * ns + w]);
    };
    const auto dE = [&](std::size_t k, std::size_t w) {
        return cplx(s.c[k * ns + w], s.ct[k * ns + w]);
    };
    std::vector<cplx> m(n);
    cplx e_mean{};
    for (std::size_t w = 0; w < ns; ++w) {
        e_mean += cplx(s.e[w], s.f[w]) / nd;
        for (std::size_t k = 0; k < n; ++k) m[k] += O(k, w) / nd;
    }
    CHECK(H(0, 0) == Catch::Approx(e_mean.real()));
    for (std::size_t i = 0; i < n; ++i) {
        cplx h0{};
        for (std::size_t w = 0; w < ns; ++w) h0 += std::conj(O(i, w) - m[i]) * cplx(s.e[w], s.f[w]);
        CHECK(H(i + 1, 0) == Catch::Approx(h0.real() / nd).margin(1e-13));
        CHECK(H(0, i + 1) == Catch::Approx(h0.real() / nd).margin(1e-13));
        for (std::size_t j = 0; j < n; ++j) {
            cplx sij{}, hij{};
            for (std::size_t w = 0; w < ns; ++w) {
                const cplx Di = O(i, w) - m[i], Dj = O(j, w) - m[j];
                const cplx el(s.e[w], s.f[w]);
                sij += std::conj(Di) * Dj;
                hij += std::conj(Di) * (el * Dj + dE(j, w));
            }
            CHECK(S(i + 1, j + 1) == Catch::Approx(sij.real() / nd).margin(1e-13));
            CHECK(H(i + 1, j + 1) == Catch::Approx(hij.real() / nd).margin(1e-13));
        }
    }
}

TEST_CASE("the complex matrices are invariant under O -> O + a complex constant",
          "[cost_function][complex][gauge]") {
    // A p-dependent normalisation AND a p-dependent global phase e^{iα(p)}
    // shift every sample's O_k by the same complex constant.
    const ComplexSamples s = random_complex_samples(2, 300, 13);
    Eigen::MatrixXd S0, H0;
    complex_matrices(s, S0, H0);
    for (const cplx shift : {cplx(1.5, 0.0), cplx(0.0, -2.0), cplx(-3.0, 7.5)}) {
        ComplexSamples t = s;
        for (std::size_t w = 0; w < t.ns; ++w) {
            t.d[t.ns + w] += shift.real();    // parameter 1 only
            t.dt[t.ns + w] += shift.imag();
        }
        Eigen::MatrixXd S, H;
        complex_matrices(t, S, H);
        for (Eigen::Index i = 0; i < S.rows(); ++i)
            for (Eigen::Index j = 0; j < S.cols(); ++j) {
                CHECK(S(i, j) == Catch::Approx(S0(i, j)).margin(1e-11));
                CHECK(H(i, j) == Catch::Approx(H0(i, j)).margin(1e-11));
            }
    }
}

TEST_CASE("complex accumulations merge, and real and complex ones do not mix",
          "[cost_function][complex]") {
    const ComplexSamples s = random_complex_samples(2, 300, 14);
    Eigen::MatrixXd S_all, H_all;
    complex_matrices(s, S_all, H_all);
    // The same samples in two halves, merged.
    const std::size_t half = s.ns / 2;
    CostFunction a, b;
    for (int part = 0; part < 2; ++part) {
        const std::size_t w0 = part == 0 ? 0 : half, nw = part == 0 ? half : s.ns - half;
        std::vector<double> e(nw), f(nw), d(s.n * nw), dt(s.n * nw), c(s.n * nw), ct(s.n * nw);
        for (std::size_t w = 0; w < nw; ++w) {
            e[w] = s.e[w0 + w];
            f[w] = s.f[w0 + w];
            for (std::size_t k = 0; k < s.n; ++k) {
                d[k * nw + w] = s.d[k * s.ns + w0 + w];
                dt[k * nw + w] = s.dt[k * s.ns + w0 + w];
                c[k * nw + w] = s.c[k * s.ns + w0 + w];
                ct[k * nw + w] = s.ct[k * s.ns + w0 + w];
            }
        }
        (part == 0 ? a : b).add_complex_samples(e, f, d, dt, c, ct, s.n, nw);
    }
    a.merge(b);
    Eigen::MatrixXd S, H;
    a.build_matrices(S, H, s.n);
    for (Eigen::Index i = 0; i < S.rows(); ++i)
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            CHECK(S(i, j) == Catch::Approx(S_all(i, j)).margin(1e-13));
            CHECK(H(i, j) == Catch::Approx(H_all(i, j)).margin(1e-13));
        }

    const std::vector<double> one = {1.0}, zero = {0.0}, two = {0.5, 0.25}, none = {};
    CostFunction real;
    real.add_samples(one, two, 2, 1);
    CHECK_THROWS_AS(real.merge(a), std::logic_error);
    CHECK_THROWS_AS(a.add_samples(one, two, 2, 1), std::logic_error);
    CHECK_THROWS_AS(real.add_complex_samples(one, zero, two, two, none, none, 2, 1),
                    std::logic_error);
}

TEST_CASE("a boost whose log-derivative is purely imaginary takes the closed-form step",
          "[cost_function][complex][curvature]") {
    // One particle in 1-D, H = −½∂² + ½x², ψ = e^{−x²/2 + iκx}: the ground
    // state boosted by κ, energy ½ + κ²/2, minimum at κ = 0.  O = ∂_κ ln ψ =
    // ix has no real part, so only the complex accumulation sees this
    // parameter at all (the real one would have S = 0).  Per sample
    //   E_L = ½ + κ²/2 + iκx,   ∂_κ E_L = κ + ix,
    // and with s = <x²>: S_11 = s, H_00 = ½ + κ²/2 = E, H_01 = H_10 = κs,
    // H_11 = (E + 1)s.  The 2×2 eigenproblem over a real step Δ gives
    //   κ s Δ² − Δ − κ = 0,   Δ = [1 − √(1 + 4κ²s)]/(2κs),
    // which for |ψ|² (s = ½) is [1 − √(1 + 2κ²)]/κ: the exact minimiser of
    // the energy of ψ(1 + iΔx) over real Δ, ≈ −κ for small κ.
    const double kappa = 0.5;
    const std::size_t ns = 200000;
    std::mt19937_64 rng(15);
    std::normal_distribution<double> gx(0.0, std::sqrt(0.5));   // |ψ|² = e^{−x²}
    std::vector<double> e(ns), f(ns), d(ns, 0.0), dt(ns), c(ns, kappa), ct(ns);
    for (std::size_t w = 0; w < ns; ++w) {
        const double x = gx(rng);
        e[w] = 0.5 + 0.5 * kappa * kappa;
        f[w] = kappa * x;
        dt[w] = x;
        ct[w] = x;
    }
    CostFunction cf;
    cf.add_complex_samples(e, f, d, dt, c, ct, 1, ns);
    Eigen::MatrixXd S, H;
    cf.build_matrices(S, H, 1);
    CHECK(S(1, 1) == Catch::Approx(0.5).epsilon(0.01));
    CHECK(H(0, 1) == Catch::Approx(0.5 * kappa).epsilon(0.01));

    alea::LinearMethodConfig lc;
    lc.shift_i = 0.0;
    lc.rescale = 1.0;   // raw eigenvector ratio
    lc.xi      = 1.0;
    const double step = alea::LinearMethod(lc).solve(S, H).update[0];
    const double exact = (1.0 - std::sqrt(1.0 + 2.0 * kappa * kappa)) / kappa;
    CHECK(step == Catch::Approx(exact).epsilon(0.01));
}
