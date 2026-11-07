// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <Eigen/Dense>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "qmc/linear_method.h"

namespace alea {
namespace {

TEST_CASE("LinearMethod: Eigen is available", "[linear_method]") {
    // Trivial test to verify Eigen compiles and basic operations work.
    Eigen::Matrix<double, 2, 2> A;
    A << 1.0, 0.0, 0.0, 2.0;

    Eigen::Vector<double, 2> b(1.0, 1.0);
    Eigen::Vector<double, 2> x = A.ldlt().solve(b);

    REQUIRE(x(0) == Catch::Approx(1.0));
    REQUIRE(x(1) == Catch::Approx(0.5));
}

TEST_CASE("LinearMethod: generalized eigenvalue solver available", "[linear_method]") {
    // Verify Eigen's GeneralizedSelfAdjointEigenSolver is available.
    Eigen::Matrix<double, 3, 3> S;
    S << 1.0, 0.1, 0.2,
         0.1, 1.0, 0.1,
         0.2, 0.1, 1.0;

    Eigen::Matrix<double, 3, 3> H;
    H << 0.5, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.5;

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> solver;
    solver.compute(H, S);

    REQUIRE(solver.info() == Eigen::Success);
    // The smallest eigenvalue should be ~0.5 (the first diagonal element of H
    // since S is close to identity).
    REQUIRE(solver.eigenvalues()(0) == Catch::Approx(0.5).margin(0.1));
}

TEST_CASE("LinearMethod solves 1-parameter problem", "[linear_method]") {
    // Non-degenerate case with S = I for simplicity.
    // H = [[3, 0.5], [0.5, 1]].
    // Eigenvalues of H: λ = (4 ± √5)/2 ≈ {0.882, 3.118}.
    // For λ₁ ≈ 0.882: (3-0.882)c₀ + 0.5c₁ = 0 → c₁ ≈ -4.236c₀.
    // Δp = -rescale·c₁/c₀ ≈ 4.236.
    Eigen::Matrix<double, 2, 2> Sbar;
    Sbar << 1.0, 0.0, 0.0, 1.0;

    Eigen::Matrix<double, 2, 2> Hbar;
    Hbar << 3.0, 0.5, 0.5, 1.0;

    alea::LinearMethodConfig cfg;
    cfg.shift_i = 0.0;
    cfg.shift_s = 0.0;
    cfg.rescale  = 1.0;
    cfg.xi       = 1.0;   // raw c/c0 arithmetic; ξ is tested separately

    alea::LinearMethod lm(cfg);
    auto result = lm.solve(Sbar, Hbar);

    REQUIRE(result.update.size() == 1);
    CHECK(result.update[0] == Catch::Approx(-4.2360679775).margin(1e-6));
    CHECK(result.optimal_energy == Catch::Approx(0.88196601125).margin(1e-6));
}

TEST_CASE("LinearMethod shift_i damps the parameter update", "[linear_method]") {
    // Without shift, the update is large (~20).
    // With large shift_i, the update is damped (~0.01).
    Eigen::Matrix<double, 2, 2> Sbar;
    Sbar << 1.0, 0.0, 0.0, 1.0;

    Eigen::Matrix<double, 2, 2> Hbar;
    Hbar << 5.0, 0.1, 0.1, 3.0;

    // No shift: large update expected.
    alea::LinearMethodConfig cfg;
    cfg.shift_i = 0.0;
    cfg.shift_s = 0.0;
    cfg.rescale  = 1.0;
    cfg.xi       = 1.0;   // raw c/c0 arithmetic; ξ is tested separately

    alea::LinearMethod lm_no_shift(cfg);
    auto result_no = lm_no_shift.solve(Sbar, Hbar);
    const double update_no_shift = std::abs(result_no.update[0]);

    // Large shift: update should be smaller.
    cfg.shift_i = 10.0;
    alea::LinearMethod lm_shift(cfg);
    auto result_shift = lm_shift.solve(Sbar, Hbar);
    const double update_shifted = std::abs(result_shift.update[0]);

    CHECK(update_shifted < update_no_shift);
}

TEST_CASE("LinearMethod rescale factor controls step size", "[linear_method]") {
    // Same matrices as first test but with rescale = 0.25.
    // Expected update should be 0.25 of the full step.
    Eigen::Matrix<double, 2, 2> Sbar;
    Sbar << 1.0, 0.0, 0.0, 1.0;

    Eigen::Matrix<double, 2, 2> Hbar;
    Hbar << 3.0, 0.5, 0.5, 1.0;

    alea::LinearMethodConfig cfg;
    cfg.shift_i = 0.0;
    cfg.shift_s = 0.0;
    cfg.rescale  = 0.25;
    cfg.xi       = 1.0;   // raw c/c0 arithmetic; ξ is tested separately

    alea::LinearMethod lm(cfg);
    auto result = lm.solve(Sbar, Hbar);

    REQUIRE(result.update.size() == 1);
    CHECK(result.update[0] == Catch::Approx(-4.2360679775 * 0.25).margin(1e-6));
}

TEST_CASE("LinearMethod returns zero update for degenerate case", "[linear_method]") {
    // Diagonal matrices: H = diag(5, 3), S = I.
    // Lowest eigenvalue is 3 with eigenvector [0, 1]^T (c_0 = 0).
    // Should return zero update rather than dividing by zero.
    Eigen::Matrix<double, 2, 2> Sbar;
    Sbar << 1.0, 0.0, 0.0, 1.0;

    Eigen::Matrix<double, 2, 2> Hbar;
    Hbar << 5.0, 0.0, 0.0, 3.0;

    alea::LinearMethodConfig cfg;
    cfg.shift_i = 0.0;
    cfg.shift_s = 0.0;
    cfg.rescale  = 1.0;
    cfg.xi       = 1.0;   // raw c/c0 arithmetic; ξ is tested separately

    alea::LinearMethod lm(cfg);
    auto result = lm.solve(Sbar, Hbar);

    REQUIRE(result.update.size() == 1);
    // c_0 ≈ 0 → update should be 0 (guarded).
    CHECK(result.update[0] == Catch::Approx(0.0).margin(1e-12));
    CHECK(result.optimal_energy == Catch::Approx(3.0).margin(1e-10));
}

}  // namespace
}  // namespace alea

// ── Nonlinear normalisation ξ (Toulouse & Umrigar, JCP 126, 084102, Eqs. 32, 34)
namespace {

// Δp for the given ξ, rescale 1, no shifts.
std::vector<double> step(const Eigen::MatrixXd& S, const Eigen::MatrixXd& H, double xi) {
    alea::LinearMethodConfig cfg;
    cfg.shift_i = 0.0;
    cfg.shift_s = 0.0;
    cfg.rescale = 1.0;
    cfg.xi      = xi;
    return alea::LinearMethod(cfg).solve(S, H).update;
}

// Eq. (32) with Eq. (34): Δp = Δp̄ / (1 + x), x = (1−ξ)D/((1−ξ)+ξ√(1+D)).
std::vector<double> paper(const std::vector<double>& pbar, const Eigen::MatrixXd& S,
                          double xi) {
    const std::size_t n = pbar.size();
    double D = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) D += pbar[i] * S(i + 1, j + 1) * pbar[j];
    const double x = (1.0 - xi) * D / ((1.0 - xi) + xi * std::sqrt(1.0 + D));
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = pbar[i] / (1.0 + x);
    return out;
}

}  // namespace

TEST_CASE("LinearMethod xi rescale matches Toulouse-Umrigar Eqs. 32 and 34",
          "[linear_method]") {
    // Two parameters with a NON-diagonal S̄, so D = Δp̄ᵀ S̄ Δp̄ really uses it.
    Eigen::MatrixXd S(3, 3), H(3, 3);
    S << 1.0, 0.0, 0.0,
         0.0, 0.8, 0.3,
         0.0, 0.3, 0.5;
    H << -1.0, 0.20, -0.10,
          0.20, 0.4,  0.05,
         -0.10, 0.05, 0.3;
    const std::vector<double> pbar = step(S, H, 1.0);   // ξ = 1: raw Δp̄
    for (double xi : {0.0, 0.25, 0.5, 0.75}) {
        INFO("xi = " << xi);
        const auto got = step(S, H, xi);
        const auto ref = paper(pbar, S, xi);
        for (std::size_t i = 0; i < 2; ++i)
            CHECK(got[i] == Catch::Approx(ref[i]).epsilon(1e-10));
    }
    // ξ = 0 is Δp̄ / (1 + D): the most shortened.
    double D = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) D += pbar[i] * S(i + 1, j + 1) * pbar[j];
    const auto p0 = step(S, H, 0.0);
    CHECK(p0[0] == Catch::Approx(pbar[0] / (1.0 + D)).epsilon(1e-10));
}

TEST_CASE("LinearMethod step is covariant when the parameters span many decades",
          "[linear_method]") {
    // p'_i = p_i / s_i gives ∂ψ/∂p'_i = s_i ∂ψ/∂p_i, so S̄'_ij = s_i s_j S̄_ij and
    // H̄'_ij likewise (row and column 0 unscaled), and the exact step is
    // Δp'_i = Δp_i / s_i.  Scales 1e-5 … 1e8 mimic a Drummond basis r^m (r − L)^3,
    // whose derivatives span ~1e3 … 1e10; the overlap shift is scale-invariant
    // and stays on, the identity shift (which is not) stays off.
    Eigen::MatrixXd S(4, 4), H(4, 4);
    S << 1.0, 0.0, 0.0, 0.0,
         0.0, 0.8, 0.3, 0.1,
         0.0, 0.3, 0.5, 0.2,
         0.0, 0.1, 0.2, 0.9;
    H << -1.0, 0.20, -0.10, 0.05,
          0.20, 0.4,  0.05, 0.02,
         -0.10, 0.05, 0.3,  0.01,
          0.05, 0.02, 0.01, 0.6;
    const Eigen::Vector4d s(1.0, 1e-5, 1e8, 3e3);
    const Eigen::MatrixXd S_scaled = s.asDiagonal() * S * s.asDiagonal();
    const Eigen::MatrixXd H_scaled = s.asDiagonal() * H * s.asDiagonal();
    alea::LinearMethodConfig cfg;
    cfg.shift_i = 0.0;
    cfg.shift_s = 0.5;
    const auto ref = alea::LinearMethod(cfg).solve(S, H);
    const auto got = alea::LinearMethod(cfg).solve(S_scaled, H_scaled);
    CHECK(got.optimal_energy == Catch::Approx(ref.optimal_energy).epsilon(1e-10));
    for (int i = 0; i < 3; ++i) {
        INFO("parameter " << i);
        CHECK(got.update[i] * s(i + 1) == Catch::Approx(ref.update[i]).epsilon(1e-9));
    }
}

TEST_CASE("LinearMethod xi = 1/2 shortens a large step and keeps it finite at c0 = 0",
          "[linear_method]") {
    // The 1-parameter problem above: Δp̄ = −4.236 (D ≈ 17.9) → ξ = ½ gives
    // Δp̄ / (1 + D/(1 + √(1+D))) ≈ −0.973.
    Eigen::MatrixXd S(2, 2), H(2, 2);
    S << 1.0, 0.0, 0.0, 1.0;
    H << 3.0, 0.5, 0.5, 1.0;
    const double pbar = -4.2360679775;
    const double D    = pbar * pbar;
    CHECK(step(S, H, 0.5)[0] ==
          Catch::Approx(pbar / (1.0 + D / (1.0 + std::sqrt(1.0 + D)))).epsilon(1e-9));

    // c0 = 0 (H diagonal): ξ = 1 has no finite step (guarded to zero), while
    // ξ = ½ gives the paper's finite limit ξ/(1−ξ)·v/√(vᵀS̄v), of norm 1 here.
    H << 5.0, 0.0, 0.0, 3.0;
    CHECK(step(S, H, 1.0)[0] == Catch::Approx(0.0).margin(1e-12));
    CHECK(std::abs(step(S, H, 0.5)[0]) == Catch::Approx(1.0).epsilon(1e-10));
}
