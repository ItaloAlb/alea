// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "qmc/linear_method.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace alea {

LinearMethod::LinearMethod(LinearMethodConfig cfg) : cfg_(cfg) {}

LinearMethodResult LinearMethod::solve(const Eigen::MatrixXd& Sbar,
                                       const Eigen::MatrixXd& Hbar) const {
    const int N = static_cast<int>(Hbar.rows());
    if (N < 2)
        throw std::invalid_argument("LinearMethod::solve: matrix size must be ≥ 2");

    const int n_params = N - 1;

    // Copy matrices and apply stabilisation shifts.
    Eigen::MatrixXd H = Hbar;
    if (cfg_.shift_s != 0.0) {
        for (int i = 1; i < N; ++i)
            for (int j = 1; j < N; ++j)
                H(i, j) += cfg_.shift_s * Sbar(i, j);
    }
    if (cfg_.shift_i != 0.0) {
        for (int i = 1; i < N; ++i)
            H(i, i) += cfg_.shift_i;
    }

    // Solve the generalized eigenvalue problem H c = E S c in the basis
    // scaled to unit overlap diagonal: D H D c̃ = E D S̄ D c̃ with
    // D = diag(1, 1/√S̄_11, …, 1/√S̄_nn) and c = D c̃.  This is an exact change
    // of variables, p̃_i = p_i √S̄_ii (same eigenvalues, the shifts above stay
    // in the original units), and it only conditions the numerics: a basis
    // spanning many decades, such as Drummond's r^m (r − L)^3 up to m = 8
    // (∂ln ψ from ~1e3 to ~1e10), leaves S̄ with a condition number near
    // 1e20, and the unscaled solver then returns spurious eigenpairs.
    Eigen::VectorXd d = Eigen::VectorXd::Ones(N);
    for (int i = 1; i < N; ++i)
        if (Sbar(i, i) > 0.0) d(i) = 1.0 / std::sqrt(Sbar(i, i));
    const Eigen::MatrixXd H_scaled = d.asDiagonal() * H * d.asDiagonal();
    const Eigen::MatrixXd S_scaled = d.asDiagonal() * Sbar * d.asDiagonal();

    // The generalised (non-symmetric) solver avoids SelfAdjoint template
    // issues with some compiler/version combinations.
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver;
    solver.compute(H_scaled, S_scaled);

    if (solver.info() != Eigen::Success)
        throw std::runtime_error("LinearMethod::solve: eigenvalue solver failed");

    const Eigen::VectorXcd& alphas = solver.alphas();
    const Eigen::VectorXcd& betas  = solver.betas();
    const Eigen::MatrixXcd  eivecs = d.cast<std::complex<double>>().asDiagonal() *
                                     solver.eigenvectors();

    // Find the lowest real eigenvalue.
    int    lowest_idx = 0;
    double lowest_eig = std::numeric_limits<double>::max();
    for (int i = 0; i < N; ++i) {
        if (std::abs(betas(i)) < 1e-14) continue;  // degenerate
        const std::complex<double> eig = alphas(i) / betas(i);
        if (std::abs(eig.imag()) > 1e-10) continue;  // skip complex eigenvalues
        const double e_real = eig.real();
        if (e_real < lowest_eig) {
            lowest_eig = e_real;
            lowest_idx = i;
        }
    }

    LinearMethodResult result;
    result.optimal_energy = lowest_eig;
    result.update.resize(n_params);

    // Eigenvector (real part; the imaginary part of a real eigenvalue's
    // vector is only a global phase), normalised so that c_0 ≥ 0 — the
    // eigenvector's overall sign is arbitrary and Δp must not depend on it.
    // A zero-phase reference: the component of largest modulus.
    int ref = 0;
    for (int i = 1; i < N; ++i)
        if (std::abs(eivecs(i, lowest_idx)) > std::abs(eivecs(ref, lowest_idx))) ref = i;
    const std::complex<double> phase =
        std::abs(eivecs(ref, lowest_idx)) > 0.0
            ? std::conj(eivecs(ref, lowest_idx)) / std::abs(eivecs(ref, lowest_idx))
            : std::complex<double>(1.0, 0.0);
    double c0 = (eivecs(0, lowest_idx) * phase).real();
    Eigen::VectorXd v(n_params);
    for (int i = 0; i < n_params; ++i) v(i) = (eivecs(i + 1, lowest_idx) * phase).real();
    if (c0 < 0.0) { c0 = -c0; v = -v; }

    // q = vᵀ S̄ v  (so D = q / c_0²), on the UNSHIFTED overlap.
    const double q = std::max(0.0, v.dot(Sbar.bottomRightCorner(n_params, n_params) * v));
    const double xi = cfg_.xi;
    // Δp = rescale · Δp̄/(1+x) rewritten without dividing by c_0:
    //   Δp = rescale · v · [(1−ξ)c0 + ξ√(c0²+q)] / [(1−ξ)(c0²+q) + ξ c0 √(c0²+q)]
    // (multiply Eq. (32) through by c0²; c0 ≥ 0 here).  For ξ = 1 this is v/c0.
    const double r     = std::sqrt(c0 * c0 + q);
    const double numer = (1.0 - xi) * c0 + xi * r;
    const double denom = (1.0 - xi) * (c0 * c0 + q) + xi * c0 * r;
    if ((xi >= 1.0 && c0 < 1e-14) || !(denom > 1e-300) ||
        !std::isfinite(numer / denom)) {
        // ξ = 1 with c_0 ≈ 0: the step is genuinely unbounded.  Return zero.
        std::fill(result.update.begin(), result.update.end(), 0.0);
        return result;
    }
    const double scale = cfg_.rescale * numer / denom;
    for (int i = 0; i < n_params; ++i) result.update[i] = scale * v(i);

    return result;
}

}  // namespace alea
