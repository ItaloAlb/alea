// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "numerics/finite_grid_pair.h"
// Setup is serial and shared by replica clones. Avoid an implicit dependence
// of table coefficients on a previous driver's OpenMP thread setting.
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace alea::numerics {
namespace {
using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
// exp(-tau H)=exp(-delta c) sum_n delta^n(cI-H)^n/n!, squared
// to tau. c>=max diag H makes every summand nonnegative. Unlike an
// eigensystem reconstruction this avoids subtracting eigenvector products
// to recover exponentially small positive off-diagonal matrix entries.
Matrix positive_exponential(const Sparse& h, double tau, int diameter) {
    const int states = static_cast<int>(h.rows());
    double shift = 0, row_bound = 0;
    for (int i = 0; i < states; ++i)
        shift = std::max(shift, h.coeff(i, i));
    Sparse b = -h;
    for (int i = 0; i < states; ++i)
        b.coeffRef(i, i) += shift;
    b.makeCompressed();
    for (int i = 0; i < states; ++i) {
        double sum = 0;
        for (Sparse::InnerIterator it(b, i); it; ++it) {
            if (it.value() < 0 || !std::isfinite(it.value()))
                throw std::invalid_argument(
                    "pair grid: generator is not finite and positive");
            sum += it.value();
        }
        row_bound = std::max(row_bound, sum);
    }
    if (!std::isfinite(tau * row_bound) || tau * row_bound > std::ldexp(1., 24))
        throw std::invalid_argument(
            "pair grid: exponential work limit exceeded");
    int squares = 0;
    double delta = tau;
    while (delta * row_bound > .25) {
        delta *= .5;
        ++squares;
    }
    Matrix term = Matrix::Identity(states, states), rho = term;
    // The graph diameter plus twenty terms protects relative errors in
    // distant positive entries, rather than using a diagonal norm alone.
    for (int n = 1; n <= diameter + 20; ++n) {
        term = (delta / n) * (b * term).eval();
        rho += term;
    }
    rho *= std::exp(-delta * shift);
    for (int i = 0; i < squares; ++i)
        rho = (rho * rho).eval();
    rho = ((rho + rho.transpose()) * .5).eval();
    if (!rho.allFinite() || rho.minCoeff() <= 0)
        throw std::runtime_error("pair grid: density matrix overflow or "
                                 "underflow; change grid/domain/time step");
    return rho;
}
} // namespace

FiniteGridPair::FiniteGridPair(double mu, double tau, double extent,
                               int grid_points,
                               const std::function<double(double)>& radial,
                               double pole_residue)
    : tau_(tau), extent_(extent), points_(grid_points) {
    if (!std::isfinite(mu) || mu <= 0 || !std::isfinite(tau) || tau <= 0 ||
        !std::isfinite(extent) || extent <= 0 || grid_points < 5 ||
        grid_points > 33 || grid_points % 2 == 0 || !radial ||
        !std::isfinite(pole_residue))
        throw std::invalid_argument(
            "pair grid: finite positive mass/tau/extent and odd grid_points in "
            "[5,33] required");
    spacing_ = 2 * extent / (grid_points - 1);
    const int states = grid_points * grid_points;
    const double hopping = 1 / (2 * mu * spacing_ * spacing_);
    if (!std::isfinite(hopping) || hopping <= 0)
        throw std::invalid_argument("pair grid: kinetic scale is unresolved");
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(states * 5);
    std::vector<double> potential(states); // [x*points+y]
    // Eight-point Gauss-Legendre cell averages. The singular central 1/r
    // square-cell integral is analytic: <1/r>=4 asinh(1)/h. Thus a finite
    // grid Hamiltonian is constructed without exp(+tau/r) primitive collapse.
    constexpr std::array nodes{-.9602898564975363, -.7966664774136267,
                               -.5255324099163290, -.1834346424956498,
                               .1834346424956498,  .5255324099163290,
                               .7966664774136267,  .9602898564975363};
    constexpr std::array weights{.1012285362903763, .2223810344533745,
                                 .3137066458778873, .3626837833783620,
                                 .3626837833783620, .3137066458778873,
                                 .2223810344533745, .1012285362903763};
    for (int x = 0; x < grid_points; ++x)
        for (int y = 0; y < grid_points; ++y) {
            const int i = x * grid_points + y;
            const bool central = x == grid_points / 2 && y == grid_points / 2;
            double average =
                central ? pole_residue * 4 * std::asinh(1.) / spacing_ : 0;
            for (std::size_t a = 0; a < nodes.size(); ++a)
                for (std::size_t b = 0; b < nodes.size(); ++b) {
                    const double xx = -extent + x * spacing_ +
                                      .5 * spacing_ * nodes[a],
                                 yy = -extent + y * spacing_ +
                                      .5 * spacing_ * nodes[b];
                    const double r = std::hypot(xx, yy);
                    const double v =
                        radial(r) - (central ? pole_residue / r : 0);
                    if (!std::isfinite(v))
                        throw std::invalid_argument(
                            "pair grid: nonfinite radial cell average");
                    average += .25 * weights[a] * weights[b] * v;
                }
            if (!std::isfinite(average))
                throw std::invalid_argument(
                    "pair grid: nonfinite potential scale");
            potential[i] = average;
            entries.emplace_back(i, i, 4 * hopping);
            if (x > 0)
                entries.emplace_back(i, i - grid_points, -hopping);
            if (x + 1 < grid_points)
                entries.emplace_back(i, i + grid_points, -hopping);
            if (y > 0)
                entries.emplace_back(i, i - 1, -hopping);
            if (y + 1 < grid_points)
                entries.emplace_back(i, i + 1, -hopping);
        }
    Sparse h0(states, states);
    h0.setFromTriplets(entries.begin(), entries.end());
    Sparse h = h0;
    for (int i = 0; i < states; ++i)
        h.coeffRef(i, i) += potential[i];
    const Matrix free = positive_exponential(h0, tau, 2 * (grid_points - 1));
    const Matrix rho = positive_exponential(h, tau, 2 * (grid_points - 1));
    const Matrix derivative = -(h * rho), free_derivative = -(h0 * free);
    if (!derivative.allFinite() || !free_derivative.allFinite())
        throw std::runtime_error("pair grid: nonfinite thermal derivative");
    // Row-major matrices: entry (left, right) sits at data()[left*states+right].
    const auto entries_count = static_cast<std::size_t>(rho.size());
    table_.resize(4 * entries_count);
    for (std::size_t e = 0; e < entries_count; ++e) {
        table_[4 * e + 0] = rho.data()[e];
        table_[4 * e + 1] = derivative.data()[e];
        table_[4 * e + 2] = free.data()[e];
        table_[4 * e + 3] = free_derivative.data()[e];
    }
}

FiniteGridPair::Stencil
FiniteGridPair::stencil(const std::array<double, 2>& q) const {
    std::array<int, 2> index{};
    std::array<double, 2> fraction{};
    for (int d = 0; d < 2; ++d) {
        const double scaled =
            std::clamp((q[d] + extent_) / spacing_, 0., double(points_ - 1));
        index[d] = std::min(points_ - 2, static_cast<int>(std::floor(scaled)));
        fraction[d] = scaled - index[d];
    }
    Stencil result;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y) {
            const int i = 2 * x + y;
            result.index[i] = (index[0] + x) * points_ + index[1] + y;
            result.weight[i] = (x ? fraction[0] : 1 - fraction[0]) *
                               (y ? fraction[1] : 1 - fraction[1]);
        }
    return result;
}
PairActionValue
FiniteGridPair::evaluate(const std::array<double, 2>& left,
                         const std::array<double, 2>& right) const {
    for (const auto& q : {left, right})
        for (double x : q)
            if (!std::isfinite(x) || std::abs(x) > extent_)
                return {std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
    const auto a = stencil(left), b = stencil(right);
    double rho = 0, d_rho = 0, free = 0, d_free = 0;
    const auto states = static_cast<std::size_t>(points_ * points_);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            const double* entry =
                table_.data() + 4 * (a.index[i] * states + b.index[j]);
            const double weight = a.weight[i] * b.weight[j];
            rho += weight * entry[0];
            d_rho += weight * entry[1];
            free += weight * entry[2];
            d_free += weight * entry[3];
        }
    // Separate logs avoid overflowing a perfectly finite density ratio.
    const double action = std::log(free) - std::log(rho),
                 thermal = d_free / free - d_rho / rho;
    if (!std::isfinite(action) || !std::isfinite(thermal))
        throw std::domain_error(
            "pair grid: unresolved interpolation or thermal derivative");
    return {action, thermal};
}
} // namespace alea::numerics
