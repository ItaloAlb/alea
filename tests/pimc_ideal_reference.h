// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Exact finite-M references for ideal Bose/Fermi gases (validation gates L1 and
// L4); see test_pimc_permutation.cpp for the
// recursion and the ring partition functions.
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <vector>

namespace pimc_reference {

// ln z(K) and d ln z / d tau of one ring of K beads.
struct Ring {
    double log_z;
    double dlog_z_dtau;
};

inline Ring harmonic_ring(std::size_t K, double tau, double omega, std::size_t dim) {
    Ring ring{0.0, 0.0};
    for (std::size_t j = 0; j < K; ++j) {
        const double s = std::sin(std::numbers::pi * static_cast<double>(j) / static_cast<double>(K));
        const double mode = 4.0 * s * s + tau * tau * omega * omega;
        ring.log_z -= 0.5 * static_cast<double>(dim) * std::log(mode);
        ring.dlog_z_dtau -= static_cast<double>(dim) * tau * omega * omega / mode;
    }
    return ring;
}

inline Ring periodic_ring(std::size_t K, double tau, double L, std::size_t dim) {
    const double beta_ring = static_cast<double>(K) * tau;
    double z = 0.0, e = 0.0;
    for (int n = -80; n <= 80; ++n) {
        const double eps = 0.5 * std::pow(2.0 * std::numbers::pi * n / L, 2);
        z += std::exp(-beta_ring * eps);
        e += eps * std::exp(-beta_ring * eps);
    }
    // d ln z(K tau) / d tau = K * d ln z / d beta_ring.
    return {static_cast<double>(dim) * std::log(z),
            -static_cast<double>(dim) * static_cast<double>(K) * e / z};
}

struct Canonical {
    double energy = 0.0;
    double partition = 0.0;
    std::vector<double> cycle_fraction;  // [k-1], meaningful for eta = +1
};

inline Canonical recursion(std::size_t N, std::size_t M, int eta,
                    const std::function<Ring(std::size_t)>& ring) {
    std::vector<double> z(N + 1), dz(N + 1), Z(N + 1, 0.0), dZ(N + 1, 0.0);
    for (std::size_t k = 1; k <= N; ++k) {
        const Ring r = ring(k * M);
        z[k] = std::exp(r.log_z);
        dz[k] = z[k] * r.dlog_z_dtau / static_cast<double>(M);  // d/d beta
    }
    Z[0] = 1.0;
    for (std::size_t n = 1; n <= N; ++n) {
        for (std::size_t k = 1; k <= n; ++k) {
            const double sign = (eta < 0 && k % 2 == 0) ? -1.0 : 1.0;
            Z[n] += sign * z[k] * Z[n - k];
            dZ[n] += sign * (dz[k] * Z[n - k] + z[k] * dZ[n - k]);
        }
        Z[n] /= static_cast<double>(n);
        dZ[n] /= static_cast<double>(n);
    }
    Canonical result;
    result.energy = -dZ[N] / Z[N];
    result.partition = Z[N];
    for (std::size_t k = 1; k <= N; ++k)
        result.cycle_fraction.push_back(z[k] * Z[N - k] / (static_cast<double>(N) * Z[N]));
    return result;
}

}  // namespace pimc_reference
