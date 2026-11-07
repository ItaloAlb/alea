// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "numerics/random.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "wavefunction/slater_determinant.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;

namespace {
void check_fields(const WalkerSet& current, const WalkerSet& reference) {
    auto compare = [](std::span<const double> a, std::span<const double> b) {
        REQUIRE(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            REQUIRE(std::isfinite(a[i]));
            REQUIRE(a[i] == Catch::Approx(b[i]).epsilon(2e-10).margin(2e-10));
        }
    };
    compare(current.log_psi(), reference.log_psi());
    compare(current.gradients(), reference.gradients());
    compare(current.laplacians(), reference.laplacians());
}
}

TEST_CASE("Slater long mixed move sequences survive clones and changing population strides",
          "[slater][sequence]") {
    // Fresh dense inversions are independent of the accepted-move update. Use
    // thousands of commits, multiple blocks, spectators and a production shell.
    const bool deferred = GENERATE(false, true);
    const std::size_t shell = GENERATE(2, 7, 19, 61);
    CAPTURE(deferred, shell);
    const std::size_t dim = shell == 2 ? 3 : 2;
    const std::size_t np = shell == 2 ? 4 : shell;
    const std::array<double, 4> vectors{10, 0, 5, 5 * std::sqrt(3.0)};
    const SimulationCell cell = shell == 2 ? SimulationCell(dim) : SimulationCell(2, vectors);
    auto make_psi = [&] {
        std::vector<SlaterDeterminant::Block> blocks;
        if (shell == 2) {
            blocks.push_back({0, 2, std::make_unique<GaussianOrbitalSet>(3,
                std::vector<double>{-1, 0, 0, 1, 0, 0}, std::vector<double>{0.6, 0.6})});
            blocks.push_back({2, 3, std::make_unique<GaussianOrbitalSet>(3,
                std::vector<double>{0, 0, 0}, std::vector<double>{0.4})});
        } else {
            blocks.push_back({0, shell, make_plane_wave_orbitals(cell, shell)});
        }
        TrialWavefunction psi;
        psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
        return psi;
    };
    numerics::Random rng(718);
    std::vector<std::array<double, 2>> sites;
    if (shell != 2) {
        // Commensurate triangular crystal sites give a well-conditioned
        // Fourier matrix (N=i*i+i*j+j*j); random near-nodal starts do not.
        const int i = shell == 7 ? 2 : shell == 19 ? 3 : 5;
        const int j = shell == 7 ? 1 : shell == 19 ? 2 : 4;
        const int n = static_cast<int>(shell);
        std::vector<std::array<int, 2>> fractions;
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b) {
                const std::array<int, 2> f{((i+j)*a+j*b)%n, ((-j*a+i*b)%n+n)%n};
                if (std::find(fractions.begin(), fractions.end(), f) == fractions.end())
                    fractions.push_back(f);
            }
        REQUIRE(fractions.size() == shell);
        for (auto f : fractions)
            sites.push_back({(f[0]*vectors[0]+f[1]*vectors[2])/n,
                             (f[0]*vectors[1]+f[1]*vectors[3])/n});
    }
    std::vector<Walker> walkers;
    // Uneven crowds and full batches exercise multiple accepted groups,
    // sparse masks, tails and nonzero offsets before the population shrinks.
    const std::size_t initial_walkers = shell == 2 ? 101 : shell == 61 ? 73 : 5;
    walkers.reserve(initial_walkers);
    for (std::size_t w = 0; w < initial_walkers; ++w) {
        std::vector<double> pos(np * dim); // [particle * dim + d]
        for (double& x : pos) x = rng.uniform(-4, 4);
        if (shell == 2) { pos[0] = -0.8; pos[dim] = 0.9; }
        else for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                pos[p * dim + d] = sites[p][d] + 0.01 * rng.uniform(-1, 1);
        walkers.emplace_back(np, dim);
        walkers.back().set_positions(pos);
    }
    WalkerSet current(initial_walkers, np, dim, cell);
    for (auto& walker : walkers) current.add_walker(walker);
    current.load_all();
    auto psi = make_psi();
    psi.evaluate_batch(current);

    for (std::size_t step = 0; step < 2400; ++step) {
        if (step == 800 || step == 1600) {
            // Cross the private-matrix layout boundary in both directions; retain
            // duplicates and stability counters through the population change.
            std::vector<std::size_t> src = {5, 2, 0};
            if (step == 800) {
                src.resize(79);
                for (std::size_t j = 0; j < src.size(); ++j)
                    src[j] = (3 * j + 2) % current.num_walkers();
            }
            current.apply_permutation(src);
            psi.permute_caches(src);
            psi = psi.clone(); // Continue without a full evaluation of the clone.
        }
        const std::size_t nw = current.num_walkers(), p = step % np;
        if (deferred) {
#pragma omp parallel for num_threads(3)
            for (int t = 0; t < 3; ++t)
                psi.refresh_particle_derivatives(current, p, nw * t / 3, nw * (t + 1) / 3);
        }
        std::vector<double> proposal(dim * nw); // [d * nw + w]
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                proposal[d * nw + w] = current.positions()[(p * dim + d) * nw + w]
                    + 0.004 * std::sin(0.73 * step + 1.2 * d + w);
        current.propose_move_batch(p, proposal);
        // Alternate private-batch and shared-crowd calls on the same caches.
        const int crowds = step % 2 == 0 ? 1 : 3;
#pragma omp parallel for num_threads(3)
        for (int t = 0; t < crowds; ++t)
            psi.ratio_batch(current, p, nw * t / crowds, nw * (t + 1) / crowds);
        if (step % 31 == 0) {
            WalkerSet proposed(nw, np, dim, cell);
            std::copy(current.positions().begin(), current.positions().end(), proposed.positions().begin());
            for (std::size_t d = 0; d < dim; ++d)
                for (std::size_t w = 0; w < nw; ++w)
                    proposed.positions()[(p * dim + d) * nw + w] = proposal[d * nw + w];
            auto ref = make_psi();
            ref.evaluate_batch(proposed);
            for (std::size_t w = 0; w < nw; ++w) {
                REQUIRE(current.log_ratios()[w] == Catch::Approx(
                    proposed.log_psi()[w] - current.log_psi()[w]).margin(2e-10));
                for (std::size_t d = 0; d < dim; ++d) {
                    const auto idx = (p * dim + d) * nw + w;
                    REQUIRE(current.new_grad()[d * nw + w] ==
                            Catch::Approx(proposed.gradients()[idx]).epsilon(2e-10).margin(2e-10));
                    REQUIRE(current.new_lap()[d * nw + w] ==
                            Catch::Approx(proposed.laplacians()[idx]).epsilon(2e-10).margin(2e-10));
                }
            }
        }
        auto accept = current.accept_mask();
        // Include all-rejected, all-accepted and only-the-last-walker commits.
        // The last case bounds tile scratch at the end of each population.
        for (std::size_t w = 0; w < nw; ++w)
            accept[w] = step % 17 != 0 && (step % 29 == 0 ? w + 1 == nw
                : step % 23 == 0 || (step + w) % 5 != 0);
#pragma omp parallel for num_threads(3)
        for (int t = 0; t < crowds; ++t) {
            if (deferred)
                psi.apply_move_batch_deferred(current, p, accept, nw * t / crowds, nw * (t + 1) / crowds);
            else
                psi.apply_move_batch(current, p, accept, nw * t / crowds, nw * (t + 1) / crowds);
        }
        current.apply_move(p, proposal, current.new_grad(), current.new_lap(), accept);
        if (step % 31 == 0 || step == 2399) {
            if (deferred) {
#pragma omp parallel for num_threads(3)
                for (int t = 0; t < 3; ++t)
                    psi.refresh_derivatives(current, nw * t / 3, nw * (t + 1) / 3);
            }
            WalkerSet fresh(nw, np, dim, cell);
            std::copy(current.positions().begin(), current.positions().end(), fresh.positions().begin());
            auto ref = make_psi();
            ref.evaluate_batch(fresh);
            check_fields(current, fresh);
            if (step == 2399) {
                // Derivatives after the long incremental path also agree with
                // finite differences of log|det D|, independent of inverse
                // contraction formulas (fresh dense determinant at +/-h).
                const double h = 1e-4;
                for (std::size_t d = 0; d < dim; ++d) {
                    const auto idx = d * nw;
                    auto log_at = [&](double shift) {
                        WalkerSet shifted(nw, np, dim, cell);
                        std::copy(current.positions().begin(), current.positions().end(),
                                  shifted.positions().begin());
                        shifted.positions()[idx] += shift;
                        auto oracle = make_psi();
                        oracle.evaluate_batch(shifted);
                        return shifted.log_psi()[0];
                    };
                    const double plus = log_at(h), minus = log_at(-h), zero = log_at(0);
                    REQUIRE(current.gradients()[idx] ==
                            Catch::Approx((plus-minus)/(2*h)).margin(2e-6));
                    REQUIRE(current.laplacians()[idx] ==
                            Catch::Approx((plus-2*zero+minus)/(h*h)).margin(2e-4));
                }
            }
        }
    }
}

TEST_CASE("Slater near-node accepted moves retain signs and recover after rejection",
          "[slater][sequence]") {
    const bool deferred = GENERATE(false, true);
    // For two equal-width Gaussians centred at +/-1, sign(det D) is
    // sign(x_1-x_0). This analytic sign oracle does not use inverse algebra.
    WalkerSet current(1, 2, 2);
    auto make_psi = [] {
        std::vector<SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 2, std::make_unique<GaussianOrbitalSet>(2,
            std::vector<double>{-1, 0, 1, 0}, std::vector<double>{0.6, 0.6})});
        TrialWavefunction psi;
        psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
        return psi;
    };
    current.positions()[0] = -0.8;
    current.positions()[2] = 0.9;
    auto psi = make_psi();
    psi.evaluate_batch(current);
    // Tiny determinant ratios and an ill-conditioned committed matrix, followed
    // by well-conditioned recovery. Never accept an exactly singular matrix.
    for (double x : {0.9 - 1e-9, 1.2, 0.9 + 1e-9, -0.8}) {
        if (deferred) psi.refresh_particle_derivatives(current, 0);
        const double old_sign = current.positions()[2] - current.positions()[0];
        const std::array<double, 2> proposal{x, 0};
        current.propose_move_batch(0, proposal);
        psi.ratio_batch(current, 0);
        CHECK((current.phase_diffs()[0] > 1) == ((0.9 - x) * old_sign < 0));
        const std::array<bool, 1> reject{false}, accept{true};
        if (deferred) psi.apply_move_batch_deferred(current, 0, reject);
        else psi.apply_move_batch(current, 0, reject);
        psi.ratio_batch(current, 0);
        if (deferred) psi.apply_move_batch_deferred(current, 0, accept);
        else psi.apply_move_batch(current, 0, accept);
        current.apply_move(0, proposal, current.new_grad(), current.new_lap(), accept);
        if (deferred) psi.refresh_derivatives(current);
        WalkerSet fresh(1, 2, 2);
        std::copy(current.positions().begin(), current.positions().end(), fresh.positions().begin());
        auto ref = make_psi();
        ref.evaluate_batch(fresh);
        // Near-node derivatives scale as separation^-1 and separation^-2.
        // Compare relative errors here; the long nonsingular test is stricter.
        for (std::size_t k = 0; k < 4; ++k) {
            REQUIRE(std::isfinite(current.gradients()[k]));
            REQUIRE(current.gradients()[k] == Catch::Approx(fresh.gradients()[k]).epsilon(2e-6).margin(1e-7));
            REQUIRE(current.laplacians()[k] == Catch::Approx(fresh.laplacians()[k]).epsilon(4e-6).margin(1e-7));
        }
    }
}

TEST_CASE("Slater masked commits preserve rejected nonfinite fields without floating exceptions",
          "[slater][sequence]") {
    const bool deferred = GENERATE(false, true);
    // An inactive SIMD lane must not evaluate 0*NaN, change its NaN payload,
    // or contaminate an accepted neighbour. A full tile plus tail is checked
    // against fresh dense inversions after restoring the diagnostic fields.
    constexpr std::size_t nw = 33;
    WalkerSet current(nw, 2, 2);
    auto make_psi = [] {
        std::vector<SlaterDeterminant::Block> blocks;
        blocks.push_back({0, 2, std::make_unique<GaussianOrbitalSet>(2,
            std::vector<double>{-1, 0, 1, 0}, std::vector<double>{0.6, 0.6})});
        TrialWavefunction psi;
        psi.add_component(std::make_unique<SlaterDeterminant>(std::move(blocks)));
        return psi;
    };
    for (std::size_t w = 0; w < nw; ++w) {
        current.positions()[w] = -0.8;
        current.positions()[2 * nw + w] = 0.9;
    }
    auto psi = make_psi();
    psi.evaluate_batch(current);
    const std::vector<double> old_grad(current.gradients().begin(), current.gradients().end());
    const std::vector<double> old_lap(current.laplacians().begin(), current.laplacians().end());
    std::vector<double> proposal(2 * nw, 0.0); // [d * nw + w]
    auto accept = current.accept_mask();
    for (std::size_t w = 0; w < nw; ++w) {
        proposal[w] = -0.7 + 0.001 * static_cast<double>(w);
        accept[w] = w % 3 != 0;
    }
    current.propose_move_batch(0, proposal);
    psi.ratio_batch(current, 0);
    constexpr std::uint64_t bad_bits = 0x7ff0000000000123ULL; // signalling NaN
    for (std::size_t i = 0; i < old_grad.size(); ++i) {
        if (accept[i % nw]) continue;
        current.gradients()[i] = std::bit_cast<double>(bad_bits);
        current.laplacians()[i] = std::bit_cast<double>(bad_bits);
    }
    REQUIRE(std::feclearexcept(FE_ALL_EXCEPT) == 0);
    if (deferred) psi.apply_move_batch_deferred(current, 0, accept);
    else psi.apply_move_batch(current, 0, accept);
    REQUIRE(std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW) == 0);
    for (std::size_t i = 0; i < old_grad.size(); ++i) {
        if (accept[i % nw]) continue;
        REQUIRE(std::bit_cast<std::uint64_t>(current.gradients()[i]) == bad_bits);
        REQUIRE(std::bit_cast<std::uint64_t>(current.laplacians()[i]) == bad_bits);
        current.gradients()[i] = old_grad[i];
        current.laplacians()[i] = old_lap[i];
    }
    current.apply_move(0, proposal, current.new_grad(), current.new_lap(), accept);
    if (deferred) psi.refresh_derivatives(current);
    WalkerSet fresh(nw, 2, 2);
    std::copy(current.positions().begin(), current.positions().end(), fresh.positions().begin());
    auto ref = make_psi();
    ref.evaluate_batch(fresh);
    check_fields(current, fresh);
}
