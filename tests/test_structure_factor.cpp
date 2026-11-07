// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Static structure factor S(k) for periodic cells.
//
// Single configurations make it exactly checkable: for one walker
// S_aa(k) = |Σ_{i∈a} e^{ik·r_i}|² / N_a, and a perfect crystal has Bragg peaks
// of height N on its own reciprocal lattice and exactly zero elsewhere.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <complex>
#include <span>
#include <string>
#include <vector>

#include "estimators/structure_factor.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"

using alea::ParticleGroup;
using alea::RunResult;
using alea::SimulationCell;
using alea::StructureFactor;
using alea::Walker;
using alea::WalkerSet;
using Catch::Approx;

namespace {

constexpr double kPi = 3.14159265358979323846;

SimulationCell square(double L) {
    const std::vector<double> a = {L, 0.0, 0.0, L};
    return SimulationCell(2, std::span<const double>(a));
}

ParticleGroup group(const std::string& name, std::size_t first, std::size_t last) {
    return ParticleGroup{name, {}, {}, 0.0, first, last, last - first};
}

const std::vector<double>& series(const RunResult& r, const std::string& name) {
    for (const auto& [n, v] : r.series)
        if (n == name) return v;
    FAIL("series not found: " << name);
    static const std::vector<double> empty;
    return empty;
}

// One walker at the given AoS positions [p*2 + d], measured once.
RunResult measure_once(StructureFactor& sf, const SimulationCell& cell,
                       const std::vector<double>& aos) {
    const std::size_t np = aos.size() / 2;
    WalkerSet ws(1, np, 2, cell);
    Walker w(np, 2);
    w.set_positions(aos);
    ws.add_walker(w);
    ws.load_all();
    ws.dist_tables().build_all(ws.positions());
    sf.start_block();
    sf.accumulate(ws);
    sf.stop_block();
    RunResult out;
    sf.finalize(out);
    return out;
}

}  // namespace

TEST_CASE("S(k) of one configuration is |sum e^{ik.r}|^2 / N per shell",
          "[sofk]") {
    // Square cell L = 4: reciprocal vectors (2π/4)(n1, n2).  kmax just above
    // the second shell keeps |k| = 2π/4 (4 vectors) and √2·2π/4 (4 vectors),
    // i.e. 2 + 2 in the half space.
    const SimulationCell cell = square(4.0);
    const std::vector<ParticleGroup> groups = {group("e", 0, 3)};
    const double b = 2 * kPi / 4.0;
    StructureFactor sf(groups, cell, /*kmax=*/1.5 * b);

    const std::vector<double> aos = {0.3, 0.1, 1.7, 2.9, -1.2, 0.8};
    const RunResult out = measure_once(sf, cell, aos);

    const auto& k = series(out, "sofk_k");
    const auto& s = series(out, "sofk_e_e");
    REQUIRE(k.size() == 2);
    REQUIRE(s.size() == 2);
    CHECK(k[0] == Approx(b));
    CHECK(k[1] == Approx(std::sqrt(2.0) * b));

    auto direct = [&](double kx, double ky) {
        std::complex<double> rho = 0.0;
        for (std::size_t i = 0; i < 3; ++i)
            rho += std::exp(std::complex<double>(0.0, kx * aos[2 * i] + ky * aos[2 * i + 1]));
        return std::norm(rho) / 3.0;
    };
    // Shell average over the half space: |k| = b → (b,0), (0,b); √2 b → (b,b), (b,−b).
    CHECK(s[0] == Approx(0.5 * (direct(b, 0) + direct(0, b))));
    CHECK(s[1] == Approx(0.5 * (direct(b, b) + direct(b, -b))));

    // ⟨ρ(k)⟩ per vector, for the Bragg (disconnected) part.
    const auto& kx = series(out, "sofk_kx");
    const auto& ky = series(out, "sofk_ky");
    const auto& re = series(out, "rhok_e_re");
    const auto& im = series(out, "rhok_e_im");
    REQUIRE(kx.size() == 4);
    for (std::size_t q = 0; q < kx.size(); ++q) {
        std::complex<double> rho = 0.0;
        for (std::size_t i = 0; i < 3; ++i)
            rho += std::exp(std::complex<double>(0.0, kx[q] * aos[2 * i] + ky[q] * aos[2 * i + 1]));
        CHECK(re[q] == Approx(rho.real()).margin(1e-12));
        CHECK(im[q] == Approx(rho.imag()).margin(1e-12));
    }
}

TEST_CASE("a perfect crystal has Bragg peaks of height N on its reciprocal lattice",
          "[sofk]") {
    // Four particles on a square sublattice of spacing 2 in an L = 4 cell: its
    // reciprocal lattice is (2π/2)(m1, m2), every second cell vector.  There
    // S = N = 4; on the other shells the phases cancel exactly.
    const SimulationCell cell = square(4.0);
    const std::vector<ParticleGroup> groups = {group("e", 0, 4)};
    const double b = 2 * kPi / 4.0;
    StructureFactor sf(groups, cell, /*kmax=*/2.1 * b);
    const std::vector<double> aos = {0.5, 0.5, 2.5, 0.5, 0.5, 2.5, 2.5, 2.5};
    const RunResult out = measure_once(sf, cell, aos);

    const auto& k = series(out, "sofk_k");
    const auto& s = series(out, "sofk_e_e");
    for (std::size_t i = 0; i < k.size(); ++i) {
        INFO("|k| / b = " << k[i] / b);
        const bool bragg = std::abs(k[i] - 2 * b) < 1e-9;   // (2,0), (0,2)
        CHECK(s[i] == Approx(bragg ? 4.0 : 0.0).margin(1e-12));
    }
}

TEST_CASE("cross-group S(k) is Re<rho_a rho_b*> / sqrt(N_a N_b)", "[sofk]") {
    const SimulationCell cell = square(4.0);
    const std::vector<ParticleGroup> groups = {group("e", 0, 2), group("h", 2, 5)};
    const double b = 2 * kPi / 4.0;
    StructureFactor sf(groups, cell, /*kmax=*/1.1 * b);
    const std::vector<double> aos = {0.3, 0.1, 1.7, 2.9, -1.2, 0.8, 3.1, -0.4, 0.0, 1.9};
    const RunResult out = measure_once(sf, cell, aos);

    auto rho = [&](std::size_t lo, std::size_t hi, double kx, double ky) {
        std::complex<double> r = 0.0;
        for (std::size_t i = lo; i < hi; ++i)
            r += std::exp(std::complex<double>(0.0, kx * aos[2 * i] + ky * aos[2 * i + 1]));
        return r;
    };
    double expect = 0.0;
    for (auto [kx, ky] : {std::pair{b, 0.0}, std::pair{0.0, b}})
        expect += 0.5 * (rho(0, 2, kx, ky) * std::conj(rho(2, 5, kx, ky))).real();
    CHECK(series(out, "sofk_e_h")[0] == Approx(expect / std::sqrt(2.0 * 3.0)));
    CHECK(series(out, "sofk_h_h").size() == 1);
}

TEST_CASE("S(k) needs a periodic cell and at least one k", "[sofk]") {
    const std::vector<ParticleGroup> groups = {group("e", 0, 2)};
    CHECK_THROWS_WITH(StructureFactor(groups, SimulationCell{}, 1.0),
                      Catch::Matchers::ContainsSubstring("periodic"));
    CHECK_THROWS_WITH(StructureFactor(groups, square(4.0), 0.5),
                      Catch::Matchers::ContainsSubstring("kmax"));
}
