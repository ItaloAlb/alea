// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hamiltonian/inverse_power.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"

using alea::InversePowerTerm;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::WalkerSet;
using Catch::Approx;

namespace {

ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = "g" + std::to_string(first);
    g.charge = 0.0;
    g.first = first;
    g.last = last;
    g.count = last - first;
    return g;
}

SimulationCell cell_2d(double a1x, double a1y, double a2x, double a2y) {
    const std::vector<double> v{a1x, a1y, a2x, a2y};
    return SimulationCell(2, std::span<const double>(v));
}

SimulationCell hexagonal(double L) {
    return cell_2d(L, 0.0, 0.5 * L, 0.5 * std::sqrt(3.0) * L);
}

// One configuration (nw = 1) at the given 2-D points, tables built.
void place(WalkerSet& ws, const std::vector<std::array<double, 2>>& pts) {
    const std::size_t nw = ws.num_walkers();
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t i = 0; i < pts.size(); ++i)
            for (std::size_t d = 0; d < 2; ++d)
                ws.positions()[i * 2 * nw + d * nw + w] = pts[i][d];
    ws.dist_tables().build_all(ws.positions());
}

double energy(const InversePowerTerm& term, WalkerSet& ws) {
    ws.local_energies()[0] = 0.0;
    term.evaluate_potential(ws, ws.local_energies(), 0, 1);
    return ws.local_energies()[0];
}

// Σ' 1/|R|^p over the triangular lattice of unit spacing, by brute force out to
// |R| < R_max plus the continuum beyond it (n = 2/√3).  Independent of the term.
double triangular_lattice_sum(double p, double r_max) {
    const int m = static_cast<int>(1.2 * r_max) + 2;  // covers the disk
    double sum = 0.0;
    for (int i = -m; i <= m; ++i)
        for (int j = -m; j <= m; ++j) {
            const double x = i + 0.5 * j, y = 0.5 * std::sqrt(3.0) * j;
            const double r = std::hypot(x, y);
            if (r > 0.0 && r < r_max) sum += std::pow(r, -p);
        }
    const double n = 2.0 / std::sqrt(3.0);
    return sum + n * 2.0 * std::numbers::pi * std::pow(r_max, 2.0 - p) / (p - 2.0);
}

}  // namespace

TEST_CASE("InversePowerTerm sums C/r^p once per intra-group pair", "[inverse_power]") {
    // (0,0), (1,0), (0,2): r = 1, 2, √5.
    WalkerSet ws(1, 3, 2);
    place(ws, {{0, 0}, {1, 0}, {0, 2}});
    const InversePowerTerm term(group(0, 3), group(0, 3), 2.0, 3.0, SimulationCell(2));
    CHECK(energy(term, ws) == Approx(2.0 * (1.0 + 1.0 / 8.0 + std::pow(5.0, -1.5))));
    CHECK(term.tail_energy() == 0.0);  // open cell: nothing outside
}

TEST_CASE("InversePowerTerm between two groups counts only the cross pairs",
          "[inverse_power]") {
    // a = {0}, b = {1, 2}: pairs (0,1) r = 1 and (0,2) r = 2; (1,2) is not counted.
    WalkerSet ws(1, 3, 2);
    place(ws, {{0, 0}, {1, 0}, {0, 2}});
    const InversePowerTerm term(group(0, 1), group(1, 3), 1.5, 3.0, SimulationCell(2));
    CHECK(energy(term, ws) == Approx(1.5 * (1.0 + 1.0 / 8.0)));
}

TEST_CASE("InversePowerTerm evaluates a non-integer exponent", "[inverse_power]") {
    WalkerSet ws(1, 2, 2);
    place(ws, {{0, 0}, {2, 0}});
    const InversePowerTerm term(group(0, 2), group(0, 2), 1.0, 2.5, SimulationCell(2));
    CHECK(energy(term, ws) == Approx(std::pow(2.0, -2.5)));
}

TEST_CASE("InversePowerTerm accumulates, and a crowd split equals the full batch",
          "[inverse_power]") {
    const SimulationCell cell = hexagonal(5.0);
    constexpr std::size_t nw = 4, np = 4;
    WalkerSet ws(nw, np, 2, cell);
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t i = 0; i < np; ++i)
            for (std::size_t d = 0; d < 2; ++d)
                ws.positions()[i * 2 * nw + d * nw + w] =
                    0.37 * static_cast<double>(1 + i + 3 * w + 5 * d) +
                    0.11 * static_cast<double>((i * 7 + w * 3 + d) % 5);
    ws.dist_tables().build_all(ws.positions());
    const InversePowerTerm term(group(0, np), group(0, np), 0.8, 3.0, cell);

    std::vector<double> full(nw, 1.0), split(nw, 1.0);  // nonzero: must add
    term.evaluate_potential(ws, full);
    term.evaluate_potential(ws, split, 0, 1);
    term.evaluate_potential(ws, split, 1, nw);
    const auto copy = term.clone();
    std::vector<double> cloned(nw, 1.0);
    copy->evaluate_potential(ws, cloned);
    for (std::size_t w = 0; w < nw; ++w) {
        CHECK(full[w] > 1.0);
        CHECK(split[w] == full[w]);
        CHECK(cloned[w] == full[w]);
    }
}

TEST_CASE("InversePowerTerm reports its core to the Jastrow, and a 1/r pole as a cusp",
          "[inverse_power][cusp]") {
    const ParticleGroup a = group(0, 1), b = group(1, 2), other = group(5, 6);
    const InversePowerTerm dip(a, b, 2.0, 3.0, SimulationCell(2));
    CHECK(dip.power_law_core(a, b).coefficient == 2.0);
    CHECK(dip.power_law_core(b, a).exponent == 3.0);
    CHECK(dip.power_law_core(a, other).coefficient == 0.0);
    CHECK(dip.coulomb_cusp_residue(a, b) == 0.0);

    // C = lim r^p V(r), read off the potential itself.
    WalkerSet ws(1, 2, 2);
    const double r = 1e-3;
    place(ws, {{0, 0}, {r, 0}});
    CHECK(std::pow(r, 3.0) * energy(dip, ws) == Approx(2.0));

    // p = 1 is a Coulomb pole: the Kato cusp reads it, the core does not.
    const InversePowerTerm pole(a, b, 0.5, 1.0, SimulationCell(2));
    CHECK(pole.coulomb_cusp_residue(b, a) == 0.5);
    CHECK(pole.power_law_core(a, b).coefficient == 0.0);
}

TEST_CASE("the integral outside the Wigner-Seitz cell matches closed forms",
          "[inverse_power][periodic]") {
    using alea::outside_wigner_seitz_integral;
    const double L = 3.0, h = 0.5 * L;  // h: WS inradius
    // Square: ρ(θ) = h/cos θ on 8 half-faces.  p = 3: ∮ dθ/ρ = 8 sin(π/4)/h.
    const SimulationCell square = cell_2d(L, 0, 0, L);
    CHECK(outside_wigner_seitz_integral(square, 3.0) ==
          Approx(8.0 * std::sin(std::numbers::pi / 4) / h).epsilon(1e-9));
    // p = 4: (1/2) · 8 h^{-2} ∫_0^{π/4} cos²θ dθ = 4 h^{-2} (π/8 + 1/4).
    CHECK(outside_wigner_seitz_integral(square, 4.0) ==
          Approx(4.0 / (h * h) * (std::numbers::pi / 8 + 0.25)).epsilon(1e-9));
    // Hexagon (12 half-faces of reach π/6): p = 3 gives 12 sin(π/6)/h = 6/h,
    // whatever the orientation of the cell vectors.
    CHECK(outside_wigner_seitz_integral(hexagonal(L), 3.0) ==
          Approx(6.0 / h).epsilon(1e-9));
    CHECK(outside_wigner_seitz_integral(cell_2d(L, 0, -0.5 * L, 0.5 * std::sqrt(3.0) * L),
                                        3.0) == Approx(6.0 / h).epsilon(1e-9));
    // 1-D: 2 ∫_{L/2}^∞ r^{-p} dr.
    const std::vector<double> v1{L};
    CHECK(outside_wigner_seitz_integral(SimulationCell(1, std::span<const double>(v1)),
                                        3.0) == Approx(2.0 * std::pow(h, -2.0) / 2.0));
}

TEST_CASE("InversePowerTerm in a periodic cell uses the minimum image plus the tail",
          "[inverse_power][periodic]") {
    // Square L = 4.  (0,0) and (2.2, 2.2): the nearest image of the second is
    // at (−1.8, −1.8), r = 1.8√2.  Tail: ½ N² / A · C · 8√2/L.
    const double L = 4.0, C = 0.7;
    const SimulationCell cell = cell_2d(L, 0, 0, L);
    WalkerSet ws(1, 2, 2, cell);
    place(ws, {{0, 0}, {2.2, 2.2}});
    const InversePowerTerm term(group(0, 2), group(0, 2), C, 3.0, cell);
    const double tail = 0.5 * 4.0 / (L * L) * C * 8.0 * std::sqrt(2.0) / L;
    CHECK(term.tail_energy() == Approx(tail).epsilon(1e-9));
    CHECK(energy(term, ws) == Approx(C * std::pow(1.8 * std::sqrt(2.0), -3.0) + tail));
}

TEST_CASE("a perfect triangular crystal approaches the exact 1/r^3 lattice sum",
          "[inverse_power][periodic][madelung]") {
    // E/N of the static crystal is ½ Σ' C/|R|³.  At unit spacing the sum is
    // 11.0341757, i.e. E/N = 4.446373 n^{3/2}: the classical limit quoted by
    // Astrakharchik et al., PRL 98, 060405 (2007), E_triang/N = 4.446 (n r0²)^{3/2}.
    const double S3 = triangular_lattice_sum(3.0, 300.0);
    const double n_unit = 2.0 / std::sqrt(3.0);
    CHECK(0.5 * S3 / std::pow(n_unit, 1.5) == Approx(4.446).margin(1e-3));
    const double exact = 0.5 * S3;

    // N = m² sites on the m×m hexagonal cell.  The relative error of minimum
    // image + continuum tail falls monotonically (≈ N^{-3/2}); measured with an
    // independent script: −3.7e−3, −6.9e−4, −6.4e−5 for N = 16, 64, 256.
    double previous = std::numeric_limits<double>::infinity();
    for (const int m : {4, 8, 16}) {
        const double L = m;  // unit spacing
        const SimulationCell cell = hexagonal(L);
        const std::size_t N = static_cast<std::size_t>(m * m);
        WalkerSet ws(1, N, 2, cell);
        std::vector<std::array<double, 2>> pts;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                pts.push_back({i + 0.5 * j, 0.5 * std::sqrt(3.0) * j});
        place(ws, pts);
        const InversePowerTerm term(group(0, N), group(0, N), 1.0, 3.0, cell);
        const double rel = (energy(term, ws) / static_cast<double>(N) - exact) / exact;
        INFO("N = " << N << ", relative error " << rel);
        CHECK(std::abs(rel) < previous);
        previous = std::abs(rel);
        if (m == 16) CHECK(std::abs(rel) < 1.5e-4);
    }
}

TEST_CASE("InversePowerTerm refuses inputs with no physical meaning", "[inverse_power]") {
    const SimulationCell open(2);
    const SimulationCell box = cell_2d(4, 0, 0, 4);
    const ParticleGroup g = group(0, 2);
    CHECK_THROWS_AS(InversePowerTerm(g, g, 1.0, 0.0, open), std::invalid_argument);
    CHECK_THROWS_AS(InversePowerTerm(g, g, 1.0, std::nan(""), open), std::invalid_argument);
    CHECK_THROWS_AS(InversePowerTerm(g, g, INFINITY, 3.0, open), std::invalid_argument);
    // Attractive and at least as singular as 1/r²: falls to the centre.
    CHECK_THROWS_AS(InversePowerTerm(g, g, -1.0, 3.0, open), std::invalid_argument);
    CHECK_NOTHROW(InversePowerTerm(g, g, -1.0, 1.0, open));
    // Periodic: the image sum needs p > d.
    CHECK_THROWS_AS(InversePowerTerm(g, g, 1.0, 2.0, box), std::invalid_argument);
    CHECK_NOTHROW(InversePowerTerm(g, g, 1.0, 2.0, open));
}
