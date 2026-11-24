// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Contract tests of HfdTerm (hamiltonian/hfd.h).  Reference numbers come from
// tests/references/hfd_reference.py (mpmath and scipy, independent of alea).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hamiltonian/hfd.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"

using alea::HfdParameters;
using alea::HfdTerm;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::WalkerSet;
using Catch::Approx;

namespace {

// HFD-B(HE) of Aziz, McCourt and Wong (1987), in K and Angstrom.
const HfdParameters kHe{10.948, 2.963, 1.8443101e5, 10.43329537, -2.27965105,
                        1.36745214, 0.42123807, 0.17473318, 1.4826};

ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name = "g" + std::to_string(first);
    g.first = first;
    g.last = last;
    g.count = last - first;
    return g;
}

SimulationCell box(double lx, double ly, double lz) {
    const std::vector<double> v{lx, 0, 0, 0, ly, 0, 0, 0, lz};
    return SimulationCell(3, std::span<const double>(v));
}

using Points = std::vector<std::array<double, 3>>;

// Walker w of ws at the given 3-D points; tables rebuilt.
void place(WalkerSet& ws, const Points& pts, std::size_t w) {
    const std::size_t nw = ws.num_walkers();
    for (std::size_t i = 0; i < pts.size(); ++i)
        for (std::size_t d = 0; d < 3; ++d) ws.positions()[i * 3 * nw + d * nw + w] = pts[i][d];
    ws.dist_tables().build_all(ws.positions());
}

double energy(const HfdTerm& term, WalkerSet& ws, std::size_t w = 0) {
    std::vector<double> v(ws.num_walkers(), 0.0);
    term.evaluate_potential(ws, v, w, w + 1);
    return v[w];
}

// N points spread over the box [0, L_x) x [0, L_y) x [0, L_z) with no pair
// closer than r_min at minimum image (He atoms do not overlap).  In an open
// cell the true distances are at least the folded ones.
Points spread(std::size_t n, std::array<double, 3> L, double r_min, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Points pts;
    while (pts.size() < n) {
        const std::array<double, 3> p{L[0] * u(rng), L[1] * u(rng), L[2] * u(rng)};
        bool ok = true;
        for (const auto& q : pts) {
            double r2 = 0.0;
            for (std::size_t d = 0; d < 3; ++d) {
                double del = p[d] - q[d];
                del -= L[d] * std::nearbyint(del / L[d]);
                r2 += del * del;
            }
            ok = ok && r2 > r_min * r_min;
        }
        if (ok) pts.push_back(p);
    }
    return pts;
}

}  // namespace

TEST_CASE("HfdTerm reproduces the HFD-B(HE) pair potential and its slope", "[hfd]") {
    // hfd_reference.py, part 1: mpmath at 30 digits; V' by numerical
    // differentiation of V, not by the derived slope.
    struct Row { double r, v, dv; };
    const Row rows[] = {
        {1.0, 46041.779071596703, -186066.14580350196},
        {2.0, 537.95539080726085, -2743.8361780586234},
        {2.963, -10.9480023632594, 1.1966841175776281e-5},
        {3.5, -6.2698356539602719, 9.2204937748414518},
        {4.0, -2.9336232659343558, 4.4429329455911068},
        {5.0, -0.72864717093642462, 0.91026555879338686},
        {8.0, -0.040371279894231519, 0.030733180766974135},
    };
    const HfdTerm term(group(0, 2), group(0, 2), kHe, SimulationCell(3));
    for (const Row& row : rows) {
        INFO("r = " << row.r);
        CHECK(term.pair_value(row.r) == Approx(row.v).epsilon(1e-13));
        CHECK(term.slope_over_r(row.r) * row.r == Approx(row.dv).epsilon(1e-9).margin(1e-12));
    }
}

TEST_CASE("the HFD-B(HE) well has depth epsilon at r_m", "[hfd]") {
    // A property of the published parameter set: Aziz et al. fix the minimum
    // at r_m with depth epsilon; the rounded parameters keep it to 2e-7.
    const HfdTerm term(group(0, 2), group(0, 2), kHe, SimulationCell(3));
    CHECK(term.pair_value(kHe.r_m) == Approx(-kHe.epsilon).epsilon(1e-6));
    CHECK(std::abs(term.slope_over_r(kHe.r_m) * kHe.r_m) < 1e-5 * kHe.epsilon / kHe.r_m);
}

TEST_CASE("slope_over_r agrees with finite differences of pair_value across the damping edge",
          "[hfd]") {
    const HfdTerm term(group(0, 2), group(0, 2), kHe, SimulationCell(3));
    const double edge = kHe.D * kHe.r_m;  // F switches off there
    for (const double r : {0.5, 1.7, 2.5, 3.1, edge - 1e-3, edge + 1e-3, 6.0, 12.0}) {
        const double h = 1e-5 * r;
        const double fd = (term.pair_value(r + h) - term.pair_value(r - h)) / (2.0 * h);
        INFO("r = " << r);
        CHECK(term.slope_over_r(r) * r == Approx(fd).epsilon(1e-7).margin(1e-9));
    }
    // F' = 0 on both sides of x = D, so the force is continuous across it (a
    // central difference AT the edge straddles the jump of F'' and errs by O(h)).
    CHECK(term.slope_over_r(edge * (1.0 - 1e-10)) ==
          Approx(term.slope_over_r(edge * (1.0 + 1e-10))).epsilon(1e-8));
}

TEST_CASE("HfdTerm counts each intra-group pair once and only cross pairs between groups",
          "[hfd]") {
    const SimulationCell open(3);
    WalkerSet ws(1, 3, 3, open);
    place(ws, {{0, 0, 0}, {3.0, 0, 0}, {0, 4.0, 1.0}}, 0);
    const double r01 = 3.0, r02 = std::sqrt(17.0), r12 = std::sqrt(26.0);
    const HfdTerm all(group(0, 3), group(0, 3), kHe, open);
    CHECK(energy(all, ws) ==
          Approx(all.pair_value(r01) + all.pair_value(r02) + all.pair_value(r12)));
    const HfdTerm cross(group(0, 1), group(1, 3), kHe, open);
    CHECK(energy(cross, ws) == Approx(all.pair_value(r01) + all.pair_value(r02)));
    CHECK(all.tail_energy() == 0.0);
}

TEST_CASE("HfdTerm accumulates, and a crowd split equals the full batch", "[hfd]") {
    const SimulationCell cell = box(9.0, 10.0, 11.0);
    const std::size_t nw = 4, np = 6;
    WalkerSet ws(nw, np, 3, cell);
    for (std::size_t w = 0; w < nw; ++w)
        place(ws, spread(np, {9.0, 10.0, 11.0}, 2.0, static_cast<unsigned>(11 + w)), w);
    const HfdTerm term(group(0, np), group(0, np), kHe, cell);
    std::vector<double> full(nw, 1.0), split(nw, 1.0);
    term.evaluate_potential(ws, full);
    term.evaluate_potential(ws, split, 0, 1);
    term.evaluate_potential(ws, split, 1, nw);
    for (std::size_t w = 0; w < nw; ++w) {
        CHECK(split[w] == full[w]);
        CHECK(full[w] - 1.0 == Approx(energy(term, ws, w)));
    }
}

TEST_CASE("configuration_gradient agrees with central differences of evaluate_potential",
          "[hfd][gradient]") {
    for (const bool periodic : {false, true}) {
        INFO((periodic ? "periodic box" : "open"));
        const double L = 9.0;
        const SimulationCell cell = periodic ? box(L, L + 1.0, L + 2.0) : SimulationCell(3);
        const std::size_t np = 7;
        WalkerSet ws(1, np, 3, cell);
        Points pts = spread(np, {L, L + 1.0, L + 2.0}, 2.2, 5);
        place(ws, pts, 0);
        const HfdTerm term(group(0, np), group(0, np), kHe, cell);
        std::vector<double> grad(np * 3, 0.0);
        term.configuration_gradient(ws, 0, HfdTerm::npos, np, HfdTerm::npos, grad);
        for (std::size_t i = 0; i < np; ++i)
            for (std::size_t d = 0; d < 3; ++d) {
                const double h = 1e-5;
                Points plus = pts, minus = pts;
                plus[i][d] += h;
                minus[i][d] -= h;
                place(ws, plus, 0);
                const double e_plus = energy(term, ws);
                place(ws, minus, 0);
                const double e_minus = energy(term, ws);
                INFO("particle " << i << ", axis " << d);
                CHECK(grad[i * 3 + d] ==
                      Approx((e_plus - e_minus) / (2.0 * h)).epsilon(1e-6).margin(1e-6));
            }
    }
}

TEST_CASE("bead_gradient is the change of the forces when a bead joins the configuration",
          "[hfd][gradient]") {
    const SimulationCell cell = box(9.0, 9.5, 10.0);
    const std::size_t np = 6, p = 2;
    WalkerSet ws(1, np, 3, cell);
    const Points pts = spread(np, {9.0, 9.5, 10.0}, 2.2, 21);
    place(ws, pts, 0);
    const HfdTerm term(group(0, np), group(0, np), kHe, cell);
    const auto none = HfdTerm::npos;
    std::vector<double> with(np * 3, 0.0), without(np * 3, 0.0), rows(np * 3, 0.0);
    term.configuration_gradient(ws, 0, none, np, none, with);
    term.configuration_gradient(ws, 0, p, np, none, without);
    double self[3] = {0, 0, 0};
    term.bead_gradient(ws, 0, p, std::span<const double>(pts[p].data(), 3), np, none,
                       std::span<double>(self, 3), rows);
    for (std::size_t d = 0; d < 3; ++d) CHECK(self[d] == Approx(with[p * 3 + d]));
    for (std::size_t j = 0; j < np; ++j) {
        if (j == p) continue;
        for (std::size_t d = 0; d < 3; ++d)
            CHECK(rows[j * 3 + d] ==
                  Approx(with[j * 3 + d] - without[j * 3 + d]).margin(1e-12));
    }
}

TEST_CASE("evaluate_potential_gradient equals configuration_gradient at every configuration, "
          "and a crowd split equals the full batch",
          "[hfd][gradient]") {
    // configuration_gradient is checked against finite differences above; the
    // batched gradient of the periodic virial must be the same forces, laid
    // out [np * dim * nw].
    const std::array<double, 3> L{9.0, 9.5, 10.0};
    for (const bool periodic : {false, true})
        for (const bool intragroup : {true, false}) {
            INFO((periodic ? "periodic box" : "open") << ", "
                                                       << (intragroup ? "one group" : "two groups"));
            const SimulationCell cell = periodic ? box(L[0], L[1], L[2]) : SimulationCell(3);
            const std::size_t np = 7, nw = 3;
            WalkerSet ws(nw, np, 3, cell);
            for (std::size_t w = 0; w < nw; ++w) place(ws, spread(np, L, 2.2, 31 + 7 * unsigned(w)), w);
            const HfdTerm term = intragroup ? HfdTerm(group(0, np), group(0, np), kHe, cell)
                                            : HfdTerm(group(0, 3), group(3, np), kHe, cell);
            REQUIRE(term.supports_potential_gradient());
            auto workspace = term.make_potential_gradient_workspace(ws);
            std::vector<double> full(np * 3 * nw, 0.0), split(np * 3 * nw, 0.0);
            term.evaluate_potential_gradient(ws, full, *workspace);
            term.evaluate_potential_gradient(ws, split, *workspace, 0, 1);
            term.evaluate_potential_gradient(ws, split, *workspace, 1, nw);
            CHECK(split == full);
            double largest = 0.0;
            for (std::size_t w = 0; w < nw; ++w) {
                std::vector<double> reference(np * 3, 0.0);
                term.configuration_gradient(ws, w, HfdTerm::npos, np, HfdTerm::npos, reference);
                for (std::size_t i = 0; i < np; ++i)
                    for (std::size_t d = 0; d < 3; ++d) {
                        INFO("configuration " << w << ", particle " << i << ", axis " << d);
                        CHECK(full[(i * 3 + d) * nw + w] ==
                              Approx(reference[i * 3 + d]).epsilon(1e-14).margin(1e-14));
                        largest = std::max(largest, std::abs(reference[i * 3 + d]));
                    }
            }
            CHECK(largest > 1.0);  // the configurations carry forces of a few K/A
        }
}

TEST_CASE("the batched gradient gives no force to a pair that coincides exactly, and the "
          "periodic virial is admitted only in a periodic cell that holds the groups",
          "[hfd][gradient][periodic-virial]") {
    const SimulationCell cell = box(9.0, 9.5, 10.0);
    const std::size_t np = 3;
    WalkerSet ws(1, np, 3, cell);
    // Rows 0 and 1 coincide: V is a cone there (v'(0) != 0) and its gradient
    // undefined; row 2 is an ordinary partner of both.
    place(ws, Points{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {4.1, 2.3, 2.6}}, 0);
    const HfdTerm term(group(0, np), group(0, np), kHe, cell);
    auto workspace = term.make_potential_gradient_workspace(ws);
    std::vector<double> gradient(np * 3, 0.0);
    term.evaluate_potential_gradient(ws, gradient, *workspace);
    for (std::size_t d = 0; d < 3; ++d) {
        CHECK(std::isfinite(gradient[d]));
        CHECK(gradient[d] == gradient[3 + d]);           // the same partner, the same force
        CHECK(gradient[6 + d] == Approx(-2.0 * gradient[d]));  // Newton's third law
    }
    CHECK(std::abs(gradient[0]) > 0.0);

    CHECK(term.supports_periodic_virial(cell, np));
    CHECK_FALSE(term.supports_periodic_virial(SimulationCell(3), np));
    CHECK_FALSE(term.supports_periodic_virial(cell, np - 1));
    CHECK_FALSE(term.supports_open_centroid_virial());
    WalkerSet too_few(1, np - 1, 3, cell);
    CHECK_THROWS_AS(term.make_potential_gradient_workspace(too_few), std::invalid_argument);
}

TEST_CASE("bead_energy is V(R + r) - V(R), the continuum tail included", "[hfd][gc]") {
    const SimulationCell cell = box(10.0, 10.0, 10.0);
    const std::size_t np = 6;
    const Points pts = spread(np, {10.0, 10.0, 10.0}, 2.2, 31);
    WalkerSet all(1, np, 3, cell), fewer(1, np - 1, 3, cell);
    place(all, pts, 0);
    place(fewer, Points(pts.begin(), pts.end() - 1), 0);
    const HfdTerm term_all(group(0, np), group(0, np), kHe, cell);
    const HfdTerm term_fewer(group(0, np - 1), group(0, np - 1), kHe, cell);
    REQUIRE(term_all.tail_energy() != 0.0);
    const double e = term_all.bead_energy(all, 0, np - 1,
                                          std::span<const double>(pts.back().data(), 3), np,
                                          HfdTerm::npos);
    CHECK(e == Approx(energy(term_all, all) - energy(term_fewer, fewer)).epsilon(1e-12));
}

TEST_CASE("delta_potential_particle is the energy change of a one-bead move", "[hfd]") {
    const SimulationCell cell = box(9.0, 9.0, 9.0);
    const std::size_t np = 6, p = 3;
    WalkerSet ws(2, np, 3, cell);
    const Points pts = spread(np, {9.0, 9.0, 9.0}, 2.2, 41);
    place(ws, pts, 0);
    place(ws, pts, 1);
    const HfdTerm term(group(0, np), group(0, np), kHe, cell);
    const double before = energy(term, ws, 1);
    // Across the cell boundary: the proposal is wrapped like a stored position.
    const std::array<double, 3> proposed{pts[p][0] + 8.7, pts[p][1] - 0.4, pts[p][2] + 0.3};
    std::vector<double> scratch(2, 0.0);
    const double delta = term.delta_potential_particle(
        ws, p, 1, std::span<const double>(proposed.data(), 3), scratch);
    Points moved = pts;
    moved[p] = proposed;
    place(ws, moved, 1);
    CHECK(delta == Approx(energy(term, ws, 1) - before).epsilon(1e-12));
}

TEST_CASE("the radial tail of a pure dispersion potential has its closed form", "[hfd][tail]") {
    // A = 0 and D below every x reached: v = -eps sum_n C_n r_m^n r^-n and
    // f(rho) = -eps sum_n C_n r_m^n rho^(d-n)/(n - d).
    HfdParameters q = kHe;
    q.A = 0.0;
    q.D = 1e-3;
    const HfdTerm term(group(0, 2), group(0, 2), q, SimulationCell(3));
    for (const std::size_t dim : {1u, 2u, 3u})
        for (const double rho : {1.0, 4.0, 7.15}) {
            const double d = static_cast<double>(dim);
            const std::array<std::array<double, 2>, 3> terms{
                {{q.C6, 6.0}, {q.C8, 8.0}, {q.C10, 10.0}}};  // {C_n, n}
            double exact = 0.0;
            for (const auto& [c, n] : terms)
                exact -= q.epsilon * c * std::pow(q.r_m, n) * std::pow(rho, d - n) / (n - d);
            INFO("d = " << dim << ", rho = " << rho);
            CHECK(term.radial_tail(rho, dim) == Approx(exact).epsilon(1e-12));
        }
}

TEST_CASE("the radial tail of HFD-B(HE) is converged in its quadrature panels", "[hfd][tail]") {
    const HfdTerm term(group(0, 2), group(0, 2), kHe, SimulationCell(3));
    // rho = 2 lies inside the damping range (x < D), where the t-range splits.
    for (const double rho : {2.0, 4.0, 7.15, 12.0}) {
        INFO("rho = " << rho);
        CHECK(term.radial_tail(rho, 3) == Approx(term.radial_tail(rho, 3, 64)).epsilon(1e-12));
    }
}

TEST_CASE("the tail of 64 He atoms in a cube matches an independent integral",
          "[hfd][tail][periodic]") {
    // hfd_reference.py, part 2: n = 0.02186 A^-3, the density of Brualla et
    // al., J. Chem. Phys. 121, 636 (2004), Table IV; Cartesian quadrature of
    // the exterior of the inscribed sphere minus the cube's corners.
    const double L = 14.305727189732;
    const SimulationCell cell = box(L, L, L);
    const HfdTerm term(group(0, 64), group(0, 64), kHe, cell);
    CHECK(term.tail_energy() == Approx(-49.8070424167).epsilon(1e-9));
}

TEST_CASE("HfdTerm refuses parameters with no physical meaning", "[hfd]") {
    const SimulationCell open(3);
    const ParticleGroup g = group(0, 2);
    auto with = [](auto change) {
        HfdParameters q = kHe;
        change(q);
        return q;
    };
    CHECK_NOTHROW(HfdTerm(g, g, kHe, open));
    CHECK_THROWS_AS(HfdTerm(g, g, with([](HfdParameters& q) { q.epsilon = 0.0; }), open),
                    std::invalid_argument);
    CHECK_THROWS_AS(HfdTerm(g, g, with([](HfdParameters& q) { q.r_m = -1.0; }), open),
                    std::invalid_argument);
    CHECK_THROWS_AS(HfdTerm(g, g, with([](HfdParameters& q) { q.D = 0.0; }), open),
                    std::invalid_argument);
    CHECK_THROWS_AS(HfdTerm(g, g, with([](HfdParameters& q) { q.A = -1.0; }), open),
                    std::invalid_argument);
    CHECK_THROWS_AS(HfdTerm(g, g, with([](HfdParameters& q) { q.C8 = -0.1; }), open),
                    std::invalid_argument);
    CHECK_THROWS_AS(HfdTerm(g, g, with([](HfdParameters& q) { q.alpha = std::nan(""); }), open),
                    std::invalid_argument);
    // The repulsive exponential must decay: beta > 0 grows, beta = 0 needs alpha > 0.
    CHECK_THROWS_AS(HfdTerm(g, g, with([](HfdParameters& q) { q.beta = 0.1; }), open),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        HfdTerm(g, g, with([](HfdParameters& q) { q.beta = 0.0; q.alpha = 0.0; }), open),
        std::invalid_argument);
    CHECK_NOTHROW(HfdTerm(g, g, with([](HfdParameters& q) { q.beta = 0.0; }), open));
}
