// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include "population/batched_distance_table.h"
#include "population/simulation_cell.h"

using alea::BatchedDistanceTable;
using alea::SimulationCell;
using Catch::Approx;

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi_v<double>;

SimulationCell make(std::initializer_list<double> v, std::size_t dim) {
    std::vector<double> a(v);
    return SimulationCell(dim, std::span<const double>(a));
}

SimulationCell square(double L)  { return make({L, 0.0, 0.0, L}, 2); }
SimulationCell hexagonal(double L) {
    return make({L, 0.0, 0.5 * L, L * std::sqrt(3.0) / 2.0}, 2);
}

// Nearest image found by exhaustive search over a wide shell — deliberately
// independent of anything SimulationCell does internally.
double brute_min_image(const SimulationCell& c, const double* del, int shells = 8) {
    const auto        a   = c.vectors();
    const std::size_t dim = c.dim();
    double            best = std::numeric_limits<double>::max();
    const int n1_lo = (dim > 1) ? -shells : 0;
    const int n1_hi = (dim > 1) ? shells : 0;
    for (int n0 = -shells; n0 <= shells; ++n0)
        for (int n1 = n1_lo; n1 <= n1_hi; ++n1) {
            const double n[2] = {static_cast<double>(n0), static_cast<double>(n1)};
            double       r2   = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                double s = del[d];
                for (std::size_t i = 0; i < dim; ++i) s += n[i] * a[i * dim + d];
                r2 += s * s;
            }
            best = std::min(best, r2);
        }
    return std::sqrt(best);
}

}  // namespace

// ── Geometry ──────────────────────────────────────────────────────────────

TEST_CASE("min_image matches an exhaustive image search on every cell shape",
          "[cell]") {
    struct Case { const char* name; SimulationCell cell; double span; };
    std::vector<Case> cases = {
        {"square",            square(10.0),                            40.0},
        {"rectangular",       make({30.0, 0.0, 0.0, 15.0}, 2),         60.0},
        {"rows permuted",     make({0.0, 15.0, 30.0, 0.0}, 2),         60.0},
        {"hexagonal",         hexagonal(10.0),                         40.0},
        {"skewed",            make({10.0, 0.0, 35.0, 10.0}, 2),        40.0},
        {"skewed 2",          make({12.0, 3.0, -4.0, 9.0}, 2),         40.0},
        {"square rotated 45", make({14.142135623730951, 14.142135623730951,
                                   -14.142135623730951, 14.142135623730951}, 2), 50.0},
        {"1D",                make({7.0}, 1),                          30.0},
    };

    for (auto& [name, cell, span] : cases) {
        INFO("cell: " << name);
        std::mt19937_64                        rng(20260921);
        std::uniform_real_distribution<double> U(-span, span);
        for (int t = 0; t < 4000; ++t) {
            double del[3] = {0.0, 0.0, 0.0};
            double got[3] = {0.0, 0.0, 0.0};
            for (std::size_t d = 0; d < cell.dim(); ++d) {
                del[d] = U(rng);
                got[d] = del[d];
            }
            const double ref = brute_min_image(cell, del);
            const double r   = std::sqrt(cell.min_image_dist2(got));
            REQUIRE(r == Approx(ref).margin(1e-10));
            // The fold never lengthens a separation.
            double raw = 0.0;
            for (std::size_t d = 0; d < cell.dim(); ++d) raw += del[d] * del[d];
            REQUIRE(r <= std::sqrt(raw) + 1e-10);
        }
    }
}

TEST_CASE("min_image_batch folds every walker bit-for-bit like min_image",
          "[cell]") {
    // The batch kernel vectorizes its walker loop.  Every lane — vector body
    // or remainder — must run exactly min_image's operations, so the batch
    // and the scalar fold agree to the last bit, ties on the cell boundary
    // included (callers mix the two: distance tables, Jastrows, Hamiltonians).
    const std::vector<SimulationCell> cells = {
        hexagonal(10.0), make({10.0, 0.0, 35.0, 10.0}, 2),
        make({12.0, 3.0, -4.0, 9.0}, 2), square(10.0)};
    for (const SimulationCell& cell : cells) {
        std::mt19937_64 rng(77);
        std::uniform_real_distribution<double> U(-40.0, 40.0);
        constexpr std::size_t nw = 37;   // odd: vector bodies and a remainder
        std::vector<double> xi(2 * nw), xj(2 * nw);   // [d * nw + w]
        for (double& x : xi) x = U(rng);
        for (double& x : xj) x = U(rng);
        // Separations of exactly half a lattice vector sit on the boundary.
        const auto a = cell.vectors();
        for (std::size_t w = 0; w < 4; ++w) {
            const std::size_t i = w % 2;
            xi[w]      = xj[w]      + 0.5 * a[i * 2 + 0] * (w < 2 ? 1.0 : -1.0);
            xi[nw + w] = xj[nw + w] + 0.5 * a[i * 2 + 1] * (w < 2 ? 1.0 : -1.0);
        }
        for (std::size_t w0 : {std::size_t{0}, std::size_t{3}}) {
            std::vector<double> out(2 * nw, -1.0);
            cell.min_image_batch(xi.data(), xj.data(), nw, out.data(), w0, nw);
            for (std::size_t w = 0; w < nw; ++w) {
                double del[2] = {xi[w] - xj[w], xi[nw + w] - xj[nw + w]};
                cell.min_image(del);
                const double want0 = w < w0 ? -1.0 : del[0];   // outside the range: untouched
                const double want1 = w < w0 ? -1.0 : del[1];
                REQUIRE(out[w] == want0);
                REQUIRE(out[nw + w] == want1);
            }
        }
    }
}

TEST_CASE("min_image_batch_from folds every walker bit-for-bit like min_image",
          "[cell]") {
    // The fixed-point variant (orbital centres) must agree with the scalar
    // fold to the last bit too: Gaussian orbitals switched from one to the
    // other without changing ψ.
    const std::vector<SimulationCell> cells = {
        hexagonal(10.0), make({10.0, 0.0, 35.0, 10.0}, 2),
        make({12.0, 3.0, -4.0, 9.0}, 2), square(10.0), SimulationCell(2)};
    for (const SimulationCell& cell : cells) {
        std::mt19937_64 rng(78);
        std::uniform_real_distribution<double> U(-40.0, 40.0);
        constexpr std::size_t nw = 37;   // odd: vector bodies and a remainder
        std::vector<double> xi(2 * nw);  // [d * nw + w]
        for (double& x : xi) x = U(rng);
        const double c[2] = {U(rng), U(rng)};
        // Separations of exactly half a lattice vector sit on the boundary.
        if (cell.is_periodic()) {
            const auto a = cell.vectors();
            for (std::size_t w = 0; w < 4; ++w) {
                const std::size_t i = w % 2;
                xi[w]      = c[0] + 0.5 * a[i * 2 + 0] * (w < 2 ? 1.0 : -1.0);
                xi[nw + w] = c[1] + 0.5 * a[i * 2 + 1] * (w < 2 ? 1.0 : -1.0);
            }
        }
        for (std::size_t w0 : {std::size_t{0}, std::size_t{3}}) {
            std::vector<double> out(2 * nw, -1.0);
            cell.min_image_batch_from(xi.data(), c, nw, out.data(), w0, nw);
            for (std::size_t w = 0; w < nw; ++w) {
                double del[2] = {xi[w] - c[0], xi[nw + w] - c[1]};
                cell.min_image(del);
                REQUIRE(out[w] == (w < w0 ? -1.0 : del[0]));   // outside the range: untouched
                REQUIRE(out[nw + w] == (w < w0 ? -1.0 : del[1]));
            }
        }
    }
}

// ── Basis invariance ──────────────────────────────────────────────────────
// A lattice has infinitely many primitive bases: any unimodular row operation
// on (a1, a2) — swap, negate, or add an integer multiple of one to the other —
// generates exactly the same set of lattice points.  Every observable must
// therefore be identical across them.  This is what the Gauss reduction in
// finalize_periodic() is for, and it is the subtlest piece of geometry here:
// rounding the fractional coordinates only lands in the right lattice shell on
// a reduced basis, so a reduction bug shows up as a wrong nearest image on
// skewed input and nowhere else.
//
// It doubles as a cross-check of the two fold kernels.  {(L,0),(0,L)} is
// axis-aligned and takes the orthorhombic per-axis fast path; the same lattice
// written {(L,0),(L,L)} is not, so it reduces and then takes the general
// 9-candidate path.  Equal results mean the two kernels agree.
TEST_CASE("equivalent primitive bases describe an identical lattice", "[cell]") {
    struct Family {
        const char*                          name;
        std::vector<std::vector<double>>     bases;  // all generate one lattice
    };
    const std::vector<Family> families = {
        {"square L=10",
         {{10, 0, 0, 10},      // axis-aligned  → orthorhombic fast path
          {10, 0, 10, 10},     // a2 += a1      → general path
          {0, 10, 10, 0},      // swapped
          {-10, 0, 0, -10},    // negated
          {10, 0, -30, 10},    // a2 -= 3*a1
          {30, 10, 10, 0}}},   // a1 = 3*a2'+a1', mixed
        {"hexagonal L=10",
         {{10, 0, 5, 8.6602540378443865},
          {10, 0, 15, 8.6602540378443865},        // a2 += a1
          {5, 8.6602540378443865, 10, 0},         // swapped
          {10, 0, -15, 8.6602540378443865},       // a2 -= 2*a1
          {-10, 0, -5, -8.6602540378443865}}},    // both negated
        {"rectangular 30x15",
         {{30, 0, 0, 15},
          {30, 0, 60, 15},
          {0, 15, 30, 0},
          {30, 0, -90, -15}}},
    };

    for (const Family& fam : families) {
        INFO("lattice: " << fam.name);

        // Reference: the first basis in the family.
        std::vector<double> a0 = fam.bases.front();
        const SimulationCell ref(2, std::span<const double>(a0));

        // Same random separations run through every basis.
        std::mt19937_64                        rng(90210);
        std::uniform_real_distribution<double> U(-45.0, 45.0);
        std::vector<std::array<double, 2>> probes(3000);
        for (auto& d : probes) d = {U(rng), U(rng)};

        for (std::size_t b = 1; b < fam.bases.size(); ++b) {
            INFO("basis index " << b);
            std::vector<double>  ab = fam.bases[b];
            const SimulationCell alt(2, std::span<const double>(ab));

            // |det| is invariant under a unimodular change of basis; if it is
            // not, the alternative basis was mistyped and generates a different
            // lattice, which would make the rest of the comparison meaningless.
            REQUIRE(alt.volume() == Approx(ref.volume()).margin(1e-9));
            CHECK(alt.ws_radius() == Approx(ref.ws_radius()).margin(1e-9));

            for (const auto& probe : probes) {
                double x[2] = {probe[0], probe[1]};
                double y[2] = {probe[0], probe[1]};
                const double r_ref = std::sqrt(ref.min_image_dist2(x));
                const double r_alt = std::sqrt(alt.min_image_dist2(y));
                REQUIRE(r_alt == Approx(r_ref).margin(1e-10));
            }
        }
    }
}

TEST_CASE("equivalent bases give identical distance tables", "[cell][distance_table]") {
    // The same check one level up, on the object the physics actually reads.
    const std::vector<std::vector<double>> bases = {
        {20, 0, 0, 20},   // orthorhombic fast path
        {20, 0, 20, 20},  // general path, same lattice
        {0, 20, 20, 0},   // swapped
    };
    const std::size_t nw = 24, np = 5, dim = 2;

    std::vector<double> pos(np * dim * nw);
    std::mt19937_64     rng(5150);
    std::uniform_real_distribution<double> U(-55.0, 55.0);
    for (double& x : pos) x = U(rng);

    std::vector<double>  a0 = bases.front();
    BatchedDistanceTable ref(nw, np, dim,
                             SimulationCell(2, std::span<const double>(a0)));
    ref.build_all(pos);

    for (std::size_t b = 1; b < bases.size(); ++b) {
        INFO("basis index " << b);
        std::vector<double>  ab = bases[b];
        BatchedDistanceTable alt(nw, np, dim,
                                 SimulationCell(2, std::span<const double>(ab)));
        alt.build_all(pos);
        for (std::size_t i = 0; i < np; ++i)
            for (std::size_t j = 0; j < np; ++j)
                for (std::size_t w = 0; w < nw; ++w)
                    REQUIRE(alt(w, i, j) == Approx(ref(w, i, j)).margin(1e-10));
    }
}

TEST_CASE("an axis-aligned cell takes the orthorhombic fast path", "[cell]") {
    CHECK(square(10.0).orthorhombic());
    CHECK(make({30.0, 0.0, 0.0, 15.0}, 2).orthorhombic());
    // Gauss reduction sorts rows by length; a rectangular cell given with the
    // long axis first must still be recognised as orthorhombic.
    CHECK(make({100.0, 0.0, 0.0, 50.0}, 2).orthorhombic());
    CHECK(make({0.0, 15.0, 30.0, 0.0}, 2).orthorhombic());
    CHECK_FALSE(hexagonal(10.0).orthorhombic());
}

TEST_CASE("tiny nonzero cell shear is not silently discarded by the axis fast path", "[cell][cell_shear]") {
    // A shear may be small relative to the row length but is still a real
    // lattice translation. Dropping it changes both PIMC springs and pair
    // distances: a1 = (1e13, 1) must fold exactly to the origin.
    const SimulationCell cell = make({1e13, 1.0, 0.0, 1e13}, 2);
    CHECK_FALSE(cell.orthorhombic());
    double displacement[2] = {1e13, 1.0};
    CHECK(cell.min_image_dist2(displacement) == Approx(0.0).margin(1e-12));
    double position[2] = {1e13, 1.0};
    cell.wrap(position);
    CHECK(position[0] == Approx(0.0).margin(1e-12));
    CHECK(position[1] == Approx(0.0).margin(1e-12));
}

TEST_CASE("volume and Wigner-Seitz inradius take their known values", "[cell]") {
    CHECK(square(10.0).volume() == Approx(100.0));
    CHECK(square(10.0).ws_radius() == Approx(5.0));

    CHECK(make({30.0, 0.0, 0.0, 15.0}, 2).volume() == Approx(450.0));
    CHECK(make({30.0, 0.0, 0.0, 15.0}, 2).ws_radius() == Approx(7.5));

    // Hexagonal: area = L²√3/2, shortest lattice vector is L.
    CHECK(hexagonal(10.0).volume() == Approx(100.0 * std::sqrt(3.0) / 2.0));
    CHECK(hexagonal(10.0).ws_radius() == Approx(5.0));

    // Reduction is unimodular, so a skewed basis keeps its cell area.
    CHECK(make({10.0, 0.0, 35.0, 10.0}, 2).volume() == Approx(100.0));
}

TEST_CASE("the reciprocal lattice satisfies b_i . a_j = 2 pi delta_ij", "[cell]") {
    for (const SimulationCell& c :
         {square(10.0), hexagonal(10.0), make({12.0, 3.0, -4.0, 9.0}, 2)}) {
        const auto a = c.vectors();
        const auto b = c.reciprocal();
        for (std::size_t i = 0; i < c.dim(); ++i)
            for (std::size_t j = 0; j < c.dim(); ++j) {
                double s = 0.0;
                for (std::size_t d = 0; d < c.dim(); ++d)
                    s += b[i * c.dim() + d] * a[j * c.dim() + d];
                CHECK(s == Approx(i == j ? kTwoPi : 0.0).margin(1e-9));
            }
    }
}

TEST_CASE("wrap is idempotent and leaves every minimum-image distance alone",
          "[cell]") {
    const SimulationCell cell = hexagonal(10.0);
    std::mt19937_64                        rng(7);
    std::uniform_real_distribution<double> U(-200.0, 200.0);

    for (int t = 0; t < 4000; ++t) {
        double p[2] = {U(rng), U(rng)};
        double q[2] = {U(rng), U(rng)};

        double before[2] = {p[0] - q[0], p[1] - q[1]};
        const double r_before = std::sqrt(cell.min_image_dist2(before));

        cell.wrap(p);
        cell.wrap(q);

        double again[2] = {p[0], p[1]};
        cell.wrap(again);
        REQUIRE(again[0] == Approx(p[0]).margin(1e-12));
        REQUIRE(again[1] == Approx(p[1]).margin(1e-12));

        double after[2] = {p[0] - q[0], p[1] - q[1]};
        REQUIRE(std::sqrt(cell.min_image_dist2(after)) ==
                Approx(r_before).margin(1e-10));
    }
}

TEST_CASE("an open cell leaves separations and positions untouched", "[cell]") {
    const SimulationCell open(2);
    CHECK_FALSE(open.is_periodic());

    double del[2] = {123.5, -998.25};
    open.min_image(del);
    CHECK(del[0] == 123.5);
    CHECK(del[1] == -998.25);

    double r[2] = {-4321.0, 8765.0};
    open.wrap(r);
    CHECK(r[0] == -4321.0);
    CHECK(r[1] == 8765.0);
}

TEST_CASE("a periodic cell is rejected in unsupported dimensions", "[cell]") {
    std::vector<double> v27(27, 1.0);
    CHECK_THROWS(SimulationCell(3, std::span<const double>(v27)));
    // Singular basis (the two vectors are parallel).
    std::vector<double> sing = {1.0, 1.0, 2.0, 2.0};
    CHECK_THROWS(SimulationCell(2, std::span<const double>(sing)));
}

// ── Distance table under periodic boundaries ──────────────────────────────

TEST_CASE("build_all yields minimum-image distances", "[cell][distance_table]") {
    // Two particles straddling the boundary of a 20-wide square cell: they sit
    // 18 apart by raw subtraction but are neighbours 2 apart across the edge.
    const SimulationCell cell = square(20.0);
    BatchedDistanceTable dt(/*nw=*/1, /*np=*/2, /*dim=*/2, cell);
    const std::vector<double> pos = {1.0, 19.0,    // p0 x, p1 x  (nw = 1)
                                     1.0, 1.0};   // p0 y, p1 y
    // Layout is [p*dim*nw + d*nw + w]: p0 = (1,1), p1 = (19,1).
    const std::vector<double> soa = {1.0, 1.0, 19.0, 1.0};
    dt.build_all(soa);
    CHECK(dt(0, 0, 1) == Approx(2.0));
    (void)pos;
}

TEST_CASE("periodic build_all agrees with an exhaustive image search",
          "[cell][distance_table]") {
    for (const SimulationCell& cell :
         {square(20.0), make({30.0, 0.0, 0.0, 15.0}, 2), hexagonal(20.0),
          make({12.0, 3.0, -4.0, 9.0}, 2)}) {
        const std::size_t nw = 32, np = 5, dim = 2;
        BatchedDistanceTable dt(nw, np, dim, cell);
        std::vector<double>  pos(np * dim * nw);
        std::mt19937_64      rng(99);
        std::uniform_real_distribution<double> U(-60.0, 60.0);
        for (double& x : pos) x = U(rng);
        dt.build_all(pos);

        for (std::size_t i = 0; i < np; ++i)
            for (std::size_t j = 0; j < np; ++j)
                for (std::size_t w = 0; w < nw; ++w) {
                    double del[2] = {
                        pos[(i * dim + 0) * nw + w] - pos[(j * dim + 0) * nw + w],
                        pos[(i * dim + 1) * nw + w] - pos[(j * dim + 1) * nw + w]};
                    REQUIRE(dt(w, i, j) ==
                            Approx(brute_min_image(cell, del, 14)).margin(1e-10));
                }
    }
}

TEST_CASE("incremental update + commit reproduces a full periodic rebuild",
          "[cell][distance_table]") {
    for (const SimulationCell& cell : {square(20.0), hexagonal(20.0)}) {
        const std::size_t nw = 32, np = 4, dim = 2, moved = 2;
        BatchedDistanceTable dt(nw, np, dim, cell);
        std::vector<double>  pos(np * dim * nw);
        std::mt19937_64      rng(4242);
        std::uniform_real_distribution<double> U(-40.0, 40.0);
        for (double& x : pos) x = U(rng);
        dt.build_all(pos);

        // Large moves, so many of them cross the cell boundary.
        std::vector<double> new_pos(dim * nw), out(np * nw);
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                new_pos[d * nw + w] = pos[(moved * dim + d) * nw + w] + U(rng);

        std::vector<char> accept_store(nw);
        for (std::size_t w = 0; w < nw; ++w) accept_store[w] = (w % 3) != 0;
        std::vector<bool> dummy;  // std::vector<bool> has no data(); use an array
        std::unique_ptr<bool[]> accept(new bool[nw]);
        for (std::size_t w = 0; w < nw; ++w) accept[w] = accept_store[w] != 0;

        dt.update_batch(moved, pos, new_pos, out);
        dt.commit_batch(moved, out, std::span<const bool>(accept.get(), nw));
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                if (accept[w]) pos[(moved * dim + d) * nw + w] = new_pos[d * nw + w];

        BatchedDistanceTable ref(nw, np, dim, cell);
        ref.build_all(pos);
        for (std::size_t i = 0; i < np; ++i)
            for (std::size_t j = 0; j < np; ++j)
                for (std::size_t w = 0; w < nw; ++w)
                    REQUIRE(dt(w, i, j) == Approx(ref(w, i, j)).margin(1e-12));
        (void)dummy;
    }
}

TEST_CASE("distances are invariant under a lattice translation and an origin shift",
          "[cell][distance_table]") {
    const SimulationCell cell = hexagonal(20.0);
    const auto           a    = cell.vectors();
    const std::size_t nw = 16, np = 4, dim = 2;

    std::vector<double> pos(np * dim * nw);
    std::mt19937_64     rng(31415);
    std::uniform_real_distribution<double> U(-30.0, 30.0);
    for (double& x : pos) x = U(rng);

    BatchedDistanceTable base(nw, np, dim, cell);
    base.build_all(pos);

    SECTION("translating one particle by a lattice vector changes nothing") {
        std::vector<double> moved = pos;
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                moved[(1 * dim + d) * nw + w] += 3.0 * a[0 * dim + d] -
                                                 2.0 * a[1 * dim + d];
        BatchedDistanceTable t(nw, np, dim, cell);
        t.build_all(moved);
        for (std::size_t i = 0; i < np; ++i)
            for (std::size_t j = 0; j < np; ++j)
                for (std::size_t w = 0; w < nw; ++w)
                    REQUIRE(t(w, i, j) == Approx(base(w, i, j)).margin(1e-10));
    }

    SECTION("shifting the origin changes nothing") {
        // The failure mode this guards against: "just use the copy inside the
        // box" gives a different answer depending on where the box edges fall.
        const double shift[2] = {7.3, -11.9};
        std::vector<double> shifted = pos;
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                for (std::size_t w = 0; w < nw; ++w)
                    shifted[(p * dim + d) * nw + w] += shift[d];
        // ... and then re-wrap every particle into the cell.
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t w = 0; w < nw; ++w) {
                double r[2] = {shifted[(p * dim + 0) * nw + w],
                               shifted[(p * dim + 1) * nw + w]};
                cell.wrap(r);
                shifted[(p * dim + 0) * nw + w] = r[0];
                shifted[(p * dim + 1) * nw + w] = r[1];
            }
        BatchedDistanceTable t(nw, np, dim, cell);
        t.build_all(shifted);
        for (std::size_t i = 0; i < np; ++i)
            for (std::size_t j = 0; j < np; ++j)
                for (std::size_t w = 0; w < nw; ++w)
                    REQUIRE(t(w, i, j) == Approx(base(w, i, j)).margin(1e-10));
    }
}

TEST_CASE("an open-boundary table is bit-identical with and without a cell object",
          "[cell][distance_table]") {
    const std::size_t nw = 16, np = 4, dim = 2;
    std::vector<double> pos(np * dim * nw);
    std::mt19937_64     rng(2718);
    std::uniform_real_distribution<double> U(-10.0, 10.0);
    for (double& x : pos) x = U(rng);

    BatchedDistanceTable defaulted(nw, np, dim);
    BatchedDistanceTable explicit_open(nw, np, dim, SimulationCell(dim));
    defaulted.build_all(pos);
    explicit_open.build_all(pos);

    for (std::size_t k = 0; k < np * np * nw; ++k)
        REQUIRE(defaulted.data()[k] == explicit_open.data()[k]);
}
