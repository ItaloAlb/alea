// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/hamiltonian.h"
#include "hamiltonian/kinetic.h"
#include "hamiltonian/moire.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"
#include "wavefunction/drummond_jastrow.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using alea::CoulombTerm;
using alea::DrummondJastrow;
using alea::Hamiltonian;
using alea::Kinetic;
using alea::MoireTerm;
using alea::PadeJastrow;
using alea::Parameter;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::TrialWavefunction;
using alea::WalkerSet;
using alea::WavefunctionComponent;
using Catch::Approx;

namespace {

ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name   = "g";
    g.charge = 0.0;
    g.first  = first;
    g.last   = last;
    return g;
}

SimulationCell square(double L) {
    const std::vector<double> a = {L, 0.0, 0.0, L};
    return SimulationCell(2, std::span<const double>(a));
}

SimulationCell hexagonal(double L) {
    const std::vector<double> a = {L, 0.0, 0.5 * L, L * std::sqrt(3.0) / 2.0};
    return SimulationCell(2, std::span<const double>(a));
}

std::unique_ptr<WavefunctionComponent> make_pade() {
    std::vector<Parameter> p = {{"a1", 0.7, false}, {"a2", -0.15, false},
                                {"b1", 0.3, false}};
    return std::make_unique<PadeJastrow>(2, 1, group(0, 1), group(1, 2), p);
}

std::unique_ptr<WavefunctionComponent> make_drummond() {
    // Layout is [α0, α2.., L] — L LAST.  (It used to be listed first, which
    // made the cutoff −0.2, u ≡ 0, and every Drummond check here vacuous.)
    std::vector<Parameter> p = {{"a0", 0.4, false}, {"a2", -0.02, false},
                                {"L", 6.0, false}};
    return std::make_unique<DrummondJastrow>(3, 0.0, group(0, 1), group(1, 2), p);
}

// Put particle 0 and particle 1 on opposite sides of the cell boundary, so the
// nearest image is NOT the copy stored in the box.  This is the configuration
// that a missing fold gets catastrophically wrong: with L = 20 the raw
// separation is 18 while the true one is 2.
void place_straddling(WalkerSet& ws) {
    auto pos = ws.positions();
    const std::size_t nw = ws.num_walkers();
    pos[0 * 2 * nw + 0 * nw + 0] = 1.0;   // p0 x
    pos[0 * 2 * nw + 1 * nw + 0] = 0.7;   // p0 y
    pos[1 * 2 * nw + 0 * nw + 0] = 19.0;  // p1 x
    pos[1 * 2 * nw + 1 * nw + 0] = -0.4;  // p1 y
}

}  // namespace

TEST_CASE("the fold reaches a Jastrow's gradient, not just its value",
          "[cell][wavefunction]") {
    // A component that folds r (via the distance table) but reconstructs the
    // displacement by raw subtraction produces a direction cosine far outside
    // [-1, 1] and a drift pointing the wrong way.  Central differences of
    // log psi catch exactly that.
    const SimulationCell cell = square(20.0);

    struct Case { const char* name; std::unique_ptr<WavefunctionComponent> (*make)(); };
    const Case cases[] = {{"pade", make_pade}, {"drummond", make_drummond}};

    for (const Case& c : cases) {
        INFO("component: " << c.name);
        for (std::size_t particle = 0; particle < 2; ++particle)
            for (std::size_t d = 0; d < 2; ++d) {
                INFO("particle " << particle << " dimension " << d);

                WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2, cell);
                place_straddling(ws);
                ws.dist_tables().build_all(ws.positions());

                auto jas = c.make();
                jas->evaluate_batch(ws);
                const double analytic = jas->grad_c()[(particle * 2 + d) * 1 + 0];

                const double eps = 1e-6;
                double       logs[2];
                for (int sign = 0; sign < 2; ++sign) {
                    WalkerSet probe(/*nw=*/1, /*np=*/2, /*dim=*/2, cell);
                    place_straddling(probe);
                    probe.positions()[(particle * 2 + d) * 1 + 0] +=
                        (sign == 0 ? eps : -eps);
                    probe.dist_tables().build_all(probe.positions());
                    auto j2 = c.make();
                    j2->evaluate_batch(probe);
                    logs[sign] = j2->log_psi_c()[0];
                }
                const double numeric = (logs[0] - logs[1]) / (2.0 * eps);

                // The separation here is 2, not 18: an unfolded direction would
                // be off by ~9x and flip sign, so the margin is generous but
                // still far tighter than that failure mode.
                CHECK(analytic == Approx(numeric).margin(1e-5));
            }
    }
}

TEST_CASE("Jastrow curvature derivatives match finite differences across the boundary",
          "[cell][wavefunction][optimize]") {
    // ∂(∇logψ)/∂p and ∂(∇²logψ)/∂p feed the linear method's curvature term.
    // Check them against central differences of grad_c / lap_c in each
    // parameter, on the straddling pair so the minimum-image fold is exercised.
    const SimulationCell cell = square(20.0);
    struct Case { const char* name; std::unique_ptr<WavefunctionComponent> (*make)(); };
    const Case cases[] = {{"pade", make_pade}, {"drummond", make_drummond}};

    for (const Case& c : cases) {
        INFO("component: " << c.name);
        WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2, cell);
        place_straddling(ws);
        ws.dist_tables().build_all(ws.positions());
        auto jas = c.make();
        jas->evaluate_batch(ws);
        REQUIRE(jas->supports_parameter_gradient_derivatives());

        const std::size_t n_opt = jas->num_optimizable_params();
        const std::size_t block = 2 * 2 * 1;
        std::vector<double> dg(n_opt * block), dl(n_opt * block);
        REQUIRE(jas->compute_parameter_gradient_derivatives(ws, dg, dl));

        const double h = 1e-6;
        for (std::size_t k = 0; k < n_opt; ++k) {
            INFO("parameter " << k);
            std::vector<double> g[2], l[2];
            for (int sign = 0; sign < 2; ++sign) {
                WalkerSet probe(/*nw=*/1, /*np=*/2, /*dim=*/2, cell);
                place_straddling(probe);
                probe.dist_tables().build_all(probe.positions());
                auto j2 = c.make();
                j2->parameters()[k].value += (sign == 0 ? h : -h);
                j2->rebuild();
                j2->evaluate_batch(probe);
                g[sign].assign(j2->grad_c().begin(), j2->grad_c().end());
                l[sign].assign(j2->lap_c().begin(), j2->lap_c().end());
            }
            bool any_nonzero = false;
            for (std::size_t i = 0; i < block; ++i) {
                const double fd_g = (g[0][i] - g[1][i]) / (2 * h);
                const double fd_l = (l[0][i] - l[1][i]) / (2 * h);
                CHECK(dg[k * block + i] == Approx(fd_g).epsilon(1e-5).margin(1e-7));
                CHECK(dl[k * block + i] == Approx(fd_l).epsilon(1e-5).margin(1e-7));
                any_nonzero = any_nonzero || std::abs(fd_g) > 1e-6;
            }
            CHECK(any_nonzero);   // guards against a vacuous (u ≡ 0) setup
        }
    }
}

TEST_CASE("a periodic cell far larger than every separation reproduces open results",
          "[cell][wavefunction]") {
    // The fold must be inert when no periodic image is nearer than the stored
    // copy — this pins the periodic path against the open one on real physics
    // rather than on geometry alone.
    const SimulationCell big = square(1.0e4);

    WalkerSet open_ws(/*nw=*/2, /*np=*/3, /*dim=*/2);
    WalkerSet per_ws(/*nw=*/2, /*np=*/3, /*dim=*/2, big);
    const std::vector<double> coords = {
        0.0, 0.3,  1.5, 1.1,   // p0 x (w0, w1), p0 y
        2.5, 2.1, -0.5, 0.2,   // p1
        1.0, -1.4, 3.0, 2.6,   // p2
    };
    for (std::size_t k = 0; k < coords.size(); ++k) {
        open_ws.positions()[k] = coords[k];
        per_ws.positions()[k]  = coords[k];
    }
    open_ws.dist_tables().build_all(open_ws.positions());
    per_ws.dist_tables().build_all(per_ws.positions());

    std::vector<Parameter> p = {{"a1", 0.7, false}, {"a2", -0.15, false},
                                {"b1", 0.3, false}};
    PadeJastrow open_j(2, 1, group(0, 2), group(2, 3), p);
    PadeJastrow per_j(2, 1, group(0, 2), group(2, 3), p);
    open_j.evaluate_batch(open_ws);
    per_j.evaluate_batch(per_ws);

    for (std::size_t w = 0; w < 2; ++w)
        CHECK(per_j.log_psi_c()[w] == Approx(open_j.log_psi_c()[w]).margin(1e-12));
    for (std::size_t k = 0; k < open_j.grad_c().size(); ++k)
        CHECK(per_j.grad_c()[k] == Approx(open_j.grad_c()[k]).margin(1e-12));
    for (std::size_t k = 0; k < open_j.lap_c().size(); ++k)
        CHECK(per_j.lap_c()[k] == Approx(open_j.lap_c()[k]).margin(1e-12));
}

TEST_CASE("a full periodic move cycle matches a fresh evaluate",
          "[cell][wavefunction]") {
    // ratio_batch and apply_move_batch fold independently of evaluate_batch, so
    // a fold missing from either one shows up as drift between the incremental
    // caches and a rebuild.  Moves are large enough to cross the boundary.
    for (const SimulationCell& cell : {square(20.0), hexagonal(20.0)}) {
        const std::size_t nw = 1, np = 2, dim = 2, moved = 0;

        WalkerSet ws(nw, np, dim, cell);
        place_straddling(ws);
        ws.dist_tables().build_all(ws.positions());

        auto jas = make_pade();
        jas->evaluate_batch(ws);

        // Propose a move that leaves the cell entirely.
        std::vector<double> new_pos = {1.0 + 27.0, 0.7 - 31.0};
        ws.propose_move_batch(moved, new_pos);
        jas->ratio_batch(ws, moved);

        const bool  accept_raw[1] = {true};
        std::span<const bool> accept(accept_raw, 1);
        jas->apply_move_batch(ws, moved, accept);
        ws.apply_move(moved, new_pos, ws.new_grad(), ws.new_lap(), accept);

        // Rebuild from scratch at the committed (and wrapped) configuration.
        WalkerSet fresh(nw, np, dim, cell);
        for (std::size_t k = 0; k < np * dim * nw; ++k)
            fresh.positions()[k] = ws.positions()[k];
        fresh.dist_tables().build_all(fresh.positions());
        auto ref = make_pade();
        ref->evaluate_batch(fresh);

        CHECK(jas->log_psi_c()[0] == Approx(ref->log_psi_c()[0]).margin(1e-10));
        for (std::size_t k = 0; k < ref->grad_c().size(); ++k)
            CHECK(jas->grad_c()[k] == Approx(ref->grad_c()[k]).margin(1e-10));
        for (std::size_t k = 0; k < ref->lap_c().size(); ++k)
            CHECK(jas->lap_c()[k] == Approx(ref->lap_c()[k]).margin(1e-10));
    }
}


// ── Whole-system translation invariance of the ENERGY ────────────────────────
// The distance-table test already pins invariance at the geometry level.  This
// one pins it where it actually matters: on the local energy of a full
// wavefunction-plus-Hamiltonian stack, so a fold missing from a Jastrow's
// gradient or from a potential's pair loop cannot hide behind a correct
// distance table.
//
// Two distinct invariances, and they fail differently:
//   • translating ONE particle by a lattice vector is a no-op on a torus — the
//     particle did not move at all, it is the same point of the cell.
//   • translating the WHOLE system by an arbitrary, non-lattice vector is the
//     origin-shift test: "use the copy inside the box" is exactly the rule that
//     breaks here, because re-wrapping puts different particles in different
//     images.
namespace {

struct System {
    std::vector<alea::ParticleGroup> groups;
    TrialWavefunction                psi;
    Hamiltonian                      ham;
};

alea::ParticleGroup charged_group(const char* name, std::size_t first,
                                  std::size_t last, double charge) {
    alea::ParticleGroup g;
    g.name     = name;
    g.mass     = {1.0, 1.0};
    g.inv_mass = {1.0, 1.0};
    g.charge   = charge;
    g.first    = first;
    g.last     = last;
    g.count    = last - first;
    return g;
}

// 3 electrons + 3 holes, an e-h and an e-e Jastrow, kinetic + all three Coulomb
// pair types.  Every fold site in the codebase is on this path.
System build_system() {
    System sys;
    const alea::ParticleGroup e = charged_group("e", 0, 3, -1.0);
    const alea::ParticleGroup h = charged_group("h", 3, 6, +1.0);
    sys.groups = {e, h};

    std::vector<Parameter> eh = {{"a1", -0.5, true}, {"b1", 0.3, true}};
    std::vector<Parameter> ee = {{"a1", 0.25, true}, {"b1", 0.4, true}};
    sys.psi.add_component(std::make_unique<PadeJastrow>(1, 1, e, h, eh));
    sys.psi.add_component(std::make_unique<PadeJastrow>(1, 1, e, e, ee));

    sys.ham.add_component(std::make_unique<Kinetic>(sys.groups));
    sys.ham.add_component(std::make_unique<CoulombTerm>(e, h, 1.0, 0.0));
    sys.ham.add_component(std::make_unique<CoulombTerm>(e, e, 1.0, 0.0));
    sys.ham.add_component(std::make_unique<CoulombTerm>(h, h, 1.0, 0.0));
    return sys;
}

// Random positions in the cell, rejecting configurations with a near-coincident
// pair so the 1/r potential cannot produce a huge value that swamps the
// comparison tolerance.
std::vector<double> scatter(const SimulationCell& cell, std::size_t np,
                            std::size_t nw, std::uint64_t seed) {
    std::mt19937_64                        rng(seed);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    const auto        a   = cell.vectors();
    const std::size_t dim = 2;
    std::vector<double> pos(np * dim * nw);
    for (std::size_t w = 0; w < nw; ++w)
        for (bool ok = false; !ok;) {
            for (std::size_t p = 0; p < np; ++p) {
                double f[2] = {U(rng), U(rng)};
                for (std::size_t d = 0; d < dim; ++d)
                    pos[(p * dim + d) * nw + w] =
                        f[0] * a[0 * dim + d] + f[1] * a[1 * dim + d];
            }
            ok = true;
            for (std::size_t i = 0; i < np && ok; ++i)
                for (std::size_t j = i + 1; j < np && ok; ++j) {
                    double del[2] = {
                        pos[(i * dim + 0) * nw + w] - pos[(j * dim + 0) * nw + w],
                        pos[(i * dim + 1) * nw + w] - pos[(j * dim + 1) * nw + w]};
                    if (std::sqrt(cell.min_image_dist2(del)) < 0.75) ok = false;
                }
        }
    return pos;
}

std::vector<double> local_energies(const SimulationCell& cell,
                                   const std::vector<double>& pos,
                                   std::size_t np, std::size_t nw) {
    System    sys = build_system();
    WalkerSet ws(nw, np, 2, cell);
    for (std::size_t k = 0; k < pos.size(); ++k) ws.positions()[k] = pos[k];
    ws.dist_tables().build_all(ws.positions());
    sys.psi.evaluate_batch(ws);
    sys.ham.evaluate_batch(ws);
    const std::span<const double> le = ws.local_energies();
    return std::vector<double>(le.begin(), le.end());
}

}  // namespace

TEST_CASE("the local energy is invariant under periodic translations",
          "[cell][wavefunction][hamiltonian]") {
    const SimulationCell cell = hexagonal(20.0);
    const auto           a    = cell.vectors();
    const std::size_t    np = 6, nw = 12, dim = 2;

    const std::vector<double> pos  = scatter(cell, np, nw, 24601);
    const std::vector<double> base = local_energies(cell, pos, np, nw);
    for (double e : base) REQUIRE(std::isfinite(e));

    SECTION("one particle moved by a lattice vector is the same point") {
        std::vector<double> moved = pos;
        for (std::size_t d = 0; d < dim; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                moved[(2 * dim + d) * nw + w] +=
                    3.0 * a[0 * dim + d] - 2.0 * a[1 * dim + d];
        const std::vector<double> got = local_energies(cell, moved, np, nw);
        for (std::size_t w = 0; w < nw; ++w)
            CHECK(got[w] == Approx(base[w]).margin(1e-9 * (1.0 + std::abs(base[w]))));
    }

    SECTION("shifting the origin by a non-lattice vector changes nothing") {
        // The failure mode: "use the copy inside the box" gives a different
        // answer once re-wrapping reassigns particles to different images.
        const double shift[2] = {6.1, -13.7};
        std::vector<double> moved = pos;
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t w = 0; w < nw; ++w) {
                double r[2] = {moved[(p * dim + 0) * nw + w] + shift[0],
                               moved[(p * dim + 1) * nw + w] + shift[1]};
                cell.wrap(r);
                moved[(p * dim + 0) * nw + w] = r[0];
                moved[(p * dim + 1) * nw + w] = r[1];
            }
        const std::vector<double> got = local_energies(cell, moved, np, nw);
        for (std::size_t w = 0; w < nw; ++w)
            CHECK(got[w] == Approx(base[w]).margin(1e-9 * (1.0 + std::abs(base[w]))));
    }
}


// ── Moiré commensurability: testing the gate's premise, not its proxy ────────
// The commensurability gate in App does not check the thing it cares about. It
// cares whether MoireTerm's V(r) is periodic under the cell translations; it
// CHECKS whether g_i . a_j is an integer multiple of 2*pi. The step between the
// two is a derivation:
//
//   V(r + a_j) = V0 + 2*Vt * sum_i cos(g_i.r + g_i.a_j + phi)
//
// which equals V(r) for all r exactly when every g_i.a_j = 0 (mod 2*pi). That
// derivation is the premise, and this test checks it end to end against the
// term's actual output rather than against the algebra.
//
// It has to run in both directions or it is vacuous: a gate that rejected every
// cell would trivially never admit a discontinuous potential. So the rejected
// case must genuinely be discontinuous too.
//
// The specific drift it guards: the gate reads only the FIRST SHELL of three g
// vectors from moire_reciprocal_vectors. If MoireTerm ever gains second-shell
// harmonics (a routine refinement of moiré potentials), the gate would keep
// passing while V stopped being periodic. A test written against the term's
// output catches that; one written against the g list cannot.
namespace {

constexpr double kTwoPiLocal = 6.283185307179586476925286766559;

SimulationCell make_cell(std::initializer_list<double> v, std::size_t dim) {
    const std::vector<double> a(v);
    return SimulationCell(dim, std::span<const double>(a));
}

// The gate's proxy condition, reproduced here so the test can classify a cell
// the same way App does.
bool commensurate(const SimulationCell& cell, double a_M, double theta0) {
    const auto        gs  = alea::moire_reciprocal_vectors(a_M, theta0, cell.dim());
    const auto        a   = cell.vectors();
    const std::size_t dim = cell.dim();
    for (const auto& g : gs)
        for (std::size_t j = 0; j < dim; ++j) {
            double dot = 0.0;
            for (std::size_t d = 0; d < dim; ++d) dot += g[d] * a[j * dim + d];
            const double turns = dot / kTwoPiLocal;
            if (std::abs(turns - std::nearbyint(turns)) > 1e-10) return false;
        }
    return true;
}

// Largest |V(r) - V(r + a_j)| over a scatter of positions, taken from the term
// itself rather than from the cosine formula.
double worst_translation_mismatch(const SimulationCell& cell, double a_M,
                                  double theta0) {
    const std::size_t np = 4, nw = 16, dim = 2;
    const auto        a  = cell.vectors();

    auto energies = [&](const std::vector<double>& pos) {
        MoireTerm term(dim, /*V0=*/0.35, /*Vtilde=*/0.8, /*phi=*/0.4,
                       alea::moire_reciprocal_vectors(a_M, theta0, dim),
                       {{0, np}});
        WalkerSet ws(nw, np, dim, cell);
        for (std::size_t k = 0; k < pos.size(); ++k) ws.positions()[k] = pos[k];
        ws.dist_tables().build_all(ws.positions());
        for (std::size_t w = 0; w < nw; ++w) ws.local_energies()[w] = 0.0;
        term.evaluate_batch(ws);
        const std::span<const double> le = ws.local_energies();
        return std::vector<double>(le.begin(), le.end());
    };

    std::mt19937_64                        rng(13579);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    std::vector<double> pos(np * dim * nw);
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t p = 0; p < np; ++p) {
            const double f[2] = {U(rng), U(rng)};
            for (std::size_t d = 0; d < dim; ++d)
                pos[(p * dim + d) * nw + w] =
                    f[0] * a[0 * dim + d] + f[1] * a[1 * dim + d];
        }

    const std::vector<double> base = energies(pos);

    double worst = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
        std::vector<double> shifted = pos;
        for (std::size_t p = 0; p < np; ++p)
            for (std::size_t d = 0; d < dim; ++d)
                for (std::size_t w = 0; w < nw; ++w)
                    shifted[(p * dim + d) * nw + w] += a[j * dim + d];
        const std::vector<double> got = energies(shifted);
        for (std::size_t w = 0; w < nw; ++w)
            worst = std::max(worst, std::abs(got[w] - base[w]));
    }
    return worst;
}

}  // namespace

TEST_CASE("a commensurate cell makes the moire potential periodic",
          "[cell][moire]") {
    const double a_M = 1.0, s3 = std::sqrt(3.0);

    // The `shape: hexagonal` shortcut builds a cell aligned with the x axis.
    // With theta0 = 0 the moiré lattice sits at 30/90 degrees, so no x-aligned
    // cell of side a_M matches it; the smallest x-aligned one is the sqrt(3)
    // supercell.  This is about the shortcut's fixed orientation, not about
    // commensurability, which does not care how either lattice is oriented.
    // The direct case: a cell built FROM the moiré lattice vectors themselves.
    // Commensurability is a property of the two lattices, independent of
    // orientation, so L = a_M already works, with no supercell.  It is the
    // `shape:` shortcut that forces an alignment with the x axis and creates
    // the sqrt(3) requirement; that is a limit of the interface, not physics.
    SECTION("a cell built from the moire lattice vectors is commensurate at L = a_M") {
        const auto gs = alea::moire_reciprocal_vectors(a_M, 0.0, 2);
        const double det = gs[0][0] * gs[1][1] - gs[0][1] * gs[1][0];
        const double k   = kTwoPiLocal / det;
        const SimulationCell cell = make_cell(
            {k * gs[1][1], -k * gs[1][0], -k * gs[0][1], k * gs[0][0]}, 2);
        // One moiré cell, not three: the area of the moiré primitive cell.
        CHECK(cell.volume() == Approx(a_M * a_M * s3 / 2.0).margin(1e-9));
        REQUIRE(commensurate(cell, a_M, 0.0));
        CHECK(worst_translation_mismatch(cell, a_M, 0.0) < 1e-12);
    }

    SECTION("theta0 = 0: hexagonal L = sqrt(3) a_M is commensurate") {
        const SimulationCell cell =
            make_cell({s3 * a_M, 0.0, s3 * a_M * 0.5, s3 * a_M * s3 / 2.0}, 2);
        REQUIRE(commensurate(cell, a_M, 0.0));
        CHECK(worst_translation_mismatch(cell, a_M, 0.0) < 1e-12);
    }

    SECTION("theta0 = 0: hexagonal L = 2 sqrt(3) a_M is commensurate") {
        const double L = 2.0 * s3 * a_M;
        const SimulationCell cell = make_cell({L, 0.0, L * 0.5, L * s3 / 2.0}, 2);
        REQUIRE(commensurate(cell, a_M, 0.0));
        CHECK(worst_translation_mismatch(cell, a_M, 0.0) < 1e-12);
    }

    SECTION("theta0 = 30 deg: hexagonal L = a_M is commensurate") {
        const double         t30  = std::atan(1.0) * 4.0 / 6.0;  // pi/6
        const SimulationCell cell = make_cell({a_M, 0.0, a_M * 0.5, a_M * s3 / 2.0}, 2);
        REQUIRE(commensurate(cell, a_M, t30));
        CHECK(worst_translation_mismatch(cell, a_M, t30) < 1e-12);
    }

    // The other direction: the gate must not be merely conservative. A cell it
    // rejects has to be genuinely discontinuous across the boundary, or the
    // checks above would be testing nothing.
    SECTION("cells the gate rejects really are non-periodic") {
        struct Bad { const char* name; SimulationCell cell; };
        std::vector<Bad> bad;
        bad.push_back({"hexagonal L = a_M (the obvious wrong choice)",
                       make_cell({a_M, 0.0, a_M * 0.5, a_M * s3 / 2.0}, 2)});
        bad.push_back({"square L = a_M", make_cell({a_M, 0.0, 0.0, a_M}, 2)});
        bad.push_back({"square L = sqrt(3) a_M",
                       make_cell({s3 * a_M, 0.0, 0.0, s3 * a_M}, 2)});

        for (auto& [name, cell] : bad) {
            INFO("cell: " << name);
            CHECK_FALSE(commensurate(cell, a_M, 0.0));
            // Not just "different" — different by an O(Vtilde) amount.
            CHECK(worst_translation_mismatch(cell, a_M, 0.0) > 0.1);
        }
    }
}
