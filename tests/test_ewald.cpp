// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

#include "hamiltonian/ewald_term.h"
#include "numerics/ewald_breakup.h"
#include "numerics/ewald_kernel.h"
#include "numerics/keldysh_kernel.h"
#include "numerics/random.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"

using alea::EwaldTerm;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::WalkerSet;
using Catch::Approx;

namespace {

ParticleGroup grp(const char* n, std::size_t f, std::size_t l, double q) {
    ParticleGroup g;
    g.name     = n;
    g.charge   = q;
    g.first    = f;
    g.last     = l;
    g.count    = l - f;
    g.mass     = {1.0, 1.0};
    g.inv_mass = {1.0, 1.0};
    return g;
}

alea::numerics::PotentialSpec coul(double eps = 1.0) {
    return alea::numerics::PotentialSpec::coulomb(eps);
}

SimulationCell cell_from(std::initializer_list<double> v) {
    const std::vector<double> a(v);
    return SimulationCell(2, std::span<const double>(a));
}

// The 2-D "NaCl": a square lattice of alternating charges, nearest-neighbour
// distance 1.  A 2x2 cell holds four ions, ordered so each sign is contiguous.
struct Madelung2D {
    SimulationCell             cell = cell_from({2, 0, 0, 2});
    std::vector<ParticleGroup> groups{grp("p", 0, 2, +1.0), grp("m", 2, 4, -1.0)};
    // (0,0)+ (1,1)+ | (1,0)- (0,1)-
    std::vector<double> pos{0, 0, 1, 1, 1, 0, 0, 1};
};

double energy_of(const Madelung2D& sys, double tol) {
    EwaldTerm term(sys.groups, 1.0, sys.cell, tol);
    WalkerSet ws(1, 4, 2, sys.cell);
    for (std::size_t i = 0; i < 4; ++i) {
        ws.positions()[i * 2 + 0] = sys.pos[2 * i + 0];
        ws.positions()[i * 2 + 1] = sys.pos[2 * i + 1];
    }
    ws.dist_tables().build_all(ws.positions());
    ws.local_energies()[0] = 0.0;
    term.evaluate_batch(ws);
    return ws.local_energies()[0];
}

}  // namespace

// ── The erfc table ──────────────────────────────────────────────────────────

TEST_CASE("the long-range half is the breakup's own spline", "[ewald][breakup]") {
    // Under the erfc split this checked V_long against erf(a r)/r, the closed
    // form that split has.  The optimized breakup has no such form: V_long IS
    // the quintic spline it solved for.  What must hold instead is that the
    // spline is the exact complement of the reciprocal half — the property the
    // whole decomposition rests on — and that it has compact support, which is
    // what lets the real-space sum stop at the nearest image.
    const SimulationCell cell = cell_from({8, 0, 0, 8});
    const auto s = alea::numerics::build_ewald(cell, 1e-6, coul(1.0));

    // Compact support: beyond r_cut the short-range half must be exactly zero,
    // not merely small, or the nearest-image truncation is an approximation
    // rather than an identity.
    CHECK(s.long_range.eval(s.r_cut) == 0.0);
    CHECK(s.long_range.eval(1.5 * s.r_cut) == 0.0);

    // Smoothness: V_long carries none of V's singularity, so it stays bounded
    // into the origin.  (V_short = V - V_long keeps the 1/r, which is why the
    // hot loop evaluates V itself rather than tabulating V_short.)
    double worst = 0.0;
    for (int i = 1; i < 5000; ++i) {
        worst = std::max(worst, std::abs(s.long_range.eval(s.r_cut * i / 5000.0)));
    }
    INFO("max |V_long| inside r_cut = " << worst);
    CHECK(std::isfinite(worst));
    CHECK(worst < 10.0);

    // The self-energy constant is V_long(0), which for this basis is a single
    // coefficient: every basis function but the first vanishes at the origin.
    CHECK(s.v_long_zero == Approx(s.long_range.eval(1e-12)).epsilon(1e-9));

    // The gate the breakup actually offers: what the truncated reciprocal sum
    // drops.  chi is deliberately not exposed — it is a weighted RMS over a
    // k-range of our own choosing, not an error bar.
    INFO("residual above k_cut = " << s.residual << ", cond(C) = " << s.condition);
    CHECK(s.residual < 1e-6);
    CHECK(s.condition > 1.0);
}

TEST_CASE("build_ewald produces a sane splitting", "[ewald][breakup]") {
    const SimulationCell cell = cell_from({10, 0, 0, 10});
    const auto s = alea::numerics::build_ewald(cell, 1e-6, coul());

    CHECK(s.area == Approx(100.0));
    CHECK(s.r_cut == Approx(cell.ws_radius()));
    CHECK(s.n_k() > 0);

    // The real-space half vanishes EXACTLY at the inradius rather than merely
    // becoming negligible there: the breakup constrains the short-range part to
    // die at r_cut with two derivatives, so the lattice sum collapsing to the
    // nearest image is an identity, not a tolerance.
    CHECK(s.long_range.eval(s.r_cut) == 0.0);

    const auto b = cell.reciprocal();
    for (std::size_t i = 0; i < s.n_k(); ++i) {
        // Every k inside the cutoff, and none at the origin.
        const double k = std::hypot(s.kx[i], s.ky[i]);
        REQUIRE(k > 0.0);
        REQUIRE(k <= s.k_cut + 1e-12);
        // Half of k-space only: lexicographically positive indices.
        REQUIRE((s.n1[i] > 0 || (s.n1[i] == 0 && s.n2[i] > 0)));
        // The stored Cartesian components agree with the integer indices.
        CHECK(s.kx[i] == Approx(s.n1[i] * b[0] + s.n2[i] * b[2]));
        CHECK(s.ky[i] == Approx(s.n1[i] * b[1] + s.n2[i] * b[3]));
    }

    // Weights need NOT all be positive.  Under the erfc split they were, since
    // (2pi/A)erfc(k/2a)/k is positive by construction.  Here the weight is the
    // FITTED V_long(k), which the breakup drives toward zero as k approaches
    // the cutoff and can carry slightly past it.  What must hold is that the
    // weights carrying the physics — the small-k ones, where V_long is the
    // potential itself — dominate, and that any overshoot is negligible beside
    // them.  (residual is not a bound here: it is measured ABOVE k_cut, while
    // these weights live below it.)
    {
        double w_max = 0.0, w_min = 0.0;
        for (std::size_t i = 0; i < s.n_k(); ++i) {
            w_max = std::max(w_max, s.weight[i]);
            w_min = std::min(w_min, s.weight[i]);
        }
        INFO("weights span [" << w_min << ", " << w_max << "]");
        CHECK(w_max > 0.0);
        CHECK(-w_min < 1e-3 * w_max);
        for (std::size_t i = 0; i < s.n_k(); ++i) {
            CHECK(std::isfinite(s.weight[i]));
        }
    }
}

TEST_CASE("build_ewald rejects what it cannot handle", "[ewald][breakup]") {
    CHECK_THROWS(alea::numerics::build_ewald(SimulationCell(2), 1e-6, coul()));  // open
    const SimulationCell cell = cell_from({10, 0, 0, 10});
    CHECK_THROWS(alea::numerics::build_ewald(cell, 0.0, coul()));
    CHECK_THROWS(alea::numerics::build_ewald(cell, 1.0, coul()));
    CHECK_THROWS(alea::numerics::build_ewald(cell, -1e-6, coul()));
}

// ── The two validations that matter ─────────────────────────────────────────

TEST_CASE("the Ewald energy does not depend on the reciprocal cutoff",
          "[ewald][physics]") {
    // The erfc split had one knob, alpha, and this test swept it.  The
    // optimized breakup has no alpha: accuracy is set by the FIT and cost by
    // the reciprocal cutoff.  That separation is the whole point of the method,
    // so the invariant worth testing is that k_cut moves the cost and not the
    // answer.
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    std::vector<double> energies;
    std::vector<std::size_t> counts;
    for (double tol : {1e-5, 1e-7, 1e-9}) {
        const auto s = alea::numerics::build_ewald(cell, tol, coul(1.0));
        // One +1/-1 pair separated by r along x, assembled from the setup's own
        // pieces.  All three terms are needed: omitting the self-energy makes
        // the total drift with tolerance, because v_long(0) is a fitted
        // quantity and changes with the fit.
        const double r = 1.37;
        const double e_real = -(1.0 / r - s.long_range.eval(r));
        double e_recip = 0.0;
        for (std::size_t i = 0; i < s.n_k(); ++i) {
            // rho(k) = 1 - exp(-i k.d) for d = (r, 0), so |rho|^2 = 2(1 - cos k_x r).
            e_recip += s.weight[i] * 2.0 * (1.0 - std::cos(s.kx[i] * r));
        }
        const double e_self = -0.5 * 2.0 * s.v_long_zero;
        energies.push_back(e_real + e_recip + e_self);
        counts.push_back(s.n_k());
        INFO("tol " << tol << "  n_k " << s.n_k() << "  residual " << s.residual);
        CHECK(s.residual < tol);
    }
    // The answer must agree across tolerances far better than the loosest of
    // them; a drift here would mean the split is leaking into the physics.
    CHECK(std::abs(energies[1] - energies[0]) < 1e-5);
    CHECK(std::abs(energies[2] - energies[1]) < 1e-7);
}

TEST_CASE("the Ewald sum reproduces the 2D square Madelung constant",
          "[ewald][physics]") {
    // External validation: the only case with a published reference value, and
    // the reason Coulomb is wired up before Rytova-Keldysh or the ETM stack.
    // For N ions of charge +-q at nearest-neighbour distance a, the total
    // electrostatic energy of the cell is -(N/2)*M*q^2/a.
    const Madelung2D sys;
    const double E = energy_of(sys, 1e-8);
    const double M = -2.0 * E / 4.0;
    INFO("Madelung constant implied by the Ewald sum: " << M);
    CHECK(M == Approx(1.6155426267).margin(1e-6));
}

// ── Invariances ─────────────────────────────────────────────────────────────

TEST_CASE("the Ewald energy is invariant under periodic translations",
          "[ewald][physics][cell]") {
    const Madelung2D sys;
    const double     base = energy_of(sys, 1e-8);
    const auto       a    = sys.cell.vectors();

    SECTION("one ion moved by a lattice vector") {
        Madelung2D moved = sys;
        moved.pos[2 * 1 + 0] += 3.0 * a[0] - 2.0 * a[2];
        moved.pos[2 * 1 + 1] += 3.0 * a[1] - 2.0 * a[3];
        CHECK(energy_of(moved, 1e-8) == Approx(base).margin(1e-8));
    }

    SECTION("the whole lattice shifted by a non-lattice vector") {
        Madelung2D moved = sys;
        for (std::size_t i = 0; i < 4; ++i) {
            moved.pos[2 * i + 0] += 0.37;
            moved.pos[2 * i + 1] -= 0.81;
        }
        CHECK(energy_of(moved, 1e-8) == Approx(base).margin(1e-8));
    }
}

TEST_CASE("EwaldTerm accepts a charged cell once the background exists",
          "[ewald][background]") {
    // This used to be "EwaldTerm refuses a charged cell", and refusing was the
    // right behaviour while the k = 0 term was simply omitted: omitting it is
    // exact only when the charges sum to zero, and returning a number anyway
    // would have meant returning one that does not converge to anything as the
    // cell grows at fixed density.
    //
    // With the uniform neutralising background the k = 0 term is legitimately
    // zero, because the background makes the total charge density vanish there.
    // The system being solved is jellium, which is a modelling choice and not a
    // numerical device: in a real doped system the compensating charge is
    // physically present, just outside the simulated layers.
    const SimulationCell cell = cell_from({10, 0, 0, 10});
    CHECK_NOTHROW(EwaldTerm({grp("e", 0, 4, -1.0)}, 1.0, cell, 1e-6));
    CHECK_NOTHROW(
        EwaldTerm({grp("e", 0, 2, -1.0), grp("h", 2, 4, +1.0)}, 1.0, cell, 1e-6));
}

TEST_CASE("EwaldTerm exposes the cusp residue the Jastrow builder needs",
          "[ewald]") {
    // erfc(ar)/r still goes as 1/r at the origin and the reciprocal half is
    // smooth there, so the cusp condition is unchanged by the splitting.
    const SimulationCell cell = cell_from({10, 0, 0, 10});
    const ParticleGroup  e = grp("e", 0, 2, -1.0), h = grp("h", 2, 4, +1.0);
    EwaldTerm            term({e, h}, 2.0, cell, 1e-6);
    CHECK(term.coulomb_cusp_residue(e, h) == Approx(-0.5));
    CHECK(term.coulomb_cusp_residue(e, e) == Approx(0.5));
}


// The erfc split could not do Rytova-Keldysh.  V_short = V - V_long decays
// like erfc(a r) only when 1/eps(k) is analytic at k = 0; for Keldysh
// 1/(1 + r0 k) is not, and V_short kept a 1/r^3 tail whose coefficient fell
// merely as 1/alpha.  Measured on this cell at tol = 1e-6: V_short at the
// inradius was 1.0e-6 for Coulomb (the tolerance, as designed) but 7.2e-3 for
// Keldysh, so truncating at the nearest image never converged.
//
// The optimized breakup removes the premise: it CONSTRAINS the short-range
// half to vanish at r_cut with two derivatives, for any radial function, so
// the nearest-image truncation is an identity rather than an approximation
// that happens to hold for Coulomb.
TEST_CASE("the optimized breakup gives Rytova-Keldysh compact support",
          "[ewald][physics][keldysh]") {
    const SimulationCell cell = cell_from({8, 0, 0, 8});
    constexpr double eps = 1.0, r0 = 0.8, tol = 1e-6;

    const auto sc = alea::numerics::build_ewald(cell, tol, coul(eps));
    const auto sk = alea::numerics::build_ewald(
        cell, tol, alea::numerics::PotentialSpec::rytova_keldysh(eps, r0));

    // What must vanish at the cutoff is the SHORT-range half, f - f_long, not
    // f_long: the boundary condition makes f_long(r_cut) = f(r_cut) exactly, so
    // f_long approaches the potential's own value there (0.25 on this cell) and
    // their difference is what dies.  That is the property the erfc split could
    // only deliver for Coulomb.
    const double pi = std::acos(-1.0);
    const auto v_keldysh = [&](double r) {
        return (pi / (2.0 * eps * r0)) * alea::numerics::keldysh_kernel(r / r0);
    };
    const auto v_coulomb = [&](double r) { return 1.0 / (eps * r); };

    for (int which = 0; which < 2; ++which) {
        const auto& s = which == 0 ? sc : sk;
        const double R = s.r_cut;
        // Approached from inside, f_short must already be at tolerance.
        double worst = 0.0;
        for (double f : {1.0 - 1e-9, 1.0 - 1e-7, 1.0 - 1e-5}) {
            const double r = R * f;
            const double v = which == 0 ? v_coulomb(r) : v_keldysh(r);
            worst = std::max(worst, std::abs(v - s.long_range.eval(r)));
        }
        INFO((which == 0 ? "Coulomb" : "Keldysh")
             << ": max |f - f_long| approaching r_cut = " << worst);
        CHECK(worst < 1e-6);
    }

    // Both reach the requested tolerance on the gate that matters: what the
    // truncated reciprocal sum actually drops.
    INFO("residual: Coulomb " << sc.residual << ", Keldysh " << sk.residual);
    CHECK(sc.residual < tol);
    CHECK(sk.residual < tol);

    // And Keldysh is genuinely a different potential, not Coulomb in disguise:
    // its long-range half differs materially inside the cell.
    const double probe = 0.4 * sc.r_cut;
    CHECK(std::abs(sc.long_range.eval(probe) - sk.long_range.eval(probe)) > 1e-3);
}

TEST_CASE("EwaldTerm handles a charged cell through a neutralising background",
          "[ewald][background]") {
    // Before the background existed this configuration was refused, because the
    // omitted k = 0 term is only zero when the charges sum to zero.  What the
    // background buys is not accuracy but EXISTENCE: without it the periodic
    // potential diverges, and the energy is not a number that a tighter
    // tolerance would recover.
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    const std::vector<ParticleGroup> charged{grp("e", 0, 4, -1.0)};

    std::unique_ptr<EwaldTerm> term;
    REQUIRE_NOTHROW(term = std::make_unique<EwaldTerm>(charged, 1.0, cell, 1e-6));
    CHECK(term->net_charge() == Approx(-4.0));
    CHECK(std::isfinite(term->background_energy()));
    // Q^2 = 16 and sr_k0 > 0 for a repulsive short-range half, so the constant
    // is negative: the background is attractive to the like charges it cancels.
    CHECK(term->background_energy() < 0.0);

    // A neutral cell must be untouched -- the background term vanishes
    // identically rather than merely being small, since it carries (sum q)^2.
    const std::vector<ParticleGroup> neutral{grp("e", 0, 2, -1.0),
                                             grp("h", 2, 4, +1.0)};
    const EwaldTerm n_term(neutral, 1.0, cell, 1e-6);
    CHECK(n_term.net_charge() == Approx(0.0).margin(1e-12));
    CHECK(n_term.background_energy() == 0.0);
}

// ── Per-pair kernels ────────────────────────────────────────────────────────
//
// One kernel per GROUP PAIR is what lets a multilayer cell be summed at all: a
// cross-layer pair feels 1/sqrt(r^2+d^2), an in-plane pair 1/r, and a single
// structure factor cannot tell them apart.  These tests pin the generalised
// term down before any layer offset exists.

namespace {

using alea::EwaldPair;
using alea::numerics::PotentialSpec;

double energy_with(const EwaldTerm& term, const SimulationCell& cell,
                   const std::vector<double>& pos) {
    const std::size_t np = pos.size() / 2;
    WalkerSet ws(1, np, 2, cell);
    for (std::size_t i = 0; i < np; ++i) {
        ws.positions()[i * 2 + 0] = pos[2 * i + 0];
        ws.positions()[i * 2 + 1] = pos[2 * i + 1];
    }
    ws.dist_tables().build_all(ws.positions());
    ws.local_energies()[0] = 0.0;
    EwaldTerm t = term;  // evaluate_batch is non-const
    t.evaluate_batch(ws);
    return ws.local_energies()[0];
}

// Every unordered pair of groups {0..n-1}, self-pairs included, on `k`.
std::vector<EwaldPair> uniform_pairs(std::size_t n, const PotentialSpec& k) {
    std::vector<EwaldPair> out;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a; b < n; ++b) out.push_back({a, b, k});
    return out;
}

}  // namespace

TEST_CASE("per-pair kernels, all the same, reproduce the Madelung constant",
          "[ewald][pairs]") {
    const Madelung2D sys;
    const EwaldTerm  term(sys.groups, uniform_pairs(2, coul(1.0)), sys.cell, 1e-8);
    const double     M = -2.0 * energy_with(term, sys.cell, sys.pos) / 4.0;
    INFO("Madelung constant from per-pair kernels: " << M);
    CHECK(M == Approx(1.6155426267).margin(1e-6));
}

TEST_CASE("mixed kernels equal all-Coulomb plus a direct sum of the difference",
          "[ewald][pairs][physics]") {
    // Coulomb in-group, Rytova-Keldysh across groups, one shared eps.  The
    // reference needs no Ewald at all for the part that differs: the kernels
    // share their 1/r tail, so dV = V_RK - V_C decays as -r0^2/(eps r^3) and its
    // lattice sum converges ABSOLUTELY.  Summed directly to radius R, with the
    // continuum tail beyond R added analytically:
    //
    //   E_mixed = E_Coulomb + sum_{e,h} q_e q_h sum_L dV(|r_eh + L|)
    //
    // This also tests the k = 0 term.  dV integrates to -2 pi r0/eps over the
    // plane, which the direct sum contains (on average every image contributes
    // (1/A) int dV) and an Ewald sum that simply drops k = 0 would lose -- here
    // 8 pi r0/(eps A) ~ 0.4, nowhere near a rounding error.
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    constexpr double     eps = 1.5, r0 = 0.8, tol = 1e-8;
    const std::vector<ParticleGroup> groups{grp("e", 0, 2, -1.0),
                                            grp("h", 2, 4, +1.0)};
    const std::vector<double> pos{0.3, 0.4, 2.1, 4.4,   // e
                                  1.0, 0.9, 4.2, 2.6};  // h

    const PotentialSpec C = PotentialSpec::coulomb(eps);
    const PotentialSpec K = PotentialSpec::rytova_keldysh(eps, r0);
    const EwaldTerm all_c(groups, uniform_pairs(2, C), cell, tol);
    const EwaldTerm mixed(groups, {{0, 0, C}, {1, 1, C}, {0, 1, K}}, cell, tol);

    const double pi = std::acos(-1.0);
    const auto dV = [&](double r) {
        return (pi / (2.0 * eps * r0)) * alea::numerics::keldysh_kernel(r / r0) -
               1.0 / (eps * r);
    };
    constexpr double R = 400.0;
    const int        nL = static_cast<int>(R / 6.0) + 2;
    double           direct = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 2; j < 4; ++j) {
            const double dx = pos[2 * i] - pos[2 * j];
            const double dy = pos[2 * i + 1] - pos[2 * j + 1];
            double       s  = 0.0;
            for (int n1 = -nL; n1 <= nL; ++n1)
                for (int n2 = -nL; n2 <= nL; ++n2) {
                    const double r = std::hypot(dx + 6.0 * n1, dy + 6.0 * n2);
                    if (r < R) s += dV(r);
                }
            // Continuum beyond R, images at density 1/A:
            // int_R^inf dV(r) 2 pi r dr / A with dV ~ -r0^2/(eps r^3).
            s += -2.0 * pi * r0 * r0 / (eps * 36.0 * R);
            direct += -1.0 * s;  // q_e q_h
        }

    const double e_c = energy_with(all_c, cell, pos);
    const double e_m = energy_with(mixed, cell, pos);
    INFO("E_coulomb " << e_c << "  E_mixed " << e_m << "  direct dE " << direct
                      << "  Ewald dE " << e_m - e_c);
    CHECK(e_m - e_c == Approx(direct).margin(2e-6));
}

TEST_CASE("EwaldTerm rejects pair kernels that do not define an energy",
          "[ewald][pairs]") {
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    const PotentialSpec  C1   = PotentialSpec::coulomb(1.0);
    const PotentialSpec  C2   = PotentialSpec::coulomb(2.0);
    const PotentialSpec  K1   = PotentialSpec::rytova_keldysh(1.0, 0.8);
    const std::vector<ParticleGroup> neutral{grp("e", 0, 2, -1.0),
                                             grp("h", 2, 4, +1.0)};

    SECTION("every group pair must be assigned exactly once") {
        CHECK_THROWS(EwaldTerm(neutral, {{0, 0, C1}, {1, 1, C1}}, cell, 1e-6));
        CHECK_THROWS(EwaldTerm(neutral, {{0, 0, C1}, {1, 1, C1}, {0, 1, C1},
                                         {1, 0, C1}}, cell, 1e-6));
        CHECK_THROWS(EwaldTerm(neutral, {{0, 0, C1}, {1, 1, C1}, {0, 2, C1}},
                               cell, 1e-6));
    }
    SECTION("kernels must share their long-range tail") {
        // 2 pi/(eps k) with different eps: the 1/k divergences of the pairs no
        // longer cancel against neutrality, so the periodic sum diverges.
        CHECK_THROWS(EwaldTerm(neutral, {{0, 0, C1}, {1, 1, C1}, {0, 1, C2}},
                               cell, 1e-6));
    }
    SECTION("a charged cell needs a single kernel") {
        // The uniform background couples to every charge through ONE kernel;
        // with several, which one it uses is undefined.
        const std::vector<ParticleGroup> charged{grp("e", 0, 2, -1.0),
                                                 grp("h", 2, 3, +1.0)};
        CHECK_THROWS(EwaldTerm(charged, {{0, 0, C1}, {1, 1, C1}, {0, 1, K1}},
                               cell, 1e-6));
    }
}

TEST_CASE("the cusp residue follows each pair's own kernel", "[ewald][pairs]") {
    // The Jastrow builder cancels a 1/r pole, and only a Coulomb kernel has
    // one: Rytova-Keldysh diverges only logarithmically at the origin, so its
    // residue is zero -- as the open-boundary KeldyshTerm already reports.
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    const ParticleGroup  e = grp("e", 0, 2, -1.0), h = grp("h", 2, 4, +1.0);
    const PotentialSpec  C = PotentialSpec::coulomb(1.5);
    const PotentialSpec  K = PotentialSpec::rytova_keldysh(1.5, 0.8);

    const EwaldTerm mixed({e, h}, {{0, 0, C}, {1, 1, C}, {0, 1, K}}, cell, 1e-6);
    CHECK(mixed.coulomb_cusp_residue(e, e) == Approx(1.0 / 1.5));
    CHECK(mixed.coulomb_cusp_residue(h, h) == Approx(1.0 / 1.5));
    CHECK(mixed.coulomb_cusp_residue(e, h) == 0.0);
    CHECK(mixed.coulomb_cusp_residue(h, e) == 0.0);

    const EwaldTerm keldysh_only({e, h}, K, 1.5, cell, 1e-6);
    CHECK(keldysh_only.coulomb_cusp_residue(e, h) == 0.0);
}

// ── Layer offsets ───────────────────────────────────────────────────────────

namespace {

// Independent reference: the erfc-split Ewald sum for point charges in planes
// z_i, periodic in x and y only (Parry 1975; Heyes, Barber & Clarke 1977).
// With rho_ij the in-plane and z_ij the out-of-plane separation,
//
//   E =  1/2 sum_ij sum'_L q_i q_j erfc(a R)/R,   R = sqrt(|rho_ij + L|^2 + z_ij^2)
//      + pi/(2A) sum_ij q_i q_j sum_{k!=0} cos(k.rho_ij)/k
//            * [e^{k z} erfc(k/2a + a z) + e^{-k z} erfc(k/2a - a z)]
//      - pi/A sum_ij q_i q_j [z_ij erf(a z_ij) + e^{-a^2 z_ij^2}/(a sqrt(pi))]
//      - a/sqrt(pi) sum_i q_i^2
//
// (all over eps).  Shares nothing with the optimized breakup: no spline, no
// fit, no Hankel transform.  Its own correctness is checked by requiring the
// result to be independent of the splitting parameter a.
double parry_energy(const std::vector<double>& pos, const std::vector<double>& z,
                    const std::vector<double>& q, double L, double eps, double a) {
    const double      pi = std::acos(-1.0), A = L * L;
    const std::size_t n  = q.size();
    const int         nr = 4, nk = 24;
    double            e  = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double dx = pos[2 * i] - pos[2 * j], dy = pos[2 * i + 1] - pos[2 * j + 1];
            const double dz = z[i] - z[j], qq = q[i] * q[j];
            for (int n1 = -nr; n1 <= nr; ++n1)
                for (int n2 = -nr; n2 <= nr; ++n2) {
                    if (i == j && n1 == 0 && n2 == 0) continue;
                    const double R = std::sqrt(std::pow(dx + L * n1, 2) +
                                               std::pow(dy + L * n2, 2) + dz * dz);
                    e += 0.5 * qq * std::erfc(a * R) / R;
                }
            for (int m1 = -nk; m1 <= nk; ++m1)
                for (int m2 = -nk; m2 <= nk; ++m2) {
                    if (m1 == 0 && m2 == 0) continue;
                    const double kx = 2 * pi * m1 / L, ky = 2 * pi * m2 / L;
                    const double k  = std::hypot(kx, ky);
                    const double f  = std::exp(k * dz) * std::erfc(k / (2 * a) + a * dz) +
                                     std::exp(-k * dz) * std::erfc(k / (2 * a) - a * dz);
                    e += pi / (2 * A) * qq * std::cos(kx * dx + ky * dy) / k * f;
                }
            e -= pi / A * qq *
                 (dz * std::erf(a * dz) + std::exp(-a * a * dz * dz) / (a * std::sqrt(pi)));
        }
    for (std::size_t i = 0; i < n; ++i) e -= a / std::sqrt(pi) * q[i] * q[i];
    return e / eps;
}

}  // namespace

TEST_CASE("a Coulomb bilayer matches the slab Ewald sum of Parry",
          "[ewald][pairs][layers][physics]") {
    // Electrons in z = 0, holes in z = d.  The cross-layer kernel is
    // 1/(eps sqrt(r^2 + d^2)) with transform 2 pi e^{-kd}/(eps k); in-plane
    // pairs stay 1/(eps r).  Its k = 0 remainder, -2 pi d/eps, is what makes the
    // charged layers a capacitor, and the reference contains it independently
    // (the erf/exp term above, which tends to -pi/A sum q q |z| as a grows).
    // d from nearly in-plane (sharp kernel) to beyond the inradius (flat one).
    const double d = GENERATE(0.3, 1.0, 5.0);
    constexpr double L = 6.0, eps = 1.5;
    const SimulationCell cell = cell_from({L, 0, 0, L});
    const std::vector<double> pos{0.3, 0.4, 2.1, 4.4,   // e
                                  1.0, 0.9, 4.2, 2.6};  // h
    const std::vector<double> z{0.0, 0.0, d, d};
    const std::vector<double> q{-1.0, -1.0, +1.0, +1.0};

    const double ref_a = parry_energy(pos, z, q, L, eps, 1.0);
    const double ref_b = parry_energy(pos, z, q, L, eps, 1.4);
    INFO("d = " << d << "  Parry reference at a = 1.0: " << ref_a << ", at a = 1.4: " << ref_b);
    REQUIRE(ref_a == Approx(ref_b).margin(1e-10));  // the reference itself

    const PotentialSpec in_plane = PotentialSpec::coulomb(eps);
    const PotentialSpec across   = PotentialSpec::coulomb(eps, d);
    const EwaldTerm bilayer({grp("e", 0, 2, -1.0), grp("h", 2, 4, +1.0)},
                            {{0, 0, in_plane}, {1, 1, in_plane}, {0, 1, across}},
                            cell, 1e-8);
    const double e = energy_with(bilayer, cell, pos);
    INFO("EwaldTerm bilayer: " << e);
    CHECK(e == Approx(ref_a).margin(1e-6));
}

TEST_CASE("EwaldTerm reports the capacitor energy of charged layers",
          "[ewald][pairs][layers]") {
    // All electrons in one plane, all holes in the other: two uniformly charged
    // sheets at mean-field level, whose electrostatic energy per cell is
    //   E_cap = 2 pi sigma^2 d A/eps = -(pi/(eps A)) sum_{g,h} Q_g Q_h d_gh.
    // It is a constant, already inside E_L (the Parry test covers that); this
    // pins down the separately REPORTED number, so that energies can be
    // compared with work that quotes them without it.
    constexpr double eps = 1.5, d = 1.0, A = 36.0;
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    const PotentialSpec  C    = PotentialSpec::coulomb(eps);
    const std::vector<ParticleGroup> groups{grp("e", 0, 2, -1.0),
                                            grp("h", 2, 4, +1.0)};
    const double pi = std::acos(-1.0);

    const EwaldTerm bilayer(groups,
                            {{0, 0, C}, {1, 1, C}, {0, 1, PotentialSpec::coulomb(eps, d)}},
                            cell, 1e-6);
    // Q_e = -2, Q_h = +2: sum over ordered pairs = 2 * (-2)(+2) d = -8 d.
    CHECK(bilayer.capacitor_energy() == Approx(8.0 * pi * d / (eps * A)));

    // One plane: no capacitor.
    const EwaldTerm mono(groups, C, eps, cell, 1e-6);
    CHECK(mono.capacitor_energy() == 0.0);
}

TEST_CASE("the capacitor energy of Keldysh layers is their mean-field energy",
          "[ewald][pairs][layers][keldysh]") {
    // Two sheets of charge -Q and +Q, d apart, interacting through V_RK in-plane
    // and V_RK(sqrt(r^2+d^2)) across.  Smeared uniformly, their energy is
    //   E = (Q^2/A) * integral [V_RK(r) - V_RK(sqrt(r^2 + d^2))] d^2r,
    // computed here in REAL space, independently of any k = 0 limit.  (The r0
    // screening of each layer cancels between the pairs; what is left is not
    // 2 pi d/eps per unit charge^2, because the cross kernel is not Coulomb.)
    constexpr double eps = 1.5, r0 = 0.8, d = 1.0, A = 36.0, Q = 2.0;
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    const PotentialSpec  K    = PotentialSpec::rytova_keldysh(eps, r0);
    const EwaldTerm bilayer({grp("e", 0, 2, -1.0), grp("h", 2, 4, +1.0)},
                            {{0, 0, K}, {1, 1, K},
                             {0, 1, PotentialSpec::rytova_keldysh(eps, r0, d)}},
                            cell, 1e-6);

    const double pi = std::acos(-1.0);
    const auto   V  = [&](double s) {
        return pi / (2.0 * eps * r0) * alea::numerics::keldysh_kernel(s / r0);
    };
    // 2 pi int_0^R [V(r) - V(sqrt(r^2+d^2))] r dr on a geometric mesh (V(r) has
    // a log at 0), plus the tail beyond R where the bracket ~ d^2/(2 eps r^3).
    constexpr double R = 2000.0;
    double integral = 0.0;
    for (double a = 1e-9; a < R; a *= 1.02) {
        const double b = std::min(1.02 * a, R), m = 0.5 * (a + b), h = 0.5 * (b - a);
        const double g = 0.5773502691896257 * h;  // 2-point Gauss-Legendre
        for (double r : {m - g, m + g}) integral += h * r * (V(r) - V(std::hypot(r, d)));
    }
    integral = 2.0 * pi * (integral + d * d / (2.0 * eps * R));
    const double expected = Q * Q / A * integral;
    INFO("mean-field sheets " << expected << "  bare Coulomb capacitor "
                              << 2.0 * pi * Q * Q * d / (eps * A));
    CHECK(bilayer.capacitor_energy() == Approx(expected).epsilon(1e-5));
}

TEST_CASE("a cross-plane pair has no cusp", "[ewald][pairs][layers]") {
    // 1/sqrt(r^2 + d^2) is finite at r = 0, so the Jastrow builder must not
    // try to cancel a pole there (as the open-boundary CoulombTerm with d > 0
    // already reports).
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    const ParticleGroup  e = grp("e", 0, 2, -1.0), h = grp("h", 2, 4, +1.0);
    const PotentialSpec  C = PotentialSpec::coulomb(1.0);
    const EwaldTerm term({e, h},
                         {{0, 0, C}, {1, 1, C}, {0, 1, PotentialSpec::coulomb(1.0, 1.0)}},
                         cell, 1e-6);
    CHECK(term.coulomb_cusp_residue(e, h) == 0.0);
    CHECK(term.coulomb_cusp_residue(e, e) == Approx(1.0));
}

TEST_CASE("the cross-plane Keldysh transform reduces to its analytic limits",
          "[ewald][layers][keldysh]") {
    // V(r) = V_RK(sqrt(r^2 + d^2)) has no closed-form transform, so it is
    // computed numerically.  Its two limits do have one:
    //   d -> 0 : 2 pi/(eps k (1 + r0 k)),   c -> -2 pi r0/eps
    //   r0 -> 0: 2 pi e^{-kd}/(eps k),       c -> -2 pi d/eps
    // Both are approached from inside the numerical path (tiny but nonzero d or
    // r0), so they check the quadrature and not a shortcut around it.
    constexpr double eps = 1.5, r0 = 0.8, d = 1.2;
    const double pi = std::acos(-1.0);

    const PotentialSpec near_plane = PotentialSpec::rytova_keldysh(eps, r0, 1e-9);
    const PotentialSpec near_coul  = PotentialSpec::rytova_keldysh(eps, 1e-9, d);
    for (double k : {1e-3, 0.05, 0.7, 3.0, 40.0, 900.0}) {
        INFO("k = " << k);
        CHECK(near_plane.v_fourier(k) ==
              Approx(2 * pi / (eps * k * (1 + r0 * k))).epsilon(1e-8));
        CHECK(near_coul.v_fourier(k) ==
              Approx(2 * pi * std::exp(-k * d) / (eps * k)).epsilon(1e-8).margin(1e-300));
    }
    CHECK(near_plane.k0_offset == Approx(-2 * pi * r0 / eps).epsilon(1e-8));
    CHECK(near_coul.k0_offset == Approx(-2 * pi * d / eps).epsilon(1e-8));

    // The real-space value is the open-boundary KeldyshTerm's: V_RK at the 3-D
    // separation, which the Ewald sum must reproduce and not redefine.
    const PotentialSpec across = PotentialSpec::rytova_keldysh(eps, r0, d);
    const double s = std::hypot(0.9, d);
    CHECK(across.v_real(0.9) ==
          Approx(pi / (2 * eps * r0) * alea::numerics::keldysh_kernel(s / r0)));
}

TEST_CASE("a Keldysh bilayer equals the Coulomb-across one plus a direct sum",
          "[ewald][pairs][layers][keldysh][physics]") {
    // Rytova-Keldysh everywhere (in-plane and across), electrons at z = 0 and
    // holes at z = d.  Swapping only the cross-plane kernel for Coulomb-at-d
    // (validated against Parry above) changes the energy by the lattice sum of
    //   dV(r) = V_RK(s) - 1/(eps s),   s = sqrt(r^2 + d^2),
    // which decays as -r0^2/(eps r^3) and so converges absolutely: summed
    // directly to R with the continuum tail -2 pi r0^2/(eps A R) beyond it.
    // This checks the numerical transform AND its k = 0 constant at once.
    constexpr double L = 6.0, eps = 1.5, r0 = 0.8, d = 1.0, tol = 1e-7;
    const SimulationCell cell = cell_from({L, 0, 0, L});
    const std::vector<ParticleGroup> groups{grp("e", 0, 2, -1.0),
                                            grp("h", 2, 4, +1.0)};
    const std::vector<double> pos{0.3, 0.4, 2.1, 4.4, 1.0, 0.9, 4.2, 2.6};

    const PotentialSpec K  = PotentialSpec::rytova_keldysh(eps, r0);
    const EwaldTerm rk_across(groups,
                              {{0, 0, K}, {1, 1, K},
                               {0, 1, PotentialSpec::rytova_keldysh(eps, r0, d)}},
                              cell, tol);
    const EwaldTerm c_across(groups,
                             {{0, 0, K}, {1, 1, K}, {0, 1, PotentialSpec::coulomb(eps, d)}},
                             cell, tol);

    const double pi = std::acos(-1.0);
    const auto dV = [&](double r) {
        const double s = std::hypot(r, d);
        return (pi / (2.0 * eps * r0)) * alea::numerics::keldysh_kernel(s / r0) -
               1.0 / (eps * s);
    };
    constexpr double R  = 400.0;
    const int        nL = static_cast<int>(R / L) + 2;
    double           direct = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 2; j < 4; ++j) {
            const double dx = pos[2 * i] - pos[2 * j], dy = pos[2 * i + 1] - pos[2 * j + 1];
            double       sum = 0.0;
            for (int n1 = -nL; n1 <= nL; ++n1)
                for (int n2 = -nL; n2 <= nL; ++n2) {
                    const double r = std::hypot(dx + L * n1, dy + L * n2);
                    if (r < R) sum += dV(r);
                }
            sum += -2.0 * pi * r0 * r0 / (eps * L * L * R);
            direct += -1.0 * sum;  // q_e q_h
        }

    const double e_rk = energy_with(rk_across, cell, pos);
    const double e_c  = energy_with(c_across, cell, pos);
    INFO("E(RK across) " << e_rk << "  E(C across) " << e_c << "  Ewald dE "
                         << e_rk - e_c << "  direct dE " << direct);
    CHECK(e_rk - e_c == Approx(direct).margin(2e-6));
}

TEST_CASE("mixed kernels are invariant under periodic translations",
          "[ewald][pairs][cell]") {
    // With one structure factor per group a bookkeeping slip (a particle summed
    // into the wrong class, a cross term with the wrong sign) breaks the
    // periodicity of the energy; translations catch it without any reference.
    const SimulationCell cell = cell_from({6, 0, 0, 6});
    const PotentialSpec  C    = PotentialSpec::coulomb(1.5);
    const PotentialSpec  K    = PotentialSpec::rytova_keldysh(1.5, 0.8);
    const EwaldTerm mixed({grp("e", 0, 2, -1.0), grp("h", 2, 4, +1.0)},
                          {{0, 0, C}, {1, 1, C}, {0, 1, K}}, cell, 1e-8);
    const std::vector<double> pos{0.3, 0.4, 2.1, 4.4, 1.0, 0.9, 4.2, 2.6};
    const double base = energy_with(mixed, cell, pos);

    std::vector<double> hole_wrapped = pos;  // one hole by (-6, +12)
    hole_wrapped[2 * 3 + 0] -= 6.0;
    hole_wrapped[2 * 3 + 1] += 12.0;
    CHECK(energy_with(mixed, cell, hole_wrapped) == Approx(base).margin(1e-8));

    std::vector<double> shifted = pos;  // everything by a non-lattice vector
    for (std::size_t i = 0; i < 4; ++i) {
        shifted[2 * i + 0] += 0.37;
        shifted[2 * i + 1] -= 0.81;
    }
    CHECK(energy_with(mixed, cell, shifted) == Approx(base).margin(1e-8));
}

// ── Single-particle ΔV (the PIMC path) ──────────────────────────────────────
// EwaldTerm::delta_potential_particle must equal the change of the full
// energy, evaluate(after) − evaluate(before), for every kernel kind, and its
// structure-factor cache must stay exact whatever happens to the batch
// between calls: accepted and rejected proposals, several particles moved at
// once, table rebuilds, and the periodic rebuild that bounds rounding drift.
namespace {

struct DeltaSystem {
    SimulationCell cell = cell_from({5.0, 0.0, 0.0, 5.0});
    std::vector<ParticleGroup> groups;
    EwaldTerm term;
};

// Brute force: set p's position in configuration k, evaluate, restore.
double brute_delta(const EwaldTerm& term, WalkerSet& ws, std::size_t p, std::size_t k,
                   const double* proposed) {
    const std::size_t nw = ws.num_walkers();
    std::vector<double> v(nw, 0.0);
    term.evaluate_potential(ws, v, k, k + 1);
    const double before = v[k];
    const double old_x = ws.positions()[p * 2 * nw + k];
    const double old_y = ws.positions()[p * 2 * nw + nw + k];
    ws.positions()[p * 2 * nw + k] = proposed[0];
    ws.positions()[p * 2 * nw + nw + k] = proposed[1];
    ws.refresh_particle_tables(p, k, k + 1);
    v[k] = 0.0;
    term.evaluate_potential(ws, v, k, k + 1);
    const double after = v[k];
    ws.positions()[p * 2 * nw + k] = old_x;
    ws.positions()[p * 2 * nw + nw + k] = old_y;
    ws.refresh_particle_tables(p, k, k + 1);
    return after - before;
}

void move_particle(WalkerSet& ws, std::size_t p, std::size_t k, const double* r) {
    const std::size_t nw = ws.num_walkers();
    ws.positions()[p * 2 * nw + k] = r[0];
    ws.positions()[p * 2 * nw + nw + k] = r[1];
    ws.refresh_particle_tables(p, k, k + 1);
}

// `which` selects systems by index (all when empty): Ewald tables at 1e-8
// take seconds per kernel, so each test builds only what it uses.
std::vector<std::pair<const char*, EwaldTerm>> delta_systems(const SimulationCell& cell,
                                                             std::vector<int> which = {}) {
    std::vector<std::pair<const char*, EwaldTerm>> out;
    auto wanted = [&](int i) {
        return which.empty() || std::find(which.begin(), which.end(), i) != which.end();
    };
    // Charged one-component cell: the jellium background constant.
    if (wanted(0)) out.emplace_back("jellium", EwaldTerm({grp("b", 0, 6, +1.0)}, 1.0, cell, 1e-8));
    // Neutral two-component in-plane plasma, epsilon = 2.
    if (wanted(1)) out.emplace_back("in-plane e-h",
                     EwaldTerm({grp("e", 0, 3, -1.0), grp("h", 3, 6, +1.0)}, 2.0, cell, 1e-8));
    // Coulomb bilayer: two structure-factor classes, a cross-plane kernel.
    if (wanted(2)) out.emplace_back("Coulomb bilayer",
                     EwaldTerm({grp("e", 0, 3, -1.0), grp("h", 3, 6, +1.0)},
                               {{0, 0, PotentialSpec::coulomb(1.0)},
                                {1, 1, PotentialSpec::coulomb(1.0)},
                                {0, 1, PotentialSpec::coulomb(1.0, 0.7)}},
                               cell, 1e-8));
    // Rytova-Keldysh bilayer: the Keldysh kind with its cutoff correction.
    if (wanted(3)) out.emplace_back("Keldysh bilayer",
                     EwaldTerm({grp("e", 0, 3, -1.0), grp("h", 3, 6, +1.0)},
                               {{0, 0, PotentialSpec::rytova_keldysh(1.5, 2.0)},
                                {1, 1, PotentialSpec::rytova_keldysh(1.5, 2.0)},
                                {0, 1, PotentialSpec::rytova_keldysh(1.5, 2.0, 0.7)}},
                               cell, 1e-8));
    return out;
}

}  // namespace

TEST_CASE("Ewald single-particle delta equals the change of the full energy",
          "[ewald][delta]") {
    const SimulationCell cell = cell_from({5.0, 0.0, 0.0, 5.0});
    for (auto& [label, term] : delta_systems(cell)) {
        constexpr std::size_t nw = 4, np = 6;
        WalkerSet ws(nw, np, 2, cell);
        alea::numerics::Random rng(911);
        for (auto& x : ws.positions()) x = 5.0 * rng.uniform();
        ws.dist_tables().build_all(ws.positions());
        std::vector<double> scratch(nw, 0.0);
        double worst = 0.0;
        for (int t = 0; t < 300; ++t) {
            const auto p = static_cast<std::size_t>(rng.uniform() * np);
            const auto k = static_cast<std::size_t>(rng.uniform() * nw);
            // Small and cell-spanning displacements, outside the cell too.
            const double step = (t % 3 == 0) ? 7.0 : 0.4;
            double r[2] = {ws.positions()[p * 2 * nw + k] + step * (rng.uniform() - 0.5),
                           ws.positions()[p * 2 * nw + nw + k] + step * (rng.uniform() - 0.5)};
            const double fast = term.delta_potential_particle(ws, p, k, r, scratch);
            const double exact = brute_delta(term, ws, p, k, r);
            worst = std::max(worst, std::abs(fast - exact));
        }
        INFO(label << ": worst |delta - (E_after - E_before)| = " << worst);
        CHECK(worst < 1e-10);
    }
}

TEST_CASE("the Ewald delta cache follows accepted, rejected and many-particle changes",
          "[ewald][delta]") {
    const SimulationCell cell = cell_from({5.0, 0.0, 0.0, 5.0});
    for (auto& [label, term] : delta_systems(cell, {0, 2})) {   // jellium, bilayer
        constexpr std::size_t nw = 2, np = 6;
        WalkerSet ws(nw, np, 2, cell);
        alea::numerics::Random rng(4242);
        for (auto& x : ws.positions()) x = 5.0 * rng.uniform();
        ws.dist_tables().build_all(ws.positions());
        std::vector<double> scratch(nw, 0.0);
        double worst = 0.0;
        // 9000 steps on 2 configurations push each past the 4096-increment
        // drift rebuild.
        for (int t = 0; t < 9000; ++t) {
            const auto k = static_cast<std::size_t>(rng.uniform() * nw);
            if (t % 97 == 0) {
                // Several particles at once: 3 (incremental) or all (rebuild).
                const std::size_t count = (t % 2 == 0) ? 3 : np;
                for (std::size_t m = 0; m < count; ++m) {
                    const double r[2] = {5.0 * rng.uniform(), 5.0 * rng.uniform()};
                    move_particle(ws, m, k, r);
                }
            }
            if (t % 1013 == 0) ws.dist_tables().build_all(ws.positions());
            const auto p = static_cast<std::size_t>(rng.uniform() * np);
            const double r[2] = {ws.positions()[p * 2 * nw + k] + 0.6 * (rng.uniform() - 0.5),
                                 ws.positions()[p * 2 * nw + nw + k] + 0.6 * (rng.uniform() - 0.5)};
            const double fast = term.delta_potential_particle(ws, p, k, r, scratch);
            if (t % 7 == 0) worst = std::max(worst, std::abs(fast - brute_delta(term, ws, p, k, r)));
            if (rng.uniform() < 0.8) {                    // accept: commit like a move
                double w[2] = {r[0], r[1]};
                cell.wrap(w);
                move_particle(ws, p, k, w);
            }
        }
        // A clone starts empty and must agree on the same batch.
        const auto clone = term.clone();
        const double r[2] = {1.3, 4.1};
        const double from_clone = clone->delta_potential_particle(ws, 2, 1, r, scratch);
        const double from_term = term.delta_potential_particle(ws, 2, 1, r, scratch);
        INFO(label << ": worst |delta - brute| over the sequence = " << worst);
        CHECK(worst < 1e-10);
        CHECK(std::abs(from_clone - from_term) < 1e-10);
    }
}

TEST_CASE("Excluding the capacitor removes exactly the reported capacitor energy",
          "[ewald][pairs][layers][capacitor]") {
    // A neutral bilayer: with `capacitor: exclude` the k = 0 remainders c_gh
    // of the pair kernels are dropped, which is the capacitor energy and
    // nothing else.
    constexpr double L = 6.0, eps = 1.5, d = 0.7;
    const SimulationCell cell = cell_from({L, 0, 0, L});
    const std::vector<double> pos{0.3, 0.4, 2.2, 3.9, 4.8, 1.1,     // e
                                  1.0, 0.9, 4.2, 2.6, 3.3, 5.0};    // h
    const std::vector<ParticleGroup> groups{grp("e", 0, 3, -1.0), grp("h", 3, 6, +1.0)};
    const std::vector<EwaldPair> pairs{{0, 0, PotentialSpec::coulomb(eps)},
                                       {1, 1, PotentialSpec::coulomb(eps)},
                                       {0, 1, PotentialSpec::coulomb(eps, d)}};
    const EwaldTerm with(groups, pairs, cell, 1e-8);
    const EwaldTerm without(groups, pairs, cell, 1e-8, /*exclude_capacitor=*/true);
    const double pi = std::acos(-1.0);
    REQUIRE(with.capacitor_energy() == Approx(2.0 * pi * d * 9.0 / (eps * L * L)));
    CHECK(without.capacitor_energy() == 0.0);
    CHECK(without.capacitor_excluded());
    CHECK(energy_with(with, cell, pos) - energy_with(without, cell, pos) ==
          Approx(with.capacitor_energy()).margin(1e-10));
}

TEST_CASE("A charged bilayer without the capacitor matches Parry's slab sum minus its "
          "interlayer k = 0 term", "[ewald][pairs][layers][capacitor][physics]") {
    // CASINO's convention: every layer neutralized in its own plane, i.e.
    // Parry's 2-D slab Ewald sum (which carries a uniform background for the
    // net charge) without −(π/εA) Σ_ij q_i q_j |z_i − z_j|.  The reference is
    // independent of its splitting a; Alea's of its tolerance.  Three
    // electrons and two holes: net charge −1, the N+1/2 case.
    constexpr double L = 6.0, eps = 1.0;
    const double d = GENERATE(0.4, 2.0);
    const SimulationCell cell = cell_from({L, 0, 0, L});
    const std::vector<double> pos{0.3, 0.4, 2.2, 3.9, 4.8, 1.1,   // e
                                  1.0, 0.9, 4.2, 2.6};            // h
    const std::vector<double> z{0.0, 0.0, 0.0, d, d};
    const std::vector<double> q{-1.0, -1.0, -1.0, +1.0, +1.0};
    const double pi = std::acos(-1.0), A = L * L;
    auto reference = [&](double a) {
        double interlayer = 0.0;
        for (std::size_t i = 0; i < q.size(); ++i)
            for (std::size_t j = 0; j < q.size(); ++j)
                interlayer += q[i] * q[j] * std::abs(z[i] - z[j]);
        return parry_energy(pos, z, q, L, eps, a) + pi / (eps * A) * interlayer;
    };
    const double ref = reference(1.0);
    REQUIRE(ref == Approx(reference(1.4)).margin(1e-10));
    const std::vector<ParticleGroup> groups{grp("e", 0, 3, -1.0), grp("h", 3, 5, +1.0)};
    const std::vector<EwaldPair> pairs{{0, 0, PotentialSpec::coulomb(eps)},
                                       {1, 1, PotentialSpec::coulomb(eps)},
                                       {0, 1, PotentialSpec::coulomb(eps, d)}};
    CHECK_THROWS(EwaldTerm(groups, pairs, cell, 1e-8));   // undefined with the capacitor
    for (const double tol : {1e-6, 1e-9}) {
        CAPTURE(d, tol);
        const EwaldTerm term(groups, pairs, cell, tol, /*exclude_capacitor=*/true);
        CHECK(energy_with(term, cell, pos) == Approx(ref).margin(20 * tol));
    }
}

TEST_CASE("One charge in a square cell has CASINO's self-image energy",
          "[ewald][capacitor][physics]") {
    // The energy of one unit charge with its images and the uniform
    // background is v_M/2, v_M the Madelung potential of the square lattice.
    // CASINO prints v_M = −0.144469170127861 au for the cell of side
    // 26.9972127382 (López Ríos et al. 2018 data, rs2/N/out.6).
    constexpr double L = 26.9972127382;
    const SimulationCell cell = cell_from({L, 0, 0, L});
    const EwaldTerm one({grp("e", 0, 1, -1.0)}, coul(), 1.0, cell, 1e-10);
    CHECK(energy_with(one, cell, {3.0, 7.0}) == Approx(-0.144469170127861 / 2).margin(1e-9));
}

TEST_CASE("Without the capacitor an extra electron and an extra hole cost the same",
          "[ewald][pairs][layers][capacitor]") {
    // q → −q and z → d − z leave every pair's interaction unchanged, so three
    // electrons over two holes and three holes over two electrons, at mirrored
    // positions, have one energy: the e ↔ h symmetry of the convention.
    constexpr double L = 6.0, d = 0.4;
    const SimulationCell cell = cell_from({L, 0, 0, L});
    const std::vector<double> three{0.3, 0.4, 2.2, 3.9, 4.8, 1.1}, two{1.0, 0.9, 4.2, 2.6};
    auto energy = [&](double q_three, double z_three) {
        std::vector<double> pos = three;
        pos.insert(pos.end(), two.begin(), two.end());
        const std::vector<ParticleGroup> groups{grp("a", 0, 3, q_three), grp("b", 3, 5, -q_three)};
        const double sep = std::abs(z_three - (d - z_three));
        const EwaldTerm term(groups,
                             {{0, 0, PotentialSpec::coulomb(1.0)}, {1, 1, PotentialSpec::coulomb(1.0)},
                              {0, 1, PotentialSpec::coulomb(1.0, sep)}},
                             cell, 1e-9, /*exclude_capacitor=*/true);
        return energy_with(term, cell, pos);
    };
    CHECK(energy(-1.0, 0.0) == Approx(energy(+1.0, d)).margin(1e-12));
}
