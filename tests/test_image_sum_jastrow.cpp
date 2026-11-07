// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Pair Jastrows summed over every periodic image (Ewald), not only the nearest.
//
// Ψ must be periodic AND smooth for the variational bound to hold: Ceperley,
// "Introduction to QMC methods applied to the electron gas", §2.2 (∇Ψ continuous
// at the edges of the box) and §2.5 (the Ewald image method for the trial
// function).  A Padé read at the minimum-image distance is periodic but has a
// kink on the Wigner–Seitz boundary, because ∇u ≠ 0 there and the nearest image
// switches sides.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <limits>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

#include <random>

#include "numerics/long_range_breakup.h"
#include "numerics/pair_radial.h"
#include "population/simulation_cell.h"
#include "population/walker_set.h"
#include "wavefunction/drummond_jastrow.h"
#include "wavefunction/pade_jastrow.h"
#include "wavefunction/trial_wavefunction.h"

using alea::DrummondJastrow;
using alea::PadeJastrow;
using alea::Parameter;
using alea::ParticleGroup;
using alea::SimulationCell;
using alea::TrialWavefunction;
using alea::WalkerSet;
using Catch::Approx;

namespace {

ParticleGroup group(std::size_t first, std::size_t last) {
    ParticleGroup g;
    g.name   = first == 0 ? "a" : "b";
    g.first  = first;
    g.last   = last;
    return g;
}

SimulationCell square(double L) {
    const std::vector<double> a = {L, 0.0, 0.0, L};
    return SimulationCell(2, std::span<const double>(a));
}

// u = a₁r/(1 + b₁r) between particle 0 (group a) and particle 1 (group b).
std::unique_ptr<PadeJastrow> eh_pade(const SimulationCell* image_sum_cell) {
    std::vector<Parameter> p = {{"a1", -1.0, true}, {"b1", 0.5, false}};
    auto j = std::make_unique<PadeJastrow>(1, 1, group(0, 1), group(1, 2), p);
    if (image_sum_cell) j->sum_over_images(*image_sum_cell);
    return j;
}

// Evaluate the component for particle 0 at the origin and particle 1 at (x, y).
struct Eval { double log_psi, gx0, gy0; };
Eval evaluate(PadeJastrow& j, const SimulationCell& cell, double x, double y) {
    WalkerSet ws(/*nw=*/1, /*np=*/2, /*dim=*/2, cell);
    auto pos = ws.positions();
    pos[0] = 0.0; pos[1] = 0.0;   // [p*dim*nw + d*nw + w] at nw = 1
    pos[2] = x;   pos[3] = y;
    ws.dist_tables().build_all(ws.positions());
    j.evaluate_batch(ws);
    return {j.log_psi_c()[0], j.grad_c()[0], j.grad_c()[1]};
}

SimulationCell hexagonal(double L) {
    const std::vector<double> a = {L, 0.0, 0.5 * L, L * std::sqrt(3.0) / 2.0};
    return SimulationCell(2, std::span<const double>(a));
}

// [2/2] Padé, u → a₂/b₂: the general case (constant tail removed, 1/r tail
// summed in k-space).  Q > 0 for r > 0.
std::vector<Parameter> pade22() {
    return {{"a1", -0.6, true}, {"a2", 0.2, false}, {"b1", 0.8, false}, {"b2", 0.3, false}};
}

// Three particles in group A (0..2), two in group B (3..4).
constexpr std::size_t kNp = 5;
const ParticleGroup kA = group(0, 3);
const ParticleGroup kB = [] { ParticleGroup g = group(3, 5); g.name = "b"; return g; }();

std::unique_ptr<PadeJastrow> summed(const ParticleGroup& a, const ParticleGroup& b,
                                    const SimulationCell& cell) {
    auto j = std::make_unique<PadeJastrow>(2, 2, a, b, pade22());
    j->sum_over_images(cell);
    return j;
}

// Uniform positions in the cell, from fractional coordinates.
void scatter(WalkerSet& ws, const SimulationCell& cell, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const auto a = cell.vectors();   // [a1x, a1y, a2x, a2y]
    const std::size_t nw = ws.num_walkers();
    auto pos = ws.positions();
    for (std::size_t w = 0; w < nw; ++w)
        for (std::size_t p = 0; p < ws.n_particles(); ++p) {
            const double f1 = u(rng), f2 = u(rng);
            pos[(p * 2 + 0) * nw + w] = f1 * a[0] + f2 * a[2];
            pos[(p * 2 + 1) * nw + w] = f1 * a[1] + f2 * a[3];
        }
    ws.dist_tables().build_all(ws.positions());
}

}  // namespace

TEST_CASE("image-summed Pade gradient and laplacian are the derivatives of its log",
          "[image_sum][pade]") {
    // Both halves — minimum-image short range and reciprocal space — must be
    // differentiated consistently, for one group (|ρ_A|²) and two (Re ρ_A ρ_B*).
    const SimulationCell cell = hexagonal(5.0);
    std::mt19937_64 rng(7);
    WalkerSet base(1, kNp, 2, cell);
    scatter(base, cell, rng);
    const std::vector<double> x0(base.positions().begin(), base.positions().end());

    struct Case { const char* name; ParticleGroup a, b; };
    for (const Case& c : {Case{"same group", kA, kA}, Case{"two groups", kA, kB}}) {
        INFO(c.name);
        auto j = summed(c.a, c.b, cell);
        auto log_at = [&](std::size_t idx, double shift) {
            WalkerSet ws(1, kNp, 2, cell);
            std::copy(x0.begin(), x0.end(), ws.positions().begin());
            ws.positions()[idx] += shift;
            ws.dist_tables().build_all(ws.positions());
            j->evaluate_batch(ws);
            return j->log_psi_c()[0];
        };
        WalkerSet ws(1, kNp, 2, cell);
        std::copy(x0.begin(), x0.end(), ws.positions().begin());
        ws.dist_tables().build_all(ws.positions());
        j->evaluate_batch(ws);
        const std::vector<double> g(j->grad_c().begin(), j->grad_c().end());
        const std::vector<double> l(j->lap_c().begin(), j->lap_c().end());
        const double l0 = j->log_psi_c()[0];

        const double h = 1e-4;
        for (std::size_t idx = 0; idx < kNp * 2; ++idx) {
            INFO("particle " << idx / 2 << " dim " << idx % 2);
            const double lp = log_at(idx, h), lm = log_at(idx, -h);
            CHECK(g[idx] == Approx((lp - lm) / (2 * h)).margin(1e-7));
            CHECK(l[idx] == Approx((lp - 2 * l0 + lm) / (h * h)).margin(1e-5));
        }
    }
}

TEST_CASE("image-summed Pade moves keep the caches equal to a fresh evaluation",
          "[image_sum][pade]") {
    // ratio/accept update ρ(k) and EVERY particle that sees it.  After many
    // accepted and rejected moves the running ln Ψ, ∇ and ∇² must still equal a
    // from-scratch evaluation of the final configuration.
    const SimulationCell cell = hexagonal(5.0);
    const std::size_t nw = 3;
    std::mt19937_64 rng(11);

    auto make = [&] {
        TrialWavefunction psi;
        psi.add_component(summed(kA, kA, cell));
        psi.add_component(summed(kA, kB, cell));
        return psi;
    };
    TrialWavefunction psi = make();
    WalkerSet ws(nw, kNp, 2, cell);
    scatter(ws, cell, rng);
    psi.evaluate_batch(ws);

    std::normal_distribution<double> step(0.0, 0.7);
    std::bernoulli_distribution accept(0.6);
    std::unique_ptr<bool[]> mask(new bool[nw]);
    for (int move = 0; move < 300; ++move) {
        const std::size_t p = static_cast<std::size_t>(move) % kNp;
        std::vector<double> np(2 * nw);
        for (std::size_t d = 0; d < 2; ++d)
            for (std::size_t w = 0; w < nw; ++w)
                np[d * nw + w] = ws.positions()[(p * 2 + d) * nw + w] + step(rng);
        ws.propose_move_batch(p, np);
        psi.ratio_batch(ws, p);
        for (std::size_t w = 0; w < nw; ++w) mask[w] = accept(rng);
        const std::span<const bool> m(mask.get(), nw);
        psi.apply_move_batch(ws, p, m);
        ws.apply_move(p, ws.new_pos(), ws.new_grad(), ws.new_lap(), m);
    }

    TrialWavefunction fresh = make();
    WalkerSet ref(nw, kNp, 2, cell);
    std::copy(ws.positions().begin(), ws.positions().end(), ref.positions().begin());
    ref.dist_tables().build_all(ref.positions());
    fresh.evaluate_batch(ref);

    for (std::size_t w = 0; w < nw; ++w)
        CHECK(ws.log_psi()[w] == Approx(ref.log_psi()[w]).margin(1e-9));
    for (std::size_t i = 0; i < ws.gradients().size(); ++i) {
        INFO("index " << i);
        CHECK(ws.gradients()[i] == Approx(ref.gradients()[i]).margin(1e-9));
        CHECK(ws.laplacians()[i] == Approx(ref.laplacians()[i]).margin(1e-9));
    }
}

TEST_CASE("image-summed Pade parameter derivatives match finite differences through rebuild",
          "[image_sum][pade][optimize]") {
    // Every Padé coefficient except the cusp is optimisable, so ∂ ln Ψ/∂p, and
    // the curvature terms ∂∇/∂p, ∂∇²/∂p, must be the derivatives of what is
    // actually evaluated — including the k-space weights, which depend on p
    // through the re-split in rebuild().
    //
    // ∂ ln Ψ/∂p is compared BETWEEN two configurations.  The split carries
    // configuration-independent constants (self-image ½N·u_long(0), the k = 0
    // offset) whose size depends on which of the many near-equivalent splits
    // the ill-conditioned fit lands on, so the analytic split of ∂u/∂p and the
    // finite difference of the split of u differ by a walker-independent
    // constant.  That constant is the normalisation gauge the linear method
    // projects out (it centres d_i), and it never touches ∇ or ∇².
    const SimulationCell cell = hexagonal(5.0);
    std::mt19937_64 rng(3);
    const std::size_t nw = 2;
    WalkerSet base(nw, kNp, 2, cell);
    scatter(base, cell, rng);
    const std::vector<double> x0(base.positions().begin(), base.positions().end());

    for (const bool same : {true, false}) {
        INFO((same ? "same group" : "two groups"));
        const ParticleGroup& gb = same ? kA : kB;
        auto j = summed(kA, gb, cell);
        WalkerSet ws(nw, kNp, 2, cell);
        std::copy(x0.begin(), x0.end(), ws.positions().begin());
        ws.dist_tables().build_all(ws.positions());
        j->evaluate_batch(ws);

        const std::size_t n_opt = j->num_optimizable_params();
        REQUIRE(n_opt == 3);
        const std::size_t stride = kNp * 2 * nw;
        std::vector<double> d(n_opt * nw), dg(n_opt * stride), dl(n_opt * stride);
        j->compute_parameter_derivatives(ws, d);
        REQUIRE(j->compute_parameter_gradient_derivatives(ws, dg, dl));

        // Finite differences: shift one parameter, rebuild (re-split), evaluate.
        auto at = [&](std::size_t q, double shift) {
            auto c = summed(kA, gb, cell);
            c->parameters()[q].value += shift;
            c->rebuild();
            WalkerSet w2(nw, kNp, 2, cell);
            std::copy(x0.begin(), x0.end(), w2.positions().begin());
            w2.dist_tables().build_all(w2.positions());
            c->evaluate_batch(w2);
            return std::make_tuple(
                std::vector<double>(c->log_psi_c().begin(), c->log_psi_c().end()),
                std::vector<double>(c->grad_c().begin(), c->grad_c().end()),
                std::vector<double>(c->lap_c().begin(), c->lap_c().end()));
        };
        const double h = 1e-4;
        std::size_t s = 0;
        for (std::size_t q = 0; q < j->parameters().size(); ++q) {
            if (j->parameters()[q].is_fixed) continue;
            INFO("parameter " << j->parameters()[q].name);
            const auto [lp, gp, lap_p] = at(q, h);
            const auto [lm, gm, lap_m] = at(q, -h);
            const double fd_contrast = ((lp[1] - lm[1]) - (lp[0] - lm[0])) / (2 * h);
            CHECK(d[s * nw + 1] - d[s * nw + 0] == Approx(fd_contrast).margin(1e-5));
            for (std::size_t i = 0; i < stride; ++i) {
                CHECK(dg[s * stride + i] == Approx((gp[i] - gm[i]) / (2 * h)).margin(1e-5));
                CHECK(dl[s * stride + i] == Approx((lap_p[i] - lap_m[i]) / (2 * h)).margin(1e-4));
            }
            ++s;
        }
    }
}

TEST_CASE("a periodic Drummond cutoff is bounded by the inradius",
          "[image_sum][drummond]") {
    // With L <= R_WS only the nearest image is ever in reach, so the minimum
    // image IS the exact image sum.  L stays optimisable: an optimizer step is
    // clamped at R_WS, and any other route past it (which would silently drop
    // the farther images) is refused by rebuild().
    std::vector<Parameter> p = {{"a0", 0.4, false}, {"a2", -0.02, false}, {"L", 2.9, false}};
    DrummondJastrow j(3, -0.5, kA, kB, p);
    j.bound_cutoff(3.0);
    CHECK_NOTHROW(j.rebuild());
    const Parameter& L = j.parameters().back();
    CHECK(L.upper_bound == 3.0);
    CHECK(j.parameters().front().upper_bound == std::numeric_limits<double>::infinity());
    CHECK(alea::stepped_value(L, 0.5) == 3.0);                      // clamped
    CHECK(alea::stepped_value(L, -0.4) == Catch::Approx(2.5));      // inward: untouched
    j.parameters().back().value = 3.0;
    CHECK_NOTHROW(j.rebuild());
    j.parameters().back().value = 3.2;
    CHECK_THROWS_WITH(j.rebuild(), Catch::Matchers::ContainsSubstring("Wigner"));
}

TEST_CASE("an optimizer step keeps a sign-locked parameter's sign",
          "[image_sum][optimizer]") {
    // p = s·exp(θ): the step Δθ multiplies by exp(Δθ) and never crosses zero.
    const Parameter locked{"b", -0.3, false, true};
    CHECK(alea::stepped_value(locked, std::log(2.0)) == Catch::Approx(-0.6));
    CHECK(alea::stepped_value(locked, -50.0) < 0.0);
    const Parameter free_p{"a", -0.3, false};
    CHECK(alea::stepped_value(free_p, 0.5) == Catch::Approx(0.2));
}

TEST_CASE("image-summed Pade setup cost", "[.timing]") {
    // Hidden ([.]) — run by name to see what the split costs.
    const SimulationCell cell = hexagonal(5.0);
    const auto t0 = std::chrono::steady_clock::now();
    auto j = summed(kA, kB, cell);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    WARN("one [2/2] Padé with 3 free parameters (4 breakups): " << s << " s");

    const double rc = cell.ws_radius();
    const std::vector<double> a{-0.6, 0.2}, b{0.8, 0.3};
    const auto f = alea::numerics::pade_radial(a, b, rc);
    auto t1 = std::chrono::steady_clock::now();
    double sink = 0.0;
    for (int i = 0; i < 100; ++i) sink += f.tail((1.0 + 2.99 * i) / rc);   // k·r_c 1..300
    const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    WARN("100 tail transforms over k r_c = 1..300: " << tt << " s (" << sink << ")");

    alea::numerics::BreakupOptions opt{};
    opt.k_continuum = 20.0 * 3.14159265358979323846 / rc;
    opt.k_max = 300.0 / rc;
    opt.n_continuum = 150;
    auto cheap = f;
    cheap.tail = [](double) { return 0.0; };
    t1 = std::chrono::steady_clock::now();
    (void)alea::numerics::build_long_range_breakup(cell, cheap, opt);
    const double tb = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    WARN("one breakup with a free tail (design matrix + QR only): " << tb << " s");
}

TEST_CASE("an image-summed Pade has a continuous gradient across the Wigner-Seitz boundary",
          "[image_sum][pade]") {
    const SimulationCell cell = square(6.0);   // boundary at x = ±3
    const double eps = 1e-7;

    // The minimum-image form jumps: the nearest image swaps sides at x = 3, so
    // ∂_x u flips sign while |∇u| = u'(r) ≠ 0.  (Kept as a check that this test
    // can see the defect at all.)
    {
        auto j = eh_pade(nullptr);
        const Eval left  = evaluate(*j, cell, 3.0 - eps, 0.4);
        const Eval right = evaluate(*j, cell, 3.0 + eps, 0.4);
        CHECK(std::abs(left.gx0 - right.gx0) > 1e-2);
    }

    auto j = eh_pade(&cell);
    const Eval left  = evaluate(*j, cell, 3.0 - eps, 0.4);
    const Eval right = evaluate(*j, cell, 3.0 + eps, 0.4);
    CHECK(right.log_psi == Approx(left.log_psi).margin(1e-6));
    CHECK(right.gx0 == Approx(left.gx0).margin(1e-5));
    CHECK(right.gy0 == Approx(left.gy0).margin(1e-5));
}
