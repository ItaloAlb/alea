// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <initializer_list>
#include <span>
#include <vector>

#include "numerics/long_range_breakup.h"
#include "numerics/special_functions.h"
#include "population/simulation_cell.h"

using alea::SimulationCell;
using alea::numerics::BreakupOptions;
using alea::numerics::build_long_range_breakup;
using alea::numerics::LongRangeBreakup;
using alea::numerics::RadialFunction;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kPi = 3.14159265358979323846;

// 2-D square Madelung constant for the NaCl arrangement, from an independent
// erfc Ewald sum: identical to 12 digits for every alpha in [3, 8].  This is
// the external anchor — everything else in this file is self-consistency.
constexpr double kMadelung2D = 1.615542626713;

SimulationCell square_cell(double side) {
    const std::array<double, 4> v{side, 0.0, 0.0, side};
    return SimulationCell(2, std::span<const double>(v));
}

RadialFunction coulomb(double eps) {
    return {[eps](double r) { return 1.0 / (eps * r); },
            [eps](double k) { return 2.0 * kPi / (eps * k); },
            [eps](double r) { return -1.0 / (eps * r * r); },
            [eps](double r) { return 2.0 / (eps * r * r * r); }};
}

// Energy of the four-ion NaCl cell (side 2, lattice constant 1) through the
// breakup: minimum-image short range, retained k in reciprocal space, and the
// self-energy constant.
double madelung(const LongRangeBreakup& br, const RadialFunction& f,
                double side) {
    const std::array<double, 4> px{0, 1, 0, 1}, py{0, 0, 1, 1};
    const std::array<double, 4> q{1, -1, -1, 1};

    double e_real = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            double best = 1e300;
            for (int mi = -2; mi <= 2; ++mi) {
                for (int mj = -2; mj <= 2; ++mj) {
                    best = std::min(best, std::hypot(px[i] - px[j] + mi * side,
                                                     py[i] - py[j] + mj * side));
                }
            }
            if (best < br.r_cut()) {
                e_real += q[i] * q[j] * (f.value(best) - br.long_range_real(best));
            }
        }
    }
    double e_recip = 0.0;
    for (std::size_t n = 0; n < br.kx().size(); ++n) {
        double rr = 0.0, ri = 0.0;
        for (int p = 0; p < 4; ++p) {
            const double ph = br.kx()[n] * px[p] + br.ky()[n] * py[p];
            rr += q[p] * std::cos(ph);
            ri += q[p] * std::sin(ph);
        }
        e_recip += br.weight()[n] * (rr * rr + ri * ri);
    }
    double q2 = 0.0;
    for (double x : q) {
        q2 += x * x;
    }
    const double e = e_real + e_recip - 0.5 * q2 * br.long_range_zero();
    return -2.0 * e / 4.0;
}

}  // namespace

TEST_CASE("Breakup reproduces the 2-D Madelung constant", "[breakup]") {
    const auto cell = square_cell(2.0);
    const auto f = coulomb(1.0);
    BreakupOptions opt;
    opt.n_knots = 20;
    const auto br = build_long_range_breakup(cell, f, opt);
    const double m = madelung(br, f, 2.0);
    INFO("M = " << m << ", error = " << std::abs(m - kMadelung2D)
                << ", cond(C) = " << br.condition()
                << ", residual = " << br.residual_above_cut());
    CHECK_THAT(m, WithinAbs(kMadelung2D, 1e-8));
}

TEST_CASE("Breakup error falls as knots are added", "[breakup]") {
    // With the normal equations this was non-monotone and capped at ~1e-7:
    // chi kept improving while the answer did not.  Under QR the two move
    // together, which is the signature of a fit that is not noise-limited.
    const auto cell = square_cell(2.0);
    const auto f = coulomb(1.0);
    std::vector<double> err;
    for (int nk : {8, 14, 20}) {
        BreakupOptions opt;
        opt.n_knots = nk;
        const auto br = build_long_range_breakup(cell, f, opt);
        err.push_back(std::abs(madelung(br, f, 2.0) - kMadelung2D));
    }
    INFO("errors: " << err[0] << ", " << err[1] << ", " << err[2]);
    CHECK(err[1] < err[0]);
    CHECK(err[2] < err[1]);
}

TEST_CASE("Breakup accuracy does not depend on the reciprocal cutoff",
          "[breakup]") {
    // The point of the method: accuracy is set by the fit, cost by k_cut.
    // Doubling the retained shells must not move the answer.
    const auto cell = square_cell(2.0);
    const auto f = coulomb(1.0);
    const double b1 = kPi;   // 2π/side
    double first = 0.0;
    for (int i = 0; i < 3; ++i) {
        BreakupOptions opt;
        opt.n_knots = 20;
        opt.k_cut = (6.0 + 3.0 * i) * b1;
        const auto br = build_long_range_breakup(cell, f, opt);
        const double m = madelung(br, f, 2.0);
        INFO("k_cut = " << opt.k_cut / b1 << " b1, M = " << m);
        CHECK_THAT(m, WithinAbs(kMadelung2D, 1e-7));
        if (i == 0) {
            first = m;
        } else {
            CHECK_THAT(m, WithinAbs(first, 1e-7));
        }
    }
}

TEST_CASE("Breakup is well conditioned under QR", "[breakup]") {
    // cond(C) ~ 1e5-1e6 here.  Forming the normal equations would square that
    // to ~1e12 and throw away half the digits before the solve begins; this
    // check documents the scale that makes QR non-optional.
    const auto cell = square_cell(2.0);
    const auto br = build_long_range_breakup(cell, coulomb(1.0), {});
    INFO("cond(C) = " << br.condition());
    CHECK(br.condition() > 1e3);
    CHECK(br.condition() < 1e9);
    CHECK(br.max_coefficient() < 1e3);
}

TEST_CASE("Breakup of Rytova-Keldysh tends to Coulomb as r0 shrinks",
          "[breakup][keldysh]") {
    // Rytova-Keldysh has no published Madelung constant, but it becomes
    // Coulomb as r0 -> 0, which borrows the external anchor.  The approach is
    // QUADRATIC, not linear: the kernel's own asymptotics give
    // K(x) ~ (2/π)(1/x − 1/x³), hence v_RK − v_C = −r0²/(eps r³).
    const auto cell = square_cell(2.0);
    std::vector<double> d;
    for (double r0 : {0.08, 0.04, 0.02}) {
        const double pre = kPi / (2.0 * r0);
        RadialFunction f{
            [pre, r0](double r) {
                return pre * alea::numerics::keldysh_kernel(r / r0);
            },
            [r0](double k) { return 2.0 * kPi / (k * (1.0 + r0 * k)); },
            [pre, r0](double r) {
                const double h = 1e-6;
                return pre * (alea::numerics::keldysh_kernel((r + h) / r0) -
                              alea::numerics::keldysh_kernel((r - h) / r0)) /
                       (2 * h);
            },
            [pre, r0](double r) {
                const double h = 1e-4;
                return pre * (alea::numerics::keldysh_kernel((r + h) / r0) -
                              2 * alea::numerics::keldysh_kernel(r / r0) +
                              alea::numerics::keldysh_kernel((r - h) / r0)) /
                       (h * h);
            }};
        BreakupOptions opt;
        opt.n_knots = 20;
        const auto br = build_long_range_breakup(cell, f, opt);
        d.push_back(std::abs(madelung(br, f, 2.0) - kMadelung2D));
    }
    INFO("|M_RK - M_C| = " << d[0] << ", " << d[1] << ", " << d[2]);
    // Halving r0 must shrink the gap by clearly more than 2x (quadratic would
    // be 4x); a loose 3x keeps this from being brittle on higher-order terms.
    CHECK(d[0] / d[1] > 3.0);
    CHECK(d[1] / d[2] > 3.0);
}

TEST_CASE("Breakup reproduces an off-lattice configuration", "[breakup]") {
    // The Madelung test above is BLIND to the real-space half: in the NaCl cell
    // every pair sits at distance 1.0 or sqrt(2), and r_c is the inradius 1.0,
    // so f_short is identically zero and only the reciprocal sum is exercised.
    // This configuration puts pairs INSIDE r_c, and checks against an
    // independent erfc Ewald sum whose own alpha-independence is 8.9e-12.
    const double side = 2.0;
    const auto cell = square_cell(side);
    const auto f = coulomb(1.0);
    const std::array<double, 4> px{0.31, 1.27, 0.62, 1.55};
    const std::array<double, 4> py{0.44, 0.18, 1.36, 1.09};
    const std::array<double, 4> q{1, -1, 1, -1};
    const auto wrap = [side](double d) {
        return d - side * std::nearbyint(d / side);
    };

    const auto ewald_erfc = [&](double a, int kmax) {
        double e = 0.0;
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                const double r = std::hypot(wrap(px[i] - px[j]),
                                            wrap(py[i] - py[j]));
                e += q[i] * q[j] * std::erfc(a * r) / r;
            }
        }
        const double b1 = 2.0 * kPi / side;
        for (int i = -kmax; i <= kmax; ++i) {
            for (int j = -kmax; j <= kmax; ++j) {
                if (i == 0 && j == 0) {
                    continue;
                }
                const double kx = b1 * i, ky = b1 * j;
                const double k = std::hypot(kx, ky);
                double rr = 0.0, ri = 0.0;
                for (int p = 0; p < 4; ++p) {
                    const double ph = kx * px[p] + ky * py[p];
                    rr += q[p] * std::cos(ph);
                    ri += q[p] * std::sin(ph);
                }
                e += 0.5 * (2.0 * kPi / k) * std::erfc(k / (2.0 * a)) *
                     (rr * rr + ri * ri) / (side * side);
            }
        }
        double q2 = 0.0;
        for (double x : q) {
            q2 += x * x;
        }
        return e - a * q2 / std::sqrt(kPi);
    };
    const double reference = ewald_erfc(6.0, 80);
    REQUIRE(std::abs(ewald_erfc(4.5, 80) - reference) < 1e-10);

    BreakupOptions opt;
    opt.n_knots = 20;
    const auto br = build_long_range_breakup(cell, f, opt);

    double e_real = 0.0;
    int pairs_inside = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const double r = std::hypot(wrap(px[i] - px[j]), wrap(py[i] - py[j]));
            if (r < br.r_cut()) {
                ++pairs_inside;
                e_real += q[i] * q[j] * (f.value(r) - br.spline().eval(r));
            }
        }
    }
    // Guard the guard: if this ever drops to zero the test has silently become
    // the Madelung one again.
    REQUIRE(pairs_inside > 0);

    double e_recip = 0.0;
    for (std::size_t n = 0; n < br.kx().size(); ++n) {
        double rr = 0.0, ri = 0.0;
        for (int p = 0; p < 4; ++p) {
            const double ph = br.kx()[n] * px[p] + br.ky()[n] * py[p];
            rr += q[p] * std::cos(ph);
            ri += q[p] * std::sin(ph);
        }
        e_recip += br.weight()[n] * (rr * rr + ri * ri);
    }
    double q2 = 0.0;
    for (double x : q) {
        q2 += x * x;
    }
    const double e = e_real + e_recip - 0.5 * q2 * br.long_range_zero();
    INFO("pairs inside r_c = " << pairs_inside << ", E = " << e
                               << ", reference = " << reference);
    CHECK_THAT(e, WithinAbs(reference, 1e-8));
}

TEST_CASE("Breakup spline matches the generic basis evaluation", "[breakup]") {
    // The hot-loop evaluator exploits that only six basis functions overlap any
    // panel.  If it drifted from the generic sum, the real-space half would
    // stop being the exact complement of the reciprocal half — the one error
    // that no amount of k-space resolution can remove.
    const auto cell = square_cell(2.0);
    const auto br = build_long_range_breakup(cell, coulomb(1.0), {});
    double worst = 0.0;
    for (int i = 0; i < 20000; ++i) {
        const double r = br.r_cut() * (i + 0.5) / 20000;
        worst = std::max(worst,
                         std::abs(br.spline().eval(r) - br.long_range_real(r)));
    }
    INFO("max |spline - basis| = " << worst);
    CHECK(worst < 1e-13);
    // Compact support: f_short must vanish at and beyond r_c.
    CHECK(br.spline().eval(br.r_cut()) == 0.0);
    CHECK(br.spline().eval(2.0 * br.r_cut()) == 0.0);
}

TEST_CASE("Breakup spline derivatives are the analytic f_long' and f_long''",
          "[breakup]") {
    // A Jastrow split by the breakup needs ∇ and ∇² of its short-range half,
    // u_short = u − u_long, so the spline has to supply u_long′ and u_long″.
    const auto cell = square_cell(2.0);
    const RadialFunction f = coulomb(1.0);
    const auto br = build_long_range_breakup(cell, f, {});
    const auto& sp = br.spline();

    double worst1 = 0.0, worst2 = 0.0;
    const double h = 1e-4;
    for (int i = 1; i < 2000; ++i) {
        const double r = br.r_cut() * (i + 0.37) / 2001;
        if (r - h <= 0.0 || r + h >= br.r_cut()) continue;
        double v = 0, d1 = 0, d2 = 0;
        sp.eval(r, v, d1, d2);
        CHECK(v == sp.eval(r));
        const double fd1 = (sp.eval(r + h) - sp.eval(r - h)) / (2 * h);
        const double fd2 = (sp.eval(r + h) - 2 * sp.eval(r) + sp.eval(r - h)) / (h * h);
        worst1 = std::max(worst1, std::abs(d1 - fd1) / (1 + std::abs(d1)));
        worst2 = std::max(worst2, std::abs(d2 - fd2) / (1 + std::abs(d2)));
    }
    INFO("worst relative error: f' " << worst1 << ", f'' " << worst2);
    CHECK(worst1 < 1e-7);
    CHECK(worst2 < 1e-4);   // central second difference carries eps/h² ~ 1e-8·f

    // At r_c the long-range half takes over f's own slope and curvature, so
    // f_short = f − f_long leaves r_c with zero value, slope AND curvature: the
    // Jastrow built on it has a continuous gradient and Laplacian there.
    double v = 0, d1 = 0, d2 = 0;
    const double rc = br.r_cut() * (1.0 - 1e-12);
    sp.eval(rc, v, d1, d2);
    CHECK_THAT(v,  WithinAbs(f.value(br.r_cut()), 1e-9));
    CHECK_THAT(d1, WithinAbs(f.d1(br.r_cut()), 1e-9));
    CHECK_THAT(d2, WithinAbs(f.d2(br.r_cut()), 1e-8));

    // Beyond r_c the evaluator returns zero, like the value-only form.
    sp.eval(1.5 * br.r_cut(), v, d1, d2);
    CHECK(v == 0.0);
    CHECK(d1 == 0.0);
    CHECK(d2 == 0.0);
}

TEST_CASE("Breakup exposes the background coupling for a charged cell",
          "[breakup][background]") {
    // A cell with net charge has no finite energy under PBC: a shell of image
    // cells at distance r holds ~2*pi*r/A of them, each contributing ~Q/r, so
    // every shell adds the same amount and the sum grows without bound.  A
    // uniform background of density -Q/A removes the monopole and makes the
    // quantity exist.  All of its effect is one constant, -(1/2)Q^2*sr_k0, with
    //     sr_k0 = (1/A) int f_short d2r.
    const auto cell = square_cell(2.0);
    const auto f = coulomb(1.0);
    BreakupOptions opt;
    opt.n_knots = 20;
    const auto br = build_long_range_breakup(cell, f, opt);

    // Independent evaluation of the same integral, straight from the spline,
    // on a mesh that shares nothing with the basis-transform route used inside.
    const int n = 200000;
    double integral = 0.0;
    for (int i = 0; i < n; ++i) {
        const double r = br.r_cut() * (i + 0.5) / n;
        integral += 2.0 * kPi * r * (f.value(r) - br.spline().eval(r));
    }
    integral *= br.r_cut() / n;
    const double sr_k0_direct = integral / (2.0 * 2.0);   // cell area
    INFO("sr_k0 = " << br.short_range_k0() << ", direct = " << sr_k0_direct);
    CHECK_THAT(br.short_range_k0(), WithinAbs(sr_k0_direct, 1e-7));

    // It must be a real number of ordinary size, not something that had to be
    // regularised: f_short is bounded away from the origin and compact.
    CHECK(std::isfinite(br.short_range_k0()));
}

TEST_CASE("Background reproduces the 2-D Wigner crystal Madelung constants",
          "[breakup][background][physics]") {
    // External anchor for the neutralising background, and the only one
    // available: a lattice of like charges has no finite energy WITHOUT it, so
    // this number cannot be checked against a background-free calculation.
    //
    // The literature quotes these per particle in units of e^2/r_s, with r_s
    // the 2-D Wigner-Seitz radius (pi r_s^2 n = 1): -1.106103 for the
    // triangular lattice, -1.100244 for the square (Bonsall & Maradudin,
    // Phys. Rev. B 15, 1959 (1977)).  The values checked below are in THIS
    // file's units -- one particle per unit area, e^2 = 1 -- so they differ by
    // a factor r_s = 1/sqrt(pi) = 0.5641896:
    //
    //     -1.960516 * 0.5641896 = -1.106103   (triangular)
    //     -1.950132 * 0.5641896 = -1.100244   (square)
    //
    // The triangular lattice being the lower of the two is why it, and not the
    // square, is the stable 2-D Wigner crystal -- a check on the ORDERING that
    // needs no unit convention at all, and the ratio 1.0053244 likewise.
    //
    // The same numbers must come out for N = 1 and N = 4 at equal density:
    // those are the same crystal grouped into different cells, and E/N ceasing
    // to depend on the cell is exactly what the background restores.  Without
    // it the value drifts with cell size (-1.664, -1.379, -1.093 for N = 1, 4,
    // 9) because each shell of image cells adds the same monopole contribution
    // however far away it sits.
    const auto f = coulomb(1.0);
    std::vector<double> ratios;
    struct Case { const char* name; bool triangular; double published; };
    for (const Case& c : {Case{"square", false, -1.950132},
                          Case{"triangular", true, -1.960516}}) {
        std::vector<double> per_particle;
        for (int m : {1, 2}) {
            const double d = c.triangular ? std::sqrt(2.0 / std::sqrt(3.0)) : 1.0;
            const double L = m * d;
            const std::array<double, 4> v =
                c.triangular
                    ? std::array<double, 4>{L, 0.0, L * 0.5, L * std::sqrt(3.0) / 2.0}
                    : std::array<double, 4>{L, 0.0, 0.0, L};
            const SimulationCell cell(2, std::span<const double>(v));
            BreakupOptions opt;
            opt.n_knots = 20;
            const auto br = build_long_range_breakup(cell, f, opt);

            const int n = m * m;
            std::vector<double> px, py;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    const double fi = double(i) / m, fj = double(j) / m;
                    px.push_back(fi * v[0] + fj * v[2]);
                    py.push_back(fi * v[1] + fj * v[3]);
                }
            }
            double e_real = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    double best = 1e300;
                    for (int a = -2; a <= 2; ++a) {
                        for (int b = -2; b <= 2; ++b) {
                            best = std::min(best,
                                std::hypot(px[i] - px[j] + a * v[0] + b * v[2],
                                           py[i] - py[j] + a * v[1] + b * v[3]));
                        }
                    }
                    if (best < br.r_cut()) {
                        e_real += f.value(best) - br.spline().eval(best);
                    }
                }
            }
            double e_recip = 0.0;
            for (std::size_t q = 0; q < br.kx().size(); ++q) {
                double rr = 0.0, ri = 0.0;
                for (int p = 0; p < n; ++p) {
                    const double ph = br.kx()[q] * px[p] + br.ky()[q] * py[p];
                    rr += std::cos(ph);
                    ri += std::sin(ph);
                }
                e_recip += br.weight()[q] * (rr * rr + ri * ri);
            }
            const double e_self = -0.5 * n * br.long_range_zero();
            const double e_bg = -0.5 * double(n) * double(n) * br.short_range_k0();
            per_particle.push_back((e_real + e_recip + e_self + e_bg) / n);
        }
        INFO(c.name << ": N=1 " << per_particle[0] << ", N=4 " << per_particle[1]
                    << ", reference " << c.published << " (= "
                    << c.published / std::sqrt(kPi) << " in e^2/r_s)");
        CHECK_THAT(per_particle[0], WithinAbs(c.published, 1e-6));
        // Size consistency: the same crystal, grouped differently.
        CHECK_THAT(per_particle[1], WithinAbs(per_particle[0], 1e-8));
        ratios.push_back(per_particle[0]);
    }

    // Unit-free statement of the same physics: the ratio carries no
    // normalisation convention, so it cannot be right by accident of units.
    const double ratio = ratios[1] / ratios[0];
    INFO("triangular / square = " << ratio);
    CHECK_THAT(ratio, WithinAbs(1.0053244, 1e-6));
}
