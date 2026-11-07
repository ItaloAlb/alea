// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <vector>

#include "estimators/bond_order.h"
#include "numerics/statistics.h"

using namespace alea;
using Catch::Approx;

namespace {
const SimulationCell triangular(2, std::vector<double>{3., 0., 1.5, 1.5*std::sqrt(3.)});
const std::vector<ParticleGroup> groups{{"e", {}, {}, 0., 0, 9, 9}};
class Batch final : public ParticleBatch {
public:
    explicit Batch(const SimulationCell& cell = triangular, std::size_t walkers = 1)
        : ParticleBatch(walkers, 9, 2, cell) {}
    void set(double distortion = 0., double shift = 0., bool reverse = false, std::size_t w = 0) {
        for (std::size_t p = 0; p < 9; ++p) {
            const auto index = reverse ? 8-p : p;
            const auto i = index/3, j = index%3;
            const double x = static_cast<double>(i)+.5*static_cast<double>(j);
            const double y = .5*std::sqrt(3.)*static_cast<double>(j);
            positions()[(p*2)*num_walkers()+w] = x+distortion*std::sin(2.*x+y)+shift;
            positions()[(p*2+1)*num_walkers()+w] = y+distortion*std::cos(x-2.*y)-shift;
        }
    }
};
const std::vector<double>& series(const RunResult& r, const std::string& key) {
    const auto it = std::find_if(r.series.begin(), r.series.end(), [&](const auto& s) { return s.first == key; });
    REQUIRE(it != r.series.end());
    return it->second;
}
void measure(Estimator& e, const Batch& batch, std::span<const double> weights) {
    e.start_block(); e.accumulate(batch, weights); e.stop_block();
}
// Independent broad-image oracle uses angles/trigonometry and all pair
// distances sorted, rather than production's bounded order statistics/z^6.
std::array<double, 3> oracle(const Batch& batch, std::size_t w = 0) {
    std::complex<double> global{};
    double local = 0., distortion = 0.;
    const auto cell = batch.cell().vectors();
    for (std::size_t p = 0; p < 9; ++p) {
        std::vector<std::pair<double, double>> neighbors;
        for (std::size_t j = 0; j < 9; ++j) {
            if (j == p) continue;
            double best = 1e100, angle = 0.;
            for (int a = -3; a <= 3; ++a) for (int b = -3; b <= 3; ++b) {
                const double dx = batch.positions()[j*2*batch.num_walkers()+w]-batch.positions()[p*2*batch.num_walkers()+w]+a*cell[0]+b*cell[2];
                const double dy = batch.positions()[(j*2+1)*batch.num_walkers()+w]-batch.positions()[(p*2+1)*batch.num_walkers()+w]+a*cell[1]+b*cell[3];
                const double radius = std::hypot(dx, dy);
                if (radius < best) { best = radius; angle = std::atan2(dy, dx); }
            }
            neighbors.emplace_back(best, angle);
        }
        std::sort(neighbors.begin(), neighbors.end());
        std::complex<double> psi{};
        for (std::size_t j = 0; j < 6; ++j) {
            psi += std::polar(1./6., 6*neighbors[j].second);
            distortion += std::pow(neighbors[j].first-1., 2)/54.;
        }
        global += psi/9.; local += std::norm(psi)/9.;
    }
    return {std::norm(global), local, distortion};
}
}  // namespace

TEST_CASE("triangular bond order is exact under translation relabeling and periodic images", "[bond_order]") {
    Batch batch;
    BondOrder e(groups, "e", triangular, 1.); e.configure_reblocking(64);
    for (std::size_t block = 0; block < 64; ++block) {
        batch.set(0., .011*static_cast<double>(block), block%2 == 0);
        for (std::size_t p = 0; p < 9; ++p) {
            batch.positions()[p*2] += triangular.vectors()[0]*static_cast<double>(p%3);
            batch.positions()[p*2+1] += triangular.vectors()[1]*static_cast<double>(p%3);
        }
        measure(e, batch, std::array<double, 1>{1.});
    }
    RunResult out; e.finalize(out);
    CHECK(series(out, "bond_e_global6")[0] == Approx(1.).margin(2e-14));
    CHECK(series(out, "bond_e_local6")[0] == Approx(1.).margin(2e-14));
    CHECK(series(out, "bond_e_distortion")[0] == Approx(0.).margin(1e-27));
    CHECK(series(out, "bond_e_global6_blocks").size() == 64);
}

TEST_CASE("bond order agrees with broad image angular oracle for deformed lattices and clone reduction", "[bond_order]") {
    BondOrder e(groups, "e", triangular, 1.); e.configure_reblocking(128);
    auto a = e.clone(), b = e.clone();
    Batch batch;
    std::array<std::vector<double>, 3> expected;
    for (std::size_t block = 0; block < 128; ++block) {
        const double amplitude = .01*static_cast<double>(1+(block/4)%8);
        batch.set(amplitude); auto va = oracle(batch);
        measure(*a, batch, std::array<double, 1>{1.});
        batch.set(amplitude*.7, .031); auto vb = oracle(batch);
        measure(*b, batch, std::array<double, 1>{1.});
        for (std::size_t i = 0; i < 3; ++i) expected[i].push_back(.5*(va[i]+vb[i]));
    }
    e.collect(*a); e.collect(*b);
    RunResult out; e.finalize(out);
    const std::array<std::string, 3> names{"global6", "local6", "distortion"};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto stats = numerics::reblock(expected[i]);
        CHECK(series(out, "bond_e_"+names[i])[0] == Approx(stats.mean).margin(2e-14));
        CHECK(series(out, "bond_e_"+names[i]+"_error")[0] == Approx(stats.error).margin(2e-14));
        const auto& history = series(out, "bond_e_"+names[i]+"_blocks");
        REQUIRE(history.size() == expected[i].size());
        for (std::size_t block = 0; block < history.size(); ++block)
            CHECK(history[block] == Approx(expected[i][block]).margin(2e-14));
    }
}

TEST_CASE("neighbor cutoff ties share weight and preserve label invariance", "[bond_order]") {
    const SimulationCell square(2, std::vector<double>{3., 0., 0., 3.});
    Batch batch(square);
    BondOrder e(groups, "e", square, 1.); e.configure_reblocking(2);
    for (std::size_t block = 0; block < 2; ++block) {
        for (std::size_t p = 0; p < 9; ++p) {
            const auto j = block == 0 ? p : 8-p;
            batch.positions()[p*2] = static_cast<double>(j/3);
            batch.positions()[p*2+1] = static_cast<double>(j%3);
        }
        measure(e, batch, std::array<double, 1>{1.});
    }
    RunResult out; e.finalize(out);
    CHECK(series(out, "bond_e_global6")[0] == Approx(0.).margin(1e-27));
    CHECK(series(out, "bond_e_local6")[0] == Approx(0.).margin(1e-27));
    CHECK(series(out, "bond_e_distortion")[0] == Approx(std::pow(std::sqrt(2.)-1., 2)/3.));
}

TEST_CASE("signed bond order keeps paired numerators denominators and skipped worm measurements", "[bond_order]") {
    BondOrder e(groups, "e", triangular, 1.); e.configure_signed_weights(); e.configure_reblocking(128);
    Batch batch(triangular, 2);
    std::array<std::vector<double>, 3> numerators;
    std::vector<double> denominators;
    for (std::size_t block = 0; block < 128; ++block) {
        batch.set(.015*static_cast<double>(1+(block/4)%6), 0., false, 0);
        batch.set(.02*static_cast<double>(1+(block/3)%4), 0., true, 1);
        const auto a = oracle(batch, 0), b = oracle(batch, 1);
        const double sign = block%5 == 0 ? -1. : 1.;
        e.start_block(); e.accumulate(batch, std::array<double, 2>{sign, 1.});
        e.skip_measurement(2); e.stop_block();
        for (std::size_t i = 0; i < 3; ++i) numerators[i].push_back((sign*a[i]+b[i])/4.);
        denominators.push_back((sign+1.)/4.);
    }
    RunResult out; e.finalize(out);
    const std::array<std::string, 3> names{"global6", "local6", "distortion"};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto stats = numerics::ratio_of_means(numerators[i], denominators);
        CHECK(series(out, "bond_e_"+names[i])[0] == Approx(stats.mean).margin(2e-14));
        CHECK(series(out, "bond_e_"+names[i]+"_error")[0] == Approx(stats.error).margin(2e-14));
        CHECK(series(out, "bond_e_"+names[i]+"_denominator_blocks") == denominators);
    }
}

TEST_CASE("bond order refuses undefined geometry groups spacing collisions and history overflow", "[bond_order]") {
    CHECK_THROWS(BondOrder(groups, "e", SimulationCell{}, 1.));
    CHECK_THROWS(BondOrder(groups, "missing", triangular, 1.));
    CHECK_THROWS(BondOrder(std::vector<ParticleGroup>{{"e", {}, {}, 0., 0, 6, 6}}, "e", triangular, 1.));
    CHECK_THROWS(BondOrder(groups, "e", triangular, 0.));
    CHECK_THROWS(BondOrder(groups, "e", triangular, 1e-300));
    CHECK_THROWS(BondOrder(groups, "e", triangular, std::numeric_limits<double>::quiet_NaN()));
    BondOrder e(groups, "e", triangular, 1.);
    CHECK_THROWS(e.skip_measurement(1));
    CHECK_THROWS(e.configure_reblocking(EstimatorReblocking::kMaxHistoryBytes/(4*sizeof(double))+1));
    e.configure_reblocking(1);
    Batch batch; batch.set(); batch.positions()[2] = batch.positions()[0]; batch.positions()[3] = batch.positions()[1];
    CHECK_THROWS(measure(e, batch, std::array<double, 1>{1.}));
}

// ── VMC/DMC path: no PIMC reblocking, independent-block moments ─────────────

TEST_CASE("without PIMC reblocking the perfect triangular lattice has unit sixfold order",
          "[bond_order][vmc_dmc]") {
    Batch batch;
    BondOrder e(groups, "e", triangular, 1.);
    for (std::size_t block = 0; block < 8; ++block) {
        batch.set(0., .019 * static_cast<double>(block), block % 2 == 1);
        measure(e, batch, std::array<double, 1>{.5 + static_cast<double>(block)});
    }
    RunResult out; e.finalize(out);
    CHECK(series(out, "bond_e_global6")[0] == Approx(1.).margin(2e-14));
    CHECK(series(out, "bond_e_local6")[0] == Approx(1.).margin(2e-14));
    CHECK(series(out, "bond_e_distortion")[0] == Approx(0.).margin(1e-27));
    CHECK(series(out, "bond_e_global6_error")[0] == Approx(0.).margin(2e-14));
    CHECK(std::none_of(out.series.begin(), out.series.end(),
                       [](const auto& s) { return s.first == "bond_e_global6_blocks"; }));
}

TEST_CASE("VMC/DMC bond-order blocks are walker-weighted means and clones merge by Chan's rule",
          "[bond_order][vmc_dmc]") {
    BondOrder prototype(groups, "e", triangular, 1.);
    auto a = prototype.clone(), b = prototype.clone();
    Batch batch(triangular, 2);
    std::array<std::vector<double>, 3> blocks;
    for (std::size_t block = 0; block < 8; ++block) {
        batch.set(.012 * static_cast<double>(1 + block % 5), 0., false, 0);
        batch.set(.02 * static_cast<double>(1 + block % 3), .05, true, 1);
        const auto va = oracle(batch, 0), vb = oracle(batch, 1);
        const std::array<double, 2> weights{.4 + .1 * static_cast<double>(block), 1.};
        measure(block < 4 ? *a : *b, batch, weights);
        for (std::size_t i = 0; i < 3; ++i)
            blocks[i].push_back((weights[0] * va[i] + weights[1] * vb[i]) / (weights[0] + weights[1]));
    }
    prototype.collect(*a); prototype.collect(*b);
    RunResult out; prototype.finalize(out);
    const std::array<std::string, 3> names{"global6", "local6", "distortion"};
    for (std::size_t i = 0; i < 3; ++i) {
        INFO(names[i]);
        double mean = 0.;
        for (double v : blocks[i]) mean += v / 8.;
        double m2 = 0.;
        for (double v : blocks[i]) m2 += (v - mean) * (v - mean);
        CHECK(series(out, "bond_e_" + names[i])[0] == Approx(mean).margin(2e-14));
        CHECK(series(out, "bond_e_" + names[i] + "_error")[0] ==
              Approx(std::sqrt(m2 / (7. * 8.))).margin(2e-14));
    }
}

TEST_CASE("VMC/DMC bond order refuses an unweighted block and mixing the two statistics paths",
          "[bond_order][vmc_dmc]") {
    BondOrder e(groups, "e", triangular, 1.);
    Batch batch; batch.set();
    CHECK_THROWS(measure(e, batch, std::array<double, 1>{0.}));
    CHECK_THROWS(measure(e, batch, std::array<double, 1>{-1.}));
    BondOrder pimc(groups, "e", triangular, 1.);
    pimc.configure_reblocking(4);
    BondOrder vmc(groups, "e", triangular, 1.);
    CHECK_THROWS(pimc.collect(vmc));
}
