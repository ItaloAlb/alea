// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

#include "estimators/bragg_order.h"
#include "numerics/statistics.h"

using namespace alea;
using Catch::Approx;

namespace {
const SimulationCell test_cell(2, std::vector<double>{4., 0., 0., 4.});
const std::vector<ParticleGroup> groups{{"e", {}, {}, 0., 0, 4, 4}};
const double k = std::numbers::pi;
const std::vector<double> star{k, 0., 0., k};

class Batch : public ParticleBatch {
public:
    Batch() : ParticleBatch(1, 4, 2, test_cell) {}
    void set(double shift, double distortion = 0.) {
        const std::vector<double> aos{.5, .5, 2.5, .5, .5, 2.5, 2.5 + distortion, 2.5};
        for (std::size_t p = 0; p < 4; ++p)
            for (std::size_t d = 0; d < 2; ++d)
                positions()[p * 2 + d] = aos[p * 2 + d] + shift;
    }
};

const std::vector<double>& series(const RunResult& r, const std::string& key) {
    const auto it = std::find_if(r.series.begin(), r.series.end(),
        [&](const auto& s) { return s.first == key; });
    REQUIRE(it != r.series.end());
    return it->second;
}

double direct(const Batch& batch, double x, double y) {
    std::complex<double> rho{};
    for (std::size_t p = 0; p < 4; ++p)
        rho += std::exp(std::complex<double>(0., x * batch.positions()[p * 2] +
                                                 y * batch.positions()[p * 2 + 1]));
    return std::norm(rho) / 16.;
}

void measure(Estimator& e, Batch& batch, double weight = 1.) {
    e.start_block();
    const std::vector<double> weights{weight};
    e.accumulate(batch, weights);
    e.stop_block();
}
}  // namespace

TEST_CASE("selected Bragg intensity survives global translation and excludes other directions",
          "[bragg]") {
    Batch batch;
    const std::vector<double> directions{k, 0., 0., k, k / 2, 0.};
    BraggOrder e(groups, "e", test_cell, directions);
    e.configure_reblocking(64);
    for (std::size_t block = 0; block < 64; ++block) {
        batch.set(static_cast<double>(block) * .173);
        measure(e, batch);
    }
    RunResult out; e.finalize(out);
    CHECK(series(out, "bragg_e")[0] == Approx(2. / 3.).margin(1e-14));
    CHECK(series(out, "bragg_e_directions")[0] == Approx(1.));
    CHECK(series(out, "bragg_e_directions")[1] == Approx(1.));
    CHECK(series(out, "bragg_e_directions")[2] == Approx(0.).margin(1e-28));
    REQUIRE(series(out, "bragg_e_blocks").size() == 64);
}

TEST_CASE("Bragg scalar error retains directional covariance and synchronized replica histories",
          "[bragg]") {
    BraggOrder prototype(groups, "e", test_cell, star);
    prototype.configure_reblocking(256);
    auto a = prototype.clone(), b = prototype.clone();
    Batch batch;
    std::vector<double> expected;
    for (std::size_t block = 0; block < 256; ++block) {
        batch.set(.23 * static_cast<double>(block), .2 * static_cast<double>((block / 8) % 5));
        const double va = .5 * (direct(batch, k, 0.) + direct(batch, 0., k));
        measure(*a, batch);
        batch.set(-.17 * static_cast<double>(block), .3 * static_cast<double>((block / 8 + 1) % 5));
        const double vb = .5 * (direct(batch, k, 0.) + direct(batch, 0., k));
        measure(*b, batch);
        expected.push_back(.5 * (va + vb));
    }
    prototype.collect(*a); prototype.collect(*b);
    RunResult out; prototype.finalize(out);
    const auto stats = numerics::reblock(expected);
    CHECK(series(out, "bragg_e")[0] == Approx(stats.mean).margin(1e-13));
    CHECK(series(out, "bragg_e_error")[0] == Approx(stats.error).margin(1e-13));
    const auto actual = series(out, "bragg_e_blocks");
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        CHECK(actual[i] == Approx(expected[i]).margin(1e-13));
}

TEST_CASE("signed Bragg blocks use a ratio of means including skipped worm measurements",
          "[bragg]") {
    BraggOrder e(groups, "e", test_cell, star);
    e.configure_signed_weights(); e.configure_reblocking(256);
    Batch batch;
    std::vector<double> numerator, denominator;
    for (std::size_t block = 0; block < 256; ++block) {
        batch.set(.11 * static_cast<double>(block), .25 * static_cast<double>((block / 4) % 4));
        const double value = .5 * (direct(batch, k, 0.) + direct(batch, 0., k));
        const double sign = block % 5 == 0 ? -1. : 1.;
        e.start_block();
        e.accumulate(batch, std::vector<double>{sign});
        e.skip_measurement(1); e.stop_block();
        numerator.push_back(sign * value / 2.); denominator.push_back(sign / 2.);
    }
    RunResult out; e.finalize(out);
    const auto stats = numerics::ratio_of_means(numerator, denominator);
    CHECK(series(out, "bragg_e")[0] == Approx(stats.mean).margin(1e-13));
    CHECK(series(out, "bragg_e_error")[0] == Approx(stats.error).margin(1e-13));
    const auto actual = series(out, "bragg_e_numerator_blocks");
    for (std::size_t i = 0; i < numerator.size(); ++i)
        CHECK(actual[i] == Approx(numerator[i]).margin(1e-13));
    CHECK(series(out, "bragg_e_denominator_blocks") == denominator);
}

TEST_CASE("Bragg input refuses nonreciprocal zero duplicate and malformed directions",
          "[bragg]") {
    CHECK_THROWS(BraggOrder(groups, "e", SimulationCell{}, star));
    CHECK_THROWS(BraggOrder(groups, "missing", test_cell, star));
    CHECK_THROWS(BraggOrder(groups, "e", test_cell, std::vector<double>{}));
    CHECK_THROWS(BraggOrder(groups, "e", test_cell, std::vector<double>{k}));
    CHECK_THROWS(BraggOrder(groups, "e", test_cell, std::vector<double>{0., 0.}));
    CHECK_THROWS(BraggOrder(groups, "e", test_cell, std::vector<double>{1., 0.}));
    CHECK_THROWS(BraggOrder(groups, "e", test_cell, std::vector<double>{k, 0., -k, 0.}));
    CHECK_THROWS(BraggOrder(groups, "e", test_cell,
        std::vector<double>{std::numeric_limits<double>::infinity(), 0.}));
    BraggOrder e(groups, "e", test_cell, star);
    CHECK_THROWS(e.skip_measurement(1));
    CHECK_THROWS(e.configure_reblocking(10000000));
    BraggOrder signed_order(groups, "e", test_cell, star);
    signed_order.configure_signed_weights();
    CHECK(signed_order.reblocking_storage_bytes(100) == 100 * 4 * sizeof(double));
    CHECK_THROWS(signed_order.configure_reblocking(EstimatorReblocking::kMaxHistoryBytes / (4 * sizeof(double)) + 1));
}

TEST_CASE("independent uniform phases give the ideal gas Bragg baseline one over N", "[bragg]") {
    BraggOrder e(groups, "e", test_cell, star);
    e.configure_reblocking(16);
    Batch batch;
    // Product quadrature of independent equiprobable +/- phases for four
    // particles. Diagonal self terms survive; all cross terms average to zero.
    for (unsigned mask = 0; mask < 16; ++mask) {
        for (unsigned p = 0; p < 4; ++p) {
            batch.positions()[p * 2] = static_cast<double>((mask >> p) & 1);
            batch.positions()[p * 2 + 1] = batch.positions()[p * 2];
        }
        measure(e, batch);
    }
    RunResult out; e.finalize(out);
    CHECK(series(out, "bragg_e")[0] == Approx(.25).margin(1e-15));
}

TEST_CASE("selected Bragg vectors remain physical under a skewed canonical cell basis",
          "[bragg]") {
    const SimulationCell skewed(2, std::vector<double>{4., 0., 2., 4.});
    const std::vector<double> vectors{std::numbers::pi, 0., 0., std::numbers::pi / 2};
    BraggOrder e(groups, "e", skewed, vectors);
    e.configure_reblocking(1);
    Batch batch; batch.set(.731, .47);
    measure(e, batch);
    RunResult out; e.finalize(out);
    CHECK(series(out, "bragg_e")[0] == Approx(.5 * (direct(batch, vectors[0], vectors[1]) +
                                                            direct(batch, vectors[2], vectors[3]))));
}


// ── VMC/DMC path: no PIMC reblocking, independent-block moments ─────────────

namespace {
// Two walkers (as a VMC/DMC crowd), layout [p * dim * nw + d * nw + w].
class Crowd : public ParticleBatch {
public:
    Crowd() : ParticleBatch(2, 4, 2, test_cell) {}
    void set(std::size_t w, double shift, double distortion) {
        const std::vector<double> aos{.5, .5, 2.5, .5, .5, 2.5, 2.5 + distortion, 2.5};
        for (std::size_t p = 0; p < 4; ++p)
            for (std::size_t d = 0; d < 2; ++d)
                positions()[(p * 2 + d) * 2 + w] = aos[p * 2 + d] + shift;
    }
    // The star intensity of walker w, summed directly.
    [[nodiscard]] double star_intensity(std::size_t w) const {
        double total = 0.;
        for (const auto& [x, y] : {std::pair{k, 0.}, std::pair{0., k}}) {
            std::complex<double> rho{};
            for (std::size_t p = 0; p < 4; ++p)
                rho += std::exp(std::complex<double>(0., x * positions()[(p * 2) * 2 + w] +
                                                         y * positions()[(p * 2 + 1) * 2 + w]));
            total += std::norm(rho) / 16.;
        }
        return total / 2.;
    }
};
}  // namespace

TEST_CASE("without PIMC reblocking Bragg order is one on every translated perfect lattice",
          "[bragg][vmc_dmc]") {
    BraggOrder e(groups, "e", test_cell, star);
    Batch batch;
    for (std::size_t block = 0; block < 8; ++block) {
        batch.set(static_cast<double>(block) * .173);
        measure(e, batch, .5 + static_cast<double>(block));
    }
    RunResult out; e.finalize(out);
    CHECK(series(out, "bragg_e")[0] == Approx(1.).margin(1e-14));
    CHECK(series(out, "bragg_e_error")[0] == Approx(0.).margin(1e-14));
    CHECK(series(out, "bragg_e_directions")[0] == Approx(1.).margin(1e-14));
    CHECK(series(out, "bragg_e_directions")[1] == Approx(1.).margin(1e-14));
    CHECK(series(out, "bragg_e_kx").size() == 2);
    // No PIMC block history on this path.
    CHECK(std::none_of(out.series.begin(), out.series.end(),
                       [](const auto& s) { return s.first == "bragg_e_blocks"; }));
}

TEST_CASE("VMC/DMC Bragg blocks are walker-weighted means and clones merge by Chan's rule",
          "[bragg][vmc_dmc]") {
    // DMC weights each walker; the block estimate is sum_w w I_w / sum_w w.
    // Two clones (threads) take four blocks each; the merged mean and SEM are
    // those of all eight block values, as if one estimator had seen them all.
    BraggOrder prototype(groups, "e", test_cell, star);
    auto a = prototype.clone(), b = prototype.clone();
    Crowd crowd;
    std::vector<double> blocks;
    for (std::size_t block = 0; block < 8; ++block) {
        crowd.set(0, .1 * static_cast<double>(block), .15 * static_cast<double>(block % 3));
        crowd.set(1, -.2 * static_cast<double>(block), .4 + .1 * static_cast<double>(block % 4));
        const std::vector<double> weights{1., .3 + .2 * static_cast<double>(block)};
        auto& target = block < 4 ? *a : *b;
        target.start_block(); target.accumulate(crowd, weights); target.stop_block();
        blocks.push_back((weights[0] * crowd.star_intensity(0) + weights[1] * crowd.star_intensity(1)) /
                         (weights[0] + weights[1]));
    }
    prototype.collect(*a); prototype.collect(*b);
    RunResult out; prototype.finalize(out);
    double mean = 0.;
    for (double v : blocks) mean += v / 8.;
    double m2 = 0.;
    for (double v : blocks) m2 += (v - mean) * (v - mean);
    CHECK(series(out, "bragg_e")[0] == Approx(mean).margin(1e-14));
    CHECK(series(out, "bragg_e_error")[0] == Approx(std::sqrt(m2 / (7. * 8.))).margin(1e-14));
}

TEST_CASE("VMC/DMC Bragg refuses an unweighted block and mixing the two statistics paths",
          "[bragg][vmc_dmc]") {
    BraggOrder e(groups, "e", test_cell, star);
    Batch batch; batch.set(0.);
    CHECK_THROWS(measure(e, batch, 0.));
    CHECK_THROWS(measure(e, batch, -1.));
    BraggOrder pimc(groups, "e", test_cell, star);
    pimc.configure_reblocking(4);
    BraggOrder vmc(groups, "e", test_cell, star);
    CHECK_THROWS(pimc.collect(vmc));
}
