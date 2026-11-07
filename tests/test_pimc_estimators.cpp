// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "estimators/collective_density.h"
#include "estimators/density.h"
#include "estimators/pair_correlation.h"
#include "estimators/structure_factor.h"
#include "numerics/statistics.h"
#include "population/particle_batch.h"
#include "qmc/run_result.h"

using namespace alea;
using Catch::Approx;

namespace {
class TestBatch : public ParticleBatch {
public:
    TestBatch(std::size_t dim, const SimulationCell& cell)
        : ParticleBatch(1, 2, dim, cell) {}
    void set(double x) {
        std::fill(positions().begin(), positions().end(), 0.0);
        positions()[dim()] = x;
        if (dim() == 2) positions()[dim() + 1] = x * 0.37;
        dist_tables().build_all(positions());
    }
};

// Before the new API exists this still compiles, allowing a runtime RED
// against real naive-error output rather than a compilation-only failure.
template<class E> void opt_in(E& e, std::size_t blocks) {
    if constexpr (requires { e.configure_reblocking(blocks); })
        e.configure_reblocking(blocks);
}
std::span<const double> series(const RunResult& r, const std::string& key) {
    const auto it = std::find_if(r.series.begin(), r.series.end(),
        [&](const auto& s) { return s.first == key; });
    INFO("missing output series: " << key);
    REQUIRE(it != r.series.end());
    return it->second;
}
std::unique_ptr<Estimator> make(const std::string& kind,
                               const std::vector<ParticleGroup>& groups,
                               const SimulationCell& cell) {
    if (kind == "density") return std::make_unique<Density>(groups, 1, 2.0, 4);
    if (kind == "collective_density") {
        const std::vector<CollectiveTerm> terms{{"g", 1.0}};
        return std::make_unique<CollectiveDensity>(groups, 1, 2.0, 4, "X", terms);
    }
    if (kind == "gofr") return std::make_unique<PairCorrelation>(groups, 1, 2.0, 4);
    return std::make_unique<StructureFactor>(groups, cell, 2.0);
}
void block(Estimator& e, TestBatch& batch, double x) {
    batch.set(x);
    e.start_block();
    const std::vector<double> weights{1.0};
    e.accumulate(batch, weights);
    e.stop_block();
}
} // namespace

TEST_CASE("PIMC estimators reblock every observable from synchronized replica blocks",
          "[pimc][estimators][reblock]") {
    const std::string kind = GENERATE(std::string("density"),
        std::string("collective_density"), std::string("gofr"), std::string("sofk"));
    INFO(kind);
    ParticleGroup g{};
    g.name = "g"; g.first = 0; g.last = 2;
    const std::vector<ParticleGroup> groups{g};
    const SimulationCell cell = kind == "sofk"
        ? SimulationCell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0})
        : SimulationCell{};
    TestBatch batch(kind == "sofk" ? 2 : 1, cell);
    constexpr std::size_t n = 256;
    auto a = make(kind, groups, cell);
    auto b = make(kind, groups, cell);
    opt_in(*a, n); opt_in(*b, n);
    std::map<std::string, std::vector<std::vector<double>>> chains;
    for (std::size_t i = 0; i < n; ++i) {
        // Persistent deterministic occupancy/phase changes; the replica chain
        // boundary is NOT an additional transition in the Markov process.
        const double xa = ((i / 16) % 3 == 0) ? 0.25 : 1.25;
        const double xb = ((i / 16 + 1) % 3 == 0) ? 0.25 : 1.25;
        block(*a, batch, xa); block(*b, batch, xb);
        auto one_a = make(kind, groups, cell);
        auto one_b = make(kind, groups, cell);
        block(*one_a, batch, xa); block(*one_b, batch, xb);
        RunResult ra, rb; one_a->finalize(ra); one_b->finalize(rb);
        for (const auto& [key, values] : ra.series) {
            if (!key.ends_with("_error")) continue;
            const std::string observable = key.substr(0, key.size() - 6);
            const auto& va = series(ra, observable);
            const auto& vb = series(rb, observable);
            auto& bins = chains[observable];
            if (bins.empty()) bins.resize(va.size());
            for (std::size_t j = 0; j < bins.size(); ++j)
                bins[j].push_back(0.5 * (va[j] + vb[j]));
        }
    }
    a->collect(*b);
    RunResult result; a->finalize(result);
    bool saw_correlated = false;
    for (const auto& [key, bins] : chains) {
        const auto& mean = series(result, key);
        const auto& error = series(result, key + "_error");
        // Verify the numerical correction first, independently of diagnostics.
        for (std::size_t j = 0; j < bins.size(); ++j) {
            const auto expected = numerics::reblock(bins[j]);
            CHECK(error[j] == Approx(expected.error).margin(1e-12));
        }
        const auto& naive = series(result, key + "_error_naive");
        const auto& plateau = series(result, key + "_reblock_plateau");
        const auto& level = series(result, key + "_reblock_level");
        const auto& count = series(result, key + "_reblock_n_blocks");
        const auto& tau = series(result, key + "_tau_int");
        const auto& eoe = series(result, key + "_error_on_error");
        const auto& reliable = series(result, key + "_error_reliable");
        const auto& onset = series(result, key + "_reblock_onset_level");
        for (std::size_t j = 0; j < bins.size(); ++j) {
            const auto expected = numerics::reblock(bins[j]);
            CHECK(mean[j] == Approx(expected.mean).margin(1e-12));
            CHECK(error[j] == Approx(expected.error).margin(1e-12));
            CHECK(naive[j] == Approx(expected.naive_error).margin(1e-12));
            CHECK(plateau[j] == (expected.plateau_found ? 1.0 : 0.0));
            // Parent adds selected-row diagnostics; use them once available.
            const auto selected = []<class R>(const R& stats) {
                if constexpr (requires { stats.selected_level; }) return stats.selected_level;
                else return stats.level;
            }(expected);
            CHECK(level[j] == selected);
            CHECK(count[j] == n / (std::size_t{1} << selected));
            CHECK(tau[j] == Approx(expected.tau_int).margin(1e-12));
            CHECK(eoe[j] == Approx(expected.error_on_error).margin(1e-12));
            CHECK(onset[j] == expected.level);
            CHECK(reliable[j] == (expected.plateau_found && expected.selected_n_blocks >= 16 ? 1.0 : 0.0));
            saw_correlated |= expected.error > 1.5 * expected.naive_error;
        }
    }
    CHECK(saw_correlated);
}

TEST_CASE("PIMC estimator reblocking refuses unsupported, oversized and mismatched histories",
          "[pimc][estimators][reblock]") {
    class Unsupported : public Estimator {
    public:
        void start_block() override {}
        void accumulate(const ParticleBatch&, std::span<const double>) override {}
        void stop_block() override {}
        void collect(const Estimator&) override {}
        void finalize(RunResult&) const override {}
        std::unique_ptr<Estimator> clone() const override { return std::make_unique<Unsupported>(); }
        std::string type() const override { return "unsupported"; }
        std::vector<std::string> output_namespaces() const override { return {"unsupported"}; }
    };
    Unsupported unsupported;
    REQUIRE_THROWS_AS(unsupported.configure_reblocking(32), std::runtime_error);
    REQUIRE_THROWS_AS(unsupported.reblocking_storage_bytes(32), std::runtime_error);
    ParticleGroup g{}; g.name = "g"; g.first = 0; g.last = 2;
    const std::vector<ParticleGroup> groups{g};
    Density a(groups, 1, 2.0, 4), b(groups, 1, 2.0, 4);
    CHECK(a.reblocking_storage_bytes(32) == 32 * 4 * sizeof(double));
    REQUIRE_THROWS_AS(a.configure_reblocking(3 * 1024 * 1024), std::runtime_error);
    a.configure_reblocking(16); b.configure_reblocking(16);
    auto collector = a.clone();
    TestBatch batch(1, SimulationCell{});
    for (int i = 0; i < 16; ++i) block(a, batch, i % 2 ? 0.25 : 1.25);
    collector->collect(a);
    RunResult result; collector->finalize(result);
    CHECK(series(result, "density_g_error").size() == 4);
    CHECK(series(result, "density_g_error_reliable").size() == 4);
    REQUIRE_THROWS_AS(a.reblocking_storage_bytes(std::numeric_limits<std::size_t>::max()),
                      std::runtime_error);
    Density short_run(groups, 1, 2.0, 4);
    short_run.configure_reblocking(2);
    block(short_run, batch, 0.25); block(short_run, batch, 1.25);
    RunResult short_result; short_run.finalize(short_result);
    for (double flag : series(short_result, "density_g_error_reliable")) CHECK(flag == 0.0);
    REQUIRE_THROWS_AS(collector->collect(b), std::runtime_error);
    RunResult incomplete;
    REQUIRE_THROWS_AS(b.finalize(incomplete), std::runtime_error);
    REQUIRE_THROWS_AS(block(a, batch, 0.25), std::runtime_error);
    Density legacy(groups, 1, 2.0, 4);
    REQUIRE_THROWS_AS(legacy.collect(a), std::runtime_error);
}

TEST_CASE("PIMC estimator opt-in leaves legacy outputs and clone isolation unchanged",
          "[pimc][estimators][reblock]") {
    ParticleGroup g{}; g.name = "g"; g.first = 0; g.last = 2;
    const std::vector<ParticleGroup> groups{g};
    Density original(groups, 1, 2.0, 4);
    auto configured = original.clone();
    opt_in(*configured, 16);
    TestBatch batch(1, SimulationCell{});
    block(original, batch, 0.25);
    block(original, batch, 1.25);
    RunResult result; original.finalize(result);
    CHECK(result.series.size() == 3); // edges, density, density_error only
    CHECK(series(result, "density_g_error")[2] == Approx(0.5));
}
