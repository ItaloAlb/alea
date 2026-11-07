// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Estimator kinds (builders/estimator_builders.h): each spec builds its
// estimator, a kind is refused only by a sampler that lacks what it needs, and
// every instance of a stage writes its own result keys.  The key contract —
// every key an estimator writes lies in its output_namespaces() — is what lets
// write_results() label an instance, so it is checked for every kind on both
// statistics paths (VMC/DMC block moments and PIMC reblocking).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "builders/estimator_builders.h"
#include "io/parser.h"
#include "population/particle_group.h"
#include "population/simulation_cell.h"
#include "population/walker.h"
#include "population/walker_set.h"
#include "qmc/run_result.h"
#include "wavefunction/trial_wavefunction.h"

using namespace alea;
using Catch::Matchers::ContainsSubstring;

namespace {

ParticleGroup group(std::string name, std::size_t first, std::size_t last, double charge) {
    ParticleGroup g{};
    g.name = std::move(name);
    g.mass = {1.0, 1.0};
    g.inv_mass = {1.0, 1.0};
    g.charge = charge;
    g.first = first;
    g.last = last;
    g.count = last - first;
    return g;
}

// Two electrons (particles 0, 1) and two holes (2, 3) in 2-D, two walkers.
struct System {
    std::vector<ParticleGroup> groups{group("e", 0, 2, -1.0), group("h", 2, 4, +1.0)};
    SimulationCell             cell;
    std::vector<Walker>        walkers;  // the WalkerSet keeps pointers: never resized
    WalkerSet                  ws;

    explicit System(const SimulationCell& c) : cell(c), ws(2, 4, 2, c) {
        const std::vector<std::vector<double>> pos = {  // [w][p * dim + d]
            {0.1, 0.2, 0.9, -0.3, 0.3, 0.1, -0.8, 0.6},
            {-0.5, 0.4, 0.2, 0.7, -0.2, -0.6, 0.6, -0.1}};
        walkers.reserve(pos.size());
        for (const auto& p : pos) {
            walkers.emplace_back(4, 2);
            walkers.back().set_positions(p);
        }
        for (Walker& w : walkers) ws.add_walker(w);
        ws.load_all();
    }
    EstimatorContext context(Sampler sampler) const {
        return EstimatorContext{groups, cell, 2, sampler};
    }
};

SimulationCell open_cell() { return SimulationCell(2); }
SimulationCell square_cell() { return SimulationCell(2, std::vector<double>{4.0, 0.0, 0.0, 4.0}); }

// Seven particles of one group in the periodic 4×4 cell, two walkers: the
// smallest system bond_order accepts (six neighbours each).
struct Crowd {
    std::vector<ParticleGroup> groups{group("e", 0, 7, -1.0)};
    SimulationCell             cell = square_cell();
    std::vector<Walker>        walkers;  // the WalkerSet keeps pointers: never resized
    WalkerSet                  ws;

    Crowd() : ws(2, 7, 2, square_cell()) {
        walkers.reserve(2);
        for (int w = 0; w < 2; ++w) {
            std::vector<double> pos;  // [p * dim + d]
            for (int p = 0; p < 7; ++p) {
                pos.push_back(0.55 * p + 0.1 * w + 0.07 * (p % 3));
                pos.push_back(0.4 * ((3 * p) % 7) - 0.2 * w);
            }
            walkers.emplace_back(7, 2);
            walkers.back().set_positions(pos);
        }
        for (Walker& w : walkers) ws.add_walker(w);
        ws.load_all();
    }
    EstimatorContext context(Sampler sampler) const {
        return EstimatorContext{groups, cell, 2, sampler};
    }
};

// One reciprocal vector of the 4×4 cell, 2π/4 along x.
io::BraggSpec bragg(std::string name = {}) {
    return io::BraggSpec{std::move(name), "e", {1.5707963267948966, 0.0}};
}
io::BondOrderSpec bond(std::string name = {}) { return io::BondOrderSpec{std::move(name), "e", 1.0}; }

// `blocks` blocks of one measurement each.  The second walker's weight changes
// from block to block (as DMC weights do), so the block means differ.
// pimc_reblocking: the PIMC driver's opt-in history instead of the VMC/DMC
// block moments.
RunResult measure(Estimator& estimator, const WalkerSet& ws, bool pimc_reblocking,
                  int blocks = 8) {
    if (pimc_reblocking) estimator.configure_reblocking(static_cast<std::size_t>(blocks));
    for (int b = 0; b < blocks; ++b) {
        const std::vector<double> weights{1.0, 0.25 + 0.1 * b};
        estimator.start_block();
        estimator.accumulate(ws, weights);
        estimator.stop_block();
    }
    RunResult out;
    estimator.write_results(out);
    return out;
}

// momentum_distribution measures only on PIMC worm paths: closed measurements
// carry the path sign, open ones the extra-bead samples.
RunResult measure_momentum(Estimator& estimator, const WalkerSet& ws, int blocks = 8) {
    estimator.configure_signed_weights();
    estimator.configure_reblocking(static_cast<std::size_t>(blocks));
    estimator.configure_open_paths(2);
    for (int b = 0; b < blocks; ++b) {
        const std::vector<double> sign{1.0, 1.0};
        const std::vector<double> displacement{0.1 + 0.05 * b, 0.2};  // [samples * dim]
        const std::vector<double> weight{0.5};                        // [samples]
        estimator.start_block();
        estimator.accumulate(ws, sign);
        estimator.accumulate_open(displacement, weight);
        estimator.stop_block();
    }
    RunResult out;
    estimator.write_results(out);
    return out;
}

std::vector<std::string> keys_of(const RunResult& r) {
    std::vector<std::string> keys;
    for (const auto& [k, v] : r.series) keys.push_back(k);
    for (const auto& [k, v] : r.scalars) keys.push_back(k);
    for (const auto& [k, v] : r.metadata) keys.push_back(k);
    return keys;
}

bool in_namespace(const std::string& key, const std::string& ns) {
    return key == ns ||
           (key.size() > ns.size() && key.starts_with(ns) && key[ns.size()] == '_');
}

// The contract write_results relies on.
void check_keys_in_namespaces(const Estimator& estimator, const RunResult& r) {
    const auto namespaces = estimator.output_namespaces();
    const auto keys = keys_of(r);
    REQUIRE_FALSE(keys.empty());
    for (const std::string& key : keys) {
        INFO(estimator.type() << " key " << key);
        CHECK(std::any_of(namespaces.begin(), namespaces.end(),
                          [&](const std::string& ns) { return in_namespace(key, ns); }));
    }
}

// `labelled` holds exactly the keys of `plain`, each with "_<label>" inserted
// after its namespace, and the same values.
void check_relabelled(const Estimator& estimator, const RunResult& plain,
                      const RunResult& labelled, const std::string& label) {
    const auto namespaces = estimator.output_namespaces();
    const auto relabel = [&](const std::string& key) {
        std::string best;
        for (const std::string& ns : namespaces)
            if (in_namespace(key, ns) && ns.size() > best.size()) best = ns;
        REQUIRE_FALSE(best.empty());
        return best + "_" + label + key.substr(best.size());
    };
    std::map<std::string, std::vector<double>> expected, got;
    for (const auto& [k, v] : plain.series) expected[relabel(k)] = v;
    for (const auto& [k, v] : plain.scalars) expected[relabel(k)] = {v};
    for (const auto& [k, v] : labelled.series) got[k] = v;
    for (const auto& [k, v] : labelled.scalars) got[k] = {v};
    CHECK(got == expected);
    std::map<std::string, std::string> expected_meta, got_meta;
    for (const auto& [k, v] : plain.metadata) expected_meta[relabel(k)] = v;
    for (const auto& [k, v] : labelled.metadata) got_meta[k] = v;
    CHECK(got_meta == expected_meta);
}

std::vector<std::string> labels_of(const std::vector<std::unique_ptr<Estimator>>& ests) {
    std::vector<std::string> out;
    for (const auto& e : ests) out.push_back(e->output_label());
    return out;
}

io::GofrSpec gofr(std::string name = {}) { return io::GofrSpec{std::move(name), 1.5, 6}; }

}  // namespace

// ── Builders ────────────────────────────────────────────────────────────────

TEST_CASE("every estimator kind builds its estimator from its spec", "[estimator_builders]") {
    const System open(open_cell());
    const System periodic(square_cell());
    const auto vmc = open.context(Sampler::vmc);
    CHECK(build_estimator(gofr(), vmc)->type() == "gofr");
    CHECK(build_estimator(io::DensitySpec{{}, 1.5, 4, false}, vmc)->type() == "density");
    CHECK(build_estimator(io::StructureFactorSpec{{}, 3.5}, periodic.context(Sampler::vmc))
              ->type() == "sofk");
    CHECK(build_estimator(io::CollectiveDensitySpec{"v", 2.0, 4, {{"e", 1.0}}}, vmc)
              ->type() == "collective_density");
    CHECK(build_estimator(io::ElectronHoleProximitySpec{{}, "e", "h", 0.5}, vmc)
              ->type() == "electron_hole_proximity");
    CHECK(build_estimator(io::MomentumDistributionSpec{{}, "e", 1.0, 4, 2.0, 3, 1},
                          open.context(Sampler::pimc))
              ->type() == "momentum_distribution");
    CHECK(build_estimator(bragg(), periodic.context(Sampler::vmc))->type() == "bragg");
    const Crowd crowd;
    CHECK(build_estimator(bond(), crowd.context(Sampler::vmc))->type() == "bond_order");
    CHECK(build_estimator(io::DensityMatrixSpec{{}, {"e"}, {}, 1.0, 2, 1}, vmc)->type() ==
          "density_matrix");
}

TEST_CASE("density_matrix is built for VMC and DMC and refused in PIMC, which has no psi_T",
          "[estimator_builders][density_matrix]") {
    const System s(open_cell());
    const io::DensityMatrixSpec spec{{}, {"e"}, {{"e", "h"}}, 1.0, 2, 1};
    for (Sampler sampler : {Sampler::vmc, Sampler::dmc}) {
        auto est = build_estimator(spec, s.context(sampler));
        CHECK(est->type() == "density_matrix");
        CHECK(est->needs_wavefunction());
    }
    CHECK_THROWS_WITH(build_estimator(spec, s.context(Sampler::pimc)),
                      ContainsSubstring("VMC/DMC") && ContainsSubstring("momentum_distribution"));
}

TEST_CASE("density_matrix keys lie in its namespace and take a label right after it",
          "[estimator_builders][density_matrix][contract]") {
    // A trial with no factor: every ratio is 1, so rho1 = rho2 = 1 and the
    // pair condensate fraction is (N_e N_h / N_p)(1 − 1·1) = 0.
    const System s(open_cell());
    const io::DensityMatrixSpec spec{{}, {"e"}, {{"e", "h"}}, 1.0, 2, 1};
    const TrialWavefunction psi;
    auto plain = build_estimator(spec, s.context(Sampler::vmc));
    auto named = build_estimator(spec, s.context(Sampler::vmc));
    named->set_output_label("fine");
    plain->bind_wavefunction(psi, 1, 1);
    named->bind_wavefunction(psi, 1, 1);
    const RunResult r = measure(*plain, s.ws, false);
    check_keys_in_namespaces(*plain, r);
    check_relabelled(*plain, r, measure(*named, s.ws, false), "fine");
    std::size_t checked = 0;
    for (const auto& [key, values] : r.series) {
        if (key == "density_matrix_rho1_e" || key == "density_matrix_rho2_e_h") {
            for (double v : values) CHECK(v == Catch::Approx(1.0));
            ++checked;
        }
        if (key == "density_matrix_pair_condensate_fraction_e_h") {
            for (double v : values) CHECK(v == Catch::Approx(0.0).margin(1e-12));
            ++checked;
        }
    }
    CHECK(checked == 3);
}

TEST_CASE("a density spec forwards per_particle to its estimator", "[estimator_builders]") {
    System s(open_cell());
    auto est = build_estimator(io::DensitySpec{{}, 2.0, 4, true}, s.context(Sampler::vmc));
    const RunResult r = measure(*est, s.ws, false);
    const auto keys = keys_of(r);
    CHECK(std::count(keys.begin(), keys.end(), "density_e_p0") == 1);
    CHECK(std::count(keys.begin(), keys.end(), "density_h_p1") == 1);
}

TEST_CASE("a collective_density spec forwards its name and terms", "[estimator_builders]") {
    // v = r_e0 + r_e1 (the holes are not in the terms): walker 0 (1.0, -0.1),
    // walker 1 (-0.3, 1.1).  rmax 2 with 4 bins per axis (width 1) puts them in
    // cells (3, 1) and (1, 3): flat indices 13 and 7 whichever axis is
    // fastest.  Summing the holes as well would move both elsewhere.
    System s(open_cell());
    auto est = build_estimator(io::CollectiveDensitySpec{"s", 2.0, 4, {{"e", 1.0}}},
                               s.context(Sampler::vmc));
    est->start_block();
    est->accumulate(s.ws);  // unit weights
    est->stop_block();
    RunResult r;
    est->write_results(r);
    const auto it = std::find_if(r.series.begin(), r.series.end(),
                                 [](const auto& e) { return e.first == "cdensity_s"; });
    REQUIRE(it != r.series.end());
    std::vector<std::size_t> occupied;
    for (std::size_t c = 0; c < it->second.size(); ++c)
        if (it->second[c] != 0.0) occupied.push_back(c);
    CHECK(occupied == std::vector<std::size_t>{7, 13});
    CHECK(it->second[7] == Catch::Approx(it->second[13]));
}

TEST_CASE("momentum_distribution is refused outside PIMC, which alone has worm paths",
          "[estimator_builders]") {
    const System s(open_cell());
    const io::MomentumDistributionSpec spec{{}, "e", 1.0, 4, 2.0, 3, 1};
    for (Sampler sampler : {Sampler::vmc, Sampler::dmc})
        CHECK_THROWS_WITH(build_estimator(spec, s.context(sampler)),
                          ContainsSubstring("PIMC-only") && ContainsSubstring("worm"));
    CHECK_NOTHROW(build_estimator(spec, s.context(Sampler::pimc)));
}

TEST_CASE("electron_hole_proximity is built for every sampler", "[estimator_builders]") {
    const System s(open_cell());
    const io::ElectronHoleProximitySpec spec{{}, "e", "h", 0.5};
    for (Sampler sampler : {Sampler::vmc, Sampler::dmc, Sampler::pimc})
        CHECK_NOTHROW(build_estimator(spec, s.context(sampler)));
}

TEST_CASE("bragg and bond_order are built for every sampler", "[estimator_builders]") {
    // Diagonal observables of R: no sampler lacks what they need.
    const System s(square_cell());
    const Crowd crowd;
    for (Sampler sampler : {Sampler::vmc, Sampler::dmc, Sampler::pimc}) {
        CHECK_NOTHROW(build_estimator(bragg(), s.context(sampler)));
        CHECK_NOTHROW(build_estimator(bond(), crowd.context(sampler)));
    }
}

TEST_CASE("bragg and bond_order are refused in an open cell", "[estimator_builders]") {
    const System s(open_cell());
    CHECK_THROWS_WITH(build_estimator(bragg(), s.context(Sampler::vmc)),
                      ContainsSubstring("periodic"));
    CHECK_THROWS_WITH(build_estimator(bond(), s.context(Sampler::vmc)),
                      ContainsSubstring("periodic"));
}

TEST_CASE("collective_density is refused under periodic boundaries", "[estimator_builders]") {
    const System s(square_cell());
    CHECK_THROWS_WITH(build_estimator(io::CollectiveDensitySpec{"v", 2.0, 4, {{"e", 1.0}}},
                                      s.context(Sampler::pimc)),
                      ContainsSubstring("collective_density"));
}

// ── Output labels ───────────────────────────────────────────────────────────

TEST_CASE("one unnamed instance keeps the plain keys", "[estimator_builders][labels]") {
    const System s(open_cell());
    const std::vector<io::EstimatorItem> items{gofr(), io::DensitySpec{{}, 1.5, 4, false}};
    const auto ests = build_estimators(items, s.context(Sampler::vmc));
    CHECK(labels_of(ests) == std::vector<std::string>{"", ""});
}

TEST_CASE("repeated unnamed instances of one kind are numbered from 2",
          "[estimator_builders][labels]") {
    const System s(open_cell());
    const std::vector<io::EstimatorItem> items{gofr(), gofr(), io::DensitySpec{{}, 1.5, 4, false},
                                              gofr()};
    const auto ests = build_estimators(items, s.context(Sampler::vmc));
    CHECK(labels_of(ests) == std::vector<std::string>{"", "2", "", "3"});
}

TEST_CASE("a name labels its instance and automatic numbers skip it",
          "[estimator_builders][labels]") {
    const System s(open_cell());
    const std::vector<io::EstimatorItem> items{gofr(), gofr("2"), gofr(), gofr("fine")};
    const auto ests = build_estimators(items, s.context(Sampler::vmc));
    CHECK(labels_of(ests) == std::vector<std::string>{"", "2", "3", "fine"});
}

TEST_CASE("instances whose keys already differ need no label", "[estimator_builders][labels]") {
    const System s(open_cell());
    // Different groups → electron_hole_proximity_e_h vs momentum nk_e / nk_h.
    const std::vector<io::EstimatorItem> items{
        io::MomentumDistributionSpec{{}, "e", 1.0, 4, 2.0, 3, 1},
        io::MomentumDistributionSpec{{}, "h", 1.0, 4, 2.0, 3, 1},
        io::CollectiveDensitySpec{"x", 2.0, 4, {{"e", 1.0}}},
        io::CollectiveDensitySpec{"y", 2.0, 4, {{"h", 1.0}}}};
    const auto ests = build_estimators(items, s.context(Sampler::pimc));
    CHECK(labels_of(ests) == std::vector<std::string>{"", "", "", ""});
}

TEST_CASE("two instances given the same name are refused", "[estimator_builders][labels]") {
    const System s(open_cell());
    const std::vector<io::EstimatorItem> twice{gofr("fine"), gofr("fine")};
    CHECK_THROWS_WITH(build_estimators(twice, s.context(Sampler::vmc)),
                      ContainsSubstring("duplicate"));
    const std::vector<io::EstimatorItem> collective{
        io::CollectiveDensitySpec{"x", 2.0, 4, {{"e", 1.0}}},
        io::CollectiveDensitySpec{"x", 1.0, 4, {{"h", 1.0}}}};
    CHECK_THROWS_WITH(build_estimators(collective, s.context(Sampler::vmc)),
                      ContainsSubstring("duplicate"));
}

// ── The key contract and write_results ──────────────────────────────────────

TEST_CASE("every key an estimator writes lies in its output namespaces",
          "[estimator_builders][contract]") {
    const System open(open_cell());
    const System periodic(square_cell());
    struct Case {
        std::string      label;
        const System*    system;
        io::EstimatorItem spec;
    };
    const std::vector<Case> cases{
        {"gofr open", &open, gofr()},
        {"gofr periodic", &periodic, gofr()},
        {"density", &open, io::DensitySpec{{}, 1.5, 4, false}},
        {"density per particle, periodic", &periodic, io::DensitySpec{{}, 1.5, 4, true}},
        {"sofk", &periodic, io::StructureFactorSpec{{}, 3.5}},
        {"collective_density", &open, io::CollectiveDensitySpec{"v", 2.0, 4, {{"e", 1.0}}}},
        {"electron_hole_proximity", &periodic, io::ElectronHoleProximitySpec{{}, "e", "h", 0.5}},
        {"bragg", &periodic, bragg()}};
    const Crowd crowd;
    for (bool pimc : {false, true}) {
        INFO("bond_order" << (pimc ? " (PIMC reblocking)" : " (VMC/DMC block moments)"));
        auto est = build_estimator(bond(), crowd.context(Sampler::pimc));
        check_keys_in_namespaces(*est, measure(*est, crowd.ws, pimc));
    }
    for (const Case& c : cases)
        for (bool pimc : {false, true}) {
            INFO(c.label << (pimc ? " (PIMC reblocking)" : " (VMC/DMC block moments)"));
            auto est = std::visit(
                [&](const auto& spec) {
                    return build_estimator(spec, c.system->context(Sampler::pimc));
                },
                c.spec);
            check_keys_in_namespaces(*est, measure(*est, c.system->ws, pimc));
        }
    auto nk = build_estimator(io::MomentumDistributionSpec{{}, "e", 1.0, 4, 2.0, 3, 1},
                              open.context(Sampler::pimc));
    check_keys_in_namespaces(*nk, measure_momentum(*nk, open.ws));
}

TEST_CASE("a label goes right after the namespace of every result key",
          "[estimator_builders][contract]") {
    const System open(open_cell());
    const System periodic(square_cell());
    struct Case {
        std::string      label;
        const System*    system;
        io::EstimatorItem spec;
    };
    // sofk writes two namespaces (sofk_*, rhok_*); proximity's namespace
    // carries its groups.
    const std::vector<Case> cases{
        {"gofr", &open, gofr()},
        {"density", &periodic, io::DensitySpec{{}, 1.5, 4, true}},
        {"sofk", &periodic, io::StructureFactorSpec{{}, 3.5}},
        {"electron_hole_proximity", &open, io::ElectronHoleProximitySpec{{}, "e", "h", 0.5}},
        {"bragg", &periodic, bragg()}};
    {
        INFO("bond_order");
        const Crowd crowd;
        auto plain = build_estimator(bond(), crowd.context(Sampler::vmc));
        auto named = build_estimator(bond(), crowd.context(Sampler::vmc));
        named->set_output_label("fine");
        check_relabelled(*plain, measure(*plain, crowd.ws, false),
                         measure(*named, crowd.ws, false), "fine");
    }
    for (const Case& c : cases) {
        INFO(c.label);
        const auto build = [&] {
            return std::visit(
                [&](const auto& spec) {
                    return build_estimator(spec, c.system->context(Sampler::vmc));
                },
                c.spec);
        };
        auto plain = build();
        auto named = build();
        named->set_output_label("fine");
        check_relabelled(*plain, measure(*plain, c.system->ws, false),
                         measure(*named, c.system->ws, false), "fine");
    }
    auto plain = build_estimator(io::MomentumDistributionSpec{{}, "e", 1.0, 4, 2.0, 3, 1},
                                 open.context(Sampler::pimc));
    auto named = plain->clone();  // clones carry the label
    named->set_output_label("2");
    CHECK(named->clone()->output_label() == "2");
    check_relabelled(*plain, measure_momentum(*plain, open.ws),
                     measure_momentum(*named, open.ws), "2");
}

TEST_CASE("an unlabelled instance writes exactly what finalize writes",
          "[estimator_builders][contract]") {
    const System s(open_cell());
    auto a = build_estimator(gofr(), s.context(Sampler::vmc));
    auto b = build_estimator(gofr(), s.context(Sampler::vmc));
    const RunResult via_labels = measure(*a, s.ws, false);
    for (int blk = 0; blk < 8; ++blk) {
        const std::vector<double> weights{1.0, 0.25 + 0.1 * blk};
        b->start_block();
        b->accumulate(s.ws, weights);
        b->stop_block();
    }
    RunResult direct;
    b->finalize(direct);
    CHECK(via_labels.series == direct.series);
    CHECK(via_labels.scalars == direct.scalars);
    CHECK(via_labels.metadata == direct.metadata);
}

// ── RunResult backstop ──────────────────────────────────────────────────────

TEST_CASE("a result with a repeated key is refused before it reaches the writer",
          "[estimator_builders][run_result]") {
    RunResult r;
    r.method = "vmc";
    r.add_series("gofr_r", {1.0});
    r.add_scalar("energy", -1.0);
    r.add_metadata("seed", "42");
    CHECK_NOTHROW(r.check_unique_keys());
    r.add_series("gofr_r", {2.0});
    CHECK_THROWS_WITH(r.check_unique_keys(),
                      ContainsSubstring("vmc") && ContainsSubstring("duplicate") &&
                          ContainsSubstring("gofr_r"));
    RunResult attrs;
    attrs.method = "dmc";
    attrs.add_scalar("x", 1.0);
    attrs.add_metadata("x", "one");  // scalars and metadata share HDF5 attributes
    CHECK_THROWS_WITH(attrs.check_unique_keys(), ContainsSubstring("duplicate"));
}
