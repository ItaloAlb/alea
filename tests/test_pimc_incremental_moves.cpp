// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Incremental move acceptance is compared with complete slice evaluation.
// The reference uses the same Hamiltonian but never its optimized delta API;
// the existing many-body oracle separately checks the physical pair kernels.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "hamiltonian/coulomb.h"
#include "hamiltonian/harmonic_trap.h"
#include "pimc/move.h"
#include "pimc/pimc.h"
#include "pimc/primitive_action.h"

namespace {
using namespace alea;

struct Calls {
    std::size_t full = 0;
    std::size_t delta = 0;
};

// Deliberately omit the delta override: custom slice-local action components
// must remain correct when moves start requesting incremental differences.
class FullSliceAction final : public ActionComponent {
public:
    bool slice_local_deltas() const noexcept override { return true; }
    explicit FullSliceAction(const Hamiltonian& ham) : ham_(ham.clone()) {}
    FullSliceAction(const FullSliceAction& other)
        : ham_(other.ham_.clone()), scratch_(other.scratch_) {}
    void prepare(std::size_t slices) override { scratch_.resize(slices); }
    double potential_action_slice(const PathSet& path, std::size_t slice) override {
        ham_.evaluate_potential(path, scratch_, slice, slice + 1);
        return path.tau() * scratch_[slice];
    }
    double total_action(const PathSet& path) override {
        double value = 0.0;
        for (std::size_t slice = 0; slice < path.num_slices(); ++slice)
            value += potential_action_slice(path, slice);
        return value;
    }
    bool supports_free_bridge() const noexcept override { return true; }
    std::unique_ptr<ActionComponent> clone() const override {
        return std::make_unique<FullSliceAction>(*this);
    }
private:
    Hamiltonian ham_;
    std::vector<double> scratch_; // [M]
};

// Analytic slice-local S_V = tau * sum_{p,d} r_{p,d}^2. Counter state is
// borrowed by this single-threaded test and survives composite cloning.
class TrackedQuadraticAction final : public ActionComponent {
public:
    bool slice_local_deltas() const noexcept override { return true; }
    explicit TrackedQuadraticAction(Calls& calls, bool throws = false)
        : calls_(calls), throws_(throws) {}
    double potential_action_slice(const PathSet& path, std::size_t slice) override {
        ++calls_.full;
        double value = 0.0;
        for (std::size_t particle = 0; particle < path.n_particles(); ++particle)
            for (std::size_t axis = 0; axis < path.dim(); ++axis)
                value += std::pow(path.pos(particle, axis, slice), 2);
        return path.tau() * value;
    }
    double total_action(const PathSet& path) override {
        double value = 0.0;
        for (std::size_t slice = 0; slice < path.num_slices(); ++slice)
            value += potential_action_slice(path, slice);
        return value;
    }
    double delta_potential_particle(PathSet& path, std::size_t particle,
            std::size_t slice, std::span<const double> proposed) override {
        ++calls_.delta;
        if (throws_ && calls_.delta == 2)
            throw std::runtime_error("test delta evaluation failed");
        double value = 0.0;
        for (std::size_t axis = 0; axis < path.dim(); ++axis) {
            const double old = path.pos(particle, axis, slice);
            value += proposed[axis] * proposed[axis] - old * old;
        }
        return path.tau() * value;
    }
    bool supports_free_bridge() const noexcept override { return true; }
    std::unique_ptr<ActionComponent> clone() const override {
        return std::make_unique<TrackedQuadraticAction>(*this);
    }
private:
    Calls& calls_;
    bool throws_;
};

ParticleGroup group(std::size_t first, std::size_t last, double charge,
                    std::vector<double> masses = {1.0, 2.0}) {
    ParticleGroup result;
    result.name = first == 0 ? "electrons" : "holes";
    result.first = first;
    result.last = last;
    result.count = last - first;
    result.mass = std::move(masses);
    for (double mass : result.mass) result.inv_mass.push_back(1.0 / mass);
    result.charge = charge;
    return result;
}

std::unique_ptr<Move> move_for(int kind) {
    if (kind == 0) return std::make_unique<SingleSliceMove>(0.45);
    if (kind == 1) return std::make_unique<BisectionMove>(3);
    if (kind == 3) return std::make_unique<BisectionMove>(2);
    return std::make_unique<CentroidMove>(0.5);
}

std::vector<double> table_snapshot(const PathSet& path) {
    const auto values = path.dist_tables().data();
    return {values.begin(), values.end()};
}
} // namespace

TEST_CASE("Every PIMC move requests deltas instead of full slice potentials",
          "[pimc][incremental-moves]") {
    const auto particles = group(0, 2, 0.0);
    for (int kind : {0, 1, 2}) {
        CAPTURE(kind);
        Calls calls;
        Action action{std::vector{particles}, Hamiltonian{}, 2, true, false};
        action.add_component(std::make_unique<TrackedQuadraticAction>(calls));
        PathSet path(8, 2, 2, 0.0625);
        path.collapse_to(std::array{0.1, 0.2, 1.2, -0.3});
        auto move = move_for(kind);
        move->prepare(path);
        action.prepare(path);
        numerics::Random rng(64219);
        move->attempt(path, action, rng);
        CHECK(calls.delta == (kind == 0 ? 1 : kind == 1 ? 7 : 8));
        CHECK(calls.full == 0);
    }
}

TEST_CASE("Incremental moves match full slice acceptance and exact rejection state",
          "[pimc][incremental-moves]") {
    const std::vector groups{group(0, 3, -1.0), group(3, 6, 1.0, {3.0, 5.0})};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 0.64));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[0], 2.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[1], groups[1], 3.0));
    ham.add_component(std::make_unique<CoulombTerm>(groups[0], groups[1], 4.0, 1.0));
    for (bool periodic : {false, true}) {
        const SimulationCell cell = periodic
            ? SimulationCell(2, std::array{12.0, 0.0, 0.0, 12.0}) : SimulationCell{};
        for (int kind : {0, 1, 2, 3}) {
            CAPTURE(periodic, kind);
            PathSet fast(8, 6, 2, 0.0625, cell), reference(8, 6, 2, 0.0625, cell);
            for (std::size_t particle = 0; particle < 6; ++particle)
                for (std::size_t slice = 0; slice < 8; ++slice) {
                    fast.pos(particle, 0, slice) = 0.4 * particle + 0.03 * slice;
                    fast.pos(particle, 1, slice) = -0.2 * particle + 0.02 * slice;
                }
            std::copy(fast.positions().begin(), fast.positions().end(), reference.positions().begin());
            fast.rebuild_tables();
            reference.rebuild_tables();
            Action fast_action(groups, ham, 2);
            Action full_action(groups, ham, 2, true, false);
            full_action.add_component(std::make_unique<FullSliceAction>(ham));
            fast_action.prepare(fast);
            full_action.prepare(reference);
            auto fast_move = move_for(kind), full_move = move_for(kind);
            fast_move->prepare(fast);
            full_move->prepare(reference);
            numerics::Random fast_rng(9157), full_rng(9157);
            std::size_t accepted = 0, rejected = 0;
            for (int attempt = 0; attempt < 256; ++attempt) {
                const auto saved_positions = fast.all_positions();
                const auto saved_tables = table_snapshot(fast);
                const bool accept_fast = fast_move->attempt(fast, fast_action, fast_rng);
                const bool accept_full = full_move->attempt(reference, full_action, full_rng);
                REQUIRE(accept_fast == accept_full);
                REQUIRE(fast.all_positions() == reference.all_positions());
                REQUIRE(table_snapshot(fast) == table_snapshot(reference));
                if (accept_fast) ++accepted;
                else {
                    ++rejected;
                    REQUIRE(fast.all_positions() == saved_positions);
                    REQUIRE(table_snapshot(fast) == saved_tables);
                }
                // Table oracle uses a fresh all-pair build after each decision.
                fast.rebuild_tables();
                REQUIRE(table_snapshot(fast) == table_snapshot(reference));
            }
            CHECK(accepted > 0);
            CHECK(rejected > 0);
        }
    }
}

TEST_CASE("A later bead delta exception leaves the complete proposal uncommitted",
          "[pimc][incremental-moves]") {
    const auto particles = group(0, 2, 0.0);
    for (int kind : {1, 2}) {
        CAPTURE(kind);
        Calls calls;
        Action action{std::vector{particles}, Hamiltonian{}, 2, true, false};
        action.add_component(std::make_unique<TrackedQuadraticAction>(calls, true));
        PathSet path(8, 2, 2, 0.0625);
        path.collapse_to(std::array{0.1, 0.2, 1.2, -0.3});
        const auto saved_positions = path.all_positions();
        const auto saved_tables = table_snapshot(path);
        auto move = move_for(kind);
        numerics::Random rng(64219);
        REQUIRE_THROWS_WITH(move->attempt(path, action, rng), "test delta evaluation failed");
        CHECK(path.all_positions() == saved_positions);
        CHECK(table_snapshot(path) == saved_tables);
    }
}

TEST_CASE("Custom slice action fallback computes a delta and restores state on failure",
          "[pimc][incremental-moves]") {
    class ThrowingSliceAction final : public ActionComponent {
    public:
        bool slice_local_deltas() const noexcept override { return true; }
        double total_action(const PathSet&) override { return 0.0; }
        double potential_action_slice(const PathSet& path, std::size_t slice) override {
            if (path.pos(0, 0, slice) > 10.0) throw std::runtime_error("test proposed bead");
            return path.pos(0, 0, slice) * path.dist_tables()(slice, 0, 1);
        }
        std::unique_ptr<ActionComponent> clone() const override {
            return std::make_unique<ThrowingSliceAction>(*this);
        }
    } component;
    PathSet path(4, 2, 2, 0.25);
    path.collapse_to(std::array{1.0, 0.0, 4.0, 0.0});
    const auto saved_positions = path.all_positions();
    const auto saved_tables = table_snapshot(path);
    CHECK(component.delta_potential_particle(path, 0, 3, std::array{2.0, 0.0}) == 1.0);
    REQUIRE_THROWS_WITH(component.delta_potential_particle(path, 0, 3, std::array{20.0, 0.0}),
                        "test proposed bead");
    CHECK(path.all_positions() == saved_positions);
    CHECK(table_snapshot(path) == saved_tables);
}

namespace {
// Returns a fixed (possibly nonfinite) slice-local action change.
class NonfiniteAction final : public ActionComponent {
public:
    bool slice_local_deltas() const noexcept override { return true; }
    bool supports_free_bridge() const noexcept override { return true; }
    explicit NonfiniteAction(double value) : value_(value) {}
    double total_action(const PathSet&) override { return 0.0; }
    double delta_potential_particle(PathSet&, std::size_t, std::size_t,
                                    std::span<const double>) override { return value_; }
    std::unique_ptr<ActionComponent> clone() const override {
        return std::make_unique<NonfiniteAction>(*this);
    }
private:
    double value_;
};
} // namespace

TEST_CASE("Infinite action changes are counted rejections; NaN and -inf abort "
          "without committing any bead",
          "[pimc][incremental-moves]") {
    const auto particles = group(0, 2, 0.0);
    for (double value : {std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        for (int kind : {0, 1, 2}) {
            CAPTURE(value, kind);
            Action action{std::vector{particles}, Hamiltonian{}, 2, true, false};
            action.add_component(std::make_unique<NonfiniteAction>(value));
            PathSet path(8, 2, 2, 0.0625);
            path.collapse_to(std::array{0.1, 0.2, 1.2, -0.3});
            const auto saved_positions = path.all_positions();
            const auto saved_tables = table_snapshot(path);
            auto move = move_for(kind);
            numerics::Random rng(64219);
            if (value == std::numeric_limits<double>::infinity()) {
                // exp(-S) = 0 at the proposal: an exact rejection, but counted
                // so a domain wall cannot hide inside the acceptance ratio.
                REQUIRE_FALSE(move->attempt(path, action, rng));
                CHECK(move->infinite_action_rejections() == 1);
                move->reset_diagnostics();
                CHECK(move->infinite_action_rejections() == 0);
            } else {
                // A NaN (component bug) or -inf (unbounded weight) is never a
                // Metropolis outcome: it must stop the run, not be rejected.
                REQUIRE_THROWS_AS(move->attempt(path, action, rng),
                                  std::runtime_error);
                CHECK(move->infinite_action_rejections() == 0);
            }
            CHECK(path.all_positions() == saved_positions);
            CHECK(table_snapshot(path) == saved_tables);
        }
    }
}

TEST_CASE("Pimc stops cleanly, naming the replica, when a component returns NaN",
          "[pimc][incremental-moves]") {
    // Before the fix a NaN was silently rejected; an exception thrown inside
    // the OpenMP region would instead have called std::terminate.
    const auto particles = group(0, 2, 0.0);
    Action action{std::vector{particles}, Hamiltonian{}, 2, true, false};
    action.add_component(std::make_unique<NonfiniteAction>(
        std::numeric_limits<double>::quiet_NaN()));
    PimcConfig cfg;
    cfg.samples = 3;
    cfg.threads = 2;
    cfg.beta = 0.5;
    cfg.time_step = 0.0625;
    cfg.equilibration_blocks = 2;
    cfg.accumulation_blocks = 2;
    cfg.steps_per_block = 4;
    Pimc pimc(cfg, std::vector{particles}, Hamiltonian{}, SimulationCell{}, 2, 2,
              nullptr, &action);
    REQUIRE_THROWS_WITH(
        pimc.run(),
        Catch::Matchers::ContainsSubstring("pimc: replica ") &&
            Catch::Matchers::ContainsSubstring("failed during equilibration") &&
            Catch::Matchers::ContainsSubstring("NaN action change"));
}

TEST_CASE("Unchanged large action offsets do not erase move acceptance differences",
          "[pimc][incremental-moves]") {
    class OffsetAction final : public ActionComponent {
    public:
        bool slice_local_deltas() const noexcept override { return true; }
        double total_action(const PathSet& path) override {
            return 1e300 * static_cast<double>(path.num_slices());
        }
        double potential_action_slice(const PathSet&, std::size_t) override { return 1e300; }
        double delta_potential_particle(PathSet&, std::size_t, std::size_t,
                                        std::span<const double>) override { return 0.0; }
        std::unique_ptr<ActionComponent> clone() const override {
            return std::make_unique<OffsetAction>(*this);
        }
    };
    const std::vector groups{group(0, 2, 0.0)};
    Hamiltonian ham;
    ham.add_component(std::make_unique<HarmonicTrap>(groups, 1.0));
    for (int kind : {0, 1, 2, 3}) {
        CAPTURE(kind);
        PathSet plain(8, 2, 2, 0.0625), offset(8, 2, 2, 0.0625);
        plain.collapse_to(std::array{0.1, 0.2, 1.2, -0.3});
        offset.collapse_to(std::array{0.1, 0.2, 1.2, -0.3});
        Action plain_action(groups, ham, 2), offset_action(groups, ham, 2);
        offset_action.add_component(std::make_unique<OffsetAction>());
        auto plain_move = move_for(kind), offset_move = move_for(kind);
        numerics::Random plain_rng(624), offset_rng(624);
        for (int attempt = 0; attempt < 64; ++attempt) {
            REQUIRE(plain_move->attempt(plain, plain_action, plain_rng) ==
                    offset_move->attempt(offset, offset_action, offset_rng));
            REQUIRE(plain.all_positions() == offset.all_positions());
        }
    }
}
