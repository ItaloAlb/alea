// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Config → observable estimators, one build_estimator overload per kind of
// `estimators:` entry (io::EstimatorItem).  Every rule about a kind lives in
// its builder: which samplers can measure it, what its cell must be, and how
// its spec maps onto the estimator.  build_estimators (called by App for each
// stage) visits the stage's list in document order and gives every instance
// distinct result keys (assign_output_labels).
//
// Adding a kind: add its spec to io::EstimatorItem (io/parser.h), its reader to
// parse_estimator (io/parser.cpp) and a build_estimator overload here.  A kind
// without an overload does not compile.

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "../estimators/estimator.h"
#include "../io/parser.h"
#include "../population/particle_group.h"
#include "../population/simulation_cell.h"

namespace alea {

// The sampler an estimator instance is attached to.  A kind is refused only
// where it needs something the sampler does not have (momentum_distribution
// needs the worm's open paths, which only PIMC has).  A diagonal observable of
// the configuration R is accepted everywhere: VMC measures it in |ψ_T|², DMC
// as the mixed estimate ⟨ψ_T|O|Φ⟩/⟨ψ_T|Φ⟩, PIMC at temperature 1/β.
enum class Sampler { vmc, dmc, pimc };

// Everything a builder may read besides its own spec.
struct EstimatorContext {
    std::span<const ParticleGroup> groups;
    const SimulationCell&          cell;
    std::size_t                    dim;
    Sampler                        sampler;
};

[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::GofrSpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::DensitySpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::StructureFactorSpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::CollectiveDensitySpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::ElectronHoleProximitySpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::MomentumDistributionSpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::BraggSpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::BondOrderSpec& spec, const EstimatorContext& ctx);
[[nodiscard]] std::unique_ptr<Estimator> build_estimator(
    const io::DensityMatrixSpec& spec, const EstimatorContext& ctx);

// One stage's `estimators:` list, item i → estimator i, labelled so that no
// two instances write the same result keys.
[[nodiscard]] std::vector<std::unique_ptr<Estimator>> build_estimators(
    std::span<const io::EstimatorItem> items, const EstimatorContext& ctx);

// Give every instance of one stage distinct result keys.  Instance i writes
// under its output_namespaces(), each extended by its output label when it has
// one; instances whose extended namespaces coincide would overwrite each other
// in the results file.  `named[i]` says whether the YAML gave instance i a
// `name:` (already set as its label, or — collective_density — already part of
// its namespace).
//   named instances keep their keys; two that coincide are refused, because
//                   the config asked for the same keys twice;
//   unnamed ones    keep the plain keys unless an earlier instance holds them;
//                   then they take the smallest free label 2, 3, ... and the
//                   relabelling is logged.  A stage may therefore hold several
//                   instances of one kind (e.g. a fine and a coarse g(r)).
// Named instances are placed first, so an automatic label never takes a name
// the YAML asked for.
void assign_output_labels(std::span<const std::unique_ptr<Estimator>> ests,
                          std::span<const char> named);

}  // namespace alea
