// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "builders/estimator_builders.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "estimators/bond_order.h"
#include "estimators/bragg_order.h"
#include "estimators/collective_density.h"
#include "estimators/density.h"
#include "estimators/density_matrix.h"
#include "estimators/electron_hole_proximity.h"
#include "estimators/momentum_distribution.h"
#include "estimators/pair_correlation.h"
#include "estimators/structure_factor.h"
#include "io/logger.h"

namespace alea {

namespace {

// A `name:` labels every result key of the instance (gofr_<name>_r, ...).
std::unique_ptr<Estimator> labelled(std::unique_ptr<Estimator> estimator,
                                    const std::string& name) {
    if (!name.empty()) estimator->set_output_label(name);
    return estimator;
}

}  // namespace

std::unique_ptr<Estimator> build_estimator(const io::GofrSpec& spec,
                                           const EstimatorContext& ctx) {
    return labelled(std::make_unique<PairCorrelation>(ctx.groups, ctx.dim, spec.rmax,
                                                      spec.n_bins, ctx.cell),
                    spec.name);
}

std::unique_ptr<Estimator> build_estimator(const io::DensitySpec& spec,
                                           const EstimatorContext& ctx) {
    return labelled(std::make_unique<Density>(ctx.groups, ctx.dim, spec.rmax, spec.n_bins,
                                              spec.per_particle, ctx.cell),
                    spec.name);
}

std::unique_ptr<Estimator> build_estimator(const io::StructureFactorSpec& spec,
                                           const EstimatorContext& ctx) {
    return labelled(std::make_unique<StructureFactor>(ctx.groups, ctx.cell, spec.kmax),
                    spec.name);
}

// The collective coordinate v = Σ coeff·Σ_{p∈group} r_p is not defined modulo a
// lattice translation: wrapping any one particle shifts it by a lattice vector,
// so under PBC the histogram would depend on where each configuration happened
// to be folded.  Its `name` is part of its namespace (cdensity_<name>), so it
// is never set as a label as well.
std::unique_ptr<Estimator> build_estimator(const io::CollectiveDensitySpec& spec,
                                           const EstimatorContext& ctx) {
    if (ctx.cell.is_periodic())
        throw std::runtime_error(
            "periodic lattice combined with a `collective_density` "
            "estimator: a sum of absolute positions is not invariant under "
            "the lattice translations that wrapping applies, so the "
            "histogram would be meaningless.");
    std::vector<CollectiveTerm> terms;
    terms.reserve(spec.terms.size());
    for (const auto& [group, coeff] : spec.terms) terms.push_back({group, coeff});
    return std::make_unique<CollectiveDensity>(ctx.groups, ctx.dim, spec.rmax, spec.n_bins,
                                               spec.name,
                                               std::span<const CollectiveTerm>(terms));
}

// A diagonal observable of R: every sampler measures it (Sampler).
std::unique_ptr<Estimator> build_estimator(const io::ElectronHoleProximitySpec& spec,
                                           const EstimatorContext& ctx) {
    return labelled(std::make_unique<ElectronHoleProximity>(
                        ctx.groups, spec.electron_group, spec.hole_group, spec.r_cut,
                        ctx.cell),
                    spec.name);
}

// Off-diagonal: n(r) is sampled on the cut link of an open worm path
// (WormMove::sample_open_ends), which only the PIMC driver has.
std::unique_ptr<Estimator> build_estimator(const io::MomentumDistributionSpec& spec,
                                           const EstimatorContext& ctx) {
    if (ctx.sampler != Sampler::pimc)
        throw std::runtime_error(
            "momentum_distribution is a PIMC-only estimator: it samples the "
            "one-body density matrix on the open paths of a PIMC worm, which "
            "VMC and DMC do not have");
    return labelled(std::make_unique<MomentumDistribution>(
                        ctx.groups, spec.group, spec.rmax, spec.n_bins, spec.kmax,
                        spec.n_k, spec.samples, ctx.cell, ctx.dim),
                    spec.name);
}

// Diagonal observables of R: every sampler measures them (Sampler).  Both
// need a periodic 2-D cell; their constructors refuse any other.
std::unique_ptr<Estimator> build_estimator(const io::BraggSpec& spec,
                                           const EstimatorContext& ctx) {
    return labelled(std::make_unique<BraggOrder>(ctx.groups, spec.group, ctx.cell,
                                                 spec.wavevectors),
                    spec.name);
}

std::unique_ptr<Estimator> build_estimator(const io::BondOrderSpec& spec,
                                           const EstimatorContext& ctx) {
    return labelled(std::make_unique<BondOrder>(ctx.groups, spec.group, ctx.cell,
                                                spec.reference_spacing),
                    spec.name);
}

// Off-diagonal, from ratios ψ_T(R′)/ψ_T(R) at displaced configurations: VMC
// and DMC sample with ψ_T and bind it to the estimator; PIMC has no ψ_T to
// take ratios of (its one-body density matrix is momentum_distribution).
std::unique_ptr<Estimator> build_estimator(const io::DensityMatrixSpec& spec,
                                           const EstimatorContext& ctx) {
    if (ctx.sampler == Sampler::pimc)
        throw std::runtime_error(
            "density_matrix is a VMC/DMC estimator: it takes ratios of the trial "
            "wavefunction psi_T at displaced configurations, and PIMC samples no "
            "psi_T (its one-body density matrix is momentum_distribution)");
    return labelled(std::make_unique<DensityMatrix>(ctx.groups, spec.groups, spec.pairs,
                                                    spec.rmax, spec.n_bins, spec.every,
                                                    ctx.cell, ctx.dim),
                    spec.name);
}

std::vector<std::unique_ptr<Estimator>> build_estimators(
    std::span<const io::EstimatorItem> items, const EstimatorContext& ctx) {
    std::vector<std::unique_ptr<Estimator>> ests;
    std::vector<char> named;  // [items.size()], 1 when the YAML gave a `name:`
    ests.reserve(items.size());
    named.reserve(items.size());
    for (const io::EstimatorItem& item : items) {
        ests.push_back(std::visit(
            [&](const auto& spec) { return build_estimator(spec, ctx); }, item));
        named.push_back(std::visit(
            [](const auto& spec) { return static_cast<char>(!spec.name.empty()); }, item));
    }
    assign_output_labels(ests, named);
    return ests;
}

void assign_output_labels(std::span<const std::unique_ptr<Estimator>> ests,
                          std::span<const char> named) {
    if (named.size() != ests.size())
        throw std::logic_error("assign_output_labels: one named flag per estimator");

    // Namespaces of `estimator` once extended by `label`.
    const auto namespaces_of = [](const Estimator& estimator, const std::string& label) {
        std::vector<std::string> out = estimator.output_namespaces();
        if (!label.empty())
            for (std::string& ns : out) ns += "_" + label;
        return out;
    };
    std::vector<std::string> taken;  // extended namespaces placed so far
    const auto collides = [&taken](const std::vector<std::string>& namespaces) {
        return std::any_of(namespaces.begin(), namespaces.end(), [&](const std::string& ns) {
            return std::find(taken.begin(), taken.end(), ns) != taken.end();
        });
    };

    for (std::size_t i = 0; i < ests.size(); ++i) {
        if (!named[i]) continue;
        const Estimator& estimator = *ests[i];
        const auto namespaces = namespaces_of(estimator, estimator.output_label());
        if (collides(namespaces))
            throw std::runtime_error(
                estimator.type() + ": duplicate output namespace '" + namespaces.front() +
                "': two estimators of one stage were given the same `name:`");
        taken.insert(taken.end(), namespaces.begin(), namespaces.end());
    }
    for (std::size_t i = 0; i < ests.size(); ++i) {
        if (named[i]) continue;
        Estimator& estimator = *ests[i];
        std::string label;  // plain keys unless an earlier instance holds them
        for (int n = 2; collides(namespaces_of(estimator, label)); ++n)
            label = std::to_string(n);
        const auto namespaces = namespaces_of(estimator, label);
        if (!label.empty()) {
            estimator.set_output_label(label);
            io::Logger::milestone("estimators: " + estimator.type() +
                                  " repeats in this stage; this instance writes its "
                                  "results under " + namespaces.front() + "*");
        }
        taken.insert(taken.end(), namespaces.begin(), namespaces.end());
    }
}

}  // namespace alea
