// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The single Ewald term of a periodic run (Natoli–Ceperley optimized breakup,
// hamiltonian/ewald_term.h).
//
// A pair sum decomposes per group pair; an Ewald sum does not, because its
// reciprocal half is one sum over |ρ(k)|² covering all charges at once.  So one
// EwaldTerm replaces EVERY pair spec of a periodic run, holding one kernel per
// group pair — every pair, listed in a spec or not.  Charged layers pick up
// their capacitor energy through the kernels' k = 0 remainder.
#include "builders/hamiltonian_builders.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "hamiltonian/ewald_term.h"
#include "io/logger.h"

namespace alea {

namespace {

// Kernels of one family for every group pair (a ≤ b).
template <class Spec>
std::vector<EwaldPair> family_pairs(std::span<const Spec> specs,
                                    const HamiltonianContext& ctx) {
    std::vector<EwaldPair> pairs;
    for (std::size_t a = 0; a < ctx.groups.size(); ++a)
        for (std::size_t b = a; b < ctx.groups.size(); ++b)
            pairs.push_back(
                {a, b, ewald_kernel(specs, ctx.groups[a], ctx.groups[b], ctx)});
    return pairs;
}

}  // namespace

std::unique_ptr<HamiltonianComponent> build_ewald(const HamiltonianContext& ctx) {
    const io::Config& cfg = ctx.cfg;
    if (!ctx.cell.is_periodic() || cfg.lattice.truncation == "minimum_image")
        return nullptr;

    const int n_families = static_cast<int>(!cfg.coulombs.empty()) +
                           static_cast<int>(!cfg.keldysh.empty()) +
                           static_cast<int>(!cfg.screened.empty());
    if (n_families == 0) return nullptr;
    // EwaldTerm takes one kernel per group pair (sharing one long-range ε), but
    // mapping specs of DIFFERENT families onto one pair is not defined, so a
    // periodic run uses a single family.
    if (n_families > 1)
        throw std::runtime_error(
            "periodic lattice mixing `coulomb`, `rytova_keldysh` and/or "
            "`screened_coulomb` terms.  The Ewald term can hold a different "
            "kernel per group pair, but mapping specs of different potential "
            "families onto it is not wired up.  Use one potential family, or set "
            "`system.lattice.truncation: minimum_image` for geometry testing.");

    std::vector<EwaldPair> pairs =
        !cfg.coulombs.empty() ? family_pairs<io::CoulombSpec>(cfg.coulombs, ctx)
        : !cfg.keldysh.empty() ? family_pairs<io::KeldyshSpec>(cfg.keldysh, ctx)
                               : family_pairs<io::ScreenedCoulombSpec>(cfg.screened, ctx);
    const bool exclude = cfg.lattice.capacitor == "exclude";
    auto ewald = std::make_unique<EwaldTerm>(ctx.groups, std::move(pairs), ctx.cell,
                                             cfg.lattice.ewald_tol, exclude);
    if (ewald->capacitor_energy() != 0.0)
        io::Logger::info("Ewald: charged layers carry a capacitor energy of " +
                         std::to_string(ewald->capacitor_energy()) +
                         " Ha per cell, included in the local energy.");
    if (exclude)
        io::Logger::info("Ewald: capacitor excluded; every layer is neutralized in "
                         "its own plane (no k = 0 energy between layers).");
    return ewald;
}

}  // namespace alea
