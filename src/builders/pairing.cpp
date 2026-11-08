// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// `- pairing:` items → PairingDeterminant: ψ = Π_b det φ(r_row − r_column),
// one pair orbital φ (cosines over reciprocal stars + a cuspless cutoff
// polynomial) shared by every block, and single-particle plane waves that
// complete an unbalanced block.
#include "builders/wavefunction_builders.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "wavefunction/backflow_pairing_determinant.h"
#include "wavefunction/pairing_determinant.h"

namespace alea {

namespace {

// The group's particle spec; a block group must be one fully polarized set of
// fermions, so that det D is antisymmetric under the exchange of exactly the
// particles that are identical.
const io::ParticleSpec& block_group(const io::Config& cfg, const std::string& group,
                                    const std::string& label) {
    for (const io::ParticleSpec& p : cfg.particles) {
        if (p.group != group) continue;
        if (p.type == "bosons")
            throw std::runtime_error(label + ": group " + group + " holds bosons, and a "
                                     "determinant is antisymmetric");
        if (p.up > 0 && p.down > 0)
            throw std::runtime_error(
                label + ": group " + group + " holds both spins; a pairing block "
                "antisymmetrizes all of its rows (or columns), so give each spin "
                "its own group");
        return p;
    }
    throw std::runtime_error(label + ": no particle declares group " + group);
}

// cos/sin(k·r) with k = Σ_i n_i b_i of the cell.  Two orbitals that are equal
// up to sign (the same function at ±k, or sin at k = 0) would make the
// determinant vanish, so they are refused.
std::unique_ptr<OrbitalSet> unpaired_orbitals(const io::PairingBlockSpec& blk,
                                              const WavefunctionContext& ctx,
                                              const std::string& label) {
    if (!ctx.cell.is_periodic() || ctx.dim != 2)
        throw std::runtime_error(label + ": plane-wave unpaired orbitals need a periodic "
                                 "2-D cell");
    try {
        return make_explicit_plane_wave_orbitals(ctx.cell, blk.unpaired_k, blk.unpaired_sin);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(label + ": " + e.what());
    }
}

}  // namespace

std::unique_ptr<WavefunctionComponent> build_component(const io::PairingSpec& spec,
                                                       const WavefunctionContext& ctx) {
    const std::string label = "pairing";
    const bool periodic = ctx.cell.is_periodic();
    const bool poly = spec.poly_order >= 1;

    std::unique_ptr<PairOrbital> orbital;
    try {
        orbital = std::make_unique<PairOrbital>(ctx.cell,
                                                static_cast<std::size_t>(spec.shells),
                                                spec.poly_order);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(label + " orbital: " + e.what());
    }
    const std::size_t S = orbital->n_stars();
    const std::size_t M = poly ? static_cast<std::size_t>(spec.poly_order) : 0;

    // Linear coefficients [p_0..p_{S−1}, c_0, c_2..c_M].  The default start
    // is the bare cutoff bump, c_0 = 1, with no cosines; without a polynomial
    // there is no safe default (φ = 0 gives a singular determinant).
    std::vector<double> values = spec.params;
    if (values.empty()) {
        if (!poly)
            throw std::runtime_error(label + " orbital without a polynomial needs explicit "
                                     "`params` (φ = 0 makes every determinant vanish)");
        values.assign(S + M, 0.0);
        values[S] = 1.0;
    }
    if (values.size() != S + M)
        throw std::runtime_error(label + " orbital: " + std::to_string(values.size()) +
                                 " params, expected " + std::to_string(S) + " cosine (p_0.." +
                                 ") and " + std::to_string(M) +
                                 " polynomial (c_0, c_2..c_M) coefficients");
    std::vector<Parameter> params = make_parameters(values, spec.is_fixed, {}, label);
    for (std::size_t s = 0; s < S; ++s) params[s].name = "p" + std::to_string(s);
    if (poly) {
        params[S].name = "c0";
        for (std::size_t m = 2; m <= M; ++m) params[S + m - 1].name = "c" + std::to_string(m);
    }
    // ψ is invariant under a common scale of the linear coefficients, which
    // leaves the linear method's overlap matrix singular: keep one fixed.
    bool any_fixed = false;
    for (const Parameter& p : params) any_fixed |= p.is_fixed;
    if (!any_fixed)
        throw std::runtime_error(
            label + " orbital: every coefficient is free, but scaling all of them only "
            "rescales ψ; fix one (is_fixed), e.g. c_0 or p_0");

    if (poly) {
        double L = spec.cutoff;
        if (L <= 0.0) {
            if (!periodic)
                throw std::runtime_error(label + " orbital: an open cell needs `cutoff:` (L)");
            L = ctx.cell.ws_radius();
        }
        if (periodic && L > ctx.cell.ws_radius())
            throw std::runtime_error(
                label + " orbital: cutoff " + std::to_string(L) + " exceeds the "
                "Wigner-Seitz inradius " + std::to_string(ctx.cell.ws_radius()) +
                " of the cell; beyond it the minimum image no longer sums every "
                "periodic image.");
        params.push_back(Parameter{"L", L, spec.cutoff_fixed, /*sign_locked=*/true});
    }

    std::vector<PairingDeterminant::Block> blocks;
    for (const io::PairingBlockSpec& blk : spec.blocks) {
        const std::string blabel = label + " block (" + blk.rows + ", " + blk.columns + ")";
        block_group(ctx.cfg, blk.rows, blabel);
        block_group(ctx.cfg, blk.columns, blabel);
        PairingDeterminant::Block b{find_group(ctx.groups, blk.rows),
                                    find_group(ctx.groups, blk.columns), nullptr};
        if (!blk.unpaired_k.empty()) b.unpaired = unpaired_orbitals(blk, ctx, blabel);
        blocks.push_back(std::move(b));
    }

    try {
        if (spec.backflow.empty()) {
            auto c = std::make_unique<PairingDeterminant>(std::move(*orbital), std::move(blocks),
                                                          std::move(params));
            if (periodic && poly) c->bound_cutoff(ctx.cell.ws_radius());
            return c;
        }
        // Backflow: each channel displaces one row or column group; whether
        // c_1 is derived (a group displacing itself) is the transformation's
        // rule.  L defaults to the inradius under PBC and is required open.
        std::vector<BackflowTransformation::Channel> channels;
        for (const io::BackflowSpec& b : spec.backflow) {
            const ParticleGroup& target = find_group(ctx.groups, b.group);
            const ParticleGroup& source = find_group(ctx.groups, b.from);
            BackflowTransformation::Channel ch;
            ch.target_first = target.first;
            ch.target_last = target.last;
            ch.source_first = source.first;
            ch.source_last = source.last;
            ch.truncation = b.truncation;
            for (std::size_t k = 0; k < b.params.size(); ++k)
                ch.params.push_back(
                    Parameter{"", b.params[k], b.is_fixed.empty() ? false : b.is_fixed[k]});
            double cutoff = 0.0;
            if (b.cutoff) cutoff = *b.cutoff;
            else if (periodic) cutoff = ctx.cell.ws_radius();
            else
                throw std::runtime_error(label + ": the backflow channel of " + b.group +
                                         " from " + b.from + " needs a `cutoff` in an open "
                                         "cell (under periodic boundaries it defaults to the "
                                         "Wigner-Seitz inradius)");
            ch.params.push_back(Parameter{"", cutoff, b.cutoff_fixed});
            channels.push_back(std::move(ch));
        }
        std::vector<BackflowPairingDeterminant::Block> bf_blocks;
        for (PairingDeterminant::Block& b : blocks)
            bf_blocks.push_back({b.rows, b.columns, std::move(b.unpaired)});
        auto c = std::make_unique<BackflowPairingDeterminant>(
            std::move(*orbital), std::move(bf_blocks), std::move(params), std::move(channels),
            ctx.cell);
        if (periodic && poly) c->bound_cutoff(ctx.cell.ws_radius());
        return c;
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(label + ": " + e.what());
    }
}

}  // namespace alea
