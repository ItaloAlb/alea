// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "wavefunction_component.h"
#include "../population/simulation_cell.h"

namespace alea {

// Nosanow one-body localization factor of a quantum crystal
// (L. H. Nosanow, Phys. Rev. 146, 120 (1966); the solid-phase trial of the 2-D
//  dipolar crystal in G. E. Astrakharchik et al., Phys. Rev. Lett. 98, 060405
//  (2007), times a pair Jastrow):
//
//   log ψ_N = −α Σ_i |r_i − R_i|²
//
// over the particles of one group, each tied to its own lattice site
// (particle first + k ↔ site k).  With Δ = r_i − R_i:
//
//   ∂_d log ψ_N = −2α Δ_d,          ∂²_d log ψ_N = −2α,
//   ∂ log ψ_N/∂α = −Σ_i |Δ_i|²,     ∂(∂_d)/∂α = −2Δ_d,   ∂(∂²_d)/∂α = −2.
//
// One-body, so the committed per-particle share −α|Δ_p|² (the base value_c_
// cache) makes a move cost one evaluation, at the proposed position.
//
// ── Exchange symmetry ────────────────────────────────────────────────────────
// The factor pins particle i to site i, so it is not symmetric under exchange.
// For bosons the trial stays positive: DMC has no node to fix, and its exact
// τ → ∞ projection is the bosonic ground state — the fluid, below the freezing
// density.  Within a run the walkers do not tunnel out of the localized basin
// the trial samples, so the projected energy is that of the metastable crystal,
// the solid branch the paper reports on both sides of the transition.
//
// ── Periodic cells ───────────────────────────────────────────────────────────
// Δ is the minimum image of r_i − R_i, which makes ψ_N periodic.  Its gradient
// flips on the boundary of the cell-sized Wigner–Seitz region around R_i, where
// ψ_N has fallen to at most exp(−α R_WS²) of its peak (R_WS the inradius).  The
// constructor and rebuild() refuse α with exp(−α R_WS²) > kMaxFoldedTail, so
// |ψ|² on that boundary is below 1e-12 and the kink's surface term is far
// below any error bar.
class NosanowJastrow final : public WavefunctionComponent {
public:
    static constexpr double kMaxFoldedTail = 1e-6;

    // first: index of the group's first particle; sites: [n_sites * dim], one
    // per particle of the group; params: [α].
    NosanowJastrow(std::size_t dim, std::size_t first, std::vector<double> sites,
                   const SimulationCell& cell, std::vector<Parameter> params);

    std::string name() const override { return "nosanow"; }

    // The per-particle shares are a base cache (enable_particle_values), so the
    // base prepare and permute_caches carry every per-walker cache.
    void evaluate_batch(WalkerSet& batch) override;

    void ratio_batch(WalkerSet& batch, std::size_t p,
                     std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;

    void apply_move_batch(WalkerSet& batch,
                          std::size_t p,
                          std::span<const bool> accept_mask,
                          std::size_t w_begin = 0,
                          std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override {
        return std::make_unique<NosanowJastrow>(*this);
    }

    std::span<Parameter>       parameters()       override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }

    // α is read straight from params_; this only re-checks it (α > 0 and the
    // periodic fold bound), throwing if an optimizer step left the range.
    void rebuild() override;

    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }
    bool compute_parameter_gradient_derivatives(const WalkerSet& batch,
                                                std::span<double> dgrad,
                                                std::span<double> dlap) const override;

    [[nodiscard]] double alpha() const noexcept { return params_[0].value; }
    [[nodiscard]] std::span<const double> sites() const noexcept { return sites_; }

private:
    [[nodiscard]] std::size_t n_sites() const noexcept { return sites_.size() / dim_; }
    [[nodiscard]] bool acts_on(std::size_t p) const noexcept {
        return p >= first_ && p < first_ + n_sites();
    }

    // Δ = r − R_site over [w0, w1) for one particle at x (SoA stride nw,
    // component d at x[d*nw + w]), minimum image under periodic boundaries.
    void displacement(const double* x, std::size_t site, std::size_t nw,
                      std::size_t w0, std::size_t w1, double* del) const noexcept;

    // Share u = −α|Δ|², gradient −2αΔ and per-dimension laplacian −2α of one
    // particle; grad doubles as the Δ scratch.  Each [dim * nw] (u: [nw]).
    void particle_terms(const double* x, std::size_t site, std::size_t nw,
                        std::size_t w0, std::size_t w1, double* u, double* grad,
                        double* lap) const noexcept;

    void check_alpha(double alpha) const;

    std::size_t            dim_;
    std::size_t            first_;
    std::vector<double>    sites_;   // [n_sites * dim]
    SimulationCell         cell_;
    std::vector<Parameter> params_;  // [α]
};

}  // namespace alea
