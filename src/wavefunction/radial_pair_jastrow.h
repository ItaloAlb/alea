// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "wavefunction_component.h"
#include "../population/particle_group.h"

namespace alea {

// One group pair of a pair Jastrow; a == b is a same-group pair.
struct JastrowGroupPair {
    ParticleGroup a, b;
};

// Batched machinery shared by every pair Jastrow whose pair function is a
// radial u(r):
//
//   log ψ = Σ_{pairs (i,j)} u(r_ij), summed over every group pair (a, b) of
//   the factor:
//     same group : Σ_{i<j in a} u(r_ij)
//     two groups : Σ_{i in a, j in b} u(r_ij)
//
// Several group pairs share one u(r) when the same correlation applies to all
// of them, e.g. every same-spin pair of a four-group electron-hole bilayer.
// They must not repeat (an unordered pair given twice would double-count), so
// the partners of a particle never overlap across group pairs.
//
// with, for the displacement Δ = r_i − r_j and r = |Δ|,
//
//   ∂_{i,d} u   = u′(r) Δ_d/r                                 (−  for j)
//   ∂²_{i,d} u  = u″(r) (Δ_d/r)² + (u′(r)/r) (1 − (Δ_d/r)²)   (same for j)
//
// Under periodic boundaries Δ is the minimum image.  That is the exact sum over
// periodic images only when u is constant beyond the Wigner–Seitz inradius,
// where no second image is in reach; each derived class keeps its range there.
//
// Radial is the pair function, a value type with
//   void evaluate(std::span<const double> r, std::span<double> u,
//                 std::span<double> du, std::span<double> d2u) const noexcept;
// writing (not accumulating) u(r[w]), u′(r[w]), u″(r[w]) for every w.  The
// base owns it as radial_ and the walker caches and evaluate / ratio / commit
// loops; a derived class owns its parameters, rebuild() (which reassigns
// radial_) and parameter derivatives.
//
// The member definitions live in radial_pair_jastrow_impl.h, included only by
// the .cpp that defines Radial and explicitly instantiates this class for it,
// so Radial::evaluate inlines into the walker loops as a direct call.
template <class Radial>
class RadialPairJastrow : public WavefunctionComponent {
public:
    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;

    void permute_caches(std::span<const std::size_t> src) override;

    void evaluate_batch(WalkerSet& batch) override;

    void ratio_batch(WalkerSet& batch, std::size_t p,
                     std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;

    void apply_move_batch(WalkerSet& batch,
                          std::size_t p,
                          std::span<const bool> accept_mask,
                          std::size_t w_begin = 0,
                          std::size_t w_end = npos) override;

    // Both particles shifted by the same s keep their separation, so
    //   log ψ(R_ij)/ψ(R) = S_i + S_j − [u(|r_i + s − r_j|) − u(r_ij)]
    //                                 − [u(|r_j + s − r_i|) − u(r_ij)],
    // S_x = Σ_{k partner of x} [u(|r_x + s − r_k|) − u(r_xk)], the brackets
    // only when (i, j) is a pair of this factor (S_i and S_j counted the i–j
    // term with the partner unmoved).  One sum per particle and two radial
    // evaluations per pair; a positive factor, so no phase.
    [[nodiscard]] bool supports_shifted_pairs() const override { return true; }
    void shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                   std::size_t a_last, std::size_t b_first, std::size_t b_last,
                                   std::span<const double> shift, std::span<double> log_ratios,
                                   std::span<double> phases) override;

protected:
    // Throws std::invalid_argument for an empty list or a repeated unordered pair.
    RadialPairJastrow(std::vector<JastrowGroupPair> pairs, Radial radial)
        : radial_(std::move(radial)), pairs_(std::move(pairs)) {
        if (pairs_.empty())
            throw std::invalid_argument("pair jastrow: no group pair");
        for (std::size_t m = 0; m < pairs_.size(); ++m)
            for (std::size_t n = m + 1; n < pairs_.size(); ++n) {
                const JastrowGroupPair& x = pairs_[m];
                const JastrowGroupPair& y = pairs_[n];
                if ((x.a == y.a && x.b == y.b) || (x.a == y.b && x.b == y.a))
                    throw std::invalid_argument(
                        "pair jastrow: group pair (" + x.a.name + ", " + x.b.name +
                        ") given twice");
            }
    }

    RadialPairJastrow(ParticleGroup group_a, ParticleGroup group_b, Radial radial)
        : RadialPairJastrow(
              std::vector<JastrowGroupPair>{{std::move(group_a), std::move(group_b)}},
              std::move(radial)) {}

    // Calls f(i, j) once for every pair of this component, in a fixed order.
    template <class F>
    void for_each_pair(F&& f) const {
        for (const JastrowGroupPair& gp : pairs_) {
            if (gp.a == gp.b) {
                for (std::size_t i = gp.a.first; i < gp.a.last; ++i)
                    for (std::size_t j = i + 1; j < gp.a.last; ++j)
                        f(i, j);
            } else {
                for (std::size_t i = gp.a.first; i < gp.a.last; ++i)
                    for (std::size_t j = gp.b.first; j < gp.b.last; ++j)
                        f(i, j);
            }
        }
    }

    Radial radial_;  // u(r); the derived class rebuilds it from its parameters

private:
    // Calls f(k) for every partner k of particle p, over every group pair that
    // contains p's group; false when none does (the move leaves this factor
    // unchanged).  The pairs are distinct, so no k is visited twice.
    template <class F>
    bool for_each_partner(std::size_t p, F&& f) const {
        bool found = false;
        for (const JastrowGroupPair& gp : pairs_) {
            if (gp.a == gp.b) {
                if (!gp.a.contains(p)) continue;
                for (std::size_t k = gp.a.first; k < gp.a.last; ++k)
                    if (k != p) f(k);
            } else if (gp.a.contains(p)) {
                for (std::size_t k = gp.b.first; k < gp.b.last; ++k) f(k);
            } else if (gp.b.contains(p)) {
                for (std::size_t k = gp.a.first; k < gp.a.last; ++k) f(k);
            } else {
                continue;
            }
            found = true;
        }
        return found;
    }

    void size_scratch(std::size_t nw);

    std::vector<JastrowGroupPair> pairs_;

    // Per-walker scratch for the batched kernels, each [nw] (resized in
    // prepare, never inside a batch call).
    std::vector<double> u_, du_, d2u_, inv_r_;
    std::vector<double> u_old_, du_old_, d2u_old_, inv_r_old_;
    std::vector<double> mask_;
    // Proposal-local old/new (u,u',u'') for each partner k. Written by ratio,
    // consumed by the commit of the same proposal before geometry commits, so
    // the commit evaluates no radial function; branching only resizes it.
    std::vector<double> proposal_radial_; // [partner * 6 * nw + field * nw + w]
    // shifted_pair_ratios_batch: S_x of every moved particle, [(N_a + N_b) * nw];
    // grows to the largest request (an estimator call, not a sweep).
    std::vector<double> shifted_share_;
};

}  // namespace alea
