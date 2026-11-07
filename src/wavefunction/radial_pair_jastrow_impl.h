// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Member definitions of RadialPairJastrow<Radial>.  Include this only from the
// .cpp that defines Radial and explicitly instantiates the base for it
// (`template class RadialPairJastrow<MyFunction>;`): the walker loops and
// Radial::evaluate then share one translation unit, and the radial call is a
// direct, inlinable one.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "wavefunction/radial_pair_jastrow.h"
#include "population/walker_set.h"

namespace alea {

template <class Radial>
void RadialPairJastrow<Radial>::size_scratch(std::size_t nw) {
    u_.assign(nw, 0.0);       du_.assign(nw, 0.0);
    d2u_.assign(nw, 0.0);     inv_r_.assign(nw, 0.0);
    u_old_.assign(nw, 0.0);   du_old_.assign(nw, 0.0);
    d2u_old_.assign(nw, 0.0); inv_r_old_.assign(nw, 0.0);
    mask_.assign(nw, 0.0);
    std::size_t last = 0;   // the cache is indexed by partner particle
    for (const JastrowGroupPair& gp : pairs_) last = std::max({last, gp.a.last, gp.b.last});
    proposal_radial_.resize(6 * last * nw);
}

template <class Radial>
void RadialPairJastrow<Radial>::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    resize_caches(nw, np, dim);
    size_scratch(nw);
}

template <class Radial>
void RadialPairJastrow<Radial>::permute_caches(std::span<const std::size_t> src) {
    WavefunctionComponent::permute_caches(src);  // committed (+ pending)
    size_scratch(log_psi_c().size());            // new walker count
}

template <class Radial>
void RadialPairJastrow<Radial>::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);

    const BatchedDistanceTable& dt = batch.dist_tables();
    const std::span<const double> pos = batch.positions();

    auto accumulate_pair = [&]<class BC>(std::size_t i, std::size_t j) {
        // Fold this pair's displacements into the Wigner-Seitz cell once for
        // every walker.  The fold mixes Cartesian components on a general
        // Bravais lattice, so it cannot live inside the per-dimension loop.
        if constexpr (BC::periodic)
            batch.cell().min_image_batch(pos.data() + i * dim * nw,
                                         pos.data() + j * dim * nw, nw,
                                         fold_delta_.data(), 0, nw);
        // One batched radial call per pair (vs. one scalar call per walker).
        radial_.evaluate(dt.row(i, j), u_, du_, d2u_);

        const double* __restrict r    = dt.row(i, j).data();
        const double* __restrict u    = u_.data();
        const double* __restrict du   = du_.data();
        const double* __restrict d2u  = d2u_.data();
        double* __restrict       invr = inv_r_.data();
        double* __restrict       lpc  = log_psi_c_.data();

        // log psi and 1/r (reused below).
#pragma omp simd
        for (std::size_t w = 0; w < nw; ++w) {
            invr[w] = 1.0 / r[w];
            lpc[w] += u[w];
        }

        // Gradient and per-dimension laplacian:
        //   ∂_d u(r) = u'(r) Δ_d / r
        //   ∂²_d u(r) = u''(r) (Δ_d/r)² + (u'(r)/r) (1 - (Δ_d/r)²)
        for (std::size_t d = 0; d < dim; ++d) {
            [[maybe_unused]] const double* __restrict pi =
                pos.data() + (i * dim + d) * nw;
            [[maybe_unused]] const double* __restrict pj =
                pos.data() + (j * dim + d) * nw;
            [[maybe_unused]] const double* __restrict fd = fold_delta_.data() + d * nw;
            double* __restrict       gi = grad_c_.data() + (i * dim + d) * nw;
            double* __restrict       gj = grad_c_.data() + (j * dim + d) * nw;
            double* __restrict       li = lap_c_.data() + (i * dim + d) * nw;
            double* __restrict       lj = lap_c_.data() + (j * dim + d) * nw;
#pragma omp simd
            for (std::size_t w = 0; w < nw; ++w) {
                double del;
                if constexpr (BC::periodic) del = fd[w];
                else                        del = pi[w] - pj[w];
                const double dor = del * invr[w];  // Δ_d / r
                const double g   = du[w] * dor;
                gi[w] += g;
                gj[w] -= g;
                const double lap_d = d2u[w] * dor * dor
                                   + du[w] * invr[w] * (1.0 - dor * dor);
                li[w] += lap_d;
                lj[w] += lap_d;
            }
        }
    };

    // Select the boundary policy once per pair, never inside the walker loop:
    // the open instantiation is byte-identical to the pre-periodic code.
    for_each_pair([&](std::size_t i, std::size_t j) {
        if (batch.cell().is_periodic())
            accumulate_pair.template operator()<PeriodicBC>(i, j);
        else
            accumulate_pair.template operator()<OpenBC>(i, j);
    });

    // Accumulate this component's contribution into the batch totals.
    std::span<double> log_psi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) log_psi[w] += log_psi_c_[w];

    std::span<double> grad = batch.gradients();
    for (std::size_t k = 0; k < grad.size(); ++k) grad[k] += grad_c_[k];

    std::span<double> lap = batch.laplacians();
    for (std::size_t k = 0; k < lap.size(); ++k) lap[k] += lap_c_[k];
}

template <class Radial>
void RadialPairJastrow<Radial>::ratio_batch(WalkerSet& batch, std::size_t p,
                                    std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();   // global stride
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t cnt = w_end - w_begin;
    if (u_.size() < nw) prepare(nw, batch.n_particles(), dim);
    const BatchedDistanceTable& dt = batch.dist_tables();   // old distances
    const std::span<const double> pend = batch.pending_dist_rows();  // new [np*nw]
    const std::span<const double> pos  = batch.positions();
    const std::span<const double> new_pos = batch.new_pos();  // proposed pos of p
    std::span<double> log_ratios = batch.log_ratios();
    std::span<double> new_grad   = batch.new_grad();  // [dim*nw]
    std::span<double> new_lap    = batch.new_lap();   // [dim*nw]

    for (std::size_t d = 0; d < dim; ++d) {
        double* __restrict pg0 = pending_grad_c_.data() + d * nw;
        double* __restrict pl0 = pending_lap_c_.data() + d * nw;
        for (std::size_t w = w_begin; w < w_end; ++w) { pg0[w] = 0.0; pl0[w] = 0.0; }
    }

    auto add_partner = [&]<class BC>(std::size_t k) {
        // Fold this pair's displacements into the Wigner-Seitz cell once for
        // every walker.  The fold mixes Cartesian components on a general
        // Bravais lattice, so it cannot live inside the per-dimension loop.
        if constexpr (BC::periodic)
            batch.cell().min_image_batch(new_pos.data(),
                                         pos.data() + k * dim * nw, nw,
                                         fold_delta_.data(), w_begin, w_end);
        radial_.evaluate(dt.row(p, k).subspan(w_begin, cnt),
                         std::span<double>(u_old_.data() + w_begin, cnt),
                         std::span<double>(du_old_.data() + w_begin, cnt),
                         std::span<double>(d2u_old_.data() + w_begin, cnt));
        radial_.evaluate(pend.subspan(k * nw + w_begin, cnt),
                         std::span<double>(u_.data() + w_begin, cnt),
                         std::span<double>(du_.data() + w_begin, cnt),
                         std::span<double>(d2u_.data() + w_begin, cnt));

        const double* __restrict rnew = pend.data() + k * nw;
        const double* __restrict uo   = u_old_.data();
        const double* __restrict duo  = du_old_.data();
        const double* __restrict d2uo = d2u_old_.data();
        const double* __restrict un   = u_.data();
        const double* __restrict dun  = du_.data();
        const double* __restrict d2un = d2u_.data();
        double* __restrict       invr = inv_r_.data();
        double* __restrict       lr   = log_ratios.data();
        double* __restrict       cache = proposal_radial_.data() + 6 * k * nw;

        // 1/r and log|ψ'/ψ|; keep both radial evaluations for the commit.
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            invr[w] = 1.0 / rnew[w];
            lr[w] += un[w] - uo[w];
            cache[0 * nw + w] = uo[w];
            cache[1 * nw + w] = duo[w];
            cache[2 * nw + w] = d2uo[w];
            cache[3 * nw + w] = un[w];
            cache[4 * nw + w] = dun[w];
            cache[5 * nw + w] = d2un[w];
        }

        // Gradient and per-dimension laplacian at the proposed position.
        for (std::size_t d = 0; d < dim; ++d) {
            [[maybe_unused]] const double* __restrict np = new_pos.data() + d * nw;
            [[maybe_unused]] const double* __restrict pk =
                pos.data() + (k * dim + d) * nw;
            [[maybe_unused]] const double* __restrict fd = fold_delta_.data() + d * nw;
            double* __restrict       ng = pending_grad_c_.data() + d * nw;
            double* __restrict       nl = pending_lap_c_.data() + d * nw;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del;
                if constexpr (BC::periodic) del = fd[w];
                else                        del = np[w] - pk[w];
                const double dor = del * invr[w];  // Δ_d / r
                ng[w] += dun[w] * dor;
                nl[w] += d2un[w] * dor * dor
                       + dun[w] * invr[w] * (1.0 - dor * dor);
            }
        }
    };

    // Boundary policy chosen once per partner, above the walker loop.  A p in
    // neither group has no partner, and its pending grad/lap stay zero.
    for_each_partner(p, [&](std::size_t k) {
        if (batch.cell().is_periodic())
            add_partner.template operator()<PeriodicBC>(k);
        else
            add_partner.template operator()<OpenBC>(k);
    });

    // Add this component's contribution into the (summed) batch totals (range).
    for (std::size_t d = 0; d < dim; ++d) {
        double* __restrict ng_b = new_grad.data() + d * nw;
        double* __restrict nl_b = new_lap.data() + d * nw;
        const double* __restrict pg = pending_grad_c_.data() + d * nw;
        const double* __restrict pl = pending_lap_c_.data() + d * nw;
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            ng_b[w] += pg[w];
            nl_b[w] += pl[w];
        }
    }
}

template <class Radial>
void RadialPairJastrow<Radial>::apply_move_batch(WalkerSet& batch, std::size_t p,
                                         std::span<const bool> accept,
                                         std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();   // global stride
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    if (mask_.size() < nw) prepare(nw, batch.n_particles(), dim);
    const BatchedDistanceTable& dt = batch.dist_tables();   // OLD distances
    const std::span<const double> pend = batch.pending_dist_rows();  // new [np*nw]
    const std::span<const double> pos      = batch.positions();   // OLD positions
    const std::span<const double> new_pos  = batch.new_pos();     // proposed pos of p

    std::span<double> b_logpsi = batch.log_psi();
    std::span<double> b_grad   = batch.gradients();
    std::span<double> b_lap    = batch.laplacians();

    for (std::size_t w = w_begin; w < w_end; ++w) mask_[w] = accept[w] ? 1.0 : 0.0;

    auto update_partner = [&]<class BC>(std::size_t k) {
        // Fold this pair's displacements into the Wigner-Seitz cell once for
        // every walker.  The fold mixes Cartesian components on a general
        // Bravais lattice, so it cannot live inside the per-dimension loop.
        if constexpr (BC::periodic) {
            batch.cell().min_image_batch(pos.data() + k * dim * nw,
                                         pos.data() + p * dim * nw, nw,
                                         fold_delta_old_.data(), w_begin, w_end);
            batch.cell().min_image_batch(pos.data() + k * dim * nw,
                                         new_pos.data(), nw,
                                         fold_delta_.data(), w_begin, w_end);
        }
        // Radial values of this proposal, from ratio_batch.
        const double* __restrict rold = dt.row(p, k).data();
        const double* __restrict rnew = pend.data() + k * nw;
        const double* __restrict uo   = proposal_radial_.data() + (6 * k + 0) * nw;
        const double* __restrict duo  = proposal_radial_.data() + (6 * k + 1) * nw;
        const double* __restrict d2uo = proposal_radial_.data() + (6 * k + 2) * nw;
        const double* __restrict un   = proposal_radial_.data() + (6 * k + 3) * nw;
        const double* __restrict dun  = proposal_radial_.data() + (6 * k + 4) * nw;
        const double* __restrict d2un = proposal_radial_.data() + (6 * k + 5) * nw;
        const double* __restrict m    = mask_.data();
        double* __restrict iro = inv_r_old_.data();
        double* __restrict irn = inv_r_.data();
        double* __restrict lpc = log_psi_c_.data();
        double* __restrict blp = b_logpsi.data();

        // log ψ and 1/r.
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            iro[w] = 1.0 / rold[w];
            irn[w] = 1.0 / rnew[w];
            const double dlog = m[w] * (un[w] - uo[w]);
            lpc[w] += dlog;
            blp[w] += dlog;
        }

        // Partner k gradient and per-dimension laplacian.
        for (std::size_t d = 0; d < dim; ++d) {
            [[maybe_unused]] const double* __restrict xk =
                pos.data() + (k * dim + d) * nw;
            [[maybe_unused]] const double* __restrict xpo =
                pos.data() + (p * dim + d) * nw;
            [[maybe_unused]] const double* __restrict xpn = new_pos.data() + d * nw;
            [[maybe_unused]] const double* __restrict fdo =
                fold_delta_old_.data() + d * nw;
            [[maybe_unused]] const double* __restrict fdn = fold_delta_.data() + d * nw;
            double* __restrict gck = grad_c_.data() + (k * dim + d) * nw;
            double* __restrict bgk = b_grad.data() + (k * dim + d) * nw;
            double* __restrict lck = lap_c_.data() + (k * dim + d) * nw;
            double* __restrict blk = b_lap.data() + (k * dim + d) * nw;
#pragma omp simd
            for (std::size_t w = w_begin; w < w_end; ++w) {
                double del_old, del_new;
                if constexpr (BC::periodic) {
                    del_old = fdo[w];
                    del_new = fdn[w];
                } else {
                    del_old = xk[w] - xpo[w];
                    del_new = xk[w] - xpn[w];
                }
                const double dor_old = del_old * iro[w];
                const double dor_new = del_new * irn[w];
                const double g_old = duo[w] * dor_old;
                const double g_new = dun[w] * dor_new;
                const double dg = m[w] * (g_new - g_old);
                gck[w] += dg;
                bgk[w] += dg;

                const double lap_old = d2uo[w] * dor_old * dor_old
                                     + duo[w] * iro[w] * (1.0 - dor_old * dor_old);
                const double lap_new = d2un[w] * dor_new * dor_new
                                     + dun[w] * irn[w] * (1.0 - dor_new * dor_new);
                const double dlap = m[w] * (lap_new - lap_old);
                lck[w] += dlap;
                blk[w] += dlap;
            }
        }
    };

    // Boundary policy chosen once per partner, above the walker loop.
    const bool moves_this_factor = for_each_partner(p, [&](std::size_t k) {
        if (batch.cell().is_periodic())
            update_partner.template operator()<PeriodicBC>(k);
        else
            update_partner.template operator()<OpenBC>(k);
    });
    if (!moves_this_factor) return;  // p is not in this component's groups.

    // Particle p: overwrite with the absolute new grad/lap (accepted only),
    // pushing the delta to the batch totals.
    const double* __restrict m = mask_.data();
    for (std::size_t d = 0; d < dim; ++d) {
        const double* __restrict ng  = pending_grad_c_.data() + d * nw;
        const double* __restrict nl  = pending_lap_c_.data() + d * nw;
        double* __restrict gcp = grad_c_.data() + (p * dim + d) * nw;
        double* __restrict bgp = b_grad.data() + (p * dim + d) * nw;
        double* __restrict lcp = lap_c_.data() + (p * dim + d) * nw;
        double* __restrict blp = b_lap.data() + (p * dim + d) * nw;
#pragma omp simd
        for (std::size_t w = w_begin; w < w_end; ++w) {
            const double dg_p   = m[w] * (ng[w] - gcp[w]);
            const double dlap_p = m[w] * (nl[w] - lcp[w]);
            gcp[w] += dg_p;
            bgp[w] += dg_p;
            lcp[w] += dlap_p;
            blp[w] += dlap_p;
        }
    }
}

template <class Radial>
void RadialPairJastrow<Radial>::shifted_pair_ratios_batch(
    const WalkerSet& batch, std::size_t a_first, std::size_t a_last, std::size_t b_first,
    std::size_t b_last, std::span<const double> shift, std::span<double> log_ratios,
    std::span<double> /*phases: a positive factor adds none*/) {
    const std::size_t nw = batch.num_walkers();
    const std::size_t dim = batch.dim();
    const std::size_t n_a = a_last - a_first, n_b = b_last - b_first;
    if (u_.size() != nw)
        throw std::logic_error(name() + ": shifted pair ratios need the batch it evaluated");
    const SimulationCell& cell = batch.cell();
    const BatchedDistanceTable& dt = batch.dist_tables();   // committed distances
    const std::span<const double> pos = batch.positions();

    // |r_x + s_w − r_k| at the minimum image, every walker, into out [nw].
    const auto shifted_distance = [&](std::size_t x, std::size_t k, double* out) {
        for (std::size_t w = 0; w < nw; ++w) {
            double del[SimulationCell::kMaxDim];
            for (std::size_t d = 0; d < dim; ++d)
                del[d] = pos[(x * dim + d) * nw + w] + shift[d * nw + w] -
                         pos[(k * dim + d) * nw + w];
            out[w] = std::sqrt(cell.min_image_dist2(del));
        }
    };

    // S_x of every moved particle: a over [0, n_a), then b.
    if (shifted_share_.size() < (n_a + n_b) * nw) shifted_share_.resize((n_a + n_b) * nw);
    const auto share = [&](std::size_t x, double* s) {
        std::fill(s, s + nw, 0.0);
        for_each_partner(x, [&](std::size_t k) {
            shifted_distance(x, k, inv_r_.data());
            radial_.evaluate(dt.row(x, k), u_old_, du_old_, d2u_old_);
            radial_.evaluate(inv_r_, u_, du_, d2u_);
            for (std::size_t w = 0; w < nw; ++w) s[w] += u_[w] - u_old_[w];
        });
    };
    for (std::size_t t = 0; t < n_a; ++t) share(a_first + t, shifted_share_.data() + t * nw);
    for (std::size_t t = 0; t < n_b; ++t)
        share(b_first + t, shifted_share_.data() + (n_a + t) * nw);

    const auto is_pair = [&](std::size_t i, std::size_t j) {
        for (const JastrowGroupPair& gp : pairs_) {
            if (gp.a == gp.b) {
                if (gp.a.contains(i) && gp.a.contains(j)) return true;
            } else if ((gp.a.contains(i) && gp.b.contains(j)) ||
                       (gp.b.contains(i) && gp.a.contains(j))) {
                return true;
            }
        }
        return false;
    };
    for (std::size_t ii = 0; ii < n_a; ++ii)
        for (std::size_t jj = 0; jj < n_b; ++jj) {
            const std::size_t i = a_first + ii, j = b_first + jj;
            double* lr = log_ratios.data() + (ii * n_b + jj) * nw;
            const double* si = shifted_share_.data() + ii * nw;
            const double* sj = shifted_share_.data() + (n_a + jj) * nw;
            for (std::size_t w = 0; w < nw; ++w) lr[w] += si[w] + sj[w];
            if (!is_pair(i, j)) continue;
            // S_i and S_j each moved the i–j term with the other one fixed;
            // moved together it does not change.
            radial_.evaluate(dt.row(i, j), u_old_, du_old_, d2u_old_);
            shifted_distance(i, j, inv_r_.data());
            radial_.evaluate(inv_r_, u_, du_, d2u_);
            for (std::size_t w = 0; w < nw; ++w) lr[w] -= u_[w] - u_old_[w];
            shifted_distance(j, i, inv_r_.data());
            radial_.evaluate(inv_r_, u_, du_, d2u_);
            for (std::size_t w = 0; w < nw; ++w) lr[w] -= u_[w] - u_old_[w];
        }
}

}  // namespace alea
