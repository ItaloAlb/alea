// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "population/walker_set.h"

#include <algorithm>

#include "numerics/xoshiro_random.h"
#include "population/walker.h"

namespace alea {

WalkerSet::WalkerSet(std::size_t nw, std::size_t np, std::size_t dim,
                     const SimulationCell& cell)
    : ParticleBatch(nw, np, dim, cell) {
    gradients_.assign(np * dim * nw, 0.0);
    laplacians_.assign(np * dim * nw, 0.0);
    log_psi_.assign(nw, 0.0);
    local_energies_.assign(nw, 0.0);

    weights_.assign(nw, 1.0);
    multiplicities_.assign(nw, 1.0);
    ages_.assign(nw, 0);
    ids_.assign(nw, 0);
    parent_ids_.assign(nw, 0);

    accept_mask_ = std::make_unique<bool[]>(nw);
    accept_capacity_ = nw;

    // Proposed-move scratch (one particle at a time).
    new_grad_.assign(dim * nw, 0.0);
    new_lap_.assign(dim * nw, 0.0);
    log_ratios_.assign(nw, 0.0);
    phase_diffs_.assign(nw, 0.0);
    rr_proposed_.assign(nw, 0.0);
    rr_accepted_.assign(nw, 0.0);
    old_energies_.assign(nw, 0.0);
}

void WalkerSet::enable_phase() {
    if (has_phase_) return;
    has_phase_ = true;
    phase_gradients_.assign(np_ * dim_ * nw_, 0.0);
    phase_laplacians_.assign(np_ * dim_ * nw_, 0.0);
    new_phase_grad_.assign(dim_ * nw_, 0.0);
    new_phase_lap_.assign(dim_ * nw_, 0.0);
    local_energies_imag_.assign(nw_, 0.0);
}

void WalkerSet::load_all() {
    // Gather each walker's AoS positions [p*dim + d] into the SoA buffer
    // positions_[p*dim*nw + d*nw + w], then rebuild the distance tables.
    for (std::size_t w = 0; w < nw_; ++w) {
        const std::span<const double> src = walkers_[w]->positions();
        for (std::size_t p = 0; p < np_; ++p)
            for (std::size_t d = 0; d < dim_; ++d)
                positions_[p * dim_ * nw_ + d * nw_ + w] = src[p * dim_ + d];

        // Seed the per-walker scalar columns from the Walker objects.
        weights_[w]        = walkers_[w]->weight();
        multiplicities_[w] = walkers_[w]->multiplicity();
        ages_[w]           = walkers_[w]->age();
        ids_[w]            = walkers_[w]->id();
        parent_ids_[w]     = walkers_[w]->parent_id();
        next_id_ = std::max(next_id_, ids_[w] + 1);
    }
    wrap_all_positions();
    dist_tables_.build_all(positions_);
}

void WalkerSet::apply_move(std::size_t p,
                           std::span<const double> new_pos,
                           std::span<const double> new_grad,
                           std::span<const double> new_lap,
                           std::span<const bool>   accept_mask,
                           std::size_t w_begin, std::size_t w_end) {
    if (w_end > nw_) w_end = nw_;
    if (w_begin >= w_end) return;
    // Distance rows and positions of the accepted walkers (wrapped into the
    // cell under PBC — see commit_move for why that is safe here).  Gradients
    // and laplacians need no adjustment: with every separation folded they are
    // invariant under a lattice translation.
    commit_move(p, new_pos, accept_mask, w_begin, w_end);

    // Particle p's gradient/laplacian for accepted walkers.
    for (std::size_t w = w_begin; w < w_end; ++w) {
        if (!accept_mask[w]) continue;
        for (std::size_t d = 0; d < dim_; ++d) {
            const std::size_t soa = p * dim_ * nw_ + d * nw_ + w;
            gradients_[soa] = new_grad[d * nw_ + w];
            laplacians_[soa] = new_lap[d * nw_ + w];
        }
    }
    // Phase planes of a complex trial.  A separate loop behind one flag keeps
    // the real path exactly as it was.
    if (has_phase_) {
        for (std::size_t w = w_begin; w < w_end; ++w) {
            if (!accept_mask[w]) continue;
            for (std::size_t d = 0; d < dim_; ++d) {
                const std::size_t soa = p * dim_ * nw_ + d * nw_ + w;
                phase_gradients_[soa]  = new_phase_grad_[d * nw_ + w];
                phase_laplacians_[soa] = new_phase_lap_[d * nw_ + w];
            }
        }
    }
}

void WalkerSet::apply_permutation(std::span<const std::size_t> src) {
    const std::size_t new_nw = src.size();
    const std::size_t old_nw = nw_;

    // Gather a [np*dim*nw] SoA array: new column j ← old column src[j].
    auto gather_pdw = [&](std::vector<double>& old) {
        gather_double_.resize(np_ * dim_ * new_nw);
        for (std::size_t pd = 0; pd < np_ * dim_; ++pd) {
            const double* __restrict o = old.data() + pd * old_nw;
            double* __restrict n = gather_double_.data() + pd * new_nw;
            for (std::size_t j = 0; j < new_nw; ++j) n[j] = o[src[j]];
        }
        old.swap(gather_double_);
    };
    // Gather a [nw] per-walker scalar array.
    auto gather_w = [&](std::vector<double>& old) {
        gather_double_.resize(new_nw);
        for (std::size_t j = 0; j < new_nw; ++j) gather_double_[j] = old[src[j]];
        old.swap(gather_double_);
    };

    // Gather a [nw] per-walker std::size_t column.
    auto gather_sz = [&](std::vector<std::size_t>& old) {
        gather_size_.resize(new_nw);
        for (std::size_t j = 0; j < new_nw; ++j) gather_size_[j] = old[src[j]];
        old.swap(gather_size_);
    };

    gather_pdw(gradients_);
    gather_pdw(laplacians_);
    if (has_phase_) {
        gather_pdw(phase_gradients_);
        gather_pdw(phase_laplacians_);
        gather_w(local_energies_imag_);
    }
    gather_w(log_psi_);
    gather_w(local_energies_);
    gather_w(rr_proposed_);
    gather_w(rr_accepted_);
    gather_w(old_energies_);
    gather_w(weights_);
    gather_w(multiplicities_);
    gather_sz(ages_);
    gather_sz(ids_);
    gather_sz(parent_ids_);

    // Positions, distance tables, new_pos and pending rows; sets nw_.
    permute_configurations(src);

    // Carry the Walker* handles along (duplicates share a handle; the SoA is
    // the source of truth, so save_all must not be used on a duplicated set).
    gather_walkers_.resize(new_nw);
    for (std::size_t j = 0; j < new_nw; ++j) gather_walkers_[j] = walkers_[src[j]];
    walkers_.swap(gather_walkers_);

    // Transient per-move scratch: just resize (no data to carry).
    new_grad_.assign(dim_ * new_nw, 0.0);
    new_lap_.assign(dim_ * new_nw, 0.0);
    log_ratios_.assign(new_nw, 0.0);
    phase_diffs_.assign(new_nw, 0.0);
    if (has_phase_) {
        new_phase_grad_.assign(dim_ * new_nw, 0.0);
        new_phase_lap_.assign(dim_ * new_nw, 0.0);
    }
    if (new_nw > accept_capacity_) {
        accept_mask_ = std::make_unique<bool[]>(new_nw);
        accept_capacity_ = new_nw;
    }
    std::fill_n(accept_mask_.get(), new_nw, false);
}

std::vector<std::size_t> WalkerSet::branch(numerics::XoshiroRandom& rng,
                                           std::size_t target) {
    std::vector<std::size_t> src;
    branch(rng, target, src);
    return src;
}

void WalkerSet::branch(numerics::XoshiroRandom& rng, std::size_t target,
                      std::vector<std::size_t>& src) {
    const std::size_t n = nw_;

    // 1. Per-walker integer multiplicity m = floor(weight + ξ).  Emit m copies.
    src.clear();
    src.reserve(2 * target + 2);
    for (std::size_t w = 0; w < n; ++w) {
        const int m = static_cast<int>(weights_[w] + rng.uniform());
        for (int c = 0; c < m; ++c) src.push_back(w);
    }

    // 2. Population bounds.  On total extinction keep the highest-weight walker.
    const std::size_t n_max = 2 * target + 1;
    const std::size_t n_min = target / 5 + 1;
    if (src.empty()) {
        std::size_t best = 0;
        for (std::size_t w = 1; w < n; ++w)
            if (weights_[w] > weights_[best]) best = w;
        src.push_back(best);
    }
    while (src.size() > n_max) {  // random removal
        const std::size_t i =
            static_cast<std::size_t>(rng.uniform() * static_cast<double>(src.size()));
        src[i] = src.back();
        src.pop_back();
    }
    while (src.size() < n_min) {  // random duplication
        const std::size_t i =
            static_cast<std::size_t>(rng.uniform() * static_cast<double>(src.size()));
        src.push_back(src[i]);
    }

    // 3. Flag copies (2nd+ occurrence of a source, in final order) for genealogy.
    branch_seen_.assign(n, 0);
    branch_copy_.assign(src.size(), 0);
    for (std::size_t j = 0; j < src.size(); ++j) {
        if (branch_seen_[src[j]]) branch_copy_[j] = 1;
        else                     branch_seen_[src[j]] = 1;
    }

    // 4. Carry all per-walker state to the new layout, then reset weights and
    //    assign fresh ids/parents to the spawned copies.
    apply_permutation(src);
    for (std::size_t j = 0; j < nw_; ++j) {
        if (branch_copy_[j]) {
            parent_ids_[j] = ids_[j];   // ids_[j] currently holds the source id
            ids_[j]        = next_id_++;
        }
        weights_[j]        = 1.0;
        multiplicities_[j] = 1.0;
    }
}

void WalkerSet::save_all() {
    // Scatter SoA positions back to each walker's AoS buffer and copy energies.
    std::vector<double> aos(np_ * dim_);
    for (std::size_t w = 0; w < nw_; ++w) {
        for (std::size_t p = 0; p < np_; ++p)
            for (std::size_t d = 0; d < dim_; ++d)
                aos[p * dim_ + d] = positions_[p * dim_ * nw_ + d * nw_ + w];
        walkers_[w]->set_positions(aos);
        walkers_[w]->set_local_energy(local_energies_[w]);
    }
}

void WalkerSet::append_configurations(std::vector<double>& out) const {
    const std::size_t row = np_ * dim_;
    const std::size_t off = out.size();
    out.resize(off + nw_ * row);
    for (std::size_t w = 0; w < nw_; ++w)
        for (std::size_t p = 0; p < np_; ++p)
            for (std::size_t d = 0; d < dim_; ++d)
                out[off + w * row + p * dim_ + d] =
                    positions_[p * dim_ * nw_ + d * nw_ + w];
}

}  // namespace alea
