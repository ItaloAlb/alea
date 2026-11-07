// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "pimc/path_set.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace alea {
namespace {
// Check dimensions before ParticleBatch allocates its np*np*M distance table.
std::size_t checked_slices(std::size_t M, std::size_t np, std::size_t dim, double tau) {
    if (!M || !np || !dim || dim > SimulationCell::kMaxDim ||
        !std::isfinite(tau) || tau <= 0 || !std::isfinite(tau * static_cast<double>(M)))
        throw std::runtime_error("PathSet: positive finite tau and nonzero valid dimensions required");
    const auto max = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (np > max / np || M > max / (np * np) ||
        np > max / dim || M > max / (np * dim))
        throw std::runtime_error("PathSet: path storage dimensions overflow");
    return M;
}
}
PathSet::PathSet(std::size_t M, std::size_t np, std::size_t dim, double tau,
                 const SimulationCell& cell)
    : ParticleBatch(checked_slices(M, np, dim, tau), np, dim, cell), tau_(tau),
      next_slice_(M), prev_slice_(M), permutation_(np), inverse_(np),
      class_id_(np), fermionic_(np, 0), visited_(np, 0) {
    for (std::size_t k = 0; k < M; ++k) {
        next_slice_[k] = (k + 1) % M;
        prev_slice_[k] = k == 0 ? M - 1 : k - 1;
    }
    std::iota(permutation_.begin(), permutation_.end(), 0);
    std::iota(class_id_.begin(), class_id_.end(), 0);  // one class per particle
    refresh_cycles();
}
void PathSet::refresh_cycles() {
    const std::size_t np = n_particles();
    identity_ = true;
    for (std::size_t p = 0; p < np; ++p) {
        inverse_[permutation_[p]] = p;
        identity_ = identity_ && permutation_[p] == p;
    }
    // Walk each cycle once from its smallest member; a cycle of length L in a
    // fermionic class is an (L-1)-transposition product, sign (-1)^(L-1).
    std::fill(visited_.begin(), visited_.end(), 0);
    cycle_leaders_.clear();
    sign_ = 1.0;
    for (std::size_t p = 0; p < np; ++p) {
        if (visited_[p]) continue;
        cycle_leaders_.push_back(p);
        std::size_t length = 0;
        for (std::size_t q = p; !visited_[q]; q = permutation_[q]) {
            visited_[q] = 1;
            ++length;
        }
        if (fermionic_[p] && length % 2 == 0) sign_ = -sign_;
    }
}
void PathSet::set_exchange_classes(std::span<const std::size_t> class_id,
                                   std::span<const char> fermionic) {
    const std::size_t np = n_particles();
    if (class_id.size() != np || fermionic.size() != np)
        throw std::invalid_argument("PathSet: exchange class labels must cover every particle");
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t q = p + 1; q < np; ++q)
            if (class_id[p] == class_id[q] && (fermionic[p] != 0) != (fermionic[q] != 0))
                throw std::invalid_argument("PathSet: one exchange class mixes fermions and bosons");
    for (std::size_t p = 0; p < np; ++p)
        if (class_id[permutation_[p]] != class_id[p])
            throw std::invalid_argument("PathSet: current permutation crosses the new exchange classes");
    std::copy(class_id.begin(), class_id.end(), class_id_.begin());
    for (std::size_t p = 0; p < np; ++p) fermionic_[p] = fermionic[p] != 0;
    largest_class_ = 1;
    exchangeable_.clear();
    for (std::size_t p = 0; p < np; ++p) {
        // The first member of each class collects the class (setup only).
        if (std::find(class_id_.begin(), class_id_.begin() + static_cast<std::ptrdiff_t>(p),
                      class_id_[p]) != class_id_.begin() + static_cast<std::ptrdiff_t>(p))
            continue;
        std::vector<std::size_t> members;
        for (std::size_t q = p; q < np; ++q)
            if (class_id_[q] == class_id_[p]) members.push_back(q);
        largest_class_ = std::max(largest_class_, members.size());
        if (members.size() > 1) exchangeable_.push_back(std::move(members));
    }
    refresh_cycles();
}
void PathSet::set_permutation(std::span<const std::size_t> pi) {
    const std::size_t np = n_particles();
    if (pi.size() != np)
        throw std::invalid_argument("PathSet: permutation size must equal the particle count");
    std::fill(visited_.begin(), visited_.end(), 0);
    for (std::size_t p = 0; p < np; ++p) {
        if (pi[p] >= np || visited_[pi[p]])
            throw std::invalid_argument("PathSet: permutation is not a bijection");
        visited_[pi[p]] = 1;
        if (class_id_[pi[p]] != class_id_[p])
            throw std::invalid_argument("PathSet: permutation connects particles of different exchange classes");
    }
    std::copy(pi.begin(), pi.end(), permutation_.begin());
    refresh_cycles();
}
void PathSet::collapse_to(std::span<const double> config_aos) {
    if (config_aos.size() != n_particles() * dim())
        throw std::runtime_error("PathSet::collapse_to: config size mismatch");
    for (const double val : config_aos)
        if (!std::isfinite(val))
            throw std::runtime_error("PathSet::collapse_to: coordinates must be finite");
    for (std::size_t p = 0; p < n_particles(); ++p)
        for (std::size_t d = 0; d < dim(); ++d)
            for (std::size_t k = 0; k < num_slices(); ++k)
                pos(p, d, k) = config_aos[p * dim() + d];
    wrap_positions();
    rebuild_tables();
}
void PathSet::rebuild_tables(std::size_t begin, std::size_t end) {
    for (std::size_t p = 0; p < n_particles(); ++p)
        refresh_particle_tables(p, begin, end);
}
void PathSet::open_at(Bead head) {
    if (open_) throw std::invalid_argument("PathSet: the path is already open");
    set_head(head);
    open_ = true;
}
void PathSet::set_head(Bead head) {
    if (head.particle >= n_particles() || head.slice >= num_slices())
        throw std::invalid_argument("PathSet: worm head outside the path");
    head_ = head;
}
std::size_t PathSet::exchange_successors(std::size_t p, std::size_t s, std::size_t k) {
    const std::size_t M = num_slices(), np = n_particles();
    if (p >= np || s >= np || p == s || k >= M || class_id_[p] != class_id_[s])
        throw std::invalid_argument(
            "PathSet::exchange_successors: needs two distinct particles of one "
            "exchange class and a slice inside the path");
    // Swapping rows after k reconnects (p,k) and (s,k) to each other's
    // successors directly; swapping rows 0..k moves the two histories instead,
    // and the closing links into slice 0 follow their beads to the other row.
    const bool swap_after = M - 1 - k <= k + 1;
    const std::size_t begin = swap_after ? k + 1 : 0, end = swap_after ? M : k + 1;
    for (std::size_t d = 0; d < dim(); ++d)
        for (std::size_t j = begin; j < end; ++j) std::swap(pos(p, d, j), pos(s, d, j));
    if (begin < end) {
        refresh_particle_tables(p, begin, end);
        refresh_particle_tables(s, begin, end);
    }
    if (swap_after) {
        std::swap(permutation_[p], permutation_[s]);
    } else {
        const std::size_t into_p = inverse_[p], into_s = inverse_[s];
        permutation_[into_p] = s;
        permutation_[into_s] = p;
    }
    refresh_cycles();
    return swap_after ? p : s;
}
void PathSet::wrap_particle(std::size_t p, std::size_t begin, std::size_t end) {
    end = std::min(end, num_slices());
    if (p >= n_particles() || begin > end)
        throw std::runtime_error("PathSet::wrap_particle: invalid particle or slice range");
    if (!cell().is_periodic()) return;
    double r[SimulationCell::kMaxDim];
    for (std::size_t k = begin; k < end; ++k) {
        for (std::size_t d = 0; d < dim(); ++d) r[d] = pos(p,d,k);
        cell().wrap(r);
        for (std::size_t d = 0; d < dim(); ++d) pos(p,d,k) = r[d];
    }
}
} // namespace alea
