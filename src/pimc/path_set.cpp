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
      class_id_(np), fermionic_(np, 0), exchangeable_of_(np, npos),
      visited_(np, 0), active_(np) {
    cycle_leaders_.reserve(np);
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
    // Unused rows are fixed points outside every cycle.
    std::fill(visited_.begin(), visited_.end(), 0);
    cycle_leaders_.clear();
    sign_ = 1.0;
    for (std::size_t p = 0; p < active_; ++p) {
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
    std::fill(exchangeable_of_.begin(), exchangeable_of_.end(), npos);
    for (std::size_t p = 0; p < np; ++p) {
        // The first member of each class collects the class (setup only).
        if (std::find(class_id_.begin(), class_id_.begin() + static_cast<std::ptrdiff_t>(p),
                      class_id_[p]) != class_id_.begin() + static_cast<std::ptrdiff_t>(p))
            continue;
        std::size_t rows = 0;
        for (std::size_t q = p; q < np; ++q) rows += class_id_[q] == class_id_[p];
        largest_class_ = std::max(largest_class_, rows);
        if (rows < 2) continue;
        // Reserved for every row of the class, so later row changes never
        // reallocate; only the active rows are members.
        std::vector<std::size_t> members;
        members.reserve(rows);
        for (std::size_t q = p; q < np; ++q) {
            if (class_id_[q] != class_id_[p]) continue;
            exchangeable_of_[q] = exchangeable_.size();
            if (q < active_) members.push_back(q);
        }
        exchangeable_.push_back(std::move(members));
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
        if (p >= active_ && pi[p] != p)
            throw std::invalid_argument("PathSet: permutation moves an unused row");
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
    gap_ = 0;
}
void PathSet::open_with_gap(Bead head, std::size_t g) {
    if (open_) throw std::invalid_argument("PathSet: the path is already open");
    if (g >= num_slices())
        throw std::invalid_argument("PathSet: a worm gap must be shorter than M beads");
    set_head(head);
    open_ = true;
    gap_ = g;
}
void PathSet::set_head(Bead head) {
    if (head.particle >= active_ || head.slice >= num_slices())
        throw std::invalid_argument("PathSet: worm head outside the path");
    head_ = head;
}
void PathSet::set_gap(std::size_t g) {
    if (!open_ || g >= num_slices())
        throw std::invalid_argument("PathSet: set_gap needs an open path and g <= M - 1");
    gap_ = g;
}
void PathSet::set_active_particles(std::size_t n) {
    if (open_ || n > n_particles())
        throw std::invalid_argument(
            "PathSet: set_active_particles needs a closed path and n <= capacity");
    for (std::size_t p = n; p < n_particles(); ++p)
        if (permutation_[p] != p)
            throw std::invalid_argument(
                "PathSet: set_active_particles would cut a permutation cycle");
    active_ = n;
    for (auto& members : exchangeable_) members.clear();  // keeps the reserved storage
    for (std::size_t q = 0; q < active_; ++q)
        if (exchangeable_of_[q] != npos) exchangeable_[exchangeable_of_[q]].push_back(q);
    refresh_cycles();
}
std::size_t PathSet::append_row() {
    if (open_ || active_ >= n_particles())
        throw std::logic_error("PathSet::append_row: needs a closed path below capacity");
    const std::size_t q = active_++;
    // Row q is already a fixed point of pi (unused rows always are), so the
    // new particle starts as its own cycle.  It is the largest active index,
    // so appending keeps every member list increasing.
    if (exchangeable_of_[q] != npos) exchangeable_[exchangeable_of_[q]].push_back(q);
    refresh_cycles();
    return q;
}
void PathSet::remove_row(std::size_t p) {
    if (open_ || active_ == 0)
        throw std::logic_error("PathSet::remove_row: needs a closed path with a particle");
    const std::size_t last = active_ - 1;
    if (p > last || permutation_[p] != p || class_id_[p] != class_id_[last] ||
        fermionic_[p] != fermionic_[last])
        throw std::logic_error(
            "PathSet::remove_row: needs an active fixed point of pi in the class of the last row");
    if (p != last) {
        // Row `last` moves into row p: its coordinates, then its links.  The
        // row that led into `last` now leads into p, and p continues where
        // `last` did (into p itself when `last` was a fixed point).
        const std::size_t M = num_slices();
        for (std::size_t d = 0; d < dim(); ++d)
            for (std::size_t k = 0; k < M; ++k) pos(p, d, k) = pos(last, d, k);
        const std::size_t into_last = inverse_[last], out_of_last = permutation_[last];
        if (into_last == last) {
            permutation_[p] = p;
        } else {
            permutation_[into_last] = p;
            permutation_[p] = out_of_last;
        }
        permutation_[last] = last;
        refresh_particle_tables(p, 0, M);
    }
    if (exchangeable_of_[last] != npos) {
        auto& members = exchangeable_[exchangeable_of_[last]];
        if (members.empty() || members.back() != last)
            throw std::logic_error("PathSet::remove_row: class member list out of order");
        members.pop_back();
    }
    --active_;
    refresh_cycles();
}
std::size_t PathSet::exchange_successors(std::size_t p, std::size_t s, std::size_t k) {
    const std::size_t M = num_slices(), np = active_;
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
