// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "pimc/pair_product_action.h"
#include "pimc/path_set.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace alea {
PairProductAction::PairProductAction(std::span<const ParticleGroup> groups,
                                     const Hamiltonian& ham, std::size_t dim,
                                     double extent, int grid_points)
    : extent_(extent), points_(grid_points) {
    if (dim != 2 || !std::isfinite(extent) || extent <= 0 || grid_points < 5 ||
        grid_points > 33 || grid_points % 2 == 0)
        throw std::invalid_argument(
            "pair_product finite_grid: open 2D, positive extent, odd "
            "grid_points in [5,33] required");
    auto mass = [](const ParticleGroup& g) {
        if (g.mass.empty() || g.mass.size() > 2 || !std::isfinite(g.mass[0]) ||
            g.mass[0] <= 0 || (g.mass.size() == 2 && g.mass[0] != g.mass[1]))
            throw std::invalid_argument("pair_product finite_grid: isotropic "
                                        "finite positive masses required");
        return g.mass[0];
    };
    std::size_t covered = 0;
    for (const auto& g : groups) {
        (void)mass(g);
        if (g.last < g.first ||
            g.size() > std::numeric_limits<std::size_t>::max() - covered)
            throw std::invalid_argument(
                "pair_product: invalid particle group ranges");
        covered += g.size();
        np_ = std::max(np_, g.last);
    }
    for (std::size_t i = 0; i < groups.size(); ++i)
        for (std::size_t j = i + 1; j < groups.size(); ++j)
            if (std::max(groups[i].first, groups[j].first) <
                std::min(groups[i].last, groups[j].last))
                throw std::invalid_argument(
                    "pair_product: overlapping particle groups");
    if (covered != np_)
        throw std::invalid_argument("pair_product: uncovered particle indices");
    for (const auto& component : ham.components()) {
        if (!component->is_potential() ||
            component->pair_action_role() == PairActionRole::one_body)
            continue;
        if (component->pair_action_role() != PairActionRole::radial_pairs)
            throw std::invalid_argument("pair_product: unsupported potential " +
                                        component->name());
        const auto pairs = component->radial_pair_potentials();
        if (pairs.empty())
            throw std::invalid_argument(
                "pair_product: radial capability returned no channels");
        for (auto pair : pairs) {
            const auto a = std::find(groups.begin(), groups.end(), pair.a),
                       b = std::find(groups.begin(), groups.end(), pair.b);
            if (a == groups.end() || b == groups.end() || !pair.value ||
                !std::isfinite(pair.pole_residue))
                throw std::invalid_argument("pair_product: radial channel does "
                                            "not match system groups");
            if (a->size() == 0 || b->size() == 0 || (*a == *b && a->size() < 2))
                continue;
            auto left = *a, right = *b;
            if (left.first > right.first)
                std::swap(left, right);
            auto channel = std::find_if(
                channels_.begin(), channels_.end(),
                [&](const auto& c) { return c.a == left && c.b == right; });
            if (channel == channels_.end()) {
                const double ma = mass(left), mb = mass(right),
                             mu = 1 / (1 / ma + 1 / mb);
                channels_.push_back({left, right, mu, 0, {}, {}});
                channel = std::prev(channels_.end());
            }
            channel->pole += pair.pole_residue;
            channel->radial.push_back(std::move(pair.value));
            if (!source_notes_.empty())
                source_notes_ += "; ";
            source_notes_ += component->name() + ": " + pair.approximation;
        }
    }
    const double states = double(points_) * points_;
    if (channels_.size() * 4 * states * states * sizeof(double) >
        512 * 1024 * 1024.)
        throw std::invalid_argument("pair_product: table storage exceeds 512 "
                                    "MiB; reduce grid or channel count");
}
void PairProductAction::validate_sampling(const SimulationCell& cell,
                                          double tau, std::size_t dim) const {
    if (cell.is_periodic() || dim != 2 || !std::isfinite(tau) || tau <= 0)
        throw std::invalid_argument(
            "pair_product finite_grid: only open 2D sampling is admitted; "
            "periodic pair actions need a separate derivation");
}
void PairProductAction::prepare_time_step(double tau) {
    if (tau == tau_)
        return;
    std::vector<std::shared_ptr<const numerics::FiniteGridPair>> tables;
    tables.reserve(channels_.size());
    for (const auto& c : channels_) {
        auto radial = [&c](double r) {
            double v = 0;
            for (const auto& f : c.radial)
                v += f(r);
            return v;
        };
        tables.push_back(std::make_shared<const numerics::FiniteGridPair>(
            c.mu, tau, extent_, points_, radial, c.pole));
    }
    for (std::size_t i = 0; i < channels_.size(); ++i)
        channels_[i].table = std::move(tables[i]);
    tau_ = tau;
}
void PairProductAction::check(const PathSet& path) const {
    if (path.cell().is_periodic() || path.dim() != 2 ||
        path.n_particles() != np_ || path.tau() != tau_)
        throw std::logic_error("pair_product: prepare_time_step and matching "
                               "open 2D path required");
}
double PairProductAction::link(const PathSet& path, const Channel& c,
                               std::size_t i, std::size_t j, std::size_t k,
                               bool thermal) const {
    // Follow both worldlines across the link: at k = M-1 the next beads are
    // (pi(i), 0) and (pi(j), 0), so the relative coordinate keeps tracking
    // the same pair of worldlines (identity pi: the ring of i and j).
    const auto bi = path.bead_after(i, k), bj = path.bead_after(j, k);
    std::array<double, 2> a, b;
    for (std::size_t d = 0; d < 2; ++d) {
        a[d] = path.pos(i, d, k) - path.pos(j, d, k);
        b[d] = path.pos(bi.particle, d, bi.slice) -
               path.pos(bj.particle, d, bj.slice);
    }
    const auto value = c.table->evaluate(a, b);
    return thermal ? value.tau_derivative : value.action;
}
double PairProductAction::potential_action_slice(const PathSet& path,
                                                 std::size_t k) {
    check(path);
    double action = 0;
    for (const auto& c : channels_)
        for (auto i = c.a.first; i < c.a.last; ++i)
            for (auto j = c.a == c.b ? i + 1 : c.b.first; j < c.b.last; ++j)
                action += link(path, c, i, j, k, false);
    return action;
}
double PairProductAction::total_action(const PathSet& path) {
    check(path);
    double action = 0;
    for (std::size_t k = 0; k < path.num_slices(); ++k)
        action += potential_action_slice(path, k);
    return action;
}
double PairProductAction::beta_derivative(const PathSet& path) {
    check(path);
    double thermal = 0;
    for (const auto& c : channels_)
        for (auto i = c.a.first; i < c.a.last; ++i)
            for (auto j = c.a == c.b ? i + 1 : c.b.first; j < c.b.last; ++j)
                for (std::size_t k = 0; k < path.num_slices(); ++k)
                    thermal += link(path, c, i, j, k, true);
    return thermal / static_cast<double>(path.num_slices());
}
double PairProductAction::total_potential(const PathSet& path) {
    check(path);
    double total = 0;
    for (const auto& c : channels_)
        for (auto i = c.a.first; i < c.a.last; ++i)
            for (auto j = c.a == c.b ? i + 1 : c.b.first; j < c.b.last; ++j)
                for (std::size_t k = 0; k < path.num_slices(); ++k) {
                    const double r =
                        std::hypot(path.pos(i, 0, k) - path.pos(j, 0, k),
                                   path.pos(i, 1, k) - path.pos(j, 1, k));
                    for (const auto& f : c.radial)
                        total += f(r);
                }
    if (!std::isfinite(total))
        throw std::domain_error(
            "pair_product: bead potential at singular collision; pair action "
            "is finite but this potential observable is undefined there");
    return total;
}
ActionThermoMeasure PairProductAction::measure_thermo(const PathSet& path) {
    const double thermal = beta_derivative(path),
                 potential = total_potential(path) / path.num_slices();
    return {thermal - potential, potential, thermal};
}
double
PairProductAction::delta_potential_particle(PathSet& path, std::size_t p,
                                            std::size_t k,
                                            std::span<const double> proposed) {
    if (p >= np_ || k >= path.num_slices() || proposed.size() != 2)
        throw std::invalid_argument(
            "pair_product delta: invalid bead or dimensions");
    // A permuted closing link joins other particles' beads, which the
    // link-local sum below does not follow: use the exact full difference.
    if (!path.identity_permutation() && (k == 0 || k + 1 == path.num_slices()))
        return ActionComponent::delta_potential_particle(path, p, k, proposed);
    // Both incident links change (one link when M = 1).
    return delta_potential_window(path, p, k, proposed);
}

// Delta S = sum over changed links l = (k -> k+1) and pairs (i,j) containing
// p of u_ij(r_ij^k, r_ij^{k+1}; tau)[new] - u_ij[old].  Link l changes iff one
// of its two beads lies in the window, so the links start at slices
// first-1, ..., first+count-1 (mod M): count+1 links, or all M when the window
// is the whole ring.  Each relative vector is formed with the same orientation
// and subtraction as link(), so old values agree bit-for-bit with
// total_action; only the summation order differs.
double PairProductAction::delta_potential_window(
    const PathSet& path, std::size_t p, std::size_t first,
    std::span<const double> positions) {
    check(path);
    const std::size_t M = path.num_slices(), count = positions.size() / 2;
    if (p >= np_ || first >= M || positions.size() % 2 != 0 || count == 0 ||
        count > M)
        throw std::invalid_argument(
            "pair_product window: invalid particle, window or dimensions");
    // The sum below pairs p's links with partners' same-particle beads; Action
    // routes windows that touch a permuted closing link to the full difference.
    if (!path.identity_permutation() && (first == 0 || first + count >= M))
        throw std::logic_error(
            "pair_product window: a permuted closing link needs the full "
            "action difference");
    // Particle p at slice s under the proposal ([count*2] window positions).
    const auto proposed = [&](std::size_t s, std::size_t d) {
        const std::size_t offset = (s + M - first) % M;
        return offset < count ? positions[offset * 2 + d] : path.pos(p, d, s);
    };
    const std::size_t n_links = std::min(count + 1, M);
    const std::size_t first_link = (first + M - 1) % M;
    double delta = 0.0;
    for (std::size_t l = 0; l < n_links; ++l) {
        const std::size_t k = (first_link + l) % M, next = path.next(k);
        const std::array<double, 2> p_old_k{path.pos(p, 0, k), path.pos(p, 1, k)},
            p_old_next{path.pos(p, 0, next), path.pos(p, 1, next)},
            p_new_k{proposed(k, 0), proposed(k, 1)},
            p_new_next{proposed(next, 0), proposed(next, 1)};
        for (const auto& c : channels_) {
            const bool in_a = c.a.contains(p), in_b = c.b.contains(p);
            if (!in_a && !in_b)
                continue;
            const auto& partners = in_a ? c.b : c.a;
            for (auto j = partners.first; j < partners.last; ++j) {
                if (j == p)
                    continue;
                // link() pairs (i in a, j in b), with i < j inside one group.
                const bool p_first = c.a == c.b ? p < j : in_a;
                const auto relative = [&](const std::array<double, 2>& rp,
                                          std::size_t s) {
                    std::array<double, 2> r;
                    for (std::size_t d = 0; d < 2; ++d)
                        r[d] = p_first ? rp[d] - path.pos(j, d, s)
                                       : path.pos(j, d, s) - rp[d];
                    return r;
                };
                delta += c.table->evaluate(relative(p_new_k, k),
                                           relative(p_new_next, next))
                             .action -
                         c.table->evaluate(relative(p_old_k, k),
                                           relative(p_old_next, next))
                             .action;
            }
        }
        // +inf (a pair leaves the finite domain) cannot be undone by later
        // links: the old path is finite.
        if (!std::isfinite(delta))
            return delta;
    }
    return delta;
}
std::string PairProductAction::potential_approximation_notes() const {
    std::ostringstream out;
    out << std::setprecision(17)
        << "pair_product finite_grid; grid_points=" << points_
        << " relative_endpoint_extent=" << extent_ << " tau=" << tau_
        << "; open 2D isotropic masses; positive finite-volume pair density; "
           "bilinear density interpolation"
        << "; endpoints outside the relative square are rejected (a "
           "finite-domain restriction)"
        << "; grid/domain/cell-quadrature/interpolation/pair-product errors "
           "excluded from statistical uncertainty; "
        << source_notes_;
    return out.str();
}
} // namespace alea
