// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "pimc/grand_canonical_worm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace alea {
namespace {

// The geometric helpers below mirror those of move.cpp.  They are kept in
// this translation unit so that the canonical worm's code generation (and its
// floating-point contraction under -march=native) is untouched.

std::size_t uniform_index(numerics::Random& rng, std::size_t count) {
    return std::min(count - 1, static_cast<std::size_t>(rng.uniform() * static_cast<double>(count)));
}

bool same_bead(PathSet::Bead a, PathSet::Bead b) noexcept {
    return a.particle == b.particle && a.slice == b.slice;
}

// Separation b - a of two beads (minimum image under PBC).
void bead_separation(const PathSet& path, PathSet::Bead a, PathSet::Bead b, double* sep) {
    for (std::size_t dd = 0; dd < path.dim(); ++dd)
        sep[dd] = path.pos(b.particle, dd, b.slice) - path.pos(a.particle, dd, a.slice);
    if (path.cell().is_periodic()) path.cell().min_image(sep);
}

// The segment of `links` links from `from` (present beads) sums to the
// minimum image of its end-to-end separation: the frame a bridge redraws.
bool segment_is_minimum_image(const PathSet& path, PathSet::Bead from, std::size_t links) {
    if (!path.cell().is_periodic()) return true;
    const std::size_t d = path.dim();
    double frame[SimulationCell::kMaxDim] = {}, link[SimulationCell::kMaxDim];
    PathSet::Bead bead = from;
    for (std::size_t i = 0; i < links; ++i) {
        const PathSet::Bead next = path.bead_after(bead.particle, bead.slice);
        bead_separation(path, bead, next, link);
        for (std::size_t dd = 0; dd < d; ++dd) frame[dd] += link[dd];
        bead = next;
    }
    double sep[SimulationCell::kMaxDim], offset[SimulationCell::kMaxDim] = {},
           frac[SimulationCell::kMaxDim] = {};
    bead_separation(path, from, bead, sep);
    for (std::size_t dd = 0; dd < d; ++dd) offset[dd] = frame[dd] - sep[dd];
    path.cell().fractional(offset, frac);
    for (std::size_t dd = 0; dd < d; ++dd)
        if (!(std::abs(frac[dd]) < 1e-6)) return false;
    return true;
}

// Every link of the unwrapped chain walk[0..n] ([(n+1) * d]) is its own
// minimum image: the stored spring action reads links that way.
bool links_are_canonical(const PathSet& path, const double* walk, std::size_t n) {
    if (!path.cell().is_periodic()) return true;
    const std::size_t d = path.dim();
    for (std::size_t i = 0; i < n; ++i) {
        double link[SimulationCell::kMaxDim], folded[SimulationCell::kMaxDim];
        for (std::size_t dd = 0; dd < d; ++dd) {
            link[dd] = walk[(i + 1) * d + dd] - walk[i * d + dd];
            folded[dd] = link[dd];
        }
        path.cell().min_image(folded);
        for (std::size_t dd = 0; dd < d; ++dd)
            if (std::abs(folded[dd] - link[dd]) > 1e-12 * (1.0 + std::abs(link[dd])))
                return false;
    }
    return true;
}

}  // namespace

GrandCanonicalWormMove::GrandCanonicalWormMove(int max_length, int swap_length,
                                               double worm_constant, GrandCanonicalSpec spec)
    : max_length_(max_length), swap_length_(swap_length), constant_(worm_constant),
      tune_(std::isnan(worm_constant)), spec_(std::move(spec)) {
    if (max_length < 1)
        throw std::invalid_argument("GrandCanonicalWormMove: max_length must be >= 1");
    if (swap_length < 2)
        throw std::invalid_argument("GrandCanonicalWormMove: swap_length must be >= 2");
    if (!tune_ && !(std::isfinite(worm_constant) && worm_constant > 0.0))
        throw std::invalid_argument("GrandCanonicalWormMove: the worm constant C must be finite and > 0");
    if (!std::isfinite(spec_.chemical_potential))
        throw std::invalid_argument("GrandCanonicalWormMove: the chemical potential must be finite");
    if (spec_.n_max < 1 || spec_.n_min > spec_.n_max)
        throw std::invalid_argument("GrandCanonicalWormMove: needs 0 <= n_min <= n_max and n_max >= 1");
    if (!(std::isfinite(spec_.insertion_width) && spec_.insertion_width >= 0.0))
        throw std::invalid_argument("GrandCanonicalWormMove: insertion_width must be finite and >= 0");
    for (const double c : spec_.insertion_center)
        if (!std::isfinite(c))
            throw std::invalid_argument("GrandCanonicalWormMove: insertion_center must be finite");
}

void GrandCanonicalWormMove::set_selection_probability(double p) {
    if (!(std::isfinite(p) && p > 0.0 && p <= 1.0))
        throw std::invalid_argument("GrandCanonicalWormMove: selection probability must be in (0, 1]");
    p_select_ = p;
}

void GrandCanonicalWormMove::prepare(const PathSet& path) {
    const std::size_t M = path.num_slices(), d = path.dim();
    const auto L = static_cast<std::size_t>(max_length_);
    if (L + 2 > M || static_cast<std::size_t>(swap_length_) + 1 > M)
        throw std::runtime_error(
            "GrandCanonicalWormMove: needs max_length <= M - 2 and swap_length <= M - 1");
    if (path.capacity() != spec_.n_max)
        throw std::runtime_error("GrandCanonicalWormMove: the path capacity must equal n_max");
    if (path.cell().is_periodic() != (spec_.insertion_width == 0.0))
        throw std::runtime_error(
            "GrandCanonicalWormMove: an open cell needs insertion_width > 0, a periodic "
            "cell none");
    if (!path.cell().is_periodic() && !spec_.insertion_center.empty() &&
        spec_.insertion_center.size() != d)
        throw std::runtime_error("GrandCanonicalWormMove: insertion_center needs one entry per axis");
    if (spec_.insertion_center.empty()) spec_.insertion_center.assign(d, 0.0);
    const std::size_t staged = std::max(L + 1, static_cast<std::size_t>(swap_length_) - 1);
    beads_.resize(staged);
    positions_.resize(staged * d);
    walk_.resize((std::max(L, static_cast<std::size_t>(swap_length_)) + 2) * d);
    heat_.resize(path.capacity());
}

void GrandCanonicalWormMove::load_mass(const Action& action) {
    // One species: every row shares row 0's mass tensor (the driver checks it).
    for (std::size_t dd = 0; dd < action.dim(); ++dd) mass_[dd] = action.mass(0, dd);
}

// ln rho0 for a separation over `links` time steps: the normalized free
// propagator prod_d sqrt(m_d / (2 pi n tau)) exp(-m_d x_d^2 / (2 n tau)).
double GrandCanonicalWormMove::log_rho0(const double* separation, std::size_t links,
                                        const PathSet& path) const {
    const double time = static_cast<double>(links) * path.tau();
    double log_rho = 0.0;
    for (std::size_t dd = 0; dd < path.dim(); ++dd)
        log_rho += 0.5 * std::log(mass_[dd] / (2.0 * std::numbers::pi * time)) -
                   mass_[dd] * separation[dd] * separation[dd] / (2.0 * time);
    return log_rho;
}

double GrandCanonicalWormMove::log_insertion_density(const PathSet& path,
                                                     const double* r) const {
    if (path.cell().is_periodic()) return -std::log(path.cell().volume());
    const double w = spec_.insertion_width;
    double log_q = 0.0;
    for (std::size_t dd = 0; dd < path.dim(); ++dd) {
        const double x = (r[dd] - spec_.insertion_center[dd]) / w;
        log_q += -0.5 * std::log(2.0 * std::numbers::pi * w * w) - 0.5 * x * x;
    }
    return log_q;
}

void GrandCanonicalWormMove::initial_constant(const PathSet& path) {
    if (constant_ > 0.0) return;
    // Starting estimate (tuned in equilibration): A_open ~ 1 for a typical
    // half-length gap with coincident ends.
    const auto L = static_cast<std::size_t>(max_length_);
    const double zero[SimulationCell::kMaxDim] = {};
    const double rows = static_cast<double>(std::max<std::size_t>(path.active_particles(), 1));
    constant_ = p_select_ * kOpenZ * std::exp(log_rho0(zero, L / 2 + 1, path)) /
                (kClose * rows * static_cast<double>(path.num_slices()) *
                 static_cast<double>(L + 1));
}

double GrandCanonicalWormMove::bead_actions(const PathSet& path, const Action& action,
                                            std::size_t count, bool staged,
                                            std::size_t n_rows) const {
    const std::size_t d = path.dim();
    double total = 0.0;
    double current[SimulationCell::kMaxDim];
    for (std::size_t i = 0; i < count; ++i) {
        const auto [p, k] = beads_[i];
        const double* r = staged ? positions_.data() + i * d : current;
        if (!staged)
            for (std::size_t dd = 0; dd < d; ++dd) current[dd] = path.pos(p, dd, k);
        // The changed beads lie on distinct slices, so each one meets the
        // unchanged beads of its slice: every row but its own and the slice's
        // missing bead (which, for a bead being added, is the bead itself).
        total += action.bead_action(path, k, p, std::span<const double>(r, d), n_rows,
                                    path.missing_row(k));
    }
    return total;
}

void GrandCanonicalWormMove::commit(PathSet& path, std::size_t count) const {
    const std::size_t d = path.dim();
    for (std::size_t i = 0; i < count; ++i) {
        const auto [q, k] = beads_[i];
        for (std::size_t dd = 0; dd < d; ++dd) path.pos(q, dd, k) = positions_[i * d + dd];
        path.update_particle_tables(q, k, k + 1);
    }
}

bool GrandCanonicalWormMove::accept(double log_ratio, double delta_s, numerics::Random& rng,
                                    PathSet::Bead where) {
    if (!std::isfinite(delta_s)) {
        reject_nonfinite(delta_s, where.particle, where.slice);  // throws unless +inf
        return false;
    }
    const double log_a = log_ratio - delta_s;
    if (!(std::isfinite(log_a) && (log_a >= 0.0 || rng.uniform() < std::exp(log_a))))
        return false;
    last_delta_s_ = delta_s;
    return true;
}

bool GrandCanonicalWormMove::attempt(PathSet& path, Action& action, numerics::Random& rng) {
    if (!(p_select_ > 0.0))
        throw std::logic_error("GrandCanonicalWormMove: the driver must set the selection probability");
    if (!action.supports_bead_action())
        throw std::logic_error("GrandCanonicalWormMove: the action has no grand-canonical bead action");
    prepare(path);
    load_mass(action);
    if (!path.is_open())
        return rng.uniform() < kOpenZ ? attempt_open(path, action, rng)
                                      : attempt_insert(path, action, rng);
    const double u = rng.uniform();
    if (u < kSwap) return attempt_swap(path, action, rng);
    if (u < kSwap + kAdvance) return attempt_advance(path, action, rng);
    if (u < kSwap + kAdvance + kRecede) return attempt_recede(path, action, rng);
    if (u < kSwap + kAdvance + kRecede + kClose) return attempt_close(path, action, rng);
    return attempt_remove(path, action, rng);
}

// Open: a uniform bead of the R M becomes the head and the g beads after it
// go missing (g uniform in [0, L]); no coordinate changes.
bool GrandCanonicalWormMove::attempt_open(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), R = path.active_particles();
    const auto L = static_cast<std::size_t>(max_length_);
    ++proposed_[kOpen];
    if (R == 0) return false;  // no bead to open
    const std::size_t n_beads = R * M;
    const std::size_t pick = uniform_index(rng, n_beads);
    const PathSet::Bead head{pick / M, pick % M};
    const std::size_t g = uniform_index(rng, L + 1);
    PathSet::Bead bead = head;
    for (std::size_t i = 0; i < g; ++i) {
        bead = path.bead_after(bead.particle, bead.slice);
        beads_[i] = bead;
    }
    const PathSet::Bead tail = path.bead_after(bead.particle, bead.slice);
    if (!segment_is_minimum_image(path, head, g + 1)) {  // close draws that frame
        ++frame_rejections_;
        return false;
    }
    initial_constant(path);
    const double removed = bead_actions(path, action, g, false, R);
    double sep[SimulationCell::kMaxDim];
    bead_separation(path, head, tail, sep);
    const double tau_mu = path.tau() * spec_.chemical_potential;
    const double log_ratio = std::log(constant_) - tau_mu * static_cast<double>(g) +
                             std::log(kClose) + std::log(static_cast<double>(n_beads)) +
                             std::log(static_cast<double>(L + 1)) - std::log(p_select_) -
                             std::log(kOpenZ) - log_rho0(sep, g + 1, path);
    if (!accept(log_ratio, -removed, rng, head)) return false;
    path.open_with_gap(head, g);
    ++accepted_[kOpen];
    return true;
}

// Close: a free bridge from the head to the tail over g + 1 links fills the
// gap (g <= L, the gaps open can make).
bool GrandCanonicalWormMove::attempt_close(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), R = path.active_particles(), d = path.dim();
    const auto L = static_cast<std::size_t>(max_length_);
    const bool periodic = path.cell().is_periodic();
    ++proposed_[kCloseMove];
    const std::size_t g = path.gap();
    if (g > L) return false;
    const PathSet::Bead head = path.head(), tail = path.tail();
    PathSet::Bead bead = head;
    for (std::size_t i = 0; i < g; ++i) {
        bead = path.bead_after(bead.particle, bead.slice);
        beads_[i] = bead;
    }
    double sep[SimulationCell::kMaxDim];
    bead_separation(path, head, tail, sep);
    // Levy bridge: bead i given bead i-1 and the end n - i + 1 links away has
    // mean x_{i-1} + (x_n - x_{i-1}) / (n - i + 1) and variance
    // tau (n - i) / ((n - i + 1) m) per axis.
    const std::size_t n = g + 1;
    for (std::size_t dd = 0; dd < d; ++dd) {
        walk_[dd] = path.pos(head.particle, dd, head.slice);
        walk_[n * d + dd] = walk_[dd] + sep[dd];
    }
    for (std::size_t i = 1; i < n; ++i) {
        const double remaining = static_cast<double>(n - i + 1);
        for (std::size_t dd = 0; dd < d; ++dd) {
            const double previous = walk_[(i - 1) * d + dd];
            const double mean = previous + (walk_[n * d + dd] - previous) / remaining;
            const double variance = path.tau() * (remaining - 1.0) / (remaining * mass_[dd]);
            walk_[i * d + dd] = mean + std::sqrt(variance) * rng.gaussian();
            if (!std::isfinite(walk_[i * d + dd])) {
                reject_overflow();
                return false;
            }
        }
    }
    if (!links_are_canonical(path, walk_.data(), n)) {
        ++frame_rejections_;
        return false;
    }
    for (std::size_t i = 0; i < g; ++i) {
        double* r = positions_.data() + i * d;
        for (std::size_t dd = 0; dd < d; ++dd) r[dd] = walk_[(i + 1) * d + dd];
        if (periodic) path.cell().wrap(r);
    }
    const double added = bead_actions(path, action, g, true, R);
    const double tau_mu = path.tau() * spec_.chemical_potential;
    const double log_ratio = std::log(p_select_) + std::log(kOpenZ) + log_rho0(sep, n, path) +
                             tau_mu * static_cast<double>(g) - std::log(constant_) -
                             std::log(kClose) -
                             std::log(static_cast<double>(R) * static_cast<double>(M)) -
                             std::log(static_cast<double>(L + 1));
    if (!accept(log_ratio, added, rng, head)) return false;
    commit(path, g);
    path.close();
    ++accepted_[kCloseMove];
    return true;
}

// Insert: a new row whose tail is uniform in the cell (or drawn from q in an
// open cell) and in slice, continued by a free walk of m links.
bool GrandCanonicalWormMove::attempt_insert(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), N = path.active_particles(), d = path.dim();
    const auto L = static_cast<std::size_t>(max_length_);
    const bool periodic = path.cell().is_periodic();
    ++proposed_[kInsert];
    if (N >= spec_.n_max) {  // the open worm would have n_max + 1 rows
        ++window_rejections_;
        return false;
    }
    const std::size_t q = N, k_tail = uniform_index(rng, M);
    const std::size_t m = 1 + uniform_index(rng, L);
    if (periodic) {
        // r = sum_i f_i a_i with f_i uniform in [0, 1): uniform in the cell.
        const auto lattice = path.cell().vectors();  // [i * d + dd]
        for (std::size_t dd = 0; dd < d; ++dd) walk_[dd] = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            const double f = rng.uniform();
            for (std::size_t dd = 0; dd < d; ++dd) walk_[dd] += f * lattice[i * d + dd];
        }
    } else {
        for (std::size_t dd = 0; dd < d; ++dd)
            walk_[dd] = spec_.insertion_center[dd] + spec_.insertion_width * rng.gaussian();
    }
    const double log_q = log_insertion_density(path, walk_.data());
    for (std::size_t i = 1; i <= m; ++i)
        for (std::size_t dd = 0; dd < d; ++dd) {
            walk_[i * d + dd] = walk_[(i - 1) * d + dd] +
                                std::sqrt(path.tau() / mass_[dd]) * rng.gaussian();
            if (!std::isfinite(walk_[i * d + dd])) {
                reject_overflow();
                return false;
            }
        }
    if (!links_are_canonical(path, walk_.data(), m)) {
        ++frame_rejections_;
        return false;
    }
    for (std::size_t i = 0; i <= m; ++i) {
        beads_[i] = {q, (k_tail + i) % M};
        double* r = positions_.data() + i * d;
        for (std::size_t dd = 0; dd < d; ++dd) r[dd] = walk_[i * d + dd];
        if (periodic) path.cell().wrap(r);
    }
    initial_constant(path);
    // The new beads meet the N closed rows at their slices.
    const double added = bead_actions(path, action, m + 1, true, N);
    const double tau_mu = path.tau() * spec_.chemical_potential;
    const double log_ratio = std::log(constant_) + std::log(kRemove) +
                             std::log(static_cast<double>(M)) + std::log(static_cast<double>(L)) +
                             tau_mu * static_cast<double>(m + 1) - std::log(p_select_) -
                             std::log(1.0 - kOpenZ) - log_q;
    if (!accept(log_ratio, added, rng, beads_[0])) return false;
    if (path.append_row() != q)
        throw std::logic_error("GrandCanonicalWormMove: the new row is not the last active row");
    commit(path, m + 1);
    path.open_with_gap({q, (k_tail + m) % M}, M - m - 1);
    ++accepted_[kInsert];
    return true;
}

// Remove: a worm that is a whole row of its own (a fixed point of pi) with
// m + 1 <= L + 1 beads is deleted.
bool GrandCanonicalWormMove::attempt_remove(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), R = path.active_particles(), d = path.dim();
    const auto L = static_cast<std::size_t>(max_length_);
    ++proposed_[kRemoveMove];
    const PathSet::Bead head = path.head(), tail = path.tail();
    const std::size_t p = head.particle;
    if (path.next_particle(p) != p) return false;  // the worm spans other rows
    const std::size_t worm_beads = M - path.gap();
    if (worm_beads < 2 || worm_beads - 1 > L) return false;  // not an insertable length
    if (R - 1 < spec_.n_min) {
        ++window_rejections_;
        return false;
    }
    const std::size_t m = worm_beads - 1;
    for (std::size_t i = 0; i < worm_beads; ++i) beads_[i] = {p, (tail.slice + i) % M};
    const double removed = bead_actions(path, action, worm_beads, false, R);
    double r_tail[SimulationCell::kMaxDim];
    for (std::size_t dd = 0; dd < d; ++dd) r_tail[dd] = path.pos(p, dd, tail.slice);
    const double tau_mu = path.tau() * spec_.chemical_potential;
    const double log_ratio = std::log(p_select_) + std::log(1.0 - kOpenZ) +
                             log_insertion_density(path, r_tail) -
                             tau_mu * static_cast<double>(m + 1) - std::log(constant_) -
                             std::log(kRemove) - std::log(static_cast<double>(M)) -
                             std::log(static_cast<double>(L));
    if (!accept(log_ratio, -removed, rng, head)) return false;
    path.close();
    path.remove_row(p);
    ++accepted_[kRemoveMove];
    return true;
}

// Advance: m <= g gap beads after the head are drawn as a free walk from it.
bool GrandCanonicalWormMove::attempt_advance(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t R = path.active_particles(), d = path.dim();
    const auto L = static_cast<std::size_t>(max_length_);
    const bool periodic = path.cell().is_periodic();
    ++proposed_[kAdvanceMove];
    const std::size_t g = path.gap(), m = 1 + uniform_index(rng, L);
    if (m > g) return false;  // the head may not pass the tail
    const PathSet::Bead head = path.head();
    PathSet::Bead bead = head;
    for (std::size_t i = 0; i < m; ++i) {
        bead = path.bead_after(bead.particle, bead.slice);
        beads_[i] = bead;
    }
    for (std::size_t dd = 0; dd < d; ++dd) walk_[dd] = path.pos(head.particle, dd, head.slice);
    for (std::size_t i = 1; i <= m; ++i)
        for (std::size_t dd = 0; dd < d; ++dd) {
            walk_[i * d + dd] = walk_[(i - 1) * d + dd] +
                                std::sqrt(path.tau() / mass_[dd]) * rng.gaussian();
            if (!std::isfinite(walk_[i * d + dd])) {
                reject_overflow();
                return false;
            }
        }
    if (!links_are_canonical(path, walk_.data(), m)) {
        ++frame_rejections_;
        return false;
    }
    for (std::size_t i = 0; i < m; ++i) {
        double* r = positions_.data() + i * d;
        for (std::size_t dd = 0; dd < d; ++dd) r[dd] = walk_[(i + 1) * d + dd];
        if (periodic) path.cell().wrap(r);
    }
    const double added = bead_actions(path, action, m, true, R);
    const double log_ratio = path.tau() * spec_.chemical_potential * static_cast<double>(m) +
                             std::log(kRecede / kAdvance);
    if (!accept(log_ratio, added, rng, head)) return false;
    commit(path, m);
    path.set_head(beads_[m - 1]);
    path.set_gap(g - m);
    ++accepted_[kAdvanceMove];
    return true;
}

// Recede: the head and its m - 1 predecessors go missing (g + m <= M - 1, so
// the worm keeps its tail and the gap stays shorter than a row).
bool GrandCanonicalWormMove::attempt_recede(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), R = path.active_particles();
    const auto L = static_cast<std::size_t>(max_length_);
    ++proposed_[kRecedeMove];
    const std::size_t g = path.gap(), m = 1 + uniform_index(rng, L);
    if (g + m > M - 1) return false;
    const PathSet::Bead head = path.head();
    PathSet::Bead bead = head;
    for (std::size_t i = 0; i < m; ++i) {
        beads_[i] = bead;
        bead = path.bead_before(bead.particle, bead.slice);
    }
    const double removed = bead_actions(path, action, m, false, R);
    const double log_ratio = -path.tau() * spec_.chemical_potential * static_cast<double>(m) +
                             std::log(kAdvance / kRecede);
    if (!accept(log_ratio, -removed, rng, head)) return false;
    path.set_head(bead);
    path.set_gap(g + m);
    ++accepted_[kRecedeMove];
    return true;
}

// Swap (BPS): alpha at slice k_h + m_s by heat bath among the beads present
// there; the segment zeta -> alpha is replaced by a free bridge head -> alpha
// and zeta becomes the head, followed by the same gap.
bool GrandCanonicalWormMove::attempt_swap(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), R = path.active_particles(), d = path.dim();
    const bool periodic = path.cell().is_periodic();
    const auto ms = static_cast<std::size_t>(swap_length_);
    const PathSet::Bead head = path.head();
    const std::size_t target_slice = (head.slice + ms) % M;
    const std::size_t missing_at_target = path.missing_row(target_slice);
    ++proposed_[kSwapMove];

    // Heat-bath weights rho0(from, row j at the target; m_s tau) up to their
    // common normalization, pruned to 0 beyond exp(-kMinusLogEpsilon) like
    // CycleTable; a missing bead is never a candidate.
    const auto heat_bath = [&](PathSet::Bead from) {
        double sum = 0.0;
        for (std::size_t j = 0; j < R; ++j) {
            heat_[j] = 0.0;
            if (j == missing_at_target) continue;
            double sep[SimulationCell::kMaxDim];
            bead_separation(path, from, {j, target_slice}, sep);
            double exponent = 0.0;
            for (std::size_t dd = 0; dd < d; ++dd) exponent += mass_[dd] * sep[dd] * sep[dd];
            exponent /= 2.0 * static_cast<double>(ms) * path.tau();
            heat_[j] = exponent > CycleTable::kMinusLogEpsilon ? 0.0 : std::exp(-exponent);
            sum += heat_[j];
        }
        return sum;
    };
    const double sigma_head = heat_bath(head);
    if (!(sigma_head > 0.0)) {
        ++no_candidate_;
        return false;
    }
    double target = rng.uniform() * sigma_head;
    std::size_t chosen = R;
    for (std::size_t j = 0; j < R; ++j) {
        if (heat_[j] <= 0.0) continue;
        chosen = j;  // roundoff leaves the last positive weight chosen
        target -= heat_[j];
        if (target < 0.0) break;
    }
    const PathSet::Bead alpha{chosen, target_slice};

    // The old segment zeta -> alpha must be present beads: tracing back from
    // alpha into the tail would meet the gap (missing beads) or, with no gap,
    // reach the head as zeta.
    PathSet::Bead bead = alpha;
    for (std::size_t i = ms - 1; i-- > 0;) {
        bead = path.bead_before(bead.particle, bead.slice);
        beads_[i] = bead;
        if (path.missing_row(bead.slice) == bead.particle) {
            ++no_candidate_;
            return false;
        }
    }
    const PathSet::Bead zeta = path.bead_before(beads_[0].particle, beads_[0].slice);
    if (same_bead(zeta, head) || path.missing_row(zeta.slice) == zeta.particle) {
        ++no_candidate_;
        return false;
    }
    if (!segment_is_minimum_image(path, zeta, ms)) {  // the reverse bridge's frame
        ++frame_rejections_;
        return false;
    }

    // Free bridge head -> alpha over m_s links (minimum-image frame).
    double sep[SimulationCell::kMaxDim];
    bead_separation(path, head, alpha, sep);
    for (std::size_t dd = 0; dd < d; ++dd) {
        walk_[dd] = path.pos(head.particle, dd, head.slice);
        walk_[ms * d + dd] = walk_[dd] + sep[dd];
    }
    for (std::size_t i = 1; i < ms; ++i) {
        const double remaining = static_cast<double>(ms - i + 1);
        for (std::size_t dd = 0; dd < d; ++dd) {
            const double previous = walk_[(i - 1) * d + dd];
            const double mean = previous + (walk_[ms * d + dd] - previous) / remaining;
            const double variance = path.tau() * (remaining - 1.0) / (remaining * mass_[dd]);
            walk_[i * d + dd] = mean + std::sqrt(variance) * rng.gaussian();
            if (!std::isfinite(walk_[i * d + dd])) {
                reject_overflow();
                return false;
            }
        }
    }
    if (!links_are_canonical(path, walk_.data(), ms)) {
        ++frame_rejections_;
        return false;
    }
    for (std::size_t i = 0; i + 1 < ms; ++i) {
        double* r = positions_.data() + i * d;
        for (std::size_t dd = 0; dd < d; ++dd) r[dd] = walk_[(i + 1) * d + dd];
        if (periodic) path.cell().wrap(r);
    }

    // The reverse heat bath from zeta sees the same beads at the target slice
    // (the proposal does not touch it) and must be able to pick alpha.
    const double sigma_zeta = heat_bath(zeta);
    if (!(heat_[chosen] > 0.0)) {
        ++no_candidate_;
        return false;
    }
    const double delta_s = bead_actions(path, action, ms - 1, true, R) -
                           bead_actions(path, action, ms - 1, false, R);
    if (!accept(std::log(sigma_head) - std::log(sigma_zeta), delta_s, rng, head)) return false;

    commit(path, ms - 1);
    // Reconnect: the head continues into the new beads (zeta's old successor
    // slots) and zeta, the new head, into the gap and the tail.
    const std::size_t old_head_row =
        path.exchange_successors(head.particle, zeta.particle, head.slice);
    path.set_head({old_head_row == head.particle ? zeta.particle : head.particle, head.slice});
    ++accepted_[kSwapMove];
    return true;
}

void GrandCanonicalWormMove::tune_constant(double closed_fraction, double open_fraction,
                                           double target_ratio) {
    if (!tune_ || !(constant_ > 0.0)) return;
    // As WormMove::tune_constant: open/closed is proportional to C at fixed
    // paths; the square root of the needed ratio, clamped to [1/4, 4], damps it.
    double ratio = 4.0;
    if (closed_fraction <= 0.0) ratio = 0.25;
    else if (open_fraction > 0.0)
        ratio = std::clamp(target_ratio * closed_fraction / open_fraction, 0.25, 4.0);
    constant_ *= std::sqrt(ratio);
}

std::vector<std::pair<std::string, double>> GrandCanonicalWormMove::extra_diagnostics() const {
    static constexpr const char* kNames[kKinds] = {"open", "close", "insert", "remove",
                                                   "advance", "recede", "swap"};
    std::vector<std::pair<std::string, double>> out;
    for (std::size_t k = 0; k < kKinds; ++k) {
        out.emplace_back(std::string(kNames[k]) + "_proposals", static_cast<double>(proposed_[k]));
        out.emplace_back(std::string(kNames[k]) + "_accepts", static_cast<double>(accepted_[k]));
    }
    out.emplace_back("window_rejections", static_cast<double>(window_rejections_));
    out.emplace_back("swap_no_candidate", static_cast<double>(no_candidate_));
    out.emplace_back("frame_rejections", static_cast<double>(frame_rejections_));
    return out;
}

void GrandCanonicalWormMove::reset_diagnostics() noexcept {
    Move::reset_diagnostics();
    proposed_.fill(0);
    accepted_.fill(0);
    window_rejections_ = frame_rejections_ = no_candidate_ = 0;
}

}  // namespace alea
