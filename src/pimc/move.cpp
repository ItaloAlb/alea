// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "pimc/move.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>

namespace alea {

namespace {

int checked_max_cycle(int max_cycle) {
    if (max_cycle < 2 || max_cycle > 8)
        throw std::invalid_argument("PermutationMove max_cycle must be in [2,8]");
    return max_cycle;
}

void require_positive_finite(double value, const char* parameter) {
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string(parameter) + " must be finite and > 0");
}

} // namespace

void Move::set_weight(double w) {
    require_positive_finite(w, "Move weight");
    weight_ = w;
}

void Move::reject_nonfinite(double delta_s, std::size_t p, std::size_t k) {
    if (delta_s == std::numeric_limits<double>::infinity()) {
        ++infinite_action_rejections_;
        return;
    }
    const std::string where = name() + " move of particle " + std::to_string(p) +
                              " at slice " + std::to_string(k);
    if (std::isnan(delta_s))
        throw std::runtime_error(
            "pimc: " + where + " produced a NaN action change; an action or "
            "potential component returned NaN for a finite configuration");
    throw std::runtime_error(
        "pimc: " + where + " produced an action change of -inf: the proposed "
        "path has unbounded weight (an attractive singularity that the action "
        "admission did not refuse)");
}

SingleSliceMove::SingleSliceMove(double sigma) : sigma_(sigma) {
    require_positive_finite(sigma, "SingleSliceMove sigma");
}

BisectionMove::BisectionMove(int levels) : levels_(levels) {
    if (levels < 1 || levels > 30)
        throw std::invalid_argument("BisectionMove levels must be in [1,30]");
}

void BisectionMove::prepare(const PathSet& path) {
    const std::size_t n = std::size_t{1} << levels_;
    if (n > path.num_slices())
        throw std::runtime_error("BisectionMove: 2^levels exceeds M");
    bridge_.resize((n + 1) * path.dim());
}

CentroidMove::CentroidMove(double step) : step_(step) {
    require_positive_finite(step, "CentroidMove step");
}

void CentroidMove::prepare(const PathSet& path) {
    // A whole permutation cycle: at most the largest exchange class.
    const auto cycle = path.largest_exchange_class();
    old_pos_.resize(cycle * path.num_slices() * path.dim());
    proposed_pos_.resize(cycle * path.num_slices() * path.dim());
    members_.resize(cycle);
}

// ── SingleSliceMove ─────────────────────────────────────────────────────────

bool SingleSliceMove::attempt(PathSet& path, Action& action,
                              numerics::Random& rng) {
    const std::size_t M  = path.num_slices();
    const std::size_t np = path.n_particles();
    const std::size_t d  = path.dim();
    if (np == 0 || M == 0) return false;

    // Pick random particle and slice.
    const std::size_t p = static_cast<std::size_t>(
        rng.uniform() * static_cast<double>(np));
    const std::size_t k = static_cast<std::size_t>(
        rng.uniform() * static_cast<double>(M));

    // Symmetric Gaussian proposal: log A = -Delta S_K - Delta S_V.
    // Only the moved bead's two links change (Ceperley RMP 67, 279 (1995),
    // primitive action); the finite-tau target is unchanged by delta evaluation.
    // The two incident links follow the worldline: at k = 0 the incoming link
    // starts on pi^{-1}(p), at k = M-1 the outgoing one ends on pi(p).
    const auto before = path.bead_before(p, k);
    const auto after = path.bead_after(p, k);
    const double spring_old =
        action.spring_action(path, before.particle, before.slice, p, k) +
        action.spring_action(path, p, k, after.particle, after.slice);

    // Form the proposal before mutating a bead or its distance table. A finite
    // width times an extreme normal variate can still overflow; wrapping an
    // infinity then propagates NaNs into the entire path.
    double old[SimulationCell::kMaxDim];
    double proposed[SimulationCell::kMaxDim];
    bool finite = true;
    for (std::size_t dd = 0; dd < d; ++dd) {
        old[dd] = path.pos(p, dd, k);
        proposed[dd] = old[dd] + sigma_ * rng.gaussian();
        finite = finite && std::isfinite(proposed[dd]);
    }
    if (!finite) {
        reject_overflow();
        return false;
    }
    if (path.cell().is_periodic()) path.cell().wrap(proposed);
    for (std::size_t dd = 0; dd < d; ++dd)
        if (!std::isfinite(proposed[dd])) {
            reject_overflow();
            return false;
        }
    // All component deltas observe the accepted configuration. A fallback
    // may temporarily evaluate the bead but restores it before returning.
    const double delta_v = action.delta_potential_particle(path, p, k,
        std::span<const double>(proposed, d));
    if (!std::isfinite(delta_v)) {
        reject_nonfinite(delta_v, p, k);  // throws unless +inf
        return false;
    }
    for (std::size_t dd = 0; dd < d; ++dd)
        path.pos(p, dd, k) = proposed[dd];

    const double spring_new =
        action.spring_action(path, before.particle, before.slice, p, k) +
        action.spring_action(path, p, k, after.particle, after.slice);

    // Metropolis.
    const double delta_s = spring_new - spring_old + delta_v;
    if (std::isfinite(delta_s) &&
        (delta_s < 0.0 || rng.uniform() < std::exp(-delta_s))) {
        path.update_particle_tables(p, k, k + 1);
        return true;  // accept
    }

    // Reject: restore old position (before a nonfinite delta_s can throw).
    for (std::size_t dd = 0; dd < d; ++dd)
        path.pos(p, dd, k) = old[dd];
    if (!std::isfinite(delta_s)) reject_nonfinite(delta_s, p, k);
    return false;
}

// ── BisectionMove ───────────────────────────────────────────────────────────

bool BisectionMove::attempt(PathSet& path, Action& action,
                            numerics::Random& rng) {
    const std::size_t M  = path.num_slices();
    const std::size_t np = path.n_particles();
    const std::size_t d  = path.dim();
    if (np == 0 || M == 0) return false;

    prepare(path);
    const int n_beads = 1 << levels_;  // constructor validates the shift

    // Pick random particle and starting slice.
    const std::size_t p = static_cast<std::size_t>(
        rng.uniform() * static_cast<double>(np));
    const std::size_t k_start = static_cast<std::size_t>(
        rng.uniform() * static_cast<double>(M));

    // The bisection covers slices [k_start, k_start + n_beads] mod M.
    // Endpoints k_start and (k_start + n_beads) % M are FIXED.
    // Interior: k_start+1 .. k_start+n_beads-1 (mod M).
    const std::size_t k_left  = k_start;
    const std::size_t k_right = (k_start + static_cast<std::size_t>(n_beads)) % M;
    const bool periodic = path.cell().is_periodic();
    // Window bead i (0..n_beads) lies on the worldline through (p, k_start):
    // on p up to slice M-1, then on pi(p) (n_beads <= M: one crossing at
    // most).  The bridge links are exactly the action's links, permuted or not.
    const std::size_t p_after = path.next_particle(p);
    const auto owner = [&](int i) {
        return k_start + static_cast<std::size_t>(i) < M ? p : p_after;
    };
    const std::size_t p_right = owner(n_beads);

    auto is_min_image = [&](const double* v) {
        double f[SimulationCell::kMaxDim];
        for (std::size_t i = 0; i < d; ++i) f[i] = v[i];
        path.cell().min_image(f);
        for (std::size_t i = 0; i < d; ++i)
            if (!std::isfinite(v[i]) ||
                std::abs(f[i] - v[i]) > 1e-12 * (1.0 + std::abs(v[i])))
                return false;
        return true;
    };

    // Relative lattice frames: F_x is the sum of OLD canonical links. The new
    // frame is F = F_x + sum_i s_i a_i, s in {-1,0,+1}^d, chosen by heat bath
    // over those 3^d candidates with the free-particle weight of the window:
    //   c(s | F_x) = exp[-T(F)] / Z(F_x),   Z(G) = sum_s exp[-T(G + s.a)],
    //   T(F) = sum_d m_d F_d² / (2 n tau)   (n = window slices).
    // This is the image sum of the periodic free propagator (Ceperley, RMP
    // 67, 279 (1995): the kinetic Green's function made periodic) restricted
    // to neighbouring images; standard codes keep the image fixed in local
    // moves (Spada, Giorgini, Pilati, Condens. Matter 7, 30 (2022)) or sample
    // spring windings by their Gaussian weight.  Proposing s uniformly instead
    // wastes 1 - 3^-d of the moves whenever L² >> n tau (8/9 in 2-D), all
    // rejected by T.
    // Successive moves still reach any winding; high-|W| paths can escape.
    // The same construction applies to partial windows with fixed endpoints.
    //
    // Detailed balance (Levy bridge, Ceperley RMP 67, 279 (1995)):
    // q(y|x) = c(s|F_x) C exp[-S_K(y)+T(F)], C frame-independent; the reverse
    // move draws -s with c(-s|F) = exp[-T(F_x)]/Z(F). The T terms cancel:
    //   log A = -Delta S_V + ln Z(F_x) - ln Z(F),
    // exact for any frames, and ~ -Delta S_V when one image dominates.
    // Generated noncanonical links are rejected ONCE, not redrawn:
    // conditioning the bridge until valid would add frame-dependent
    // normalizers and invalidate this ratio. Canonical old links and the
    // relative reverse support are checked below. The target remains the
    // planned minimum-image action, not a heat kernel.
    double sep[SimulationCell::kMaxDim];   // min-image endpoint separation
    double frame[SimulationCell::kMaxDim]; // F: bridge frame (drawn)
    double fx[SimulationCell::kMaxDim];    // F_x: OLD window frame (captured
                                           // before the path is modified —
                                           // the acceptance term needs it)
    double log_z_ratio = 0.0;              // ln Z(F_x) - ln Z(F); 0 when open
    for (std::size_t dd = 0; dd < d; ++dd) {
        sep[dd] = path.pos(p_right, dd, k_right) - path.pos(p, dd, k_left);
        fx[dd] = 0.0;
    }
    if (periodic) path.cell().min_image(sep);

    if (periodic) {
        // F_x: chain the stored min-image links of the CURRENT window.
        double del[SimulationCell::kMaxDim];
        for (int i = 1; i <= n_beads; ++i) {
            const std::size_t kn =
                (k_start + static_cast<std::size_t>(i)) % M;
            const std::size_t kp =
                (k_start + static_cast<std::size_t>(i - 1)) % M;
            for (std::size_t dd = 0; dd < d; ++dd)
                del[dd] = path.pos(owner(i), dd, kn) -
                          path.pos(owner(i - 1), dd, kp);
            path.cell().min_image(del);
            for (std::size_t dd = 0; dd < d; ++dd) fx[dd] += del[dd];
        }
        // F_x and the fixed endpoint separation must differ by an integer
        // lattice vector, with NO cap on its magnitude. Reconstruct F_x from
        // integers to avoid accumulation drift in successive frame updates.
        double diff[SimulationCell::kMaxDim];
        double frac[SimulationCell::kMaxDim];
        for (std::size_t dd = 0; dd < d; ++dd) diff[dd] = fx[dd] - sep[dd];
        path.cell().fractional(diff, frac);
        const auto vec = path.cell().vectors();
        for (std::size_t dd = 0; dd < d; ++dd) fx[dd] = sep[dd];
        for (std::size_t i = 0; i < d; ++i) {
            const double integer = std::nearbyint(frac[i]);
            if (!std::isfinite(integer) || std::abs(frac[i] - integer) > 1e-6)
                return false;
            for (std::size_t dd = 0; dd < d; ++dd)
                fx[dd] += integer * vec[i * d + dd];
        }
        // Heat-bath frame: T of the 3^d candidates around a centre G, as
        // ln Z(G) (log-sum-exp) plus the candidates' weights for the draw.
        const double inv_two_n_tau =
            1.0 / (2.0 * path.tau() * static_cast<double>(n_beads));
        std::size_t n_cand = 1;
        for (std::size_t i = 0; i < d; ++i) n_cand *= 3;
        double weight[SimulationCell::kMaxCand];   // exp[-(T - T_min)]
        auto candidate = [&](const double* centre, std::size_t c, double* out) {
            for (std::size_t dd = 0; dd < d; ++dd) out[dd] = centre[dd];
            for (std::size_t i = 0; i < d; ++i, c /= 3) {
                const double s = static_cast<double>(c % 3) - 1.0;
                for (std::size_t dd = 0; dd < d; ++dd) out[dd] += s * vec[i * d + dd];
            }
        };
        auto log_z = [&](const double* centre) {
            double t[SimulationCell::kMaxCand];
            double t_min = std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < n_cand; ++c) {
                double f[SimulationCell::kMaxDim];
                candidate(centre, c, f);
                t[c] = 0.0;
                for (std::size_t dd = 0; dd < d; ++dd)
                    t[c] += action.mass(p, dd) * f[dd] * f[dd] * inv_two_n_tau;
                t_min = std::min(t_min, t[c]);
            }
            double sum = 0.0;
            for (std::size_t c = 0; c < n_cand; ++c) {
                weight[c] = std::exp(-(t[c] - t_min));
                sum += weight[c];
            }
            return std::log(sum) - t_min;
        };
        const double log_z_old = log_z(fx);
        if (!std::isfinite(log_z_old)) return false;
        double total = 0.0;
        for (std::size_t c = 0; c < n_cand; ++c) total += weight[c];
        double u = rng.uniform() * total;
        std::size_t chosen = n_cand - 1;
        for (std::size_t c = 0; c < n_cand; ++c) {
            if (u < weight[c]) { chosen = c; break; }
            u -= weight[c];
        }
        candidate(fx, chosen, frame);
        const double log_z_new = log_z(frame);
        if (!std::isfinite(log_z_new)) return false;
        log_z_ratio = log_z_old - log_z_new;
        // Explicit reverse-support check: F_x-F is an integer increment in
        // the same {-1,0,+1} support. This is relative, never |F_x| <= one cell.
        for (std::size_t dd = 0; dd < d; ++dd) diff[dd] = fx[dd] - frame[dd];
        path.cell().fractional(diff, frac);
        for (std::size_t i = 0; i < d; ++i) {
            const double integer = std::nearbyint(frac[i]);
            if (!std::isfinite(integer) || std::abs(frac[i] - integer) > 1e-6 ||
                std::abs(integer) > 1.0)
                return false;
        }
    } else {
        // Open cell: one frame (the raw separation) — draws would be
        // asymmetric with nothing to draw them in the reverse direction.
        for (std::size_t dd = 0; dd < d; ++dd) frame[dd] = sep[dd];
    }

    // ── Levy bridge in the drawn frame ───────────────────────────────────────
    // u[0] = x_left, u[n] = x_left + F; interior beads drawn level by level
    // with conditional variance tau_level/(2 m).  The path is NOT touched
    // until the whole bridge passes check (a).
    auto at = [&](int i, std::size_t dd) -> double& {
        return bridge_[static_cast<std::size_t>(i) * d + dd];
    };
    for (std::size_t dd = 0; dd < d; ++dd) {
        at(0, dd) = path.pos(p, dd, k_left);
        at(n_beads, dd) = at(0, dd) + frame[dd];
        if (!std::isfinite(at(0, dd)) || !std::isfinite(at(n_beads, dd))) {
            reject_overflow();
            return false;
        }
    }

    const double tau = path.tau();
    bool consistent = true;
    for (int level = levels_; level >= 1; --level) {
        const int stride = 1 << level;
        const int half   = stride / 2;
        const double tau_level = static_cast<double>(half) * tau;
        for (int seg = 0; seg < n_beads; seg += stride) {
            for (std::size_t dd = 0; dd < d; ++dd) {
                // Conditional variance of a free-particle midpoint given two
                // endpoints half a `tau_level` away on each side.
                const double sigma =
                    std::sqrt(tau_level / (2.0 * action.mass(p, dd)));
                // Halving before addition avoids overflowing a midpoint of
                // two large, finite, same-sign endpoint coordinates.
                at(seg + half, dd) =
                    0.5 * at(seg, dd) + 0.5 * at(seg + stride, dd) +
                    sigma * rng.gaussian();
                if (!std::isfinite(at(seg + half, dd))) {
                    reject_overflow();
                    return false;
                }
            }
        }
    }

    // (a) every CONSECUTIVE link of the finished bridge must be its own
    // min-image: only then does the stored spring action equal the bridge's
    // quadratic form and the kinetic term cancel the proposal density.
    if (periodic) {
        for (int i = 0; i < n_beads && consistent; ++i) {
            double lft[SimulationCell::kMaxDim];
            for (std::size_t dd = 0; dd < d; ++dd)
                lft[dd] = at(i + 1, dd) - at(i, dd);
            consistent = is_min_image(lft);
        }
    }
    if (!consistent) return false;  // new window not canonically linked

    // Canonicalize the staged interior before changing any path coordinate.
    // This also makes a bad cell transform reject without partial commits.
    if (periodic) {
        for (int i = 1; i < n_beads; ++i) {
            double r[SimulationCell::kMaxDim];
            for (std::size_t dd = 0; dd < d; ++dd) r[dd] = at(i, dd);
            path.cell().wrap(r);
            for (std::size_t dd = 0; dd < d; ++dd) {
                if (!std::isfinite(r[dd])) {
                    reject_overflow();
                    return false;
                }
                at(i, dd) = r[dd];
            }
        }
    }

    // A pair action couples adjacent slices, so both proposed endpoints must
    // enter the same window evaluation. The action restores accepted coordinates
    // and tables before returning or throwing; rejection leaves no partial bridge.
    const int n_interior = n_beads - 1;
    const std::size_t first_interior = (k_start + 1) % M;
    // The window API follows the worldline from its first bead's owner.
    const std::size_t first_owner = owner(1);
    const double delta_v = action.delta_potential_window(
        path, first_owner, first_interior,
        std::span<const double>(bridge_.data() + d,
                                static_cast<std::size_t>(n_interior) * d));
    if (!std::isfinite(delta_v)) {
        reject_nonfinite(delta_v, p, first_interior);  // throws unless +inf
        return false;
    }

    // Refresh the moved beads over the interior window only. A wrapping
    // window is split into at most two contiguous ranges, the second on the
    // next particle of the worldline.
    auto refresh_window = [&]() {
        const std::size_t first = (k_start + 1) % M;
        const std::size_t count = static_cast<std::size_t>(n_interior);
        const std::size_t end = std::min(M, first + count);
        path.update_particle_tables(first_owner, first, end);
        if (first + count > M)
            path.update_particle_tables(path.next_particle(first_owner), 0,
                                        first + count - M);
    };
    // Metropolis–Hastings: within one frame the Levy density and the kinetic
    // spring cancel identically (bridge identity); across frames the
    // endpoint term exp[-T(F) + T(F_x)] cancels against the heat-bath
    // selection, leaving
    //   log A = -dS_V + ln Z(F_x) - ln Z(F).
    // For a full-ring window (n = M) the stationary winding weight is still
    // pi(W) ~ exp(-m L^2 W^2 / 2 beta): the target is unchanged, only the
    // proposal is.  Open cells: frame == F_x and the factor is identically 1.
    const double log_a = -delta_v + log_z_ratio;
    if (std::isfinite(log_a) &&
        (log_a >= 0.0 || rng.uniform() < std::exp(log_a))) {
        for (int i = 1; i < n_beads; ++i) {
            const std::size_t k = (k_start + static_cast<std::size_t>(i)) % M;
            for (std::size_t dd = 0; dd < d; ++dd)
                path.pos(owner(i), dd, k) = at(i, dd);
        }
        refresh_window();
        return true;  // accept
    }

    // Delta S = -log A; an overflowing frame term is an infinite action.
    if (!std::isfinite(log_a)) reject_nonfinite(-log_a, p, first_interior);
    return false;
}

// ── CentroidMove ────────────────────────────────────────────────────────────

bool CentroidMove::attempt(PathSet& path, Action& action,
                            numerics::Random& rng) {
    const std::size_t M  = path.num_slices();
    const std::size_t np = path.n_particles();
    const std::size_t d  = path.dim();
    if (np == 0 || M == 0) return false;

    // Pick random particle.
    const std::size_t p = static_cast<std::size_t>(
        rng.uniform() * static_cast<double>(np));

    // The rigid body is p's whole permutation cycle: shifting one member
    // alone would stretch its links to the others.  A cycle of length L is
    // chosen with probability L/np in both directions (the shift leaves pi
    // unchanged), so the proposal stays symmetric.  Beads are staged along
    // the worldline from (p, 0): member j = pi^j(p) holds beads [j*M, j*M+M).
    prepare(path);
    std::size_t length = 0;
    for (std::size_t q = p; length == 0 || q != p; q = path.next_particle(q))
        members_[length++] = q;
    const std::size_t count = length * M;

    // A mathematically rigid shift preserves the kinetic links, including
    // minimum-image links after wrapping. Check that finite-precision storage
    // actually retains this invariant before using potential-only acceptance.
    double spring_old = 0.0;
    for (std::size_t j = 0; j < length; ++j)
        for (std::size_t k = 0; k < M; ++k)
            spring_old += action.link_action(path, members_[j], k);

    // Saving exact bead values is essential: adding and subtracting a large
    // shift is not an inverse in floating point, particularly after wrapping.
    for (std::size_t j = 0; j < length; ++j)
        for (std::size_t k = 0; k < M; ++k)
            for (std::size_t dd = 0; dd < d; ++dd)
                old_pos_[(j * M + k) * d + dd] = path.pos(members_[j], dd, k);

    // Stage the entire move in reusable scratch. Reject nonfinite proposals
    // before touching coordinates or tables (even if a potential would throw).
    double shift[SimulationCell::kMaxDim];
    bool finite = true;
    for (std::size_t dd = 0; dd < d; ++dd) {
        shift[dd] = step_ * rng.gaussian();
        finite = finite && std::isfinite(shift[dd]);
    }
    if (!finite) {
        reject_overflow();
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        double r[SimulationCell::kMaxDim];
        for (std::size_t dd = 0; dd < d; ++dd) {
            r[dd] = old_pos_[i * d + dd] + shift[dd];
            finite = finite && std::isfinite(r[dd]);
        }
        if (finite && path.cell().is_periodic()) path.cell().wrap(r);
        for (std::size_t dd = 0; dd < d; ++dd) {
            finite = finite && std::isfinite(r[dd]);
            proposed_pos_[i * d + dd] = r[dd];
        }
        if (!finite) {
            reject_overflow();
            return false;
        }
    }

    // A symmetric rigid translation has log A = -Delta S_V because the
    // kinetic links cancel. Evaluate the full staged window so a pair action
    // sees both proposed endpoints of every link, preserving detailed balance;
    // a cycle window (L > 1) moves L beads per slice and telescopes exactly.
    const double delta_v = action.delta_potential_window(
        path, p, 0, std::span<const double>(proposed_pos_.data(), count * d));
    if (!std::isfinite(delta_v)) {
        reject_nonfinite(delta_v, p, 0);  // throws unless +inf
        return false;
    }

    for (std::size_t j = 0; j < length; ++j)
        for (std::size_t k = 0; k < M; ++k)
            for (std::size_t dd = 0; dd < d; ++dd)
                path.pos(members_[j], dd, k) = proposed_pos_[(j * M + k) * d + dd];

    double spring_new = 0.0;
    for (std::size_t j = 0; j < length; ++j)
        for (std::size_t k = 0; k < M; ++k)
            spring_new += action.link_action(path, members_[j], k);
    // A lost bead separation (or a nonfinite link) invalidates the assumed
    // kinetic cancellation. Reject rather than accepting a different action.
    // The tolerance allows normal binary64 roundoff in a truly rigid shift.
    const bool rigid = std::isfinite(spring_old) && std::isfinite(spring_new) &&
        std::abs(spring_new - spring_old) <= 1e-12 * (1.0 + spring_old);
    if (rigid) {
        if (delta_v < 0.0 || rng.uniform() < std::exp(-delta_v)) {
            // Never wrap another particle as an incidental side effect.
            for (std::size_t j = 0; j < length; ++j)
                path.update_particle_tables(members_[j], 0, M);
            return true;
        }
    }

    // Reject: exact restore, with no wrap or subtractive roundoff.
    for (std::size_t j = 0; j < length; ++j)
        for (std::size_t k = 0; k < M; ++k)
            for (std::size_t dd = 0; dd < d; ++dd)
                path.pos(members_[j], dd, k) = old_pos_[(j * M + k) * d + dd];
    if (!std::isfinite(spring_new)) reject_overflow();
    return false;
}

// ── CycleTable ──────────────────────────────────────────────────────────────

void CycleTable::prepare(std::size_t max_members, std::size_t max_cycle) {
    table_.resize(max_members * max_members);
    sigma_.resize(max_members);
    old_ok_.resize(max_members);
    adj_start_.resize(max_members + 1);
    adj_.resize(max_members * max_members);
    stack_.resize(max_cycle);
}

void CycleTable::build(const PathSet& path, std::span<const std::size_t> members,
                       std::size_t k0, std::size_t k1, std::size_t n,
                       const double* mass, std::size_t max_cycle) {
    const std::size_t d = path.dim();
    const bool periodic = path.cell().is_periodic();
    n_ = members.size();
    length_limit_ = std::min(max_cycle, n_);
    // pi maps the class onto itself: sigma(i) is the member index of pi(m_i).
    for (std::size_t i = 0; i < n_; ++i)
        sigma_[i] = static_cast<std::size_t>(
            std::lower_bound(members.begin(), members.end(),
                             path.next_particle(members[i])) - members.begin());
    // Free-propagator table between the fixed endpoints (common normalization
    // dropped; minimum-image separations under PBC); 0 beyond the pruning radius.
    for (std::size_t i = 0; i < n_; ++i)
        for (std::size_t j = 0; j < n_; ++j) {
            double sep[SimulationCell::kMaxDim];
            for (std::size_t dd = 0; dd < d; ++dd)
                sep[dd] = path.pos(members[j], dd, k1) - path.pos(members[i], dd, k0);
            if (periodic) path.cell().min_image(sep);
            double exponent = 0.0;
            for (std::size_t dd = 0; dd < d; ++dd)
                exponent += mass[dd] * sep[dd] * sep[dd];
            exponent /= 2.0 * static_cast<double>(n) * path.tau();
            table_[i * n_ + j] = exponent > kMinusLogEpsilon ? 0.0 : std::exp(-exponent);
        }
    build_adjacency();
}

void CycleTable::build_adjacency() {
    for (std::size_t i = 0; i < n_; ++i)
        old_ok_[i] = t(i, sigma_[i]) >= kEpsilon;
    std::size_t edge_count = 0;
    for (std::size_t i = 0; i < n_; ++i) {
        adj_start_[i] = edge_count;
        if (!old_ok_[i]) continue;
        for (std::size_t j = 0; j < n_; ++j)
            if (edge(i, j)) adj_[edge_count++] = j;
    }
    adj_start_[n_] = edge_count;
}

double CycleTable::rotation_weight(const std::size_t* sequence,
                                   std::size_t length) const noexcept {
    double w = 1.0;
    for (std::size_t m = 0; m < length; ++m) {
        const std::size_t i = sequence[m], next = sequence[(m + 1) % length];
        w *= t(i, sigma_[next]) / t(i, sigma_[i]);
    }
    return w;
}

// Each cycle rooted at `root` (its smallest member) is visited once, as the
// sequence that starts at root; every cyclic order of the others is a
// distinct permutation.  The weight multiplies t(new link)/t(old link) for
// each member (i -> target of the next member).
template <class Visit>
bool CycleTable::for_each_cycle_from(std::size_t root, Visit&& visit) {
    if (!old_ok_[root]) return false;
    stack_[0] = root;
    return extend(1, 1.0, visit);
}

template <class Visit>
bool CycleTable::extend(std::size_t depth, double weight, Visit& visit) {
    const std::size_t start = stack_[0], last = stack_[depth - 1];
    const double old_link = t(last, sigma_[last]);
    for (std::size_t e = adj_start_[last]; e < adj_start_[last + 1]; ++e) {
        const std::size_t j = adj_[e];
        if (j == start) {
            if (depth >= 2 &&
                visit(stack_.data(), depth, weight * t(last, sigma_[start]) / old_link))
                return true;
            continue;
        }
        if (j < start || depth == length_limit_) continue;
        bool used = false;
        for (std::size_t m = 1; m < depth; ++m) used = used || stack_[m] == j;
        if (used) continue;
        stack_[depth] = j;
        if (extend(depth + 1, weight * t(last, sigma_[j]) / old_link, visit))
            return true;
    }
    return false;
}

double CycleTable::cycle_sum_from(std::size_t root) {
    double q = 1.0;
    for_each_cycle_from(root, [&](const std::size_t*, std::size_t, double w) {
        q += w;
        return false;
    });
    return q;
}

namespace {

// Reversibility: the reverse move redraws an old window in the minimum-image
// frame, so it can only regenerate windows whose summed links equal that
// frame.  `left` holds slices k0..M-1 of the window, `right` slices 0..k1.
bool old_frame_is_minimum_image(const PathSet& path, std::size_t left,
                                std::size_t right, std::size_t k0,
                                std::size_t k1, std::size_t n) {
    const std::size_t M = path.num_slices(), d = path.dim();
    double frame[SimulationCell::kMaxDim] = {}, sep[SimulationCell::kMaxDim];
    for (std::size_t t = k0; t < k0 + n; ++t) {
        const std::size_t p0 = t < M ? left : right, p1 = t + 1 < M ? left : right;
        double link[SimulationCell::kMaxDim];
        for (std::size_t dd = 0; dd < d; ++dd)
            link[dd] = path.pos(p1, dd, (t + 1) % M) - path.pos(p0, dd, t % M);
        path.cell().min_image(link);
        for (std::size_t dd = 0; dd < d; ++dd) frame[dd] += link[dd];
    }
    for (std::size_t dd = 0; dd < d; ++dd)
        sep[dd] = path.pos(right, dd, k1) - path.pos(left, dd, k0);
    path.cell().min_image(sep);
    double offset[SimulationCell::kMaxDim] = {}, frac[SimulationCell::kMaxDim] = {};
    for (std::size_t dd = 0; dd < d; ++dd) offset[dd] = frame[dd] - sep[dd];
    path.cell().fractional(offset, frac);
    for (std::size_t dd = 0; dd < d; ++dd)
        if (!(std::abs(frac[dd]) < 1e-6)) return false;
    return true;
}

enum class BridgeResult { ok, overflow, noncanonical };

// Free Levy bridge from (left, k0) to (right, k1) in the minimum-image frame
// (Ceperley, RMP 67, 279 (1995), Sec. V.D), level by level as in
// BisectionMove: a midpoint between endpoints h links away on each side has
// conditional variance h tau / (2 m).  The n-1 interior beads are staged into
// beads/positions (wrapped under PBC; owner `left` before slice M, `right`
// after).  Every bridge link must be its own minimum image, so the stored
// spring action equals the bridge's quadratic form.
BridgeResult stage_bridge(const PathSet& path, std::size_t left, std::size_t right,
                          std::size_t k0, std::size_t k1, std::size_t n, int levels,
                          const double* mass, numerics::Random& rng,
                          std::vector<double>& bridge, PathSet::Bead* beads,
                          double* positions) {
    const std::size_t M = path.num_slices(), d = path.dim();
    const bool periodic = path.cell().is_periodic();
    const auto at = [&](std::size_t i, std::size_t dd) -> double& { return bridge[i * d + dd]; };
    double sep[SimulationCell::kMaxDim];
    for (std::size_t dd = 0; dd < d; ++dd)
        sep[dd] = path.pos(right, dd, k1) - path.pos(left, dd, k0);
    if (periodic) path.cell().min_image(sep);
    for (std::size_t dd = 0; dd < d; ++dd) {
        at(0, dd) = path.pos(left, dd, k0);
        at(n, dd) = at(0, dd) + sep[dd];
    }
    for (int level = levels; level >= 1; --level) {
        const std::size_t stride = std::size_t{1} << level, half = stride / 2;
        const double tau_level = static_cast<double>(half) * path.tau();
        for (std::size_t seg = 0; seg < n; seg += stride)
            for (std::size_t dd = 0; dd < d; ++dd) {
                const double sigma = std::sqrt(tau_level / (2.0 * mass[dd]));
                at(seg + half, dd) = 0.5 * at(seg, dd) + 0.5 * at(seg + stride, dd) +
                                     sigma * rng.gaussian();
                if (!std::isfinite(at(seg + half, dd))) return BridgeResult::overflow;
            }
    }
    if (periodic) {
        for (std::size_t i = 0; i < n; ++i) {
            double link[SimulationCell::kMaxDim], folded[SimulationCell::kMaxDim];
            for (std::size_t dd = 0; dd < d; ++dd) {
                link[dd] = at(i + 1, dd) - at(i, dd);
                folded[dd] = link[dd];
            }
            path.cell().min_image(folded);
            for (std::size_t dd = 0; dd < d; ++dd)
                if (std::abs(folded[dd] - link[dd]) > 1e-12 * (1.0 + std::abs(link[dd])))
                    return BridgeResult::noncanonical;
        }
    }
    for (std::size_t i = 1; i < n; ++i) {
        double r[SimulationCell::kMaxDim];
        for (std::size_t dd = 0; dd < d; ++dd) r[dd] = at(i, dd);
        if (periodic) path.cell().wrap(r);
        const std::size_t t = k0 + i;
        beads[i - 1] = {t < M ? left : right, t % M};
        for (std::size_t dd = 0; dd < d; ++dd) positions[(i - 1) * d + dd] = r[dd];
    }
    return BridgeResult::ok;
}

// Commit staged beads (positions and per-bead tables).
void commit_beads(PathSet& path, std::span<const PathSet::Bead> beads,
                  std::span<const double> positions) {
    const std::size_t d = path.dim();
    for (std::size_t i = 0; i < beads.size(); ++i) {
        const auto [q, k] = beads[i];
        for (std::size_t dd = 0; dd < d; ++dd) path.pos(q, dd, k) = positions[i * d + dd];
        path.update_particle_tables(q, k, k + 1);
    }
}

std::size_t uniform_index(numerics::Random& rng, std::size_t count) {
    return std::min(count - 1, static_cast<std::size_t>(rng.uniform() * static_cast<double>(count)));
}

} // namespace

// ── PermutationMove ─────────────────────────────────────────────────────────

PermutationMove::PermutationMove(int levels, int max_cycle)
    : levels_(levels), max_cycle_(checked_max_cycle(max_cycle)),
      proposed_(static_cast<std::size_t>(max_cycle_) + 1, 0),
      accepted_(static_cast<std::size_t>(max_cycle_) + 1, 0) {
    if (levels < 1 || levels > 30)
        throw std::invalid_argument("PermutationMove levels must be in [1,30]");
}

void PermutationMove::prepare(const PathSet& path) {
    const std::size_t n = std::size_t{1} << levels_;
    if (n > path.num_slices())
        throw std::runtime_error("PermutationMove: 2^levels exceeds M");
    const auto cycle = static_cast<std::size_t>(max_cycle_);
    table_.prepare(path.largest_exchange_class(), cycle);
    chosen_.resize(cycle);
    bridge_.resize((n + 1) * path.dim());
    beads_.resize(cycle * (n - 1));
    positions_.resize(cycle * (n - 1) * path.dim());
    pi_new_.resize(path.n_particles());
}

bool PermutationMove::attempt(PathSet& path, Action& action,
                              numerics::Random& rng) {
    const auto classes = path.exchangeable_classes();
    if (classes.empty()) return false;
    prepare(path);
    const std::size_t M = path.num_slices(), d = path.dim();
    const std::size_t n = std::size_t{1} << levels_;
    const bool periodic = path.cell().is_periodic();

    // Class and window, uniform and independent of the state, so the reverse
    // move draws them with the same probability.  The window [k0, k0 + n]
    // contains the closing link M-1 -> 0 and ends at slice k1 of the target.
    const auto& members = classes[uniform_index(rng, classes.size())];
    const std::size_t nc = members.size();
    const std::size_t k0 = M - n + uniform_index(rng, n);
    const std::size_t k1 = k0 + n - M;
    // One mass tensor per class (the driver checks it when building classes).
    double mass[SimulationCell::kMaxDim];
    for (std::size_t dd = 0; dd < d; ++dd) mass[dd] = action.mass(members.front(), dd);
    table_.build(path, members, k0, k1, n, mass, static_cast<std::size_t>(max_cycle_));

    // Rooted heat bath: a uniform member, then a cycle rooted at it (the
    // identity, no change, with probability 1/Q_root).
    const std::size_t root = uniform_index(rng, nc);
    const double q_root = table_.cycle_sum_from(root);
    double target = rng.uniform() * q_root;
    if (target < 1.0) {
        ++identity_draws_;
        return false;
    }
    target -= 1.0;
    std::size_t length = 0;
    double w_chosen = 0.0;
    table_.for_each_cycle_from(root, [&](const std::size_t* cycle, std::size_t l, double w) {
        std::copy(cycle, cycle + l, chosen_.begin());
        length = l;
        w_chosen = w;
        target -= w;
        return target < 0.0;  // roundoff leaves the last cycle chosen
    });
    if (length < 2) return false;  // unreachable: q_root > 1 implies a cycle
    ++proposed_[length];

    // sigma' rotates the targets: member m now ends at the next one's target.
    std::array<std::size_t, 8> old_target{}, new_target{};
    for (std::size_t m = 0; m < length; ++m) old_target[m] = table_.sigma(chosen_[m]);
    for (std::size_t m = 0; m < length; ++m) new_target[m] = old_target[(m + 1) % length];

    if (periodic)
        for (std::size_t m = 0; m < length; ++m)
            if (!old_frame_is_minimum_image(path, members[chosen_[m]],
                                            members[old_target[m]], k0, k1, n)) {
                ++frame_rejections_;
                return false;
            }
    for (std::size_t m = 0; m < length; ++m) {
        const auto result = stage_bridge(
            path, members[chosen_[m]], members[new_target[m]], k0, k1, n, levels_,
            mass, rng, bridge_, beads_.data() + m * (n - 1),
            positions_.data() + m * (n - 1) * d);
        if (result == BridgeResult::overflow) {
            reject_overflow();
            return false;
        }
        if (result == BridgeResult::noncanonical) {
            ++frame_rejections_;
            return false;
        }
    }

    const std::size_t count = length * (n - 1);
    const std::span<const PathSet::Bead> beads(beads_.data(), count);
    const std::span<const double> positions(positions_.data(), count * d);
    const double delta_v = action.delta_potential_beads(path, beads, positions);
    if (!std::isfinite(delta_v)) {
        reject_nonfinite(delta_v, members[chosen_[0]], k0);  // throws unless +inf
        return false;
    }

    // Q at the proposed state: the same endpoint table with targets sigma'.
    for (std::size_t m = 0; m < length; ++m) table_.set_sigma(chosen_[m], new_target[m]);
    table_.build_adjacency();
    const double q_root_new = table_.cycle_sum_from(root);  // C^{-1} is rooted at root too

    const double log_a = -delta_v - std::log(w_chosen) +
                         std::log(q_root) - std::log(q_root_new);
    if (!(std::isfinite(log_a) && (log_a >= 0.0 || rng.uniform() < std::exp(log_a))))
        return false;

    commit_beads(path, beads, positions);
    const auto pi = path.permutation();
    std::copy(pi.begin(), pi.end(), pi_new_.begin());
    for (std::size_t m = 0; m < length; ++m)
        pi_new_[members[chosen_[m]]] = members[new_target[m]];
    path.set_permutation(pi_new_);
    ++accepted_[length];
    return true;
}

std::vector<std::pair<std::string, double>> PermutationMove::extra_diagnostics() const {
    std::vector<std::pair<std::string, double>> out;
    for (std::size_t l = 2; l < proposed_.size(); ++l) {
        out.emplace_back("cycle" + std::to_string(l) + "_proposals",
                         static_cast<double>(proposed_[l]));
        out.emplace_back("cycle" + std::to_string(l) + "_accepts",
                         static_cast<double>(accepted_[l]));
    }
    out.emplace_back("identity_draws", static_cast<double>(identity_draws_));
    out.emplace_back("frame_rejections", static_cast<double>(frame_rejections_));
    return out;
}

void PermutationMove::reset_diagnostics() noexcept {
    Move::reset_diagnostics();
    std::fill(proposed_.begin(), proposed_.end(), 0);
    std::fill(accepted_.begin(), accepted_.end(), 0);
    identity_draws_ = frame_rejections_ = 0;
}

// ── JointPermutationMove ────────────────────────────────────────────────────

JointPermutationMove::JointPermutationMove(int levels, int max_cycle,
                                           double pair_radius,
                                           std::vector<ClassPair> class_pairs)
    : levels_(levels), max_cycle_(checked_max_cycle(max_cycle)),
      pair_radius_(pair_radius), class_pairs_(std::move(class_pairs)),
      proposed_(static_cast<std::size_t>(max_cycle_) + 1, 0),
      accepted_(static_cast<std::size_t>(max_cycle_) + 1, 0) {
    if (levels < 1 || levels > 30)
        throw std::invalid_argument("JointPermutationMove levels must be in [1,30]");
    if (!std::isfinite(pair_radius) || !(pair_radius > 0.0))
        throw std::invalid_argument("JointPermutationMove pair_radius must be finite and > 0");
    for (const auto& [a, b] : class_pairs_)
        if (a.size() < 2 || b.size() < 2 || !std::is_sorted(a.begin(), a.end()) ||
            !std::is_sorted(b.begin(), b.end()))
            throw std::invalid_argument(
                "JointPermutationMove: each class needs two or more sorted members");
}

void JointPermutationMove::prepare(const PathSet& path) {
    const std::size_t n = std::size_t{1} << levels_;
    if (n > path.num_slices())
        throw std::runtime_error("JointPermutationMove: 2^levels exceeds M");
    const auto cycle = static_cast<std::size_t>(max_cycle_);
    const std::size_t largest = path.largest_exchange_class();
    table_a_.prepare(largest, cycle);
    table_b_.prepare(largest, cycle);
    chosen_a_.resize(cycle);
    chosen_b_.resize(cycle);
    reversed_.resize(cycle);
    partner_.resize(cycle);
    candidate_start_.resize(cycle + 1);
    candidate_.resize(cycle * largest);
    candidate_g_.resize(cycle * largest);
    bridge_.resize((n + 1) * path.dim());
    beads_.resize(2 * cycle * (n - 1));
    positions_.resize(2 * cycle * (n - 1) * path.dim());
    pi_new_.resize(path.n_particles());
}

void JointPermutationMove::build_candidates(const PathSet& path, const ClassPair& pair,
                                            const std::size_t* sequence,
                                            std::size_t length, std::size_t k0) {
    const std::size_t d = path.dim();
    const bool periodic = path.cell().is_periodic();
    const double inv_two_lambda2 = 1.0 / (2.0 * pair_radius_ * pair_radius_);
    std::size_t count = 0;
    for (std::size_t m = 0; m < length; ++m) {
        candidate_start_[m] = count;
        const std::size_t a = pair.first[sequence[m]];
        for (std::size_t j = 0; j < pair.second.size(); ++j) {
            double sep[SimulationCell::kMaxDim];
            for (std::size_t dd = 0; dd < d; ++dd)
                sep[dd] = path.pos(pair.second[j], dd, k0) - path.pos(a, dd, k0);
            if (periodic) path.cell().min_image(sep);
            double r2 = 0.0;
            for (std::size_t dd = 0; dd < d; ++dd) r2 += sep[dd] * sep[dd];
            const double exponent = r2 * inv_two_lambda2;
            if (exponent > CycleTable::kMinusLogEpsilon) continue;
            candidate_[count] = j;
            candidate_g_[count] = std::exp(-exponent);
            ++count;
        }
    }
    candidate_start_[length] = count;
}

template <class Visit>
bool JointPermutationMove::for_each_partner_cycle(std::size_t length, Visit&& visit) {
    return extend_partner(0, length, 1.0, visit);
}

// B sequences position by position: b_m among the candidates of a_m, all
// distinct, each consecutive new link (b_m -> target of b_{m+1}) admissible,
// and the closing link b_L -> target of b_1 too.
template <class Visit>
bool JointPermutationMove::extend_partner(std::size_t depth, std::size_t length,
                                          double g, Visit& visit) {
    for (std::size_t c = candidate_start_[depth]; c < candidate_start_[depth + 1]; ++c) {
        const std::size_t j = candidate_[c];
        if (!table_b_.old_ok(j)) continue;
        bool used = false;
        for (std::size_t m = 0; m < depth; ++m) used = used || partner_[m] == j;
        if (used) continue;
        if (depth > 0 && !table_b_.edge(partner_[depth - 1], j)) continue;
        partner_[depth] = j;
        const double g_next = g * candidate_g_[c];
        if (depth + 1 == length) {
            if (!table_b_.edge(j, partner_[0])) continue;
            if (visit(partner_.data(), table_b_.rotation_weight(partner_.data(), length), g_next))
                return true;
        } else if (extend_partner(depth + 1, length, g_next, visit)) {
            return true;
        }
    }
    return false;
}

double JointPermutationMove::partner_sum(std::size_t length) {
    double z = 0.0;
    for_each_partner_cycle(length, [&](const std::size_t*, double w, double g) {
        z += w * g;
        return false;
    });
    return z;
}

bool JointPermutationMove::attempt(PathSet& path, Action& action,
                                   numerics::Random& rng) {
    if (class_pairs_.empty()) return false;
    prepare(path);
    const std::size_t M = path.num_slices(), d = path.dim();
    const std::size_t n = std::size_t{1} << levels_;
    const bool periodic = path.cell().is_periodic();

    // Class pair, window and A-root: uniform, independent of the state.
    const ClassPair& pair = class_pairs_[uniform_index(rng, class_pairs_.size())];
    const auto& members_a = pair.first;
    const auto& members_b = pair.second;
    const std::size_t k0 = M - n + uniform_index(rng, n);
    const std::size_t k1 = k0 + n - M;
    double mass_a[SimulationCell::kMaxDim], mass_b[SimulationCell::kMaxDim];
    for (std::size_t dd = 0; dd < d; ++dd) {
        mass_a[dd] = action.mass(members_a.front(), dd);
        mass_b[dd] = action.mass(members_b.front(), dd);
    }
    const auto limit = static_cast<std::size_t>(max_cycle_);
    table_a_.build(path, members_a, k0, k1, n, mass_a, limit);
    table_b_.build(path, members_b, k0, k1, n, mass_b, limit);

    const std::size_t root = uniform_index(rng, members_a.size());
    const double q_root = table_a_.cycle_sum_from(root);
    double target = rng.uniform() * q_root;
    if (target < 1.0) {
        ++identity_draws_;
        return false;
    }
    target -= 1.0;
    std::size_t length = 0;
    double w_a = 0.0;
    table_a_.for_each_cycle_from(root, [&](const std::size_t* cycle, std::size_t l, double w) {
        std::copy(cycle, cycle + l, chosen_a_.begin());
        length = l;
        w_a = w;
        target -= w;
        return target < 0.0;
    });
    if (length < 2) return false;  // unreachable: q_root > 1 implies a cycle

    // B partners matched position by position to the chosen A sequence.
    build_candidates(path, pair, chosen_a_.data(), length, k0);
    const double z_forward = partner_sum(length);
    if (!(z_forward > 0.0)) {
        ++no_partner_;
        return false;
    }
    double pick = rng.uniform() * z_forward;
    double w_b = 0.0;
    bool picked = false;
    for_each_partner_cycle(length, [&](const std::size_t* sequence, double w, double g) {
        std::copy(sequence, sequence + length, chosen_b_.begin());
        w_b = w;
        picked = true;
        pick -= w * g;
        return pick < 0.0;  // roundoff leaves the last sequence chosen
    });
    if (!picked) return false;
    ++proposed_[length];

    // Both classes rotate their targets along the matched sequences.
    std::array<std::size_t, 8> old_a{}, new_a{}, old_b{}, new_b{};
    for (std::size_t m = 0; m < length; ++m) {
        old_a[m] = table_a_.sigma(chosen_a_[m]);
        old_b[m] = table_b_.sigma(chosen_b_[m]);
    }
    for (std::size_t m = 0; m < length; ++m) {
        new_a[m] = old_a[(m + 1) % length];
        new_b[m] = old_b[(m + 1) % length];
    }
    if (periodic)
        for (std::size_t m = 0; m < length; ++m)
            if (!old_frame_is_minimum_image(path, members_a[chosen_a_[m]],
                                            members_a[old_a[m]], k0, k1, n) ||
                !old_frame_is_minimum_image(path, members_b[chosen_b_[m]],
                                            members_b[old_b[m]], k0, k1, n)) {
                ++frame_rejections_;
                return false;
            }
    for (std::size_t m = 0; m < 2 * length; ++m) {
        const bool is_a = m < length;
        const std::size_t i = is_a ? m : m - length;
        const std::size_t left = is_a ? members_a[chosen_a_[i]] : members_b[chosen_b_[i]];
        const std::size_t right = is_a ? members_a[new_a[i]] : members_b[new_b[i]];
        const auto result = stage_bridge(path, left, right, k0, k1, n, levels_,
                                         is_a ? mass_a : mass_b, rng, bridge_,
                                         beads_.data() + m * (n - 1),
                                         positions_.data() + m * (n - 1) * d);
        if (result == BridgeResult::overflow) {
            reject_overflow();
            return false;
        }
        if (result == BridgeResult::noncanonical) {
            ++frame_rejections_;
            return false;
        }
    }

    const std::size_t count = 2 * length * (n - 1);
    const std::span<const PathSet::Bead> beads(beads_.data(), count);
    const std::span<const double> positions(positions_.data(), count * d);
    const double delta_v = action.delta_potential_beads(path, beads, positions);
    if (!std::isfinite(delta_v)) {
        reject_nonfinite(delta_v, members_a[chosen_a_[0]], k0);  // throws unless +inf
        return false;
    }

    // Reverse proposal at x': the same root, the reversed A sequence
    // (a_1, a_L, .., a_2) and its matched B sequences with the new targets.
    for (std::size_t m = 0; m < length; ++m) {
        table_a_.set_sigma(chosen_a_[m], new_a[m]);
        table_b_.set_sigma(chosen_b_[m], new_b[m]);
    }
    table_a_.build_adjacency();
    table_b_.build_adjacency();
    const double q_root_new = table_a_.cycle_sum_from(root);
    reversed_[0] = chosen_a_[0];
    for (std::size_t m = 1; m < length; ++m) reversed_[m] = chosen_a_[length - m];
    build_candidates(path, pair, reversed_.data(), length, k0);
    const double z_reverse = partner_sum(length);
    if (!(z_reverse > 0.0)) return false;  // unreachable: contains the reverse sequence

    const double log_a = -delta_v - std::log(w_a) - std::log(w_b) +
                         std::log(q_root) - std::log(q_root_new) +
                         std::log(z_forward) - std::log(z_reverse);
    if (!(std::isfinite(log_a) && (log_a >= 0.0 || rng.uniform() < std::exp(log_a))))
        return false;

    commit_beads(path, beads, positions);
    const auto pi = path.permutation();
    std::copy(pi.begin(), pi.end(), pi_new_.begin());
    for (std::size_t m = 0; m < length; ++m) {
        pi_new_[members_a[chosen_a_[m]]] = members_a[new_a[m]];
        pi_new_[members_b[chosen_b_[m]]] = members_b[new_b[m]];
    }
    path.set_permutation(pi_new_);
    ++accepted_[length];
    return true;
}

std::vector<std::pair<std::string, double>> JointPermutationMove::extra_diagnostics() const {
    std::vector<std::pair<std::string, double>> out;
    for (std::size_t l = 2; l < proposed_.size(); ++l) {
        out.emplace_back("cycle" + std::to_string(l) + "_proposals",
                         static_cast<double>(proposed_[l]));
        out.emplace_back("cycle" + std::to_string(l) + "_accepts",
                         static_cast<double>(accepted_[l]));
    }
    out.emplace_back("identity_draws", static_cast<double>(identity_draws_));
    out.emplace_back("no_partner", static_cast<double>(no_partner_));
    out.emplace_back("frame_rejections", static_cast<double>(frame_rejections_));
    return out;
}

void JointPermutationMove::reset_diagnostics() noexcept {
    Move::reset_diagnostics();
    std::fill(proposed_.begin(), proposed_.end(), 0);
    std::fill(accepted_.begin(), accepted_.end(), 0);
    identity_draws_ = no_partner_ = frame_rejections_ = 0;
}

// ── WormMove ────────────────────────────────────────────────────────────────

namespace {

constexpr std::size_t kNotWorm = std::numeric_limits<std::size_t>::max();

bool same_bead(PathSet::Bead a, PathSet::Bead b) noexcept {
    return a.particle == b.particle && a.slice == b.slice;
}

// Separation b - a of two beads (minimum image under PBC).
void bead_separation(const PathSet& path, PathSet::Bead a, PathSet::Bead b, double* sep) {
    for (std::size_t dd = 0; dd < path.dim(); ++dd)
        sep[dd] = path.pos(b.particle, dd, b.slice) - path.pos(a.particle, dd, a.slice);
    if (path.cell().is_periodic()) path.cell().min_image(sep);
}

// Reversibility under PBC: a bridge redraws the segment of `links` links that
// starts at `from` in the minimum-image frame of its end points, so it can
// only regenerate a segment whose summed links equal that frame.
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

// Every link of the unwrapped chain walk[0..n] ([(n+1) * d]) must be its own
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

WormMove::WormMove(std::vector<std::vector<std::size_t>> classes, int max_length,
                   int swap_length, double worm_constant)
    : classes_(std::move(classes)), max_length_(max_length),
      swap_length_(swap_length), constant_(worm_constant),
      tune_(std::isnan(worm_constant)) {
    if (classes_.empty())
        throw std::invalid_argument("WormMove: needs at least one exchange class");
    for (const auto& members : classes_) {
        if (members.size() < 2 || !std::is_sorted(members.begin(), members.end()))
            throw std::invalid_argument(
                "WormMove: each class needs two or more sorted members");
        particles_.insert(particles_.end(), members.begin(), members.end());
    }
    std::sort(particles_.begin(), particles_.end());
    if (std::adjacent_find(particles_.begin(), particles_.end()) != particles_.end())
        throw std::invalid_argument("WormMove: exchange classes overlap");
    if (max_length < 1)
        throw std::invalid_argument("WormMove: max_length must be >= 1");
    if (swap_length < 2)
        throw std::invalid_argument("WormMove: swap_length must be >= 2");
    if (!tune_ && !(std::isfinite(worm_constant) && worm_constant > 0.0))
        throw std::invalid_argument("WormMove: the worm constant C must be finite and > 0");
}

void WormMove::set_selection_probability(double p) {
    if (!(std::isfinite(p) && p > 0.0 && p <= 1.0))
        throw std::invalid_argument("WormMove: selection probability must be in (0, 1]");
    p_select_ = p;
}

void WormMove::prepare(const PathSet& path) {
    const std::size_t M = path.num_slices(), np = path.n_particles();
    if (static_cast<std::size_t>(max_length_) + 2 > M ||
        static_cast<std::size_t>(swap_length_) + 1 > M)
        throw std::runtime_error(
            "WormMove: needs max_length <= M - 2 and swap_length <= M - 1");
    if (class_of_.size() != np) {  // first use on this path shape (setup only)
        class_of_.assign(np, kNotWorm);
        for (std::size_t c = 0; c < classes_.size(); ++c)
            for (const std::size_t p : classes_[c]) {
                if (p >= np || path.exchange_class(p) != path.exchange_class(classes_[c].front()))
                    throw std::runtime_error(
                        "WormMove: a worm class is not an exchange class of the path");
                class_of_[p] = c;
            }
    }
    const auto staged = static_cast<std::size_t>(std::max(max_length_, swap_length_ - 1));
    beads_.resize(staged);
    positions_.resize(staged * path.dim());
    walk_.resize((static_cast<std::size_t>(std::max(max_length_, swap_length_)) + 2) * path.dim());
    std::size_t largest = 0;
    for (const auto& members : classes_) largest = std::max(largest, members.size());
    heat_.resize(largest);
}

bool WormMove::owns(const PathSet& path) const noexcept {
    const std::size_t p = path.head().particle;
    return path.is_open() && p < class_of_.size() && class_of_[p] != kNotWorm;
}

void WormMove::load_mass(const Action& action) {
    // One mass tensor per worm class; the driver checks every exchange class.
    for (std::size_t dd = 0; dd < action.dim(); ++dd)
        mass_[dd] = action.mass(particles_.front(), dd);
}

// ln rho0 for a separation over `links` time steps: the normalized free
// propagator prod_d sqrt(m_d / (2 pi n tau)) exp(-m_d x_d^2 / (2 n tau)).
double WormMove::log_rho0(const double* separation, std::size_t links,
                          const PathSet& path) const {
    const double time = static_cast<double>(links) * path.tau();
    double log_rho = 0.0;
    for (std::size_t dd = 0; dd < path.dim(); ++dd)
        log_rho += 0.5 * std::log(mass_[dd] / (2.0 * std::numbers::pi * time)) -
                   mass_[dd] * separation[dd] * separation[dd] / (2.0 * time);
    return log_rho;
}

bool WormMove::attempt(PathSet& path, Action& action, numerics::Random& rng) {
    if (!(p_select_ > 0.0))
        throw std::logic_error("WormMove: the driver must set the selection probability");
    prepare(path);
    load_mass(action);
    if (!path.is_open()) return attempt_open(path, action, rng);
    if (!owns(path))
        throw std::logic_error("WormMove: the path is open in another worm's class");
    const double u = rng.uniform();
    if (u < kSwap) return attempt_swap(path, action, rng);
    if (u < kSwap + kHead) return attempt_regrow(path, action, rng, true);
    if (u < kSwap + kHead + kTail) return attempt_regrow(path, action, rng, false);
    return attempt_close(path, action, rng);
}

bool WormMove::accept_staged(PathSet& path, Action& action, numerics::Random& rng,
                             std::size_t count, double log_ratio, double& delta_v) {
    const std::span<const PathSet::Bead> beads(beads_.data(), count);
    const std::span<const double> positions(positions_.data(), count * path.dim());
    delta_v = action.delta_potential_beads(path, beads, positions);
    if (!std::isfinite(delta_v)) {
        reject_nonfinite(delta_v, beads_[0].particle, beads_[0].slice);  // throws unless +inf
        return false;
    }
    const double log_a = log_ratio - delta_v;
    if (!(std::isfinite(log_a) && (log_a >= 0.0 || rng.uniform() < std::exp(log_a))))
        return false;
    commit_beads(path, beads, positions);
    return true;
}

// Open: bead b becomes the head; b and its m-1 predecessors are redrawn as a
// free random walk from the anchor a, and the link b -> t is cut.
bool WormMove::attempt_open(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), d = path.dim();
    const bool periodic = path.cell().is_periodic();
    const std::size_t n_beads = particles_.size() * M;
    const std::size_t pick = uniform_index(rng, n_beads);
    const PathSet::Bead head{particles_[pick / M], pick % M};
    const auto m = 1 + uniform_index(rng, static_cast<std::size_t>(max_length_));
    ++proposed_[kOpen];
    PathSet::Bead bead = head;
    for (std::size_t i = m; i-- > 0;) {
        beads_[i] = bead;
        bead = path.bead_before(bead.particle, bead.slice);
    }
    const PathSet::Bead anchor = bead;
    const PathSet::Bead tail = path.bead_after(head.particle, head.slice);
    if (!segment_is_minimum_image(path, anchor, m + 1)) {  // close draws that frame
        ++frame_rejections_;
        return false;
    }
    for (std::size_t dd = 0; dd < d; ++dd)
        walk_[dd] = path.pos(anchor.particle, dd, anchor.slice);
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
    if (!(constant_ > 0.0)) {
        // Starting estimate (tuned in equilibration): A_open ~ 1 for a typical
        // half-length segment.
        double zero[SimulationCell::kMaxDim] = {};
        constant_ = p_select_ * std::exp(log_rho0(zero, static_cast<std::size_t>(max_length_) / 2 + 1, path)) /
                    (kClose * static_cast<double>(n_beads));
    }
    double sep[SimulationCell::kMaxDim];
    bead_separation(path, anchor, tail, sep);  // neither end moves
    const double log_ratio = std::log(constant_) + std::log(static_cast<double>(n_beads)) +
                             std::log(kClose) - std::log(p_select_) -
                             log_rho0(sep, m + 1, path);
    double delta_v = 0.0;
    if (!accept_staged(path, action, rng, m, log_ratio, delta_v)) return false;
    path.open_at(head);
    ++accepted_[kOpen];
    return true;
}

// Close: the head and its m-1 predecessors are redrawn as a free bridge from
// the anchor a to the tail over m+1 links (minimum-image frame).
bool WormMove::attempt_close(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t d = path.dim();
    const bool periodic = path.cell().is_periodic();
    const std::size_t n_beads = particles_.size() * path.num_slices();
    const PathSet::Bead head = path.head(), tail = path.tail();
    const auto m = 1 + uniform_index(rng, static_cast<std::size_t>(max_length_));
    ++proposed_[kCloseMove];
    PathSet::Bead bead = head;
    for (std::size_t i = m; i-- > 0;) {
        beads_[i] = bead;
        bead = path.bead_before(bead.particle, bead.slice);
    }
    const PathSet::Bead anchor = bead;
    double sep[SimulationCell::kMaxDim];
    bead_separation(path, anchor, tail, sep);
    // Levy bridge: bead i given bead i-1 and the end n - i + 1 links away has
    // mean x_{i-1} + (x_n - x_{i-1}) / (n - i + 1) and variance
    // tau (n - i) / ((n - i + 1) m) per axis.
    const std::size_t n = m + 1;
    for (std::size_t dd = 0; dd < d; ++dd) {
        walk_[dd] = path.pos(anchor.particle, dd, anchor.slice);
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
    for (std::size_t i = 0; i < m; ++i) {
        double* r = positions_.data() + i * d;
        for (std::size_t dd = 0; dd < d; ++dd) r[dd] = walk_[(i + 1) * d + dd];
        if (periodic) path.cell().wrap(r);
    }
    const double log_ratio = std::log(p_select_) + log_rho0(sep, n, path) -
                             std::log(constant_) - std::log(static_cast<double>(n_beads)) -
                             std::log(kClose);
    double delta_v = 0.0;
    if (!accept_staged(path, action, rng, m, log_ratio, delta_v)) return false;
    path.close();
    ++accepted_[kCloseMove];
    return true;
}

// Head (tail) regrowth: the head and its m-1 predecessors (the tail and its
// m-1 successors) are redrawn as a free random walk from the bead m slices
// before (after) them; the walk cancels the springs, so A = exp(-dV).
bool WormMove::attempt_regrow(PathSet& path, Action& action, numerics::Random& rng,
                              bool head_end) {
    const std::size_t d = path.dim();
    const bool periodic = path.cell().is_periodic();
    const auto m = 1 + uniform_index(rng, static_cast<std::size_t>(max_length_));
    ++proposed_[head_end ? kHeadMove : kTailMove];
    PathSet::Bead anchor{};
    if (head_end) {
        PathSet::Bead bead = path.head();
        for (std::size_t i = m; i-- > 0;) {
            beads_[i] = bead;
            bead = path.bead_before(bead.particle, bead.slice);
        }
        anchor = bead;
    } else {
        PathSet::Bead bead = path.tail();
        for (std::size_t i = 0; i < m; ++i) {
            beads_[i] = bead;
            bead = path.bead_after(bead.particle, bead.slice);
        }
        anchor = bead;
    }
    for (std::size_t dd = 0; dd < d; ++dd)
        walk_[dd] = path.pos(anchor.particle, dd, anchor.slice);
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
    // Walk step i lands on the i-th bead away from the anchor: beads_[i-1]
    // after it (head end), beads_[m-i] before it (tail end).
    for (std::size_t i = 1; i <= m; ++i) {
        double* r = positions_.data() + (head_end ? i - 1 : m - i) * d;
        for (std::size_t dd = 0; dd < d; ++dd) r[dd] = walk_[i * d + dd];
        if (periodic) path.cell().wrap(r);
    }
    double delta_v = 0.0;
    if (!accept_staged(path, action, rng, m, 0.0, delta_v)) return false;
    ++accepted_[head_end ? kHeadMove : kTailMove];
    return true;
}

// Swap (BPS 2006): alpha at slice k_h + m_s by heat bath from the head; the
// segment zeta -> alpha is replaced by a free bridge head -> alpha and zeta
// becomes the head.
bool WormMove::attempt_swap(PathSet& path, Action& action, numerics::Random& rng) {
    const std::size_t M = path.num_slices(), d = path.dim();
    const bool periodic = path.cell().is_periodic();
    const auto ms = static_cast<std::size_t>(swap_length_);
    const PathSet::Bead head = path.head(), tail = path.tail();
    const auto& members = classes_[class_of_[head.particle]];
    const std::size_t target_slice = (head.slice + ms) % M;
    ++proposed_[kSwapMove];

    // Heat-bath weights rho0(head, member at target; m_s tau) up to the common
    // normalization, pruned to 0 beyond exp(-kMinusLogEpsilon) like CycleTable.
    const auto heat_bath = [&](PathSet::Bead from) {
        double sum = 0.0;
        for (std::size_t i = 0; i < members.size(); ++i) {
            double sep[SimulationCell::kMaxDim];
            bead_separation(path, from, {members[i], target_slice}, sep);
            double exponent = 0.0;
            for (std::size_t dd = 0; dd < d; ++dd) exponent += mass_[dd] * sep[dd] * sep[dd];
            exponent /= 2.0 * static_cast<double>(ms) * path.tau();
            heat_[i] = exponent > CycleTable::kMinusLogEpsilon ? 0.0 : std::exp(-exponent);
            sum += heat_[i];
        }
        return sum;
    };
    const double sigma_head = heat_bath(head);
    if (!(sigma_head > 0.0)) {
        ++no_candidate_;
        return false;
    }
    double target = rng.uniform() * sigma_head;
    std::size_t chosen = members.size();
    for (std::size_t i = 0; i < members.size(); ++i) {
        if (heat_[i] <= 0.0) continue;
        chosen = i;  // roundoff leaves the last positive weight chosen
        target -= heat_[i];
        if (target < 0.0) break;
    }
    const PathSet::Bead alpha{members[chosen], target_slice};

    // Interior of the old segment zeta -> alpha, forward order; its first bead
    // is the tail when zeta is the head, i.e. the segment crosses the cut.
    PathSet::Bead bead = alpha;
    for (std::size_t i = ms - 1; i-- > 0;) {
        bead = path.bead_before(bead.particle, bead.slice);
        beads_[i] = bead;
    }
    if (same_bead(beads_[0], tail)) {
        ++tail_hits_;
        return false;
    }
    const PathSet::Bead zeta = path.bead_before(beads_[0].particle, beads_[0].slice);
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

    // The reverse heat bath from zeta must be able to pick alpha (symmetric
    // pruning); slice k_h + m_s is not touched by the proposal.
    const double sigma_zeta = heat_bath(zeta);
    if (!(heat_[chosen] > 0.0)) {
        ++no_candidate_;
        return false;
    }
    double delta_v = 0.0;
    if (!accept_staged(path, action, rng, ms - 1,
                       std::log(sigma_head) - std::log(sigma_zeta), delta_v))
        return false;

    // Reconnect: the head continues into the new beads (zeta's old successor
    // slots) and zeta, the new head, into the tail.
    const auto start = std::chrono::steady_clock::now();
    const std::size_t old_head_row =
        path.exchange_successors(head.particle, zeta.particle, head.slice);
    relabel_seconds_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    relabel_slices_ += std::min(head.slice + 1, M - 1 - head.slice);
    path.set_head({old_head_row == head.particle ? zeta.particle : head.particle,
                   head.slice});
    ++accepted_[kSwapMove];
    return true;
}

void WormMove::tune_constant(double closed_fraction, double open_fraction,
                             double target_ratio) {
    if (!tune_ || !(constant_ > 0.0)) return;
    // open/closed is proportional to C at fixed paths.  A block's fractions
    // come from a few dozen attempts, so take the square root of the needed
    // ratio (clamped to [1/4, 4]): damped, C settles within ~20 % instead of
    // random-walking.
    double ratio = 4.0;
    if (closed_fraction <= 0.0) ratio = 0.25;
    else if (open_fraction > 0.0)
        ratio = std::clamp(target_ratio * closed_fraction / open_fraction, 0.25, 4.0);
    constant_ *= std::sqrt(ratio);
}

void WormMove::sample_open_ends(PathSet& path, Action& action, numerics::Random& rng,
                                std::span<double> displacement, std::span<double> weight) {
    const std::size_t d = path.dim(), n = weight.size();
    if (!owns(path) || n == 0 || displacement.size() != n * d || !(constant_ > 0.0))
        throw std::logic_error("WormMove: open-end samples need an open path of this worm");
    load_mass(action);
    const PathSet::Bead head = path.head(), tail = path.tail();
    const double scale = path.permutation_sign() /
        (static_cast<double>(n) * constant_ * static_cast<double>(particles_.size()) *
         static_cast<double>(path.num_slices()));
    for (std::size_t i = 0; i < n; ++i) {
        double x[SimulationCell::kMaxDim];
        for (std::size_t dd = 0; dd < d; ++dd)
            x[dd] = path.pos(head.particle, dd, head.slice) +
                    std::sqrt(path.tau() / mass_[dd]) * rng.gaussian();
        if (path.cell().is_periodic()) path.cell().wrap(x);
        // tau (V_x - V_t): the tail's interactions moved to x; both open ends
        // keep half of the primitive potential weight.
        const double delta_s = action.delta_potential_particle(
            path, tail.particle, tail.slice, std::span<const double>(x, d));
        if (std::isnan(delta_s) || delta_s == -std::numeric_limits<double>::infinity())
            throw std::runtime_error("worm: nonfinite potential change at an open end");
        weight[i] = std::isinf(delta_s) ? 0.0 : scale * std::exp(-0.5 * delta_s);
        double sep[SimulationCell::kMaxDim];
        for (std::size_t dd = 0; dd < d; ++dd)
            sep[dd] = x[dd] - path.pos(tail.particle, dd, tail.slice);
        if (path.cell().is_periodic()) path.cell().min_image(sep);
        for (std::size_t dd = 0; dd < d; ++dd) displacement[i * d + dd] = sep[dd];
    }
}

std::vector<std::pair<std::string, double>> WormMove::extra_diagnostics() const {
    static constexpr const char* kNames[kKinds] = {"open", "close", "swap", "head", "tail"};
    std::vector<std::pair<std::string, double>> out;
    for (std::size_t k = 0; k < kKinds; ++k) {
        out.emplace_back(std::string(kNames[k]) + "_proposals", static_cast<double>(proposed_[k]));
        out.emplace_back(std::string(kNames[k]) + "_accepts", static_cast<double>(accepted_[k]));
    }
    out.emplace_back("swap_tail_hits", static_cast<double>(tail_hits_));
    out.emplace_back("swap_no_candidate", static_cast<double>(no_candidate_));
    out.emplace_back("frame_rejections", static_cast<double>(frame_rejections_));
    out.emplace_back("relabel_slices", static_cast<double>(relabel_slices_));
    out.emplace_back("relabel_seconds", relabel_seconds_);
    return out;
}

void WormMove::reset_diagnostics() noexcept {
    Move::reset_diagnostics();
    proposed_.fill(0);
    accepted_.fill(0);
    tail_hits_ = no_candidate_ = frame_rejections_ = relabel_slices_ = 0;
    relabel_seconds_ = 0.0;
}

}  // namespace alea
