// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Fixed-node restriction with time-independent nodes and the image action of
// every link; see the header for the formulas and their references.
#include "pimc/fixed_node_restriction.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "pimc/path_set.h"
#include "population/simulation_cell.h"

namespace alea {
namespace {
constexpr double kInfinity = std::numeric_limits<double>::infinity();

double dot(const double* a, const double* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
}  // namespace

FixedNodeRestriction::FixedNodeRestriction(std::vector<Block> blocks, std::size_t np,
                                           std::size_t dim, std::vector<double> inv_mass,
                                           NodeCrossing crossing)
    : blocks_(std::move(blocks)), np_(np), dim_(dim), inv_mass_(std::move(inv_mass)),
      crossing_(crossing), block_of_(np, kNoBlock) {
    if (dim_ == 0 || dim_ > SimulationCell::kMaxDim)
        throw std::invalid_argument("fixed_node: dimension must be between 1 and " +
                                    std::to_string(SimulationCell::kMaxDim));
    if (blocks_.empty())
        throw std::invalid_argument("fixed_node: no determinant block to restrict");
    if (inv_mass_.size() != np_ * dim_ ||
        !std::all_of(inv_mass_.begin(), inv_mass_.end(),
                     [](double v) { return std::isfinite(v) && v > 0.0; }))
        throw std::invalid_argument(
            "fixed_node: one finite positive inverse mass per particle and axis is required");
    std::size_t n_max = 0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& block = blocks_[b];
        if (block.first >= block.last || block.last > np_)
            throw std::invalid_argument(
                "fixed_node: block " + std::to_string(b) + " covers particles [" +
                std::to_string(block.first) + ", " + std::to_string(block.last) +
                "), which is empty or outside the " + std::to_string(np_) + " particles");
        const std::size_t n = block.last - block.first;
        if (!block.orbitals || block.orbitals->size() != n)
            throw std::invalid_argument(
                "fixed_node: block " + std::to_string(b) + " needs one orbital per particle (" +
                std::to_string(n) + "), got " +
                std::to_string(block.orbitals ? block.orbitals->size() : 0));
        for (std::size_t p = block.first; p < block.last; ++p) {
            if (block_of_[p] != kNoBlock)
                throw std::invalid_argument("fixed_node: determinant blocks overlap at particle " +
                                            std::to_string(p));
            block_of_[p] = b;
        }
        fold_bound_ = std::max(fold_bound_, block.orbitals->periodic_fold_bound());
        n_max = std::max(n_max, n);
    }
    if (!(fold_bound_ <= kFoldTolerance)) {
        std::ostringstream message;
        message << "fixed_node: the minimum-image orbitals neglect periodic images up to "
                << fold_bound_ << " of their peak (tolerance " << kFoldTolerance
                << "), so their nodes would differ from those of the lattice-summed "
                   "orbitals; use a larger cell (more particles) or narrower orbitals";
        throw std::invalid_argument(message.str());
    }
    caches_.resize(blocks_.size());
    for (const Block& block : blocks_) {
        const auto n = static_cast<Eigen::Index>(block.last - block.first);
        matrix_.emplace_back(n, n);
        lu_.emplace_back(n);
        inverse_.emplace_back(n, n);
    }
    phi_.resize(n_max);
    grad_.resize(n_max * dim_);
    lap_.resize(n_max * dim_);
    row_.resize(n_max);
    column_.resize(n_max);
    proposed_.resize(n_max * n_max);
    proposed_row_.resize(dim_ * (n_max + 1));
}

FixedNodeRestriction::FixedNodeRestriction(const FixedNodeRestriction& other)
    : ActionComponent(other), np_(other.np_), dim_(other.dim_),
      inv_mass_(other.inv_mass_), crossing_(other.crossing_),
      block_of_(other.block_of_), caches_(other.blocks_.size()),
      fold_bound_(other.fold_bound_), matrix_(other.matrix_),
      inverse_(other.inverse_), phi_(other.phi_), grad_(other.grad_),
      lap_(other.lap_), row_(other.row_), column_(other.column_),
      proposed_(other.proposed_), proposed_row_(other.proposed_row_) {
    blocks_.reserve(other.blocks_.size());
    for (const Block& block : other.blocks_) {
        blocks_.push_back({block.first, block.last, block.orbitals->clone()});
        lu_.emplace_back(static_cast<Eigen::Index>(block.last - block.first));
    }
}

void FixedNodeRestriction::prepare(std::size_t M) { ensure_slices(M); }

void FixedNodeRestriction::ensure_slices(std::size_t M) {
    if (M == M_) return;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        Cache& cache = caches_[b];
        cache.inverse.assign(M * n * n, 0.0);
        cache.snapshot.assign(M * n * dim_, 0.0);
        cache.since_rebuild.assign(M, 0);
        cache.built.assign(M, 0);
        cache.grad2.assign(M, 0.0);
        cache.grad2_valid.assign(M, 0);
        cache.pending_row.assign(M, kNoBlock);
        cache.pending_pos.assign(M * dim_, 0.0);
        cache.pending_grad2.assign(M, 0.0);
        cache.gradients.assign(crossing_ == NodeCrossing::image ? M * n * dim_ * (n + 1) : 0, 0.0);
    }
    window_grad2_.assign(M, 0.0);
    M_ = M;
}

void FixedNodeRestriction::orbitals_at(std::size_t b, const double* r) {
    const std::size_t n = blocks_[b].last - blocks_[b].first;
    blocks_[b].orbitals->evaluate(std::span<const double>(r, dim_), dim_,
                                  std::span<double>(phi_.data(), n),
                                  std::span<double>(grad_.data(), n * dim_),
                                  std::span<double>(lap_.data(), n * dim_), 0, 1);
}

void FixedNodeRestriction::store_gradient_row(std::size_t b, double* row) const {
    const std::size_t n = blocks_[b].last - blocks_[b].first;
    for (std::size_t a = 0; a < dim_; ++a)
        for (std::size_t i = 0; i < n; ++i) row[a * n + i] = grad_[i * dim_ + a];
    // Row scale m_j = |φ_k(r_j)|, k = argmax_i |φ_i(r_j)|:  ∂_a ln m_j = ∂_a φ_k / φ_k
    // (the sign of φ_k cancels).  Zero if every orbital vanishes at r_j (then
    // det D = 0 and the move is refused).
    std::size_t k = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (std::abs(phi_[i]) > std::abs(phi_[k])) k = i;
    for (std::size_t a = 0; a < dim_; ++a)
        row[dim_ * n + a] = phi_[k] != 0.0 ? grad_[k * dim_ + a] / phi_[k] : 0.0;
}

int FixedNodeRestriction::rebuild(const PathSet& path, std::size_t b, std::size_t k) {
    const Block& block = blocks_[b];
    const std::size_t n = block.last - block.first;
    Cache& cache = caches_[b];
    Eigen::MatrixXd& D = matrix_[b];
    double* snapshot = &cache.snapshot[k * n * dim_];
    double r[SimulationCell::kMaxDim];
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t d = 0; d < dim_; ++d)
            snapshot[j * dim_ + d] = r[d] = path.pos(block.first + j, d, k);
        orbitals_at(b, r);
        for (std::size_t i = 0; i < n; ++i)
            D(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = phi_[i];
        if (crossing_ == NodeCrossing::image)
            store_gradient_row(b, &cache.gradients[(k * n + j) * dim_ * (n + 1)]);
    }
    ++rebuilds_;
    cache.built[k] = 0;
    cache.grad2_valid[k] = 0;
    cache.pending_row[k] = kNoBlock;
    lu_[b].compute(D);
    // det D = det P · Π U_ii (Eigen's PartialPivLU: P D = L U, unit L).
    int sign = static_cast<int>(lu_[b].permutationP().determinant());
    const Eigen::MatrixXd& LU = lu_[b].matrixLU();
    for (Eigen::Index i = 0; i < LU.rows(); ++i) {
        const double u = LU(i, i);
        if (!std::isfinite(u) || u == 0.0) return 0;
        if (u < 0.0) sign = -sign;
    }
    inverse_[b] = lu_[b].inverse();
    const double* inverse = inverse_[b].data();  // column-major: (D⁻¹)_{i j} at [j*n + i]
    if (!std::all_of(inverse, inverse + n * n, [](double v) { return std::isfinite(v); }))
        return 0;
    std::memcpy(&cache.inverse[k * n * n], inverse, n * n * sizeof(double));
    cache.built[k] = 1;
    cache.since_rebuild[k] = 0;
    return sign;
}

namespace {
// Row j of D replaced, ratio = det D'/det D, c = row_ (c_l = Σ_m φ_m(r') T[l][m]
// for l ≠ j, c_j = ratio), column = T[j]:
//   T'[l][i] = T[l][i] − T[j][i] c_l / ratio  (l ≠ j),   T'[j][i] = T[j][i] / ratio,
// because Σ_m φ_m(r_j) (D⁻¹)_{m l} = δ_{jl}.  `out` may alias T.
void row_replaced_inverse(const double* T, double* out, std::size_t n, std::size_t row,
                          double ratio, const double* phi, double* c, double* column) {
    for (std::size_t l = 0; l < n; ++l) c[l] = l == row ? ratio : dot(phi, T + l * n, n);
    std::copy_n(T + row * n, n, column);
    const double inv_ratio = 1.0 / ratio;
    for (std::size_t l = 0; l < n; ++l) {
        const double* Tl = T + l * n;
        double* out_l = out + l * n;
        if (l == row) {
            for (std::size_t i = 0; i < n; ++i) out_l[i] = column[i] * inv_ratio;
        } else {
            const double f = c[l] * inv_ratio;
            for (std::size_t i = 0; i < n; ++i) out_l[i] = Tl[i] - f * column[i];
        }
    }
}
}  // namespace

bool FixedNodeRestriction::sync(const PathSet& path, std::size_t b, std::size_t k) {
    const Block& block = blocks_[b];
    const std::size_t n = block.last - block.first;
    Cache& cache = caches_[b];
    if (!cache.built[k]) return rebuild(path, b, k) != 0;
    double* snapshot = &cache.snapshot[k * n * dim_];
    std::size_t changed = 0, row = 0;
    for (std::size_t j = 0; j < n && changed < 2; ++j)
        for (std::size_t d = 0; d < dim_; ++d)
            if (path.pos(block.first + j, d, k) != snapshot[j * dim_ + d]) {
                ++changed;
                row = j;
                break;
            }
    if (changed == 0) return true;
    if (changed > 1 || cache.since_rebuild[k] >= kRebuildEvery)
        return rebuild(path, b, k) != 0;

    double r[SimulationCell::kMaxDim];
    for (std::size_t d = 0; d < dim_; ++d) r[d] = path.pos(block.first + row, d, k);
    orbitals_at(b, r);
    double* T = &cache.inverse[k * n * n];  // T[l*n + i] = (D⁻¹)_{i l}
    const double ratio = dot(phi_.data(), T + row * n, n);
    if (!std::isfinite(ratio) || std::abs(ratio) < kSmallRatio)
        return rebuild(path, b, k) != 0;
    row_replaced_inverse(T, T, n, row, ratio, phi_.data(), row_.data(), column_.data());
    if (crossing_ == NodeCrossing::image)  // grad_ still holds the new row's gradients
        store_gradient_row(b, &cache.gradients[(k * n + row) * dim_ * (n + 1)]);
    for (std::size_t d = 0; d < dim_; ++d) snapshot[row * dim_ + d] = r[d];
    ++cache.since_rebuild[k];
    ++updates_;
    // An accepted proposal: its gradient norm was computed for exactly this row.
    cache.grad2_valid[k] = 0;
    if (cache.pending_row[k] == row &&
        std::equal(r, r + dim_, cache.pending_pos.begin() + static_cast<std::ptrdiff_t>(k * dim_))) {
        cache.grad2[k] = cache.pending_grad2[k];
        cache.grad2_valid[k] = 1;
    }
    cache.pending_row[k] = kNoBlock;
    return true;
}

double FixedNodeRestriction::grad2_of(std::size_t b, std::size_t k, const double* T,
                                      std::size_t moved, const double* moved_gradients) {
    const Block& block = blocks_[b];
    const std::size_t n = block.last - block.first;
    const double* G = &caches_[b].gradients[k * n * dim_ * (n + 1)];
    double sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        const double* Gj = j == moved ? moved_gradients : G + j * dim_ * (n + 1);
        const double* Tj = T + j * n;
        for (std::size_t a = 0; a < dim_; ++a) {
            // ∂_{ja} ln|det(D/m)| = Σ_i ∂_a φ_i(r_j) (D⁻¹)_{i j} − ∂_a ln m_j
            const double g = dot(Gj + a * n, Tj, n) - Gj[dim_ * n + a];
            sum += g * g * inv_mass_[(block.first + j) * dim_ + a];
        }
    }
    ++gradients_;
    return sum;
}

double FixedNodeRestriction::current_grad2(const PathSet&, std::size_t b, std::size_t k) {
    Cache& cache = caches_[b];
    if (!cache.grad2_valid[k]) {
        const std::size_t n = blocks_[b].last - blocks_[b].first;
        cache.grad2[k] = grad2_of(b, k, &cache.inverse[k * n * n], kNoBlock, nullptr);
        cache.grad2_valid[k] = 1;
    }
    return cache.grad2[k];
}

bool FixedNodeRestriction::propose(const PathSet& path, std::size_t b, std::size_t k,
                                   std::size_t row, const double* r_new, double& grad2) {
    const std::size_t n = blocks_[b].last - blocks_[b].first;
    Cache& cache = caches_[b];
    const double* T = &cache.inverse[k * n * n];
    orbitals_at(b, r_new);
    const double ratio = dot(phi_.data(), T + row * n, n);
    if (!std::isfinite(ratio))
        throw std::logic_error("fixed_node: nonfinite determinant ratio at a finite proposal");
    if (!(ratio > 0.0)) return false;
    grad2 = 0.0;
    if (crossing_ == NodeCrossing::beads_only) return true;
    store_gradient_row(b, proposed_row_.data());  // before anything reuses grad_
    double* Tp = proposed_.data();
    if (ratio >= kSmallRatio) {
        row_replaced_inverse(T, Tp, n, row, ratio, phi_.data(), row_.data(), column_.data());
    } else {
        // Near the node a Sherman–Morrison update loses digits: fresh LU of D'.
        Eigen::MatrixXd& D = matrix_[b];
        double r[SimulationCell::kMaxDim];
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t d = 0; d < dim_; ++d)
                r[d] = j == row ? r_new[d] : path.pos(blocks_[b].first + j, d, k);
            orbitals_at(b, r);
            for (std::size_t i = 0; i < n; ++i)
                D(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = phi_[i];
        }
        lu_[b].compute(D);
        inverse_[b] = lu_[b].inverse();
        if (!std::all_of(inverse_[b].data(), inverse_[b].data() + n * n,
                         [](double v) { return std::isfinite(v); }))
            return false;  // numerically on the node
        std::memcpy(Tp, inverse_[b].data(), n * n * sizeof(double));
    }
    grad2 = grad2_of(b, k, Tp, row, proposed_row_.data());
    cache.pending_row[k] = row;
    std::copy_n(r_new, dim_, cache.pending_pos.begin() + static_cast<std::ptrdiff_t>(k * dim_));
    cache.pending_grad2[k] = grad2;
    return true;
}

double FixedNodeRestriction::link_action(double g1, double g2, double tau) {
    if (!(g1 > 0.0) || !(g2 > 0.0)) return 0.0;  // a flat ψ_b: no node in reach
    const double x = 2.0 / (tau * std::sqrt(g1 * g2));
    return -std::log1p(-std::exp(-x));
}

double FixedNodeRestriction::link_tau_derivative(double g1, double g2, double tau) {
    if (!(g1 > 0.0) || !(g2 > 0.0)) return 0.0;
    const double x = 2.0 / (tau * std::sqrt(g1 * g2));
    if (x > 700.0) return 0.0;  // e^{-x} below double precision
    return (x / tau) / std::expm1(x);
}

double FixedNodeRestriction::total_action(const PathSet& path) {
    if (path.n_particles() != np_ || path.dim() != dim_)
        throw std::invalid_argument("fixed_node: path shape does not match the restriction");
    const std::size_t M = path.num_slices();
    ensure_slices(M);
    double total = 0.0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        int reference = 0;
        for (std::size_t k = 0; k < M; ++k) {
            const int sign = rebuild(path, b, k);
            if (sign == 0) return kInfinity;
            if (k == 0) reference = sign;
            else if (sign != reference) return kInfinity;
        }
        if (crossing_ == NodeCrossing::beads_only) continue;
        for (std::size_t k = 0; k < M; ++k) window_grad2_[k] = current_grad2(path, b, k);
        for (std::size_t k = 0; k < M; ++k)
            total += link_action(window_grad2_[k], window_grad2_[(k + 1) % M], path.tau());
    }
    return total;
}

double FixedNodeRestriction::delta_potential_particle(PathSet& path, std::size_t p,
                                                      std::size_t k,
                                                      std::span<const double> proposed) {
    if (path.n_particles() != np_ || path.dim() != dim_ || p >= np_ ||
        k >= path.num_slices() || proposed.size() != dim_)
        throw std::invalid_argument("fixed_node delta: invalid particle, slice or position");
    const std::size_t b = block_of_[p];
    if (b == kNoBlock) return 0.0;
    const std::size_t M = path.num_slices();
    ensure_slices(M);
    for (std::size_t d = 0; d < dim_; ++d)
        if (!std::isfinite(proposed[d])) return kInfinity;  // overflowing proposal
    const auto require = [&](std::size_t slice) {
        if (!sync(path, b, slice))
            throw std::runtime_error(
                "fixed_node: the current path sits on a node of block " + std::to_string(b) +
                " at slice " + std::to_string(slice) +
                " (singular determinant), so the restriction was already violated");
    };
    require(k);
    double g_new = 0.0;
    if (!propose(path, b, k, p - blocks_[b].first, proposed.data(), g_new)) return kInfinity;
    if (crossing_ == NodeCrossing::beads_only) return 0.0;
    const double tau = path.tau();
    const double g_old = current_grad2(path, b, k);
    if (M == 1) return link_action(g_new, g_new, tau) - link_action(g_old, g_old, tau);
    const std::size_t before = (k + M - 1) % M, after = (k + 1) % M;
    require(before);
    require(after);
    const double g_before = current_grad2(path, b, before);
    const double g_after = current_grad2(path, b, after);
    return link_action(g_before, g_new, tau) + link_action(g_new, g_after, tau) -
           link_action(g_before, g_old, tau) - link_action(g_old, g_after, tau);
}

double FixedNodeRestriction::delta_potential_window(const PathSet& path, std::size_t p,
                                                    std::size_t first,
                                                    std::span<const double> positions) {
    const std::size_t M = path.num_slices();
    if (path.n_particles() != np_ || path.dim() != dim_ || p >= np_ || first >= M ||
        positions.size() % dim_ || positions.size() / dim_ > M)
        throw std::invalid_argument("fixed_node window: invalid particle, slice or positions");
    const std::size_t count = positions.size() / dim_;
    const std::size_t b = block_of_[p];
    if (b == kNoBlock || count == 0) return 0.0;
    ensure_slices(M);
    if (!std::all_of(positions.begin(), positions.end(), [](double v) { return std::isfinite(v); }))
        return kInfinity;
    const auto require = [&](std::size_t slice) {
        if (!sync(path, b, slice))
            throw std::runtime_error(
                "fixed_node: the current path sits on a node of block " + std::to_string(b) +
                " at slice " + std::to_string(slice) +
                " (singular determinant), so the restriction was already violated");
    };
    const std::size_t row = p - blocks_[b].first;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t k = (first + i) % M;
        require(k);
        if (!propose(path, b, k, row, positions.data() + i * dim_, window_grad2_[i]))
            return kInfinity;
    }
    if (crossing_ == NodeCrossing::beads_only) return 0.0;
    const double tau = path.tau();
    const auto old_g = [&](std::size_t i) { return current_grad2(path, b, (first + i) % M); };
    double delta = 0.0;
    if (count == M) {  // the whole ring: every link has both ends moved
        for (std::size_t i = 0; i < M; ++i) {
            const std::size_t j = (i + 1) % M;
            delta += link_action(window_grad2_[i], window_grad2_[j], tau) -
                     link_action(old_g(i), old_g(j), tau);
        }
        return delta;
    }
    const std::size_t before = (first + M - 1) % M, after = (first + count) % M;
    require(before);
    require(after);
    const double g_before = current_grad2(path, b, before);
    const double g_after = current_grad2(path, b, after);
    delta += link_action(g_before, window_grad2_[0], tau) - link_action(g_before, old_g(0), tau);
    for (std::size_t i = 0; i + 1 < count; ++i)
        delta += link_action(window_grad2_[i], window_grad2_[i + 1], tau) -
                 link_action(old_g(i), old_g(i + 1), tau);
    delta += link_action(window_grad2_[count - 1], g_after, tau) -
             link_action(old_g(count - 1), g_after, tau);
    return delta;
}

double FixedNodeRestriction::beta_derivative(const PathSet& path) {
    if (crossing_ == NodeCrossing::beads_only) return 0.0;
    const std::size_t M = path.num_slices();
    ensure_slices(M);
    const double tau = path.tau();
    double sum = 0.0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        for (std::size_t k = 0; k < M; ++k) {
            if (!sync(path, b, k))
                throw std::runtime_error("fixed_node: a measured path sits on a node");
            window_grad2_[k] = current_grad2(path, b, k);
        }
        for (std::size_t k = 0; k < M; ++k)
            sum += link_tau_derivative(window_grad2_[k], window_grad2_[(k + 1) % M], tau);
    }
    return sum / static_cast<double>(M);
}

ActionThermoMeasure FixedNodeRestriction::measure_thermo(const PathSet& path) {
    // The image factor belongs to the free propagator: kinetic energy.
    const double kinetic = beta_derivative(path);
    return {kinetic, 0.0, kinetic};
}

void FixedNodeRestriction::validate_open_centroid(std::size_t, std::size_t) const {
    throw std::invalid_argument(
        "centroid_virial: the fixed-node restriction puts a singular force on the "
        "nodal surface, which the virial does not carry; use energy: thermodynamic");
}

void FixedNodeRestriction::validate_periodic_virial(const SimulationCell&, std::size_t,
                                                    std::size_t) const {
    throw std::invalid_argument(
        "periodic_virial: the fixed-node restriction puts a singular force on the "
        "nodal surface, which the virial does not carry; use energy: thermodynamic");
}

std::string FixedNodeRestriction::potential_approximation_notes() const {
    std::ostringstream notes;
    notes << "fixed-node restriction: time-independent nodes of " << blocks_.size()
          << " determinant block(s) of size";
    for (const Block& block : blocks_) notes << ' ' << block.last - block.first;
    if (crossing_ == NodeCrossing::image)
        notes << "; link crossings by the image action -ln(1 - exp(-2 d d'/tau)) with "
                 "d = 1/|grad ln|det(D/m)||, each row divided by its largest orbital "
                 "(first order; exact for a planar node)";
    else
        notes << "; nodes checked at beads only (O(sqrt(tau)) bias)";
    notes << "; fixed-node ground-state estimates only as beta -> infinity; "
             "periodic-fold bound "
          << fold_bound_;
    return notes.str();
}

std::vector<std::pair<std::size_t, std::size_t>> FixedNodeRestriction::restricted_blocks() const {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    ranges.reserve(blocks_.size());
    for (const Block& block : blocks_) ranges.emplace_back(block.first, block.last);
    return ranges;
}

std::unique_ptr<ActionComponent> FixedNodeRestriction::clone() const {
    return std::make_unique<FixedNodeRestriction>(*this);
}

int FixedNodeRestriction::block_sign(const PathSet& path, std::size_t b, std::size_t k) {
    if (b >= blocks_.size() || k >= path.num_slices())
        throw std::invalid_argument("fixed_node: invalid block or slice");
    ensure_slices(path.num_slices());
    return rebuild(path, b, k);
}

double FixedNodeRestriction::node_distance(const PathSet& path, std::size_t b, std::size_t k) {
    if (b >= blocks_.size() || k >= path.num_slices())
        throw std::invalid_argument("fixed_node: invalid block or slice");
    ensure_slices(path.num_slices());
    if (!sync(path, b, k)) return 0.0;
    const double g = current_grad2(path, b, k);
    return g > 0.0 ? 1.0 / std::sqrt(g) : kInfinity;
}

}  // namespace alea
