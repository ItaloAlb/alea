// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/slater_determinant.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Dense>

#include "population/walker_set.h"

namespace alea {

namespace {
// Large determinants retain contiguous per-walker matrices even
// when VMC splits its population into private batches below 64 walkers.
// Small determinants retain the previously measured small-batch SIMD route.
constexpr std::size_t min_tiled_walkers = 64;
constexpr std::size_t min_tiled_particles = 32;
constexpr std::size_t matrix_tile_walkers = 1;

constexpr std::size_t cache_stride(std::size_t nw, std::size_t n) noexcept {
    return nw < min_tiled_walkers && n < min_tiled_particles ? nw : matrix_tile_walkers;
}
constexpr std::size_t cache_slots(std::size_t nw) noexcept { return nw; }
constexpr std::size_t cache_tile_offset(std::size_t w, std::size_t width,
                                       std::size_t nw, std::size_t n) noexcept {
    return cache_stride(nw, n) == nw ? 0 : w * width;
}
constexpr std::size_t cache_lane(std::size_t w, std::size_t nw, std::size_t n) noexcept {
    return cache_stride(nw, n) == nw ? w : 0;
}
constexpr std::size_t cache_index(std::size_t element, std::size_t w,
                                 std::size_t width, std::size_t nw, std::size_t n) noexcept {
    return cache_tile_offset(w, width, nw, n) + element * cache_stride(nw, n) + cache_lane(w, nw, n);
}

// A full-bit mask zeros inactive operands before floating arithmetic, even
// if their committed value is nonfinite. Loads remain unconditional/contiguous;
// selecting committed output bitwise preserves rejected states exactly.
static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
inline double active_value(double value, std::uint64_t mask) noexcept {
    return std::bit_cast<double>(std::bit_cast<std::uint64_t>(value) & mask);
}
inline double selected_value(double next, double previous, std::uint64_t mask) noexcept {
    return std::bit_cast<double>((std::bit_cast<std::uint64_t>(next) & mask)
        | (std::bit_cast<std::uint64_t>(previous) & ~mask));
}
constexpr double kPi      = 3.14159265358979323846;
constexpr double kDetFloor = 1e-300;  // avoid log(0) on a walker sitting on a node
}  // namespace

SlaterDeterminant::SlaterDeterminant(std::vector<Block> blocks)
    : blocks_(std::move(blocks)) {
    // Flatten whatever the orbital sets expose, block-major.  The orbital owns
    // the physics of its own parameter (its default fixedness, whether its sign
    // is locked); the determinant only namespaces it by block, so a two-block
    // crystal's two sublattice widths stay distinguishable in the optimiser log.
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t n = blocks_[b].orbitals->n_parameters();
        for (std::size_t k = 0; k < n; ++k) {
            Parameter p = blocks_[b].orbitals->parameter(k);
            p.name = "b" + std::to_string(b) + "_" + p.name;
            params_.push_back(std::move(p));
            param_owner_.emplace_back(b, k);
        }
    }
}

void SlaterDeterminant::rebuild() {
    for (std::size_t i = 0; i < params_.size(); ++i) {
        const auto [b, k] = param_owner_[i];
        blocks_[b].orbitals->set_parameter(k, params_[i].value);
    }
}

// ∂ln|det D_b|/∂p = Tr(D_b⁻¹ ∂D_b/∂p).  With D_b(c,a) = φ_a(r_{first+c}) and
// A = D_b⁻¹ addressed by cache_index(a*N + c, w, N*N, nw, N), the trace is
//
//     Σ_c Σ_a A_{a,c} · ∂φ_a(r_{first+c})/∂p
//
// i.e. one pass over the block's rows, each contributing the inner product of
// the inverse's column c with the parameter derivative of the orbitals at that
// row's particle.  The committed A is exactly what the Metropolis machinery
// already maintains, so nothing is re-inverted here.
//
// Called once per measurement block (not inside the Metropolis loop), which is
// why the row scratch is a local allocation rather than a member: this method is
// const and shared read-only across the crowd threads that call it.
void SlaterDeterminant::compute_parameter_derivatives(
    const WalkerSet& batch, std::span<double> derivs) const {
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    std::fill(derivs.begin(), derivs.end(), 0.0);

    const std::span<const double> pos = batch.positions();

    // params_ is block-major, so this walks it and the blocks together.  `kopt`
    // indexes the PACKED output: a fixed parameter takes no column, which is the
    // layout TrialWavefunction and CostFunction assume.
    std::size_t first = 0;   // index of block b's first entry in params_
    std::size_t kopt  = 0;   // next free column of `derivs`
    std::vector<double> dval;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        const std::size_t n_par = blk.orbitals->n_parameters();
        std::size_t n_live = 0;
        for (std::size_t k = 0; k < n_par; ++k)
            if (!params_[first + k].is_fixed) ++n_live;
        if (n_live == 0) { first += n_par; continue; }

        const std::size_t N = blk.last - blk.first;
        const double* __restrict dinv = d_inv_[b].data();
        dval.assign(n_par * N * nw, 0.0);

        for (std::size_t c = 0; c < N; ++c) {
            // One orbital pass per row feeds every parameter of this block.
            blk.orbitals->evaluate_parameter_derivatives(
                pos.subspan((blk.first + c) * dim * nw, dim * nw), dim, dval,
                {}, {}, 0, nw);
            std::size_t col = kopt;
            for (std::size_t k = 0; k < n_par; ++k) {
                if (params_[first + k].is_fixed) continue;
                double* __restrict out = derivs.data() + col * nw;
                const double* __restrict dphi = dval.data() + k * N * nw;
                for (std::size_t a = 0; a < N; ++a) {
                    const double* __restrict da   = dphi + a * nw;
#pragma omp simd
                    for (std::size_t w = 0; w < nw; ++w) out[w] += dinv[cache_index(a * N + c, w, N * N, nw, N)] * da[w];
                }
                ++col;
            }
        }
        kopt  += n_live;
        first += n_par;
    }
}

// ∂(∂_d ln|det D|)/∂p and ∂(∂²_d ln|det D|)/∂p.  The inverse moves with p too:
//
//     G_d = Σ_m A_{m,j} ∂_d φ_m(r_j),   L_d = Σ_m A_{m,j} ∂²_d φ_m(r_j) − G_d²
//     ∂_p A = −A (∂_p D) A,             (∂_p D)_{c,a} = ∂_p φ_a(r_c)
//
// so ∂_p G_d = Σ_m [ (∂_p A)_{m,j} ∂_d φ_m + A_{m,j} ∂_p(∂_d φ_m) ]; dropping the
// first term gives a plausible wrong answer.  O(N³) per walker per parameter.
//
// Loop order is chosen for memory: ∂_p D and ∂_p A are kept whole ([N²·nw]
// without matrix-cache padding), the per-row orbital buffers one row at a time.
bool SlaterDeterminant::compute_parameter_gradient_derivatives(
    const WalkerSet& batch, std::span<double> dgrad,
    std::span<double> dlap) const {
    const std::size_t n_opt = num_optimizable_params();
    if (n_opt == 0) return true;
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    std::fill(dgrad.begin(), dgrad.end(), 0.0);
    std::fill(dlap.begin(), dlap.end(), 0.0);

    const std::span<const double> pos = batch.positions();
    const std::size_t stride = np * dim * nw;

    std::size_t first = 0, kopt = 0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        const std::size_t n_par = blk.orbitals->n_parameters();
        std::size_t n_live = 0;
        for (std::size_t k = 0; k < n_par; ++k)
            if (!params_[first + k].is_fixed) ++n_live;
        if (n_live == 0) { first += n_par; continue; }

        const std::size_t N = blk.last - blk.first;
        const double* __restrict dinv = d_inv_[b].data();
        auto row_pos = [&](std::size_t c) {
            return pos.subspan((blk.first + c) * dim * nw, dim * nw);
        };

        std::vector<double> dD_all(N * N * nw), dA_all(N * N * nw);
        std::vector<double> v(N * nw), g(N * dim * nw), l(N * dim * nw);
        std::vector<double> dv(n_par * N * nw);
        std::vector<double> dg(n_par * N * dim * nw), dl(n_par * N * dim * nw);
        Eigen::MatrixXd A(N, N), dD(N, N), dA(N, N);

        std::size_t col = kopt;
        for (std::size_t k = 0; k < n_par; ++k) {
            if (params_[first + k].is_fixed) continue;

            // (∂_p D)_{c,a} = ∂_p φ_a(r_c), one block row at a time.
            for (std::size_t c = 0; c < N; ++c) {
                blk.orbitals->evaluate_parameter_derivatives(row_pos(c), dim, dv,
                                                             {}, {}, 0, nw);
                for (std::size_t a = 0; a < N; ++a)
                    for (std::size_t w = 0; w < nw; ++w)
                        dD_all[(c * N + a) * nw + w] = dv[(k * N + a) * nw + w];
            }

            // ∂_p A = −A (∂_p D) A, per walker.
            for (std::size_t w = 0; w < nw; ++w) {
                for (std::size_t a = 0; a < N; ++a)
                    for (std::size_t c = 0; c < N; ++c) {
                        A(a, c)  = dinv[cache_index((a * N + c), w, N * N, nw, N)];
                        dD(a, c) = dD_all[(a * N + c) * nw + w];
                    }
                dA.noalias() = -A * dD * A;
                for (std::size_t a = 0; a < N; ++a)
                    for (std::size_t c = 0; c < N; ++c)
                        dA_all[(a * N + c) * nw + w] = dA(a, c);
            }

            double* __restrict og = dgrad.data() + col * stride;
            double* __restrict ol = dlap.data() + col * stride;
            for (std::size_t j = 0; j < N; ++j) {
                blk.orbitals->evaluate(row_pos(j), dim, v, g, l, 0, nw);
                blk.orbitals->evaluate_parameter_derivatives(row_pos(j), dim, dv,
                                                             dg, dl, 0, nw);
                const std::size_t particle = blk.first + j;
                for (std::size_t d = 0; d < dim; ++d) {
                    for (std::size_t w = 0; w < nw; ++w) {
                        double dG = 0.0, dLraw = 0.0;
                        for (std::size_t m = 0; m < N; ++m) {
                            const double a_mj  = dinv[cache_index((m * N + j), w, N * N, nw, N)];
                            const double da_mj = dA_all[(m * N + j) * nw + w];
                            const std::size_t oi = (m * dim + d) * nw + w;
                            const std::size_t ki = ((k * N + m) * dim + d) * nw + w;
                            dG    += da_mj * g[oi] + a_mj * dg[ki];
                            dLraw += da_mj * l[oi] + a_mj * dl[ki];
                        }
                        const std::size_t idx = (particle * dim + d) * nw + w;
                        // L = Lraw − G², so ∂_p L = ∂_p Lraw − 2 G ∂_p G.
                        og[idx] = dG;
                        ol[idx] = dLraw - 2.0 * grad_c_[idx] * dG;
                    }
                }
            }
            ++col;
        }
        kopt  += n_live;
        first += n_par;
    }
    return true;
}

std::size_t SlaterDeterminant::block_of(std::size_t p) const noexcept {
    for (std::size_t b = 0; b < blocks_.size(); ++b)
        if (p >= blocks_[b].first && p < blocks_[b].last) return b;
    return npos;
}

void SlaterDeterminant::resize_scratch(std::size_t nw, std::size_t dim) {
    std::size_t maxN = 0;
    for (const Block& blk : blocks_)
        maxN = std::max(maxN, blk.last - blk.first);
    max_block_size_ = maxN;
    column_.assign(maxN * nw, 0.0);
    row_product_.assign(maxN * nw, 0.0);
    walker_column_.assign(nw * maxN, 0.0);
    walker_row_product_.assign(nw * maxN, 0.0);
    nval_.assign(maxN * nw, 0.0);
    ngrad_.assign(maxN * dim * nw, 0.0);
    nlap_.assign(maxN * dim * nw, 0.0);
}

void SlaterDeterminant::resize_state(std::size_t nw, std::size_t dim) {
    dim_ = dim;
    nw_  = nw;
    d_inv_.resize(blocks_.size());
    values_.resize(blocks_.size());
    gradients_.resize(blocks_.size());
    laplacians_.resize(blocks_.size());
    block_log_.resize(blocks_.size());
    updates_.resize(blocks_.size());
    condition_.resize(blocks_.size());
    row_abs_sum_.resize(blocks_.size());
    work_.resize(blocks_.size());
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const std::size_t N = blocks_[b].last - blocks_[b].first;
        d_inv_[b].assign(N * N * cache_slots(nw), 0.0);
        values_[b].assign(N * N * cache_slots(nw), 0.0);
        gradients_[b].assign(N * N * dim * cache_slots(nw), 0.0);
        laplacians_[b].assign(N * N * dim * cache_slots(nw), 0.0);
        block_log_[b].assign(nw, 0.0);
        updates_[b].assign(nw, 0);
        condition_[b].assign(nw, 1.0);
        row_abs_sum_[b].assign(N * nw, 0.0);
        work_[b].assign(N * N * cache_slots(nw), 0.0);
    }
    resize_scratch(nw, dim);
}

void SlaterDeterminant::prepare(std::size_t nw, std::size_t np, std::size_t dim) {
    resize_caches(nw, np, dim);
    resize_state(nw, dim);
}

void SlaterDeterminant::evaluate_batch(WalkerSet& batch) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t np  = batch.n_particles();
    const std::size_t dim = batch.dim();
    prepare(nw, np, dim);   // zeros committed caches, (re)sizes d_inv_ + scratch

    const std::span<const double> pos = batch.positions();

    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& blk = blocks_[b];
        const std::size_t N = blk.last - blk.first;
        const std::size_t matrix_stride = cache_stride(nw, N);

        // Orbital values/derivatives at every block particle's position.
        for (std::size_t kr = 0; kr < N; ++kr) {
            const std::size_t particle = blk.first + kr;
            if (matrix_stride == nw) {
                blk.orbitals->evaluate(pos.subspan(particle * dim * nw, dim * nw),
                    dim, std::span(values_[b]).subspan(kr * N * nw, N * nw),
                    std::span(gradients_[b]).subspan(kr * N * dim * nw, N * dim * nw),
                    std::span(laplacians_[b]).subspan(kr * N * dim * nw, N * dim * nw),
                    0, nw);
            } else {
                blk.orbitals->evaluate(pos.subspan(particle * dim * nw, dim * nw),
                                       dim, nval_, ngrad_, nlap_, 0, nw);
                for (std::size_t m = 0; m < N; ++m) {
                    for (std::size_t w = 0; w < nw; ++w)
                        values_[b][cache_index((kr * N + m), w, N * N, nw, N)] = nval_[m * nw + w];
                    for (std::size_t d = 0; d < dim; ++d)
                        for (std::size_t w = 0; w < nw; ++w) {
                            const auto dst = cache_index((kr * N + m) * dim + d, w, N * N * dim, nw, N);
                            const auto src = (m * dim + d) * nw + w;
                            gradients_[b][dst] = ngrad_[src];
                            laplacians_[b][dst] = nlap_[src];
                        }
                }
            }
        }

        // Row sums in the column order every commit path uses, so a cached
        // row equals the one a commit would recompute.
        for (std::size_t kr = 0; kr < N; ++kr)
            for (std::size_t w = 0; w < nw; ++w) {
                double row_sum = 0.0;
                for (std::size_t i = 0; i < N; ++i)
                    row_sum += std::abs(values_[b][cache_index(kr * N + i, w, N * N, nw, N)]);
                row_abs_sum_[b][kr * nw + w] = row_sum;
            }

        double* __restrict dinv = d_inv_[b].data();
        for (std::size_t w = 0; w < nw; ++w) {
            Eigen::MatrixXd D(N, N);
            for (std::size_t kr = 0; kr < N; ++kr)
                for (std::size_t i = 0; i < N; ++i)
                    D(kr, i) = values_[b][cache_index((kr * N + i), w, N * N, nw, N)];
            const Eigen::MatrixXd A = D.inverse();
            block_log_[b][w] = std::log(std::max(std::abs(D.determinant()), kDetFloor));
            log_psi_c_[w] += block_log_[b][w];
            condition_[b][w] = D.cwiseAbs().rowwise().sum().maxCoeff()
                            * A.cwiseAbs().rowwise().sum().maxCoeff();

            for (std::size_t a = 0; a < N; ++a)
                for (std::size_t c = 0; c < N; ++c)
                    dinv[cache_index((a * N + c), w, N * N, nw, N)] = A(a, c);

            for (std::size_t kr = 0; kr < N; ++kr) {
                const std::size_t particle = blk.first + kr;
                for (std::size_t d = 0; d < dim; ++d) {
                    double G = 0.0, Lraw = 0.0;
                    for (std::size_t m = 0; m < N; ++m) {
                        const double a = A(m, kr);
                        G    += a * gradients_[b][cache_index(((kr * N + m) * dim + d), w, N * N * dim, nw, N)];
                        Lraw += a * laplacians_[b][cache_index(((kr * N + m) * dim + d), w, N * N * dim, nw, N)];
                    }
                    const std::size_t idx = (particle * dim + d) * nw + w;
                    grad_c_[idx] += G;
                    lap_c_[idx]  += Lraw - G * G;
                }
            }
        }
    }

    std::span<double> b_logpsi = batch.log_psi();
    for (std::size_t w = 0; w < nw; ++w) b_logpsi[w] += log_psi_c_[w];
    std::span<double> b_grad = batch.gradients();
    for (std::size_t k = 0; k < b_grad.size(); ++k) b_grad[k] += grad_c_[k];
    std::span<double> b_lap = batch.laplacians();
    for (std::size_t k = 0; k < b_lap.size(); ++k) b_lap[k] += lap_c_[k];
}

void SlaterDeterminant::ratio_batch(WalkerSet& batch, std::size_t p,
                                    std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t b = block_of(p);
    if (b == npos) return;   // particle in no block → ψ_det unchanged (ratio 1)

    const Block& blk = blocks_[b];
    const std::size_t N    = blk.last - blk.first;
    const std::size_t jrow = p - blk.first;

    blk.orbitals->evaluate(batch.new_pos(), dim, nval_, ngrad_, nlap_,
                           w_begin, w_end);

    std::span<double> log_ratios = batch.log_ratios();
    std::span<double> phase      = batch.phase_diffs();
    std::span<double> new_grad   = batch.new_grad();
    std::span<double> new_lap    = batch.new_lap();
    const double* __restrict dinv = d_inv_[b].data();

    for (std::size_t w = w_begin; w < w_end; ++w) {
        double R = 0.0;
        for (std::size_t m = 0; m < N; ++m)
            R += dinv[cache_index((m * N + jrow), w, N * N, nw, N)] * nval_[m * nw + w];
        log_ratios[w] += std::log(std::max(std::abs(R), kDetFloor));
        if (R < 0.0) phase[w] += kPi;   // node crossing → fixed-node gate rejects

        const double invR = (R != 0.0) ? 1.0 / R : 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
            double G = 0.0, Lraw = 0.0;
            for (std::size_t m = 0; m < N; ++m) {
                const double a = dinv[cache_index((m * N + jrow), w, N * N, nw, N)];  // A_old column jrow
                G    += a * ngrad_[(m * dim + d) * nw + w];
                Lraw += a * nlap_[(m * dim + d) * nw + w];
            }
            const double Gp = invR * G;   // A'_{m,jrow} = A_{m,jrow}/R
            new_grad[d * nw + w] += Gp;
            new_lap[d * nw + w]  += invR * Lraw - Gp * Gp;
        }
    }
}

void SlaterDeterminant::shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                                  std::size_t a_last, std::size_t b_first,
                                                  std::size_t b_last,
                                                  std::span<const double> shift,
                                                  std::span<double> log_ratios,
                                                  std::span<double> phases) {
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    if (nw != nw_)
        throw std::logic_error("slater determinant: shifted pair ratios need the batch it evaluated");
    const std::size_t n_a = a_last - a_first, n_b = b_last - b_first, n_t = n_a + n_b;
    const std::size_t maxN = max_block_size_;
    const std::span<const double> pos = batch.positions();
    shift_pos_.resize(dim * nw);
    if (shift_rows_.size() < n_t * nw * maxN) shift_rows_.resize(n_t * nw * maxN);
    const auto particle = [&](std::size_t t) { return t < n_a ? a_first + t : b_first + t - n_a; };

    // R_t(y) = Σ_m φ_m(r_t + s) (D⁻¹)_{m y} for every row y of t's block.
    for (std::size_t t = 0; t < n_t; ++t) {
        const std::size_t x = particle(t);
        const std::size_t b = block_of(x);
        if (b == npos) continue;
        const Block& blk = blocks_[b];
        const std::size_t N = blk.last - blk.first;
        // r + s folded into the cell, as an accepted move would leave it.
        for (std::size_t w = 0; w < nw; ++w) {
            double r[3] = {0.0, 0.0, 0.0};
            for (std::size_t d = 0; d < dim; ++d)
                r[d] = pos[(x * dim + d) * nw + w] + shift[d * nw + w];
            batch.cell().wrap(r);
            for (std::size_t d = 0; d < dim; ++d) shift_pos_[d * nw + w] = r[d];
        }
        blk.orbitals->evaluate(shift_pos_, dim, nval_, ngrad_, nlap_, 0, nw);
        const double* dinv = d_inv_[b].data();
        for (std::size_t w = 0; w < nw; ++w) {
            double* rows = shift_rows_.data() + (t * nw + w) * maxN;
            for (std::size_t y = 0; y < N; ++y) {
                double r = 0.0;
                for (std::size_t m = 0; m < N; ++m)
                    r += dinv[cache_index(m * N + y, w, N * N, nw, N)] * nval_[m * nw + w];
                rows[y] = r;
            }
        }
    }

    for (std::size_t ii = 0; ii < n_a; ++ii)
        for (std::size_t jj = 0; jj < n_b; ++jj) {
            const std::size_t i = a_first + ii, j = b_first + jj;
            const std::size_t bi = block_of(i), bj = block_of(j);
            if (bi == npos && bj == npos) continue;
            double* lr = log_ratios.data() + (ii * n_b + jj) * nw;
            double* ph = phases.data() + (ii * n_b + jj) * nw;
            const std::size_t ti = ii, tj = n_a + jj;
            const std::size_t yi = bi == npos ? 0 : i - blocks_[bi].first;
            const std::size_t yj = bj == npos ? 0 : j - blocks_[bj].first;
            for (std::size_t w = 0; w < nw; ++w) {
                const double* ri = shift_rows_.data() + (ti * nw + w) * maxN;
                const double* rj = shift_rows_.data() + (tj * nw + w) * maxN;
                const double ratio =
                    bi == bj ? ri[yi] * rj[yj] - ri[yj] * rj[yi]
                             : (bi != npos ? ri[yi] : 1.0) * (bj != npos ? rj[yj] : 1.0);
                lr[w] += std::log(std::max(std::abs(ratio), kDetFloor));
                if (ratio < 0.0) ph[w] += kPi;
            }
        }
}

void SlaterDeterminant::apply_move_batch(WalkerSet& batch, std::size_t p,
    std::span<const bool> accept, std::size_t w_begin, std::size_t w_end) {
    apply_move_impl(batch, p, accept, w_begin, w_end, true);
}

void SlaterDeterminant::apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
    std::span<const bool> accept, std::size_t w_begin, std::size_t w_end) {
    apply_move_impl(batch, p, accept, w_begin, w_end, false);
}

void SlaterDeterminant::apply_move_impl(WalkerSet& batch, std::size_t p,
    std::span<const bool> accept, std::size_t w_begin, std::size_t w_end,
    bool refresh_partners) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    if (w_end > nw) w_end = nw;
    if (w_begin >= w_end) return;
    const std::size_t b = block_of(p);
    if (b == npos) return;

    const Block& blk = blocks_[b];
    const std::size_t N    = blk.last - blk.first;
    const std::size_t matrix_stride = cache_stride(nw, N);
    const std::size_t jrow = p - blk.first;

    std::span<double> b_logpsi = batch.log_psi();
    double* b_grad = batch.gradients().data();
    double* b_lap = batch.laplacians().data();
    double* column = column_.data();
    double* row_product = row_product_.data();
    const double* nval = nval_.data();
    const double* ngrad = ngrad_.data();
    const double* nlap = nlap_.data();
    double* component_grad = grad_c_.data();
    double* component_lap = lap_c_.data();

    // Keep a small contiguous walker tile in the inner loops. The orbital
    // reduction order within each walker is unchanged; only independent
    // walkers are interleaved. Scratch is stack-local, so disjoint crowds do
    // not share it. Rebuilds retain the scalar pivot order and criteria below.
    const auto apply_tiles = [&]<std::size_t tile_size>() {
        for (std::size_t first = w_begin, count = 0; first < w_end; first += count) {
            const std::size_t lane = cache_lane(first, nw, N);
            count = std::min(std::min(tile_size, w_end - first), matrix_stride - lane);
            std::array<std::uint64_t, tile_size> active{}, incremental{}; // [tile walker]
            std::array<double, tile_size> ratio{}, ratio_scale{}, old_log{}; // [tile walker]
            bool any_accepted = false;
            for (std::size_t i = 0; i < count; ++i) {
                active[i] = accept[first + i] ? ~std::uint64_t{0} : 0;
                any_accepted = any_accepted || accept[first + i];
            }
            if (!any_accepted) continue;
            double* __restrict dinv = d_inv_[b].data() + cache_tile_offset(first, N * N, nw, N);
            double* values = values_[b].data() + cache_tile_offset(first, N * N, nw, N);
            double* gradients = gradients_[b].data() + cache_tile_offset(first, N * N * dim, nw, N);
            double* laps = laplacians_[b].data() + cache_tile_offset(first, N * N * dim, nw, N);
            // D' = D + e_j (v' - v)^T. Sherman-Morrison gives
            // A' = A - (A e_j) (v'^T A - e_j^T) / R, R = v'^T A e_j.
            // Sherman & Morrison, Ann. Math. Statist. 21, 124-127 (1950).
            // This is exact determinant algebra; no sampling approximation or
            // RNG draw changes. Periodic/ill-conditioned rebuilds bound drift.
            for (std::size_t m = 0; m < N; ++m) {
#pragma omp simd
                for (std::size_t i = 0; i < count; ++i) {
                    const std::size_t w = first + i;
                    const double inverse = active_value(dinv[(m * N + jrow) * matrix_stride + lane + i], active[i]);
                    const double value = active_value(nval[m * nw + w], active[i]);
                    const double term = inverse * value;
                    ratio[i] += term;
                    ratio_scale[i] += std::abs(term);
                    column[m * nw + w] = inverse;
                }
            }
            for (std::size_t c = 0; c < N; ++c) {
                std::array<double, tile_size> product{}; // [tile walker]
                for (std::size_t m = 0; m < N; ++m) {
#pragma omp simd
                    for (std::size_t i = 0; i < count; ++i) {
                        const std::size_t w = first + i;
                        const double value = active_value(nval[m * nw + w], active[i]);
                        const double inverse = active_value(dinv[(m * N + c) * matrix_stride + lane + i], active[i]);
                        product[i] += value * inverse;
                    }
                }
#pragma omp simd
                for (std::size_t i = 0; i < count; ++i) {
                    row_product[c * nw + first + i] = product[i] - (c == jrow ? 1.0 : 0.0);
                }
            }
            for (std::size_t m = 0; m < N; ++m) {
#pragma omp simd
                for (std::size_t i = 0; i < count; ++i) {
                    auto& target = values[(jrow * N + m) * matrix_stride + lane + i];
                    target = selected_value(nval[m * nw + first + i], target, active[i]);
                }
                for (std::size_t d = 0; d < dim; ++d) {
#pragma omp simd
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto dst = ((jrow * N + m) * dim + d) * matrix_stride + lane + i;
                        const auto src = (m * dim + d) * nw + first + i;
                        gradients[dst] = selected_value(ngrad[src], gradients[dst], active[i]);
                        laps[dst] = selected_value(nlap[src], laps[dst], active[i]);
                    }
                }
            }
            for (std::size_t i = 0; i < count; ++i) {
                if (!active[i]) continue;
                const std::size_t w = first + i;
                old_log[i] = block_log_[b][w];
                const double R = ratio[i];
                const bool rebuild = ++updates_[b][w] >= 64 || !std::isfinite(R)
                    || std::abs(R) < 1e-5 || condition_[b][w] > 1e8
                    || std::abs(R) <= 1e4 * std::numeric_limits<double>::epsilon() * ratio_scale[i];
                incremental[i] = rebuild ? 0 : ~std::uint64_t{0};
                if (rebuild) block_log_[b][w] = rebuild_inverse(b, w);
                else block_log_[b][w] += std::log(std::abs(R));
            }
            for (std::size_t a = 0; a < N; ++a) {
                std::array<double, tile_size> factor{}; // [tile walker]
#pragma omp simd
                for (std::size_t i = 0; i < count; ++i) {
                    const double numerator = active_value(column[a * nw + first + i], incremental[i]);
                    const double denominator = selected_value(ratio[i], 1.0, incremental[i]);
                    factor[i] = numerator / denominator;
                }
                for (std::size_t c = 0; c < N; ++c) {
#pragma omp simd
                    for (std::size_t i = 0; i < count; ++i) {
                        const std::size_t w = first + i;
                        const auto index = (a * N + c) * matrix_stride + lane + i;
                        const double inverse = active_value(dinv[index], incremental[i]);
                        const double product = active_value(row_product[c * nw + w], incremental[i]);
                        const double updated = inverse - factor[i] * product;
                        dinv[index] = selected_value(updated, dinv[index], incremental[i]);
                    }
                }
            }
            std::array<double, tile_size> norm_d{}, norm_a{}; // [tile walker]
            for (std::size_t a = 0; a < N; ++a) {
                std::array<double, tile_size> row_d{}, row_a{}; // [tile walker]
                for (std::size_t c = 0; c < N; ++c) {
#pragma omp simd
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto index = (a * N + c) * matrix_stride + lane + i;
                        row_d[i] += std::abs(active_value(values[index], active[i]));
                        row_a[i] += std::abs(active_value(dinv[index], active[i]));
                    }
                }
#pragma omp simd
                for (std::size_t i = 0; i < count; ++i) {
                    norm_d[i] = std::max(norm_d[i], row_d[i]);
                    norm_a[i] = std::max(norm_a[i], row_a[i]);
                }
                // Keep the cached row sums valid for the contiguous layout,
                // which a population change can switch this block to.
                for (std::size_t i = 0; i < count; ++i)
                    if (active[i]) row_abs_sum_[b][a * nw + first + i] = row_d[i];
            }
            for (std::size_t i = 0; i < count; ++i) {
                if (!active[i]) continue;
                const std::size_t w = first + i;
                condition_[b][w] = norm_d[i] * norm_a[i];
                if (incremental[i] && (!std::isfinite(condition_[b][w]) || condition_[b][w] > 1e8))
                    block_log_[b][w] = rebuild_inverse(b, w);
                const double dlog = block_log_[b][w] - old_log[i];
                log_psi_c_[w] += dlog;
                b_logpsi[w] += dlog;
            }

            // Recompute block grad/lap.  Partners (kr != jrow): push the delta into
            // the batch totals (apply_move does not touch them).  Moved particle p
            // (kr == jrow): only refresh the cache — apply_move overwrites its totals
            // from new_grad/new_lap (which equal these values).
            const std::size_t derivative_begin = refresh_partners ? 0 : jrow;
            const std::size_t derivative_end = refresh_partners ? N : jrow + 1;
            for (std::size_t kr = derivative_begin; kr < derivative_end; ++kr) {
                const std::size_t particle = blk.first + kr;
                for (std::size_t d = 0; d < dim; ++d) {
                    std::array<double, tile_size> grad{}, lap{}; // [tile walker]
                    for (std::size_t m = 0; m < N; ++m) {
#pragma omp simd
                        for (std::size_t i = 0; i < count; ++i) {
                            const auto index = ((kr * N + m) * dim + d) * matrix_stride + lane + i;
                            const double a = active_value(dinv[(m * N + kr) * matrix_stride + lane + i], active[i]);
                            const double g = active_value(gradients[index], active[i]);
                            const double l = active_value(laps[index], active[i]);
                            grad[i] += a * g;
                            lap[i] += a * l;
                        }
                    }
#pragma omp simd
                    for (std::size_t i = 0; i < count; ++i) {
                        const double Gnew = grad[i];
                        const double Lnew = lap[i] - Gnew * Gnew;
                        const std::size_t idx = (particle * dim + d) * nw + first + i;
                        if (kr != jrow) {
                            // Subtract before adding to preserve node recovery.
                            const double bg = active_value(b_grad[idx], active[i]);
                            const double bl = active_value(b_lap[idx], active[i]);
                            const double cg = active_value(component_grad[idx], active[i]);
                            const double cl = active_value(component_lap[idx], active[i]);
                            const double updated_g = (bg - cg) + Gnew;
                            const double updated_l = (bl - cl) + Lnew;
                            b_grad[idx] = selected_value(updated_g, b_grad[idx], active[i]);
                            b_lap[idx] = selected_value(updated_l, b_lap[idx], active[i]);
                        }
                        component_grad[idx] = selected_value(Gnew, component_grad[idx], active[i]);
                        component_lap[idx] = selected_value(Lnew, component_lap[idx], active[i]);
                    }
                }
            }
        }
    };
    // Small private batches vectorize across walkers. Larger batches operate
    // on one contiguous matrix at a time; no per-move packing is needed.
    if (cache_stride(nw, N) == nw) {
        apply_tiles.template operator()<32>();
        return;
    }
    for (std::size_t w = w_begin; w < w_end; ++w)
        if (accept[w]) commit_walker(batch, b, jrow, w, refresh_partners);
}

// Accepted move of particle j for one walker whose block matrices are stored
// contiguously ([w*N*N + row*N + column]).  Same Sherman–Morrison algebra,
// rebuild triggers and per-element summation order as the walker-tile path
// above; only the traversal changes, so that A = D⁻¹ is streamed twice per
// accepted move instead of four times, and D only in its changed row:
//
//   pass 1, read A:        u = A e_j,  R = v'ᵀ u,  rᵀ = v'ᵀ A − e_jᵀ
//   pass 2, read/write A:  A' = A − u rᵀ / R,  ‖A'‖∞ = max_a Σ_c |A'_ac|
//   ‖D'‖∞ = max_a Σ_c |D'_ac| from the cached row sums; only row j changed.
//
// Both passes walk rows of A, so the inner loops are unit-stride over the
// column c and vectorize. The ‖A'‖∞ row sums are SIMD reductions: they may
// differ from the tile path in the last bits, which only matters for the
// κ∞ = ‖D'‖∞‖A'‖∞ > 1e8 rebuild trigger; every other quantity is formed
// with the same operations in the same order.
// Sherman & Morrison, Ann. Math. Statist. 21, 124-127 (1950).
void SlaterDeterminant::commit_walker(WalkerSet& batch, std::size_t b, std::size_t jrow,
                                      std::size_t w, bool refresh_partners) {
    const std::size_t nw  = batch.num_walkers();
    const std::size_t dim = batch.dim();
    const Block& blk = blocks_[b];
    const std::size_t N = blk.last - blk.first;

    // A and D are not __restrict here: rebuild_inverse rewrites A and reads D
    // through the members. Loop-scoped row pointers carry the no-alias hint.
    double* inverse = d_inv_[b].data() + w * N * N;     // A [N * N]
    double* values  = values_[b].data() + w * N * N;    // D [N * N]
    double* __restrict gradients  = gradients_[b].data() + w * N * N * dim;
    double* __restrict laplacians = laplacians_[b].data() + w * N * N * dim;
    double* __restrict column      = walker_column_.data() + w * max_block_size_;      // u [N]
    double* __restrict row_product = walker_row_product_.data() + w * max_block_size_; // r [N]
    double* __restrict row_abs     = row_abs_sum_[b].data();                        // [N * nw]

    // Commit the proposed row j of D and its derivative rows first: pass 1
    // reads v' contiguously from it, and a rebuild inverts the accepted D.
    double* new_row = values + jrow * N;   // v'_m = φ_m(r_j')
    for (std::size_t m = 0; m < N; ++m) new_row[m] = nval_[m * nw + w];
    for (std::size_t m = 0; m < N; ++m)
        for (std::size_t d = 0; d < dim; ++d) {
            const std::size_t src = (m * dim + d) * nw + w;
            gradients[(jrow * N + m) * dim + d]  = ngrad_[src];
            laplacians[(jrow * N + m) * dim + d] = nlap_[src];
        }

    // Pass 1.
    double ratio = 0.0, ratio_scale = 0.0;
    std::fill_n(row_product, N, 0.0);
    for (std::size_t m = 0; m < N; ++m) {
        const double* __restrict inverse_row = inverse + m * N;
        const double v = new_row[m];
        const double term = inverse_row[jrow] * v;
        ratio += term;
        ratio_scale += std::abs(term);
        column[m] = inverse_row[jrow];
#pragma omp simd
        for (std::size_t c = 0; c < N; ++c) row_product[c] += v * inverse_row[c];
    }
    row_product[jrow] -= 1.0;

    const double old_log = block_log_[b][w];
    const bool rebuild = ++updates_[b][w] >= 64 || !std::isfinite(ratio)
        || std::abs(ratio) < 1e-5 || condition_[b][w] > 1e8
        || std::abs(ratio) <= 1e4 * std::numeric_limits<double>::epsilon() * ratio_scale;

    // Pass 2 (or a pivoted rebuild, after which only the norm is needed).
    double norm_a = 0.0;
    if (rebuild) {
        block_log_[b][w] = rebuild_inverse(b, w);
        for (std::size_t a = 0; a < N; ++a) {
            const double* __restrict inverse_row = inverse + a * N;
            double row_sum = 0.0;
#pragma omp simd reduction(+ : row_sum)
            for (std::size_t c = 0; c < N; ++c) row_sum += std::abs(inverse_row[c]);
            norm_a = std::max(norm_a, row_sum);
        }
    } else {
        block_log_[b][w] += std::log(std::abs(ratio));
        for (std::size_t a = 0; a < N; ++a) {
            double* __restrict inverse_row = inverse + a * N;
            const double factor = column[a] / ratio;
            double row_sum = 0.0;
#pragma omp simd reduction(+ : row_sum)
            for (std::size_t c = 0; c < N; ++c) {
                inverse_row[c] = inverse_row[c] - factor * row_product[c];
                row_sum += std::abs(inverse_row[c]);
            }
            norm_a = std::max(norm_a, row_sum);
        }
    }

    double new_row_sum = 0.0;   // sequential, like every other row sum
    for (std::size_t c = 0; c < N; ++c) new_row_sum += std::abs(new_row[c]);
    row_abs[jrow * nw + w] = new_row_sum;
    double norm_d = 0.0;
    for (std::size_t a = 0; a < N; ++a) norm_d = std::max(norm_d, row_abs[a * nw + w]);

    condition_[b][w] = norm_d * norm_a;
    if (!rebuild && (!std::isfinite(condition_[b][w]) || condition_[b][w] > 1e8))
        block_log_[b][w] = rebuild_inverse(b, w);
    const double dlog = block_log_[b][w] - old_log;
    log_psi_c_[w] += dlog;
    batch.log_psi()[w] += dlog;

    // Block derivatives at the committed A'. The moved particle only refreshes
    // the component cache: apply_move overwrites its totals from new_grad/new_lap.
    // Eager partners push their change into the batch totals, subtracting the
    // old component before adding the new one to preserve node recovery.
    double* __restrict b_grad = batch.gradients().data();
    double* __restrict b_lap  = batch.laplacians().data();
    const std::size_t derivative_begin = refresh_partners ? 0 : jrow;
    const std::size_t derivative_end   = refresh_partners ? N : jrow + 1;
    for (std::size_t kr = derivative_begin; kr < derivative_end; ++kr) {
        const std::size_t particle = blk.first + kr;
        for (std::size_t d = 0; d < dim; ++d) {
            double grad = 0.0, lap = 0.0;
            for (std::size_t m = 0; m < N; ++m) {
                const double a = inverse[m * N + kr];
                grad += a * gradients[(kr * N + m) * dim + d];
                lap  += a * laplacians[(kr * N + m) * dim + d];
            }
            const double lap_new = lap - grad * grad;
            const std::size_t idx = (particle * dim + d) * nw + w;
            if (kr != jrow) {
                b_grad[idx] = (b_grad[idx] - grad_c_[idx]) + grad;
                b_lap[idx]  = (b_lap[idx] - lap_c_[idx]) + lap_new;
            }
            grad_c_[idx] = grad;
            lap_c_[idx]  = lap_new;
        }
    }
}

// Differentiating log(det D) gives the same derivatives as the eager commit:
// G_p = sum_m A_mp phi'_m, L_p = sum_m A_mp phi''_m - G_p^2.
// See the class derivative convention and the Sherman-Morrison reference above.
// Deferral changes only evaluation time; drift and H request current fields.
void SlaterDeterminant::refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
    std::size_t w_begin, std::size_t w_end) {
    const std::size_t nw = batch.num_walkers(), dim = batch.dim();
    w_end = std::min(w_end, nw);
    if (w_begin >= w_end) return;
    const auto b = block_of(p);
    if (b == npos) return;
    const auto N = blocks_[b].last - blocks_[b].first;
    const auto kr = p - blocks_[b].first;
    auto bg = batch.gradients();
    auto bl = batch.laplacians();
    for (std::size_t d = 0; d < dim; ++d)
        for (std::size_t w = w_begin; w < w_end; ++w) {
            double grad = 0.0, lap = 0.0;
            for (std::size_t m = 0; m < N; ++m) {
                const double a = d_inv_[b][cache_index(m * N + kr, w, N * N, nw, N)];
                const auto j = cache_index((kr * N + m) * dim + d, w, N * N * dim, nw, N);
                grad += a * gradients_[b][j];
                lap += a * laplacians_[b][j];
            }
            lap -= grad * grad;
            const auto j = (p * dim + d) * nw + w;
            // Already-current fields must remain untouched: subtracting and
            // re-adding a component can cancel a smaller component's bits.
            if (grad != grad_c_[j]) bg[j] = (bg[j] - grad_c_[j]) + grad;
            if (lap != lap_c_[j]) bl[j] = (bl[j] - lap_c_[j]) + lap;
            grad_c_[j] = grad;
            lap_c_[j] = lap;
        }
}

void SlaterDeterminant::refresh_derivatives(WalkerSet& batch, std::size_t w_begin,
    std::size_t w_end) {
    for (const auto& block : blocks_)
        for (std::size_t p = block.first; p < block.last; ++p)
            refresh_particle_derivatives(batch, p, w_begin, w_end);
}

// Pivoted Gauss-Jordan rebuild of A=D^-1, using preallocated, walker-local
// planes. Golub & Van Loan, Matrix Computations, 4th ed., Sec. 3.1 (2013).
// Accumulate log|det D| from pivots instead of multiplying small determinants.
double SlaterDeterminant::rebuild_inverse(std::size_t b, std::size_t w) {
    const std::size_t N = blocks_[b].last - blocks_[b].first;
    const std::size_t matrix_stride = cache_stride(nw_, N);
    if (matrix_stride == 1) {
        // Contiguous per-walker matrices: the same pivots and the same
        // operations on every element of A⁻¹ as the general loop below, on
        // raw rows so the column loops vectorize.  Columns c ≤ k of the
        // reduced matrix are never read after step k, so they are skipped.
        double* __restrict m = work_[b].data() + w * N * N;     // [N * N]
        double* __restrict a = d_inv_[b].data() + w * N * N;    // [N * N]
        const double* __restrict d = values_[b].data() + w * N * N;
        for (std::size_t r = 0; r < N; ++r)
            for (std::size_t c = 0; c < N; ++c) {
                m[r * N + c] = d[r * N + c];
                a[r * N + c] = r == c ? 1.0 : 0.0;
            }
        double log_det = 0.0;
        for (std::size_t k = 0; k < N; ++k) {
            std::size_t pivot = k;
            for (std::size_t r = k + 1; r < N; ++r)
                if (std::abs(m[r * N + k]) > std::abs(m[pivot * N + k])) pivot = r;
            const double diag = m[pivot * N + k];
            if (diag == 0.0 || !std::isfinite(diag))
                throw std::runtime_error("Slater determinant: accepted matrix is singular or nonfinite");
            log_det += std::log(std::abs(diag));
            if (pivot != k) {
                std::swap_ranges(m + k * N + k, m + k * N + N, m + pivot * N + k);
                std::swap_ranges(a + k * N, a + k * N + N, a + pivot * N);
            }
            double* __restrict m_k = m + k * N;
            double* __restrict a_k = a + k * N;
#pragma omp simd
            for (std::size_t c = k + 1; c < N; ++c) m_k[c] /= diag;
#pragma omp simd
            for (std::size_t c = 0; c < N; ++c) a_k[c] /= diag;
            for (std::size_t r = 0; r < N; ++r) {
                if (r == k) continue;
                double* __restrict m_r = m + r * N;
                double* __restrict a_r = a + r * N;
                const double factor = m_r[k];
#pragma omp simd
                for (std::size_t c = k + 1; c < N; ++c) m_r[c] -= factor * m_k[c];
#pragma omp simd
                for (std::size_t c = 0; c < N; ++c) a_r[c] -= factor * a_k[c];
            }
        }
        updates_[b][w] = 0;
        return log_det;
    }
    const std::size_t offset = cache_tile_offset(w, N * N, nw_, N);
    const std::size_t lane = cache_lane(w, nw_, N);
    auto at = [&](std::size_t r, std::size_t c) { return offset + (r * N + c) * matrix_stride + lane; };
    auto& matrix = work_[b];
    auto& inverse = d_inv_[b];
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c) {
            matrix[at(r, c)] = values_[b][at(r, c)];
            inverse[at(r, c)] = r == c ? 1.0 : 0.0;
        }
    double log_det = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        std::size_t pivot = k;
        for (std::size_t r = k + 1; r < N; ++r)
            if (std::abs(matrix[at(r, k)]) > std::abs(matrix[at(pivot, k)])) pivot = r;
        const double diag = matrix[at(pivot, k)];
        if (diag == 0.0 || !std::isfinite(diag))
            throw std::runtime_error("Slater determinant: accepted matrix is singular or nonfinite");
        log_det += std::log(std::abs(diag));
        if (pivot != k)
            for (std::size_t c = 0; c < N; ++c) {
                std::swap(matrix[at(k, c)], matrix[at(pivot, c)]);
                std::swap(inverse[at(k, c)], inverse[at(pivot, c)]);
            }
        for (std::size_t c = 0; c < N; ++c) {
            matrix[at(k, c)] /= diag;
            inverse[at(k, c)] /= diag;
        }
        for (std::size_t r = 0; r < N; ++r) {
            if (r == k) continue;
            const double factor = matrix[at(r, k)];
            for (std::size_t c = 0; c < N; ++c) {
                matrix[at(r, c)] -= factor * matrix[at(k, c)];
                inverse[at(r, c)] -= factor * inverse[at(k, c)];
            }
        }
    }
    updates_[b][w] = 0;
    return log_det;
}

void SlaterDeterminant::permute_caches(std::span<const std::size_t> src) {
    WavefunctionComponent::permute_caches(src);   // log_psi_c_, grad_c_, lap_c_
    const std::size_t old_nw = nw_;
    if (old_nw == 0) return;   // never evaluated — nothing committed to carry
    const std::size_t new_nw = src.size();

    // Matrix planes dominate the branch step's traffic (183 MB for 61
    // electrons and 1024 walkers).  The destination is allocated without
    // value-initialisation and filled in parallel, which halved the serial
    // gather on the workstation; what remains is memory bandwidth.  Keeping
    // persistent spares instead did not reduce it and doubled the memory.
    auto gather_matrices = [&](std::vector<MatrixPlane>& planes) {
        for (std::size_t b = 0; b < planes.size(); ++b) {
            MatrixPlane& plane = planes[b];
            const std::size_t N = blocks_[b].last - blocks_[b].first;
            const std::size_t width = plane.size() / cache_slots(old_nw);
            MatrixPlane dst(width * cache_slots(new_nw));   // uninitialised
            if (cache_stride(old_nw, N) == 1 && cache_stride(new_nw, N) == 1) {
                // One contiguous matrix per walker on both sides: copy whole
                // matrices in parallel over destination walkers.  The static
                // schedule splits walkers like DMC's crowds, so each thread
                // touches the matrices its crowd works on next step.
                const double* from = plane.data();
                double* to = dst.data();
                const auto n_dst = static_cast<std::ptrdiff_t>(new_nw);
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t j = 0; j < n_dst; ++j)
                    std::copy_n(from + src[static_cast<std::size_t>(j)] * width, width,
                                to + static_cast<std::size_t>(j) * width);
            } else {
                const std::size_t tile = cache_stride(new_nw, N);
                for (std::size_t first = 0; first < new_nw; first += tile)
                    for (std::size_t k = 0; k < width; ++k)
                        for (std::size_t j = first; j < std::min(first + tile, new_nw); ++j)
                            dst[cache_index(k, j, width, new_nw, N)] = plane[cache_index(k, src[j], width, old_nw, N)];
            }
            plane.swap(dst);
        }
    };
    // Per-walker columns [k * nw + w]: small, gathered serially.
    auto gather_columns = [&](auto& planes) {
        for (auto& plane : planes) {
            using Vector = std::decay_t<decltype(plane)>;
            const std::size_t width = plane.size() / old_nw;
            Vector dst(width * new_nw);
            for (std::size_t k = 0; k < width; ++k)
                for (std::size_t j = 0; j < new_nw; ++j)
                    dst[k * new_nw + j] = plane[k * old_nw + src[j]];
            plane = std::move(dst);
        }
    };
    gather_matrices(d_inv_);
    gather_matrices(values_);
    gather_matrices(gradients_);
    gather_matrices(laplacians_);
    gather_columns(block_log_);
    gather_columns(updates_);
    gather_columns(condition_);
    gather_columns(row_abs_sum_);
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const auto N = blocks_[b].last - blocks_[b].first;
        work_[b].resize(N * N * cache_slots(new_nw));
    }
    nw_ = new_nw;
    resize_scratch(new_nw, dim_);
}

std::size_t SlaterDeterminant::walker_state_bytes() const noexcept {
    std::size_t doubles = 0;
    for (const Block& blk : blocks_) {
        const std::size_t N = blk.last - blk.first;
        doubles += N * N * (2 + 2 * dim_) + N + 3;
    }
    return WavefunctionComponent::walker_state_bytes() + doubles * sizeof(double);
}

std::unique_ptr<WavefunctionComponent> SlaterDeterminant::clone() const {
    std::vector<Block> bl;
    bl.reserve(blocks_.size());
    for (const Block& blk : blocks_)
        bl.push_back(Block{blk.first, blk.last, blk.orbitals->clone()});
    auto copy = std::make_unique<SlaterDeterminant>(std::move(bl));

    // Carry committed per-walker state so a clone of an already-evaluated
    // determinant is immediately consistent (matches the base-cache deep copy).
    copy->dim_   = dim_;
    copy->nw_    = nw_;
    copy->d_inv_ = d_inv_;
    copy->values_ = values_;
    copy->gradients_ = gradients_;
    copy->laplacians_ = laplacians_;
    copy->block_log_ = block_log_;
    copy->updates_ = updates_;
    copy->condition_ = condition_;
    copy->row_abs_sum_ = row_abs_sum_;
    copy->work_ = work_;
    copy->log_psi_c_ = log_psi_c_;
    copy->grad_c_    = grad_c_;
    copy->lap_c_     = lap_c_;

    // The constructor already rebuilt params_ from the cloned orbitals, so the
    // VALUES match; what it cannot know is which of them the config unfixed.
    // Overwriting the whole list carries those flags to every thread's copy.
    copy->params_ = params_;
    if (nw_ > 0) copy->resize_scratch(nw_, dim_);
    return copy;
}

}  // namespace alea
