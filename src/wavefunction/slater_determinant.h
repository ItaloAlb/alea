// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "orbital_set.h"
#include "wavefunction_component.h"

namespace alea {

// std::allocator whose value-less construction default-initialises: resizing
// a vector of doubles leaves the new elements uninitialised.  The determinant's
// population gathers overwrite every element, so value-initialisation would be
// a wasted serial pass over hundreds of MB; it would also first-touch every
// page on the thread that allocates instead of the threads that gather.
template <class T>
struct OverwriteAllocator : std::allocator<T> {
    using std::allocator<T>::allocator;
    template <class U> struct rebind { using other = OverwriteAllocator<U>; };
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

// Slater determinant trial-wavefunction factor for fermionic antisymmetry.
//
// ψ_det = Π_b det(D_b),  where each "block" b spans a contiguous range of
// identical fermions [first, last) (e.g. the spin-up electrons, then the
// spin-down electrons) and carries its own OrbitalSet.  For a block of N
// particles with N orbitals the matrix is D_b[j][i] = φ_i(r_{first+j}) (square).
//
// This is the first sign-CHANGING component: its single-particle move ratio
// R = det(D')/det(D) can be negative, and that sign is fed as a phase (π) into
// batch.phase_diffs(), which the DMC fixed-node gate rejects on a node crossing.
//
// Derivative conventions match Kinetic (grad = ∇log ψ, lap = ∂²_d log ψ):
//   ∂_d log ψ  (particle j) = Σ_m (D⁻¹)_{m,j} ∂_d φ_m(r_j)
//   ∂²_d log ψ (particle j) = Σ_m (D⁻¹)_{m,j} ∂²_d φ_m(r_j) − (∂_d log ψ)²
//   move ratio (particle j)  = Σ_m (D⁻¹)_{m,j} φ_m(r_j')
//
// Accepted rows use Sherman-Morrison with guarded, periodic dense rebuilds.
// DMC splits walkers into disjoint crowds; VMC uses a clone per thread. The
// component has no mutable state shared between walkers.
class SlaterDeterminant final : public WavefunctionComponent {
public:
    struct Block {
        std::size_t                 first;
        std::size_t                 last;     // size N = last - first == orbitals->size()
        std::unique_ptr<OrbitalSet> orbitals;
    };

    explicit SlaterDeterminant(std::vector<Block> blocks);

    std::string name() const override { return "slater_determinant"; }

    void prepare(std::size_t nw, std::size_t np, std::size_t dim) override;
    void permute_caches(std::span<const std::size_t> src) override;

    void evaluate_batch(WalkerSet& batch) override;
    void ratio_batch(WalkerSet& batch, std::size_t p, std::size_t w_begin = 0,
                     std::size_t w_end = npos) override;
    void apply_move_batch(WalkerSet& batch, std::size_t p,
                          std::span<const bool> accept_mask,
                          std::size_t w_begin = 0,
                          std::size_t w_end = npos) override;

    void apply_move_batch_deferred(WalkerSet& batch, std::size_t p,
        std::span<const bool> accept, std::size_t w_begin = 0, std::size_t w_end = npos) override;
    void refresh_particle_derivatives(WalkerSet& batch, std::size_t p,
        std::size_t w_begin = 0, std::size_t w_end = npos) override;
    void refresh_derivatives(WalkerSet& batch,
        std::size_t w_begin = 0, std::size_t w_end = npos) override;

    [[nodiscard]] std::unique_ptr<WavefunctionComponent> clone() const override;

    // Base caches plus, per block, A = D⁻¹, D, ∇D, ∇²D, the row sums and the
    // log/condition/update scalars.  The rebuild scratch is left out: it is
    // touched once per 64 accepted moves, not by every move.
    [[nodiscard]] std::size_t walker_state_bytes() const noexcept override;

    // The blocks' OrbitalSet parameters, flattened block-major and renamed
    // "b<block>_<name>".  These are the only parameters of ψ = D·e^J that move
    // the nodes.  All arrive is_fixed; the config layer unfixes what it wants.
    std::span<Parameter>       parameters() override { return params_; }
    std::span<const Parameter> parameters() const override { return params_; }

    // Push parameter values back into the orbital sets (they are what evaluates).
    void rebuild() override;

    // ∂ln|det D_b|/∂p = Tr(D_b⁻¹ ∂D_b/∂p), with D_b(c,a) = φ_a(r_{first+c}).
    // Blocks are independent: a parameter of block b sees only b.
    void compute_parameter_derivatives(const WalkerSet& batch,
                                       std::span<double> derivs) const override;

    [[nodiscard]] bool is_nodal() const override { return true; }

    // Two particles i, j shifted by the same s: in different blocks the
    // one-row ratios multiply; in one block two rows change, and
    //   det D″/det D = R_i(i) R_j(j) − R_i(j) R_j(i),
    //   R_x(y) = Σ_m φ_m(r_x + s) (D⁻¹)_{m y}
    // (the determinant lemma for two rows).  O(N²) per particle, O(1) per pair.
    [[nodiscard]] bool supports_shifted_pairs() const override { return true; }
    void shifted_pair_ratios_batch(const WalkerSet& batch, std::size_t a_first,
                                   std::size_t a_last, std::size_t b_first, std::size_t b_last,
                                   std::span<const double> shift, std::span<double> log_ratios,
                                   std::span<double> phases) override;

    [[nodiscard]] std::vector<NodalBlockView> nodal_blocks() const override {
        std::vector<NodalBlockView> views;
        views.reserve(blocks_.size());
        for (const Block& b : blocks_) views.push_back({b.first, b.last, b.orbitals.get()});
        return views;
    }

    [[nodiscard]] bool supports_parameter_gradient_derivatives() const override {
        return true;
    }

    bool compute_parameter_gradient_derivatives(
        const WalkerSet& batch, std::span<double> dgrad,
        std::span<double> dlap) const override;

private:
    // Index of the block containing particle p, or npos if p is in none.
    [[nodiscard]] std::size_t block_of(std::size_t p) const noexcept;
    void resize_state(std::size_t nw, std::size_t dim);   // d_inv_ + scratch
    void resize_scratch(std::size_t nw, std::size_t dim); // scratch only
    double rebuild_inverse(std::size_t b, std::size_t w);
    void apply_move_impl(WalkerSet& batch, std::size_t p, std::span<const bool> accept,
        std::size_t w_begin, std::size_t w_end, bool refresh_partners);
    // One accepted move of walker w in a block that stores one contiguous
    // matrix per walker (cache stride 1); see the definition for the passes.
    void commit_walker(WalkerSet& batch, std::size_t b, std::size_t jrow,
                       std::size_t w, bool refresh_partners);

    std::vector<Block> blocks_;
    std::size_t        dim_ = 0;
    std::size_t        nw_  = 0;
    std::size_t        max_block_size_ = 0;   // largest N, set with the scratch

    // Flat parameter list, block-major, and where each entry came from:
    // param_owner_[i] = {block index, parameter index within that OrbitalSet}.
    std::vector<Parameter>                              params_;
    std::vector<std::pair<std::size_t, std::size_t>>    param_owner_;

    // Private matrices use the layout policy in cache_index: walker-fast
    // [element*nw+w] for small determinants/batches, [w*width+element] otherwise.
    // Layout selection depends on both block size and private walker count.
    // Base caches/proposal scratch remain walker-fast. Clone and population
    // permutations carry both layouts, including crossings of that boundary.
    using MatrixPlane = std::vector<double, OverwriteAllocator<double>>;
    std::vector<MatrixPlane> d_inv_;              // per block [N * N * nw]
    // Committed rows, carried by cloning/branching. Only the accepted row is
    // replaced; every other particle's orbital evaluation is reusable.
    std::vector<MatrixPlane> values_;     // per block [N * N * nw]
    std::vector<MatrixPlane> gradients_;  // per block [N * N * dim * nw]
    std::vector<MatrixPlane> laplacians_; // per block [N * N * dim * nw]
    std::vector<std::vector<double>> block_log_; // per block [nw]
    std::vector<std::vector<std::size_t>> updates_; // per block [nw]
    std::vector<std::vector<double>> condition_; // per block [nw], infinity-norm estimate
    // Σ_c |D_ac| of every row a, so ‖D‖∞ after a one-row change costs O(N).
    std::vector<std::vector<double>> row_abs_sum_; // per block [N * nw], [a * nw + w]
    std::vector<MatrixPlane> work_; // per block [N * N * nw], rebuild scratch
    std::vector<double> column_, row_product_;  // [maxN * nw], rank-one scratch
    // Contiguous-matrix kernel scratch, walker-major so a walker's rows are
    // contiguous for SIMD over columns. Separate from column_/row_product_ so
    // concurrent crowds on blocks of different layout never share indices.
    std::vector<double> walker_column_, walker_row_product_; // [nw * maxN], [w * maxN + m]
    std::vector<double> nval_, ngrad_, nlap_;   // orbitals at new_pos (ratio)
    // shifted_pair_ratios_batch, per moved particle t; they grow to the
    // largest request (an estimator call, not a sweep).
    std::vector<double> shift_pos_;    // [dim * nw]
    std::vector<double> shift_rows_;   // [(t * nw + w) * maxN]  R_t(y) for every row y of its block
};

}  // namespace alea
