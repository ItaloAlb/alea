// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Fixed-node restriction of the PIMC path integral with time-independent
// (ground-state) nodes.
//
// Ceperley's restricted path integral (J. Stat. Phys. 63, 1237 (1991); Rev.
// Mod. Phys. 67, 279 (1995), Sec. VI) keeps the paths on which a trial density
// matrix ρ_T(R_k, R*; kτ) stays positive.  As t → ∞, ρ_T(R, R*; t) →
// ψ_T(R) ψ_T(R*), the reference point R* drops out, and the restriction becomes
//
//     sign det D_b(R_k) = s_b   for every block b and every slice k,
//     D_b[j][i] = φ_i(r_{first_b + j}),
//
// with s_b the block's common sign along the path.
//
//   * β → ∞: a continuous path stays in one nodal pocket Ω, Z = Tr_Ω e^{-βH_Ω}
//     (Dirichlet on the node), and the energy tends to the fixed-node energy
//     E_FN of ψ_T — what fixed-node DMC computes with the same determinant.  By
//     the tiling property every pocket gives the same E_FN.
//   * Odd permutations flip the sign and cannot occur; the run samples the
//     identity sector, whose weight also decays as e^{-βE_FN}.
//   * Declared approximation: at finite β the restricted ensemble is NOT
//     fermion thermodynamics.  It admits states of the wrong exchange symmetry
//     (two particles with the node x₁ = x₂ keep the relative state sin 2φ), so
//     results are fixed-node ground-state estimates only as β → ∞.
//   * The virial estimators would need the singular force of the nodal
//     surface and are refused.
//
// Link crossings (NodeCrossing).  Checking the sign at beads only keeps links
// that cross a node and come back, which biases E by O(√τ).
// The image action removes that term: in mass-scaled coordinates y = √m r the
// free link from y to y' that must not cross a planar node at distances d, d'
// carries the Dirichlet factor
//
//     ρ_Ω / ρ_0 = 1 − exp(−2 d d'/τ)       (ħ = 1; method of images),
//
// so each link and block adds  s = −ln(1 − e^{−x}),  x = 2 d_k d_{k+1}/τ,  with
// the distance to the node estimated as  d = |f| / |∇_y f|  for a function f
// with the same nodes as det D_b.  The usual RPIMC choice (Ceperley 1995,
// Sec. VI) is f = det D_b itself.  Here each row is first divided by its
// largest orbital, m_j = max_i |φ_i(r_j)| > 0.  That keeps the nodes, and the
// first-order distance at a node (f = 0 there, so ∇f = ∇det / Π m):
//
//     f = det(D_b) / Π_j m_j,
//     |∇_y ln|f||² = Σ_{j,a} (∂_{ja} ln|det D_b| − ∂_a ln m_j)² / m_{ja},
//     ∂_{ja} ln|det D_b| = Σ_i ∂_a φ_i(r_j) (D⁻¹)_{i j},
//     ∂_a ln m_j = ∂_a φ_k(r_j) / φ_k(r_j),   k = argmax_i |φ_i(r_j)|.
//
// Why: with localized orbitals det D_b is close to a product over particles,
// and ∂_{ja} ln det D_b ≈ ∂_a ln φ_own(r_j) is the slope of each particle's own
// orbital, not of a node.  Those slopes add in quadrature over the N
// particles, so f = det D_b puts every slice at d ∝ 1/√N from a node that is
// much farther.  For the r_s = 40 crystal (Drummond & Needs exponent,
// τ = r_s^{3/2}/16), on equilibrated paths, the closest nodal point found is
// at ~25 bohr for N = 37 and 127.  f = det D_b gives a median d of 11 and
// 6.4 bohr; the image term then adds a spurious 0.1 and 15 µHa/e to E, and
// the second grows with N.  The L2 row norm (Σ_i φ_i²)^{1/2} still adds
// 3.7 µHa/e at N = 127, through the slopes of the neighbouring orbitals.
// Dividing by the largest orbital leaves a median d of 14 bohr at N = 127
// and 0.005 µHa/e.  Near a node all of these agree with the closest nodal
// point to ~10%.  For a
// plane-wave set that contains the constant orbital, m_j = 1 and f = det D_b.
// m_j is continuous but its gradient jumps where two orbitals tie, so d is
// piecewise smooth; it is still a function of the path alone, which is all
// the Metropolis weight needs.
// Limitation: away from a node every first-order estimate overshoots when f
// bends within √τ.  For Gaussian orbitals that happens when 2 C a √τ ≳ 1
// (C the exponent, a the site spacing); the image action then under-penalizes,
// and the τ error must be measured.  The thermodynamic estimator gains
//
//     ∂S/∂β = (1/M) Σ_links ds/dτ,   ds/dτ = (x/τ) / (e^{x} − 1)  > 0,
//
// counted as kinetic energy (it is part of the free propagator).
//
// A bead move of particle p at slice k changes one row of one block:
//
//     det D' / det D = Σ_i φ_i(r'_p) (D⁻¹)_{i, p−first}        (O(N))
//
// and the move is allowed iff that ratio is > 0.  With the image action the
// proposal also needs |∇_y ln|f'||: D'⁻¹ by Sherman–Morrison and the cached
// gradient rows (∂_a φ_i(r_j), ∂_a ln m_j) of the slice, with the moved row replaced —
// O(N² dim) arithmetic and only the moved particle's N orbital evaluations,
// which the ratio needs anyway.  Recomputing every gradient instead cost
// N² evaluations per bead and made a block 13× slower at N = 37; the cache
// holds M·N(N+1)·dim doubles per block and replica (45 MB at
// N = 37, 530 MB at N = 127, M = 2048).
// D⁻¹, the gradients and |∇_y ln|f||² of every (block, slice) live in a cache that validates
// itself against the path, like EwaldTerm's structure factors: each call
// compares the slice's positions with the ones the inverse was built from
// (O(N)); one changed particle is a Sherman–Morrison row update (O(N²)), more
// changes, kRebuildEvery updates, or an update with |ratio| < kSmallRatio
// trigger a fresh LU (O(N³)).  The gradient norm of an accepted proposal is
// reused when the changed bead matches the last proposal at that slice.  No
// move has to notify the cache, so it cannot go stale.  A clone starts with an
// empty cache; PIMC gives each replica its own clone.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "pimc/action_component.h"
#include "wavefunction/orbital_set.h"

namespace alea {

class FixedNodeRestriction final : public ActionComponent {
public:
    struct Block {
        std::size_t                 first = 0;  // particles [first, last)
        std::size_t                 last  = 0;
        std::unique_ptr<OrbitalSet> orbitals;   // size() == last - first
    };
    // How a link between two allowed beads is treated.
    enum class NodeCrossing {
        beads_only,  // sign checked at beads; O(√τ) bias (diagnostics, tests)
        image,       // plus the image action of every link (production)
    };

    // Largest admitted OrbitalSet::periodic_fold_bound(): the minimum-image
    // orbitals then differ from the lattice-summed ones by at most this
    // fraction of their peak, and so do their nodes.
    static constexpr double        kFoldTolerance = 1e-6;
    static constexpr std::uint32_t kRebuildEvery  = 256;
    static constexpr double        kSmallRatio    = 1e-6;

    // inv_mass: [np * dim], 1/m of particle p along axis d at [p*dim + d].
    // Throws std::invalid_argument for no blocks, an empty, overlapping or
    // out-of-range block, an orbital count different from the block size, a
    // dimension outside [1, kMaxDim], inverse masses that are not finite and
    // positive or of the wrong size, or a fold bound above kFoldTolerance.
    FixedNodeRestriction(std::vector<Block> blocks, std::size_t np, std::size_t dim,
                         std::vector<double> inv_mass, NodeCrossing crossing);
    // Clones the orbitals; the copy starts with an empty cache.
    FixedNodeRestriction(const FixedNodeRestriction& other);
    FixedNodeRestriction& operator=(const FixedNodeRestriction&) = delete;

    void prepare(std::size_t M) override;
    // +∞ unless every block keeps one nonzero determinant sign over all
    // slices; otherwise the sum of the link image terms (0 for beads_only).
    // Fresh LU per (block, slice); refreshes the whole cache.
    double total_action(const PathSet& path) override;
    // Change of the restriction action when bead (p, k) moves: +∞ if the
    // proposal reaches or crosses the node (or has a nonfinite coordinate),
    // else the change of the two incident link terms.  Read-only on the path.
    double delta_potential_particle(PathSet& path, std::size_t p, std::size_t k,
                                    std::span<const double> proposed) override;
    // Same for `count` consecutive beads of p from slice `first` (cyclic),
    // positions [count*dim]: the links touching the window.
    double delta_potential_window(const PathSet& path, std::size_t p, std::size_t first,
                                  std::span<const double> positions) override;
    bool slice_local_deltas() const noexcept override {
        return crossing_ == NodeCrossing::beads_only;
    }
    bool link_local_window_deltas() const noexcept override {
        return crossing_ == NodeCrossing::image;
    }
    bool supports_free_bridge() const noexcept override { return true; }
    double beta_derivative(const PathSet& path) override;
    ActionThermoMeasure measure_thermo(const PathSet& path) override;
    void validate_open_centroid(std::size_t, std::size_t) const override;
    void validate_periodic_virial(const SimulationCell&, std::size_t,
                                  std::size_t) const override;
    std::string potential_approximation_notes() const override;
    std::vector<std::pair<std::size_t, std::size_t>> restricted_blocks() const override;
    std::unique_ptr<ActionComponent> clone() const override;

    // Sign of det D_b at slice k from a fresh LU: +1, -1, or 0 when singular.
    [[nodiscard]] int block_sign(const PathSet& path, std::size_t b, std::size_t k);
    // Estimated distance d = 1/|∇_y ln|det(D_b/m)|| of slice k to block b's node
    // (+∞ when the gradient vanishes); refreshes that slice's cache.
    [[nodiscard]] double node_distance(const PathSet& path, std::size_t b, std::size_t k);
    // Cache counters, for tests and diagnostics.
    [[nodiscard]] std::size_t rebuilds() const noexcept { return rebuilds_; }
    [[nodiscard]] std::size_t updates() const noexcept { return updates_; }
    [[nodiscard]] std::size_t gradient_evaluations() const noexcept { return gradients_; }

private:
    static constexpr std::size_t kNoBlock = static_cast<std::size_t>(-1);

    struct Cache {
        // Transposed inverse per slice: inverse[(k*n + j)*n + i] = (D⁻¹)_{i j},
        // so the ratio for row j is a contiguous dot product over i.  This is
        // exactly Eigen's column-major storage of D⁻¹, one slice after another.
        std::vector<double>        inverse;        // [M * n * n]
        std::vector<double>        snapshot;       // [M * n * dim], positions of the inverse
        std::vector<std::uint32_t> since_rebuild;  // [M]
        std::vector<unsigned char> built;          // [M]
        std::vector<double>        grad2;          // [M], |∇_y ln|det(D/m)||² of the snapshot
        std::vector<unsigned char> grad2_valid;    // [M]
        // Orbital gradients of the snapshot (image action only), one row of
        // dim*(n+1) values per particle j at offset (k*n + j)*dim*(n+1):
        // [a*n + i] = ∂_a φ_i(r_j), contiguous in i, then [dim*n + a] =
        // ∂_a ln m_j, the row scale of the header comment.
        std::vector<double>        gradients;      // [M * n * dim * (n + 1)]
        // Last proposal per slice, to reuse its gradient norm once accepted.
        std::vector<std::size_t>   pending_row;    // [M], row or kNoBlock
        std::vector<double>        pending_pos;    // [M * dim]
        std::vector<double>        pending_grad2;  // [M]
    };

    void ensure_slices(std::size_t M);
    // Orbital values and gradients of block b at one position r [dim] into
    // phi_ [n] and grad_ [n*dim] (grad_[i*dim + a] = ∂_a φ_i).
    void orbitals_at(std::size_t b, const double* r);
    // Fresh LU of D_b at slice k; stores D⁻¹ and the snapshot when nonsingular.
    // Returns the determinant sign (+1, -1) or 0 when singular.
    int rebuild(const PathSet& path, std::size_t b, std::size_t k);
    // Bring (b, k) in line with the path; false if D_b is singular there.
    bool sync(const PathSet& path, std::size_t b, std::size_t k);
    // |∇_y ln|det(D_b/m)||² from a transposed inverse T [n*n]
    // (T[j*n + i] = (D⁻¹)_{ij}) and slice k's cached gradient rows, row
    // `moved` (kNoBlock for none) replaced by moved_gradients [dim*(n+1)].
    double grad2_of(std::size_t b, std::size_t k, const double* T, std::size_t moved,
                    const double* moved_gradients);
    // phi_ and grad_ of the last orbitals_at into one cached-row layout
    // [dim*(n+1)]: the orbital gradients, then ∂_a ln m_j.
    void store_gradient_row(std::size_t b, double* row) const;
    // Synced slice's |∇_y ln|det(D/m)||², computed on demand.
    double current_grad2(const PathSet& path, std::size_t b, std::size_t k);
    // Proposal of row `row` at r_new on synced (b, k): ratio > 0 returns true
    // and, with the image action, |∇_y ln|f'||² (also kept as pending).
    bool propose(const PathSet& path, std::size_t b, std::size_t k, std::size_t row,
                 const double* r_new, double& grad2);
    // Link term s(x), x = 2 d d'/τ with d = 1/√g; 0 when either gradient vanishes.
    static double link_action(double g1, double g2, double tau);
    static double link_tau_derivative(double g1, double g2, double tau);

    std::vector<Block>       blocks_;
    std::size_t              np_ = 0, dim_ = 0, M_ = 0;
    std::vector<double>      inv_mass_;  // [np * dim]
    NodeCrossing             crossing_ = NodeCrossing::image;
    std::vector<std::size_t> block_of_;  // [np], block index or kNoBlock
    std::vector<Cache>       caches_;    // [blocks]
    double                   fold_bound_ = 0.0;
    // Per-block LU workspace, sized once: compute() on an equal size reuses it.
    std::vector<Eigen::MatrixXd>                      matrix_;   // [blocks], n x n
    std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>> lu_;       // [blocks]
    std::vector<Eigen::MatrixXd>                      inverse_;  // [blocks], n x n
    std::vector<double>      phi_, grad_, lap_;  // [n_max], [n_max*dim] x2
    std::vector<double>      row_;               // [n_max], c_l = Σ_m φ_m(r') (D⁻¹)_{m l}
    std::vector<double>      column_;            // [n_max], (D⁻¹)_{i j} of the replaced row j
    std::vector<double>      proposed_;          // [n_max*n_max], D'⁻¹ of a proposal (transposed)
    std::vector<double>      proposed_row_;      // [dim*(n_max+1)], one gradient row at a proposal
    std::vector<double>      window_grad2_;      // [M], proposal gradient norms of a window
    std::size_t              rebuilds_ = 0, updates_ = 0, gradients_ = 0;
};

}  // namespace alea
