// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "../population/simulation_cell.h"
#include "backflow_function.h"
#include "wavefunction_component.h"   // Parameter

namespace alea {

// The backflow transformation shared by the backflow determinants
// (P. López Ríos, A. Ma, N. D. Drummond, M. D. Towler and R. J. Needs,
//  Phys. Rev. E 74, 066701 (2006), Eqs. (3)–(4), (7)–(8)):
//
//   x_t = r_t + Σ_c [t ∈ T_c] Σ_{j ∈ S_c, j ≠ t} η_c(r_tj) r_tj,   r_tj = r_t − r_j (minimum image),
//
// for every transformed particle t (the particles a determinant evaluates its
// orbitals at) and channels c, each a target group T_c, a source group S_c and
// a function η_c (backflow_function.h).  A channel whose source is its target
// group is like-spin: η_c'(0) = 0, which keeps the P-wave cusp.
//
// What a determinant needs from it, per configuration (one walker):
//   B_c(r)_{αβ} = η δ_αβ + η' r_α r_β / r,
//   s_c(r)_{αβ} = ∂²_β [η(r) r_α] = 2η' r_β δ_αβ / r + r_α [η'' r_β²/r² + η'(1/r − r_β²/r³)],
// stored per (t, j) pair (Tables), and from them the Jacobian column of a
// particle k and dimension β, v_tα = ∂x_tα/∂r_kβ:
//   v_{t(k)} = e_β + Σ_j B(r_kj)_{·β},   v_t = −B(r_tk)_{·β}   (k a source of t),
// and Σ_t F_t · ∂²x_t/∂r_kβ² for a given F_t = ∂ln|D|/∂x_t.
//
// Under periodic boundaries every cutoff L ≤ the Wigner–Seitz inradius: the
// nearest image is then the exact periodic sum and η is continuous across the
// cell (Ref., Sec. III).  A pure value type: the per-configuration buffers
// are the caller's.
class BackflowTransformation {
public:
    // One channel: the particles [source_first, source_last) displace those of
    // [target_first, target_last).  params: [c_0, (c_1), c_2.., L] as in
    // BackflowFunction; c_1 is derived (absent) when the source is the target.
    struct Channel {
        std::size_t            target_first = 0, target_last = 0;
        std::size_t            source_first = 0, source_last = 0;
        int                    truncation = 3;
        std::vector<Parameter> params;
    };
    // Pair tables of one configuration, [n_transformed * np] entries.
    struct Tables {
        std::vector<double> b;       // [(t*np + j) * dim * dim]  B_{αβ}
        std::vector<double> s;       // [(t*np + j) * dim * dim]  s_{αβ}
        std::vector<char>   linked;  // [t*np + j]  j displaces t (within the cutoff)
    };

    BackflowTransformation() = default;
    // transformed: the particles that carry quasi-coordinates, ascending.
    // Every channel's target must lie in it.  Throws std::invalid_argument on
    // an empty or overlapping source, a source that overlaps its target
    // without equalling it, two channels with one (target, source) pair, or
    // a cutoff beyond the inradius of a periodic cell.
    BackflowTransformation(std::vector<Channel> channels, std::vector<std::size_t> transformed,
                           const SimulationCell& cell);

    // Parameters of every channel in order, named "bf<c>_c<k>" and "bf<c>_L";
    // L is sign-locked and, under PBC, bounded by the inradius.
    [[nodiscard]] const std::vector<Parameter>& initial_parameters() const noexcept {
        return params_;
    }
    [[nodiscard]] std::size_t n_parameters() const noexcept { return params_.size(); }
    // Values of the channel parameters, in initial_parameters() order.
    void set_parameters(std::span<const Parameter> params);

    // Size the particle maps for np particles of dimension dim.
    void prepare(std::size_t np, std::size_t dim);
    void size_tables(Tables& t) const;

    [[nodiscard]] std::size_t n_transformed() const noexcept { return transformed_.size(); }
    [[nodiscard]] std::size_t particle(std::size_t t) const noexcept { return transformed_[t]; }
    // Transformed index of particle p, or npos.
    [[nodiscard]] std::size_t index(std::size_t p) const noexcept { return index_[p]; }
    // Particle p is transformed or displaces a transformed one.
    [[nodiscard]] bool affects(std::size_t p) const noexcept { return affects_[p] != 0; }
    [[nodiscard]] std::size_t n_channels() const noexcept { return channels_.size(); }
    [[nodiscard]] const BackflowFunction& eta(std::size_t c) const { return channels_.at(c).eta; }

    // pos: one configuration, AoS [np * dim]; x: [n_transformed * dim].
    // Every quasi-coordinate and every table entry: O(T · S).
    void full(const double* pos, double* x, Tables& tab) const;
    // The entries the derivatives of particle k read (its own row against
    // every source, every row against k), and x_{t(k)} from scratch: O(N).
    void particle_tables(const double* pos, std::size_t k, double* x, Tables& tab) const;
    // x (the quasi-coordinates before a move of k from old_r to pos[k]) moved:
    // every t that k displaces shifts by η(r'_tk) r'_tk − η(r_tk) r_tk.  The
    // own row of k is particle_tables' job.
    void shift(const double* pos, std::size_t k, const double* old_r, double* x) const;

    // v ([n_transformed * dim]) = the Jacobian column of (k, β); returns
    // Σ_t F_t · ∂²x_t/∂r_kβ² for f = F ([n_transformed * dim]).  Reads only the
    // entries particle_tables(k) writes.  with_identity = false drops the e_β
    // of ∂x_k/∂r_k: the column of a table of parameter derivatives
    // (parameter_tangent), whose identity part does not depend on p.
    double jacobian_column(const Tables& tab, const double* f, std::size_t k, std::size_t beta,
                           double* v, bool with_identity = true) const;
    // ∂_k ln|D| ([dim]) = Σ_t F_t · ∂x_t/∂r_k (with_identity as above).
    void gradient(const Tables& tab, const double* f, std::size_t k, double* out,
                  bool with_identity = true) const;
    // Σ_t F_t · ∂x_t/∂p for every channel parameter: out[q] ([n_parameters]),
    // scratch [max channel parameters].
    void parameter_derivatives(const double* pos, const double* f, std::span<double> out,
                               std::span<double> scratch) const;
    [[nodiscard]] std::size_t max_channel_parameters() const noexcept;

    // The transformation differentiated along its parameter q (index into
    // initial_parameters()), at configuration pos: x_dot ([n_transformed*dim])
    // = ∂x_t/∂q = Σ_j ∂η(r_tj)/∂q r_tj, and tab_dot the pair tables of ∂B/∂q
    // and ∂s/∂q (B and s are linear in η, η', η'', so these are B and s with
    // η → ∂η/∂q and so on).  Only the pairs of q's channel are nonzero.
    // scratch: [3 * max_channel_parameters()].
    void parameter_tangent(const double* pos, std::size_t q, double* x_dot, Tables& tab_dot,
                           std::span<double> scratch) const;

private:
    struct ChannelState {
        std::size_t      target_first, target_last, source_first, source_last;
        BackflowFunction eta;
        std::size_t      param_offset;
    };
    // One pair: sign·η r_tj added to x; B and s filled when B is not null.
    bool pair_terms(const ChannelState& ch, const double* ri, const double* rj, double sign,
                    double* x, double* B, double* S) const;
    // B and s of one pair (del = r_tj, r = |del|) from η, η', η''.
    void pair_tensors(double eta, double e1, double e2, const double* del, double r, double* B,
                      double* S) const;

    std::vector<ChannelState> channels_;
    std::vector<std::size_t>  transformed_;
    SimulationCell            cell_;
    std::vector<Parameter>    params_;
    std::size_t np_ = 0, dim_ = 0;
    std::vector<std::size_t>  index_;    // [np] transformed index or npos
    std::vector<char>         affects_;  // [np]
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
};

}  // namespace alea
