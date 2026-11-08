// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "wavefunction/backflow_transformation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace alea {

BackflowTransformation::BackflowTransformation(std::vector<Channel> channels,
                                               std::vector<std::size_t> transformed,
                                               const SimulationCell& cell)
    : transformed_(std::move(transformed)), cell_(cell) {
    if (channels.empty())
        throw std::invalid_argument("backflow: no backflow channel");
    if (transformed_.empty())
        throw std::invalid_argument("backflow: no particle to transform");
    const auto is_transformed = [&](std::size_t p) {
        return std::find(transformed_.begin(), transformed_.end(), p) != transformed_.end();
    };
    for (std::size_t c = 0; c < channels.size(); ++c) {
        Channel& ch = channels[c];
        const std::string label = "backflow channel " + std::to_string(c);
        if (ch.source_last <= ch.source_first || ch.target_last <= ch.target_first)
            throw std::invalid_argument(label + ": an empty source or target group");
        for (std::size_t p = ch.target_first; p < ch.target_last; ++p)
            if (!is_transformed(p))
                throw std::invalid_argument(label + ": a target particle carries no "
                                            "quasi-coordinate in this determinant");
        for (const ChannelState& other : channels_)
            if (ch.target_first < other.target_last && other.target_first < ch.target_last &&
                ch.source_first < other.source_last && other.source_first < ch.source_last)
                throw std::invalid_argument(label + ": two channels share source particles "
                                            "for one target; give each source group one channel");
        // A source overlapping its target must be that group: its pairs are
        // like-spin, and η'(0) = 0 follows (backflow_function.h).
        const bool overlaps = ch.source_first < ch.target_last && ch.target_first < ch.source_last;
        if (overlaps && (ch.source_first != ch.target_first || ch.source_last != ch.target_last))
            throw std::invalid_argument(label + ": a source overlapping its target group must "
                                        "be that whole group");
        if (ch.params.size() < 2)
            throw std::invalid_argument(label + ": needs c_0 and the cutoff L");
        std::vector<double> free;
        for (std::size_t k = 0; k + 1 < ch.params.size(); ++k) free.push_back(ch.params[k].value);
        const double L = ch.params.back().value;
        if (cell_.is_periodic() && L > cell_.ws_radius())
            throw std::invalid_argument(
                label + ": cutoff " + std::to_string(L) + " exceeds the Wigner-Seitz "
                "inradius " + std::to_string(cell_.ws_radius()) + "; beyond it the "
                "minimum image is not the periodic sum");
        BackflowFunction eta(ch.truncation, L, std::move(free), overlaps);
        std::size_t power = 0;
        for (std::size_t k = 0; k + 1 < ch.params.size(); ++k, ++power) {
            if (overlaps && power == 1) ++power;
            ch.params[k].name = "bf" + std::to_string(c) + "_c" + std::to_string(power);
        }
        Parameter& cut = ch.params.back();
        cut.name = "bf" + std::to_string(c) + "_L";
        cut.sign_locked = true;
        if (cell_.is_periodic()) cut.upper_bound = cell_.ws_radius();
        channels_.push_back({ch.target_first, ch.target_last, ch.source_first, ch.source_last,
                             std::move(eta), params_.size()});
        params_.insert(params_.end(), ch.params.begin(), ch.params.end());
    }
}

void BackflowTransformation::set_parameters(std::span<const Parameter> params) {
    if (params.size() != params_.size())
        throw std::logic_error("backflow: parameter count mismatch");
    for (ChannelState& ch : channels_)
        for (std::size_t k = 0; k < ch.eta.n_parameters(); ++k) {
            const Parameter& p = params[ch.param_offset + k];
            if (k + 1 == ch.eta.n_parameters() && p.value > p.upper_bound)
                throw std::runtime_error("backflow: cutoff " + p.name + " = " +
                                         std::to_string(p.value) + " exceeds its bound " +
                                         std::to_string(p.upper_bound));
            ch.eta.set_parameter(k, p.value);
        }
    for (std::size_t q = 0; q < params_.size(); ++q) params_[q].value = params[q].value;
}

std::size_t BackflowTransformation::max_channel_parameters() const noexcept {
    std::size_t n = 1;
    for (const ChannelState& ch : channels_) n = std::max(n, ch.eta.n_parameters());
    return n;
}

void BackflowTransformation::prepare(std::size_t np, std::size_t dim) {
    for (std::size_t p : transformed_)
        if (p >= np) throw std::invalid_argument("backflow: a transformed particle beyond np");
    for (const ChannelState& ch : channels_)
        if (ch.source_last > np || ch.target_last > np)
            throw std::invalid_argument("backflow: a channel beyond the particle count");
    np_ = np;
    dim_ = dim;
    index_.assign(np, npos);
    affects_.assign(np, 0);
    for (std::size_t t = 0; t < transformed_.size(); ++t) {
        index_[transformed_[t]] = t;
        affects_[transformed_[t]] = 1;
    }
    for (const ChannelState& ch : channels_)
        for (std::size_t j = ch.source_first; j < ch.source_last; ++j) affects_[j] = 1;
}

void BackflowTransformation::size_tables(Tables& t) const {
    const std::size_t n = transformed_.size() * np_;
    t.b.assign(n * dim_ * dim_, 0.0);
    t.s.assign(n * dim_ * dim_, 0.0);
    t.linked.assign(n, 0);
}

// At r = 0 the r-hat terms of B vanish with r (B → η(0) δ) and s is bounded
// and direction-dependent; that single point is left at zero.
bool BackflowTransformation::pair_terms(const ChannelState& ch, const double* ri,
                                        const double* rj, double sign, double* x, double* B,
                                        double* S) const {
    const std::size_t dim = dim_;
    double del[SimulationCell::kMaxDim] = {};
    for (std::size_t d = 0; d < dim; ++d) del[d] = ri[d] - rj[d];
    cell_.min_image(del);
    double r2 = 0.0;
    for (std::size_t d = 0; d < dim; ++d) r2 += del[d] * del[d];
    const double r = std::sqrt(r2);
    if (r >= ch.eta.cutoff()) return false;
    double eta = 0.0, e1 = 0.0, e2 = 0.0;
    ch.eta.evaluate(r, eta, e1, e2);
    for (std::size_t d = 0; d < dim; ++d) x[d] += sign * eta * del[d];
    if (!B) return true;
    const double inv_r = r > 0.0 ? 1.0 / r : 0.0;
    for (std::size_t a = 0; a < dim; ++a)
        for (std::size_t b = 0; b < dim; ++b) {
            B[a * dim + b] = (a == b ? eta : 0.0) + e1 * del[a] * del[b] * inv_r;
            const double rb2 = del[b] * del[b] * inv_r * inv_r;
            S[a * dim + b] = (a == b ? 2.0 * e1 * del[b] * inv_r : 0.0) +
                             del[a] * (e2 * rb2 + e1 * inv_r * (1.0 - rb2));
        }
    return true;
}

void BackflowTransformation::full(const double* pos, double* x, Tables& tab) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim;
    std::fill(tab.linked.begin(), tab.linked.end(), 0);
    std::fill(tab.b.begin(), tab.b.end(), 0.0);
    std::fill(tab.s.begin(), tab.s.end(), 0.0);
    for (std::size_t t = 0; t < transformed_.size(); ++t) {
        const std::size_t P = transformed_[t];
        double* xt = x + t * dim;
        for (std::size_t d = 0; d < dim; ++d) xt[d] = pos[P * dim + d];
        for (const ChannelState& ch : channels_) {
            if (P < ch.target_first || P >= ch.target_last) continue;
            for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                if (j == P) continue;
                const std::size_t e = t * np + j;
                tab.linked[e] = pair_terms(ch, pos + P * dim, pos + j * dim, 1.0, xt,
                                           tab.b.data() + e * dd, tab.s.data() + e * dd);
            }
        }
    }
}

void BackflowTransformation::particle_tables(const double* pos, std::size_t k, double* x,
                                             Tables& tab) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim;
    const auto clear = [&](std::size_t e) {
        tab.linked[e] = 0;
        std::fill_n(tab.b.begin() + static_cast<std::ptrdiff_t>(e * dd), dd, 0.0);
        std::fill_n(tab.s.begin() + static_cast<std::ptrdiff_t>(e * dd), dd, 0.0);
    };
    const std::size_t tk = index_[k];
    if (tk != npos) {
        double* xt = x + tk * dim;
        for (std::size_t d = 0; d < dim; ++d) xt[d] = pos[k * dim + d];
        for (std::size_t j = 0; j < np; ++j) clear(tk * np + j);
        for (const ChannelState& ch : channels_) {
            if (k < ch.target_first || k >= ch.target_last) continue;
            for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                if (j == k) continue;
                const std::size_t e = tk * np + j;
                tab.linked[e] = pair_terms(ch, pos + k * dim, pos + j * dim, 1.0, xt,
                                           tab.b.data() + e * dd, tab.s.data() + e * dd);
            }
        }
    }
    for (std::size_t t = 0; t < transformed_.size(); ++t) {
        const std::size_t P = transformed_[t];
        if (P == k) continue;
        const std::size_t e = t * np + k;
        clear(e);
        for (const ChannelState& ch : channels_) {
            if (P < ch.target_first || P >= ch.target_last || k < ch.source_first ||
                k >= ch.source_last)
                continue;
            double unused[SimulationCell::kMaxDim] = {};
            tab.linked[e] = pair_terms(ch, pos + P * dim, pos + k * dim, 1.0, unused,
                                       tab.b.data() + e * dd, tab.s.data() + e * dd);
        }
    }
}

void BackflowTransformation::shift(const double* pos, std::size_t k, const double* old_r,
                                   double* x) const {
    const std::size_t dim = dim_;
    for (const ChannelState& ch : channels_) {
        if (k < ch.source_first || k >= ch.source_last) continue;
        for (std::size_t P = ch.target_first; P < ch.target_last; ++P) {
            if (P == k) continue;
            double* xt = x + index_[P] * dim;
            pair_terms(ch, pos + P * dim, pos + k * dim, 1.0, xt, nullptr, nullptr);
            pair_terms(ch, pos + P * dim, old_r, -1.0, xt, nullptr, nullptr);
        }
    }
}

double BackflowTransformation::jacobian_column(const Tables& tab, const double* f,
                                               std::size_t k, std::size_t beta,
                                               double* v) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim, T = transformed_.size();
    std::fill(v, v + T * dim, 0.0);
    double fs = 0.0;
    const std::size_t tk = index_[k];
    if (tk != npos) {
        double* vk = v + tk * dim;
        vk[beta] = 1.0;
        for (std::size_t j = 0; j < np; ++j) {
            if (!tab.linked[tk * np + j]) continue;
            const double* B = tab.b.data() + (tk * np + j) * dd;
            const double* S = tab.s.data() + (tk * np + j) * dd;
            for (std::size_t a = 0; a < dim; ++a) {
                vk[a] += B[a * dim + beta];
                fs += f[tk * dim + a] * S[a * dim + beta];
            }
        }
    }
    for (std::size_t t = 0; t < T; ++t) {
        if (t == tk || !tab.linked[t * np + k]) continue;
        const double* B = tab.b.data() + (t * np + k) * dd;
        const double* S = tab.s.data() + (t * np + k) * dd;
        double* vt = v + t * dim;
        for (std::size_t a = 0; a < dim; ++a) {
            vt[a] = -B[a * dim + beta];
            fs += f[t * dim + a] * S[a * dim + beta];
        }
    }
    return fs;
}

void BackflowTransformation::gradient(const Tables& tab, const double* f, std::size_t k,
                                      double* out) const {
    const std::size_t dim = dim_, np = np_, dd = dim * dim, T = transformed_.size();
    for (std::size_t b = 0; b < dim; ++b) out[b] = 0.0;
    const std::size_t tk = index_[k];
    if (tk != npos) {
        const double* F = f + tk * dim;
        for (std::size_t b = 0; b < dim; ++b) out[b] += F[b];
        for (std::size_t j = 0; j < np; ++j) {
            if (!tab.linked[tk * np + j]) continue;
            const double* B = tab.b.data() + (tk * np + j) * dd;
            for (std::size_t a = 0; a < dim; ++a)
                for (std::size_t b = 0; b < dim; ++b) out[b] += F[a] * B[a * dim + b];
        }
    }
    for (std::size_t t = 0; t < T; ++t) {
        if (t == tk || !tab.linked[t * np + k]) continue;
        const double* F = f + t * dim;
        const double* B = tab.b.data() + (t * np + k) * dd;
        for (std::size_t a = 0; a < dim; ++a)
            for (std::size_t b = 0; b < dim; ++b) out[b] -= F[a] * B[a * dim + b];
    }
}

void BackflowTransformation::parameter_derivatives(const double* pos, const double* f,
                                                   std::span<double> out,
                                                   std::span<double> scratch) const {
    const std::size_t dim = dim_;
    std::fill(out.begin(), out.end(), 0.0);
    for (const ChannelState& ch : channels_) {
        const std::size_t n = ch.eta.n_parameters();
        for (std::size_t P = ch.target_first; P < ch.target_last; ++P) {
            const double* F = f + index_[P] * dim;
            for (std::size_t j = ch.source_first; j < ch.source_last; ++j) {
                if (j == P) continue;
                double del[SimulationCell::kMaxDim] = {};
                for (std::size_t d = 0; d < dim; ++d) del[d] = pos[P * dim + d] - pos[j * dim + d];
                cell_.min_image(del);
                double r2 = 0.0, fd = 0.0;
                for (std::size_t d = 0; d < dim; ++d) {
                    r2 += del[d] * del[d];
                    fd += F[d] * del[d];
                }
                const double r = std::sqrt(r2);
                if (r >= ch.eta.cutoff()) continue;
                ch.eta.parameter_derivatives(r, scratch.first(n));
                for (std::size_t q = 0; q < n; ++q) out[ch.param_offset + q] += fd * scratch[q];
            }
        }
    }
}

}  // namespace alea
