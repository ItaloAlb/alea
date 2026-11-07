// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// A smooth torus field for E_pv=E_th+(F.grad S-div F)/(2 beta).
// With r=A^T f, z_pa=mean_k exp(2 pi i f_pak), use
// F_pk=A^T Im[exp(2 pi i f_pk) conj(z_p)]/(2 pi),
// div F=sum_pa (M |z_pa|^2-1). No circular angle or winding cut is needed.
// This is an Alea-derived periodic application of integration by parts;
// see docs/estimators.md for the proof and references.
// It measures the existing minimum-image action, not a full-image kernel.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>
#include "pimc/path_set.h"

namespace alea {
class PeriodicVirialField {
public:
    void prepare(const PathSet& path) {
        if (!path.cell().is_periodic() || path.cell().dim() != path.dim())
            throw std::invalid_argument(
                "periodic_virial: requires a periodic cell matching the path");
        M_ = 0; // Failed allocation must not leave a partially prepared field.
        np_ = path.n_particles();
        dim_ = path.dim();
        const auto vectors = path.cell().vectors();
        vectors_.assign(vectors.begin(), vectors.end());
        field_.resize(path.positions().size());
        sine_.resize(field_.size());
        cosine_.resize(field_.size());
        M_ = path.num_slices();
    }
    void invalidate() noexcept { M_ = 0; }
    void evaluate(const PathSet& path) {
        const auto vectors = path.cell().vectors();
        if (!path.cell().is_periodic() || path.n_particles() != np_ ||
            path.dim() != dim_ || path.num_slices() != M_ ||
            vectors.size() != vectors_.size() ||
            !std::equal(vectors.begin(), vectors.end(), vectors_.begin()))
            throw std::logic_error("periodic_virial: field is not prepared for "
                                   "this path and cell");
        divergence_ = 0.0;
        std::fill(field_.begin(), field_.end(), 0.0);
        if (M_ == 1)
            return; // Exactly zero field, avoiding roundoff in |z|^2-1.
        const double two_pi = 2.0 * std::numbers::pi,
                     inv_M = 1.0 / static_cast<double>(M_);
        for (std::size_t p = 0; p < np_; ++p) {
            for (std::size_t k = 0; k < M_; ++k) {
                double offset[SimulationCell::kMaxDim],
                    fractional[SimulationCell::kMaxDim];
                for (std::size_t d = 0; d < dim_; ++d)
                    offset[d] = path.pos(p, d, k) - path.pos(p, d, 0);
                // A common phase cancels from F. Anchoring to bead zero also
                // avoids precision loss from a large common position offset.
                path.cell().fractional(offset, fractional);
                for (std::size_t a = 0; a < dim_; ++a) {
                    const auto i = (p * dim_ + a) * M_ + k;
                    const double theta =
                        two_pi * std::remainder(fractional[a], 1.0);
                    sine_[i] = std::sin(theta);
                    cosine_[i] = std::cos(theta);
                }
            }
            for (std::size_t a = 0; a < dim_; ++a) {
                const auto base = (p * dim_ + a) * M_;
                double mean_sine = 0.0, mean_cosine = 0.0;
                for (std::size_t k = 0; k < M_; ++k) {
                    mean_sine += sine_[base + k];
                    mean_cosine += cosine_[base + k];
                }
                mean_sine *= inv_M;
                mean_cosine *= inv_M;
                divergence_ +=
                    static_cast<double>(M_) *
                        (mean_sine * mean_sine + mean_cosine * mean_cosine) -
                    1.0;
                for (std::size_t d = 0; d < dim_; ++d)
                    for (std::size_t k = 0; k < M_; ++k)
                        field_[(p * dim_ + d) * M_ + k] +=
                            vectors_[a * dim_ + d] *
                            (sine_[base + k] * mean_cosine -
                             cosine_[base + k] * mean_sine) /
                            two_pi;
            }
        }
    }
    [[nodiscard]] std::span<const double> field() const noexcept {
        return field_;
    }
    [[nodiscard]] double divergence() const noexcept { return divergence_; }

private:
    std::size_t np_ = 0, dim_ = 0, M_ = 0;
    std::vector<double> vectors_; // [dim*dim], stored reduced cell basis
    std::vector<double> field_, sine_, cosine_; // each [np*dim*M]
    double divergence_ = 0.0;
};
} // namespace alea
