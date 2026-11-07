// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// e^{ik·r} for a fixed set of reciprocal-lattice vectors k = n₁b₁ + n₂b₂ of a
// 2-D periodic cell, at one position r.
//
// Integer ladders: with θ_j = b_j·r, e^{ik·r} = e^{in₁θ₁}·e^{in₂θ₂}, and each
// factor is built by repeated multiplication from a single cos/sin pair.  So
// a whole k set costs two sincos and one complex multiply per k, instead of a
// sincos per k.  The recurrence loses ~n·ε, negligible for the |n| ≲ 30 these
// sets use.  Shared by the image-summed Jastrow and the S(k) estimator.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "population/simulation_cell.h"

namespace alea::numerics {

class ReciprocalPhases {
public:
    ReciprocalPhases() = default;

    ReciprocalPhases(const SimulationCell& cell, std::vector<int> n1, std::vector<int> n2)
        : n1_(std::move(n1)), n2_(std::move(n2)) {
        const auto b = cell.reciprocal();
        b1x_ = b[0]; b1y_ = b[1]; b2x_ = b[2]; b2y_ = b[3];
        for (std::size_t k = 0; k < n1_.size(); ++k) {
            m1_ = std::max(m1_, std::abs(n1_[k]));
            m2_ = std::max(m2_, std::abs(n2_[k]));
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return n1_.size(); }

    // (c[k], s[k]) = (cos, sin)(k·r) for every k of the set.
    void operator()(double x, double y, double* c, double* s) const {
        thread_local std::vector<double> e1r, e1i, e2r, e2i;
        ladder(b1x_ * x + b1y_ * y, m1_, e1r, e1i);
        ladder(b2x_ * x + b2y_ * y, m2_, e2r, e2i);
        const double* f1r = e1r.data() + m1_;   // index n in [−m, m]
        const double* f1i = e1i.data() + m1_;
        const double* f2r = e2r.data() + m2_;
        const double* f2i = e2i.data() + m2_;
        const int* __restrict a = n1_.data();
        const int* __restrict b = n2_.data();
        for (std::size_t k = 0; k < n1_.size(); ++k) {
            const double ar = f1r[a[k]], ai = f1i[a[k]];
            const double br = f2r[b[k]], bi = f2i[b[k]];
            c[k] = ar * br - ai * bi;
            s[k] = ar * bi + ai * br;
        }
    }

private:
    // e^{inθ} for n = −m..m, stored at [n + m].
    static void ladder(double theta, int m, std::vector<double>& re, std::vector<double>& im) {
        re.resize(2 * m + 1);
        im.resize(2 * m + 1);
        double* r = re.data() + m;
        double* i = im.data() + m;
        const double c = std::cos(theta), s = std::sin(theta);
        r[0] = 1.0; i[0] = 0.0;
        for (int n = 1; n <= m; ++n) {
            r[n] = r[n - 1] * c - i[n - 1] * s;
            i[n] = r[n - 1] * s + i[n - 1] * c;
            r[-n] = r[n];
            i[-n] = -i[n];
        }
    }

    std::vector<int> n1_, n2_;
    int m1_ = 0, m2_ = 0;
    double b1x_ = 0, b1y_ = 0, b2x_ = 0, b2y_ = 0;
};

}  // namespace alea::numerics
