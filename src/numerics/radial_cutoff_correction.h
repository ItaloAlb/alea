// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace alea::numerics {

// Quintic Hermite endpoint matching on [a,b]. C,C',C'' are zero at a;
// at b they equal the supplied value/slope/curvature residuals. With
// u=(r-a)/(b-a), C=delta_v*H0(u)+h*delta_d1*H1(u)+h²*delta_d2*H2(u),
// H0=u³(10-15u+6u²), H1=u³(-4+7u-3u²), H2=u³(1-u)²/2.
// Subtract C from a sampled short-range approximation before its strict cutoff.
// This is a declared approximation correction, not a change to the exact
// long-range spline or its Fourier transform (Natoli & Ceperley, Journal of
// Computational Physics 117, 171 (1995), optimized breakup). Outside the
// interval C is clamped constant.
class RadialCutoffCorrection {
public:
    RadialCutoffCorrection() = default; // identically zero

    RadialCutoffCorrection(double begin, double end, double value,
                           double slope, double curvature)
        : begin_(begin), width_(end-begin), inv_width_(1.0/width_),
          residual_{value,width_*slope,width_*(width_*curvature)} {
        if (!std::isfinite(begin) || !std::isfinite(end) || begin < 0.0 ||
            width_ <= 0.0 || !std::isfinite(inv_width_))
            throw std::invalid_argument("radial cutoff correction: invalid interval");
        for (double coefficient : residual_)
            if (!std::isfinite(coefficient))
                throw std::invalid_argument("radial cutoff correction: nonfinite residual");
    }

    [[nodiscard]] double begin() const noexcept { return begin_; }
    [[nodiscard]] double end() const noexcept { return begin_+width_; }

    [[nodiscard]] double eval(double r) const noexcept {
        const double u=coordinate(r);
        return u*u*u*(residual_[0]*(10.0+u*(-15.0+6.0*u))+
            residual_[1]*(-4.0+u*(7.0-3.0*u))+
            0.5*residual_[2]*(1.0-u)*(1.0-u));
    }

    [[nodiscard]] double d1(double r) const noexcept {
        if (r < begin_ || r > end()) return 0.0;
        const double u=coordinate(r);
        return inv_width_*u*u*(30.0*residual_[0]*(1.0-u)*(1.0-u)+
            residual_[1]*(-12.0+u*(28.0-15.0*u))+
            residual_[2]*(1.5+u*(-4.0+2.5*u)));
    }

    [[nodiscard]] double d2(double r) const noexcept {
        if (r < begin_ || r > end()) return 0.0;
        const double u=coordinate(r);
        const double polynomial=u*(residual_[0]*(60.0+u*(-180.0+120.0*u))+
            residual_[1]*(-24.0+u*(84.0-60.0*u))+
            residual_[2]*(3.0+u*(-12.0+10.0*u)));
        return (polynomial*inv_width_)*inv_width_;
    }

    // Exact polynomial integral int_a^b r*C(r) dr. The mean correction is
    // 2*pi/A times this, needed for BOTH jellium and multi-kernel k=0 terms.
    [[nodiscard]] double radial_integral() const noexcept {
        return width_*(begin_*(residual_[0]/2.0-residual_[1]/10.0+residual_[2]/120.0)+
            width_*(5.0*residual_[0]/14.0-13.0*residual_[1]/210.0+residual_[2]/210.0));
    }

private:
    double coordinate(double r) const noexcept {
        const double raw=(r-begin_)*inv_width_;
        return raw < 0.0 ? 0.0 : (raw > 1.0 ? 1.0 : raw);
    }
    double begin_=0.0, width_=0.0, inv_width_=0.0;
    std::array<double,3> residual_{}; // [value, h*slope, h²*curvature]
};

} // namespace alea::numerics
