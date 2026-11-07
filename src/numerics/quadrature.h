// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

namespace alea::numerics {

// 20-point Gauss-Legendre, as ten symmetric pairs.
inline constexpr double kGaussX20[10] = {0.0765265211334973, 0.2277858511416451,
                                         0.3737060887154195, 0.5108670019508271,
                                         0.6360536807265150, 0.7463319064601508,
                                         0.8391169718222188, 0.9122344282513259,
                                         0.9639719272779138, 0.9931285991850949};
inline constexpr double kGaussW20[10] = {0.1527533871307258, 0.1491729864726037,
                                         0.1420961093183820, 0.1316886384491766,
                                         0.1181945319615184, 0.1019301198172404,
                                         0.0832767415767048, 0.0626720483341091,
                                         0.0406014298003869, 0.0176140071391521};

// ∫_a^b f on one 20-point Gauss-Legendre panel.
template <class F>
double gauss20(F&& f, double a, double b) {
    const double c = 0.5 * (a + b), h = 0.5 * (b - a);
    double s = 0.0;
    for (int i = 0; i < 10; ++i) {
        s += kGaussW20[i] * (f(c + h * kGaussX20[i]) + f(c - h * kGaussX20[i]));
    }
    return s * h;
}

// ∫_a^b f on `np` equal Gauss-Legendre panels.
template <class F>
double gauss20_panels(F&& f, double a, double b, int np) {
    np = std::max(np, 1);
    const double h = (b - a) / np;
    double s = 0.0;
    for (int i = 0; i < np; ++i) {
        s += gauss20(f, a + i * h, a + (i + 1) * h);
    }
    return s;
}

// Subdivide so each panel spans about one radian of J₀(kr)'s phase: Gauss-
// Legendre integrates a smooth function, and J₀(kr) is only smooth on that
// scale.
template <class F>
double panel_integral(F&& f, double a, double b, double k) {
    return gauss20_panels(f, a, b,
                          std::clamp(1 + static_cast<int>(k * (b - a)), 1, 4000));
}

}  // namespace alea::numerics
