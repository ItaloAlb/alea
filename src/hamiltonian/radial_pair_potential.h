// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Setup-only description of a represented radial pair, in Hartree units.
// The pole is the singular part of value(), not a physical cusp hidden by a
// sampled clamp. It permits analytic cell integration of integrable 2D 1/r.
#include <functional>
#include <string>
#include "population/particle_group.h"
namespace alea {
struct RadialPairPotential {
    ParticleGroup a,b;
    std::function<double(double)> value;
    double pole_residue=0.0;
    std::string approximation;
};
enum class PairActionRole { unsupported, one_body, radial_pairs };
}
