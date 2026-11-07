// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "population/walker.h"

namespace alea {

Walker::Walker(std::size_t n_particles, std::size_t dim)
    : n_particles_(n_particles), dim_(dim), positions_(n_particles * dim, 0.0) {}

}  // namespace alea