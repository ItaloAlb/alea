// SPDX-FileCopyrightText: 2026 The Alea Authors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "population/walker.h"

using alea::Walker;

TEST_CASE("Walker sizes its position buffer to n_particles * dim", "[walker]") {
    Walker w(/*n_particles=*/3, /*dim=*/2);

    CHECK(w.n_particles() == 3);
    CHECK(w.dim() == 2);
    CHECK(w.positions().size() == 6);
}
