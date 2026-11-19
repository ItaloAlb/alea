# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Checks for tests/references/inverse_power_trap_primitive.py, the exact
finite-M path integral of two bosons in a 2-D trap with V = C / r^p.

Run with: py -3 -m unittest discover -s tests -p test_inverse_power_trap_primitive.py
NumPy/SciPy are required.
"""

import math
from pathlib import Path
import sys
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent / "references"))
warnings.filterwarnings("ignore", message="A NumPy version")
from inverse_power_trap_primitive import (  # noqa: E402
    relative_partition, relative_traces, ring_log_z, solve)


class InversePowerTrapPrimitiveTests(unittest.TestCase):
    def test_no_interaction_gives_the_discretized_rings(self):
        # C = 0: direct = z(M), exchange = z(2M) / z(M), z the 2-D ring at
        # the same tau (normalization, r dr measure and the (-1)^m sign).
        for beta, slices, omega in ((2.0, 16, 1.0), (1.0, 10, 1.5)):
            with self.subTest(beta=beta, slices=slices):
                tau = beta / slices
                z = relative_partition(relative_traces(tau, slices, omega, 0.0, 3.0, 300, 12.0))
                direct = math.exp(ring_log_z(slices, tau, omega))
                exchange = math.exp(ring_log_z(2 * slices, tau, omega)) / direct
                self.assertAlmostEqual(z["direct"] / direct, 1.0, delta=1e-11)
                self.assertAlmostEqual(z["exchange"] / exchange, 1.0, delta=1e-11)

    def test_dipolar_pair_is_converged_in_the_nodes(self):
        tau, slices = 0.125, 16
        coarse = relative_partition(relative_traces(tau, slices, 1.0, 1.0, 3.0, 300, 12.0))
        fine = relative_partition(relative_traces(tau, slices, 1.0, 1.0, 3.0, 600, 12.0))
        self.assertLess(abs(math.log(fine["boson"] / coarse["boson"])), 1e-11)

    def test_values_quoted_by_the_grand_canonical_test(self):
        # test_pimc_grand_canonical.cpp, "Two dipolar bosons in a trap".
        res = solve(2.0, 16, 1.0, 1.0, 3.0, 300, 12.0)
        self.assertAlmostEqual(res["delta_free_energy"], 1.847461147235, delta=1e-11)
        self.assertAlmostEqual(res["energy_2"], 3.442269884856, delta=1e-9)
        no_exchange = -(res["ln_z_rel_direct"] - math.log(2.0)) / 2.0
        self.assertAlmostEqual(no_exchange, 1.899507970692, delta=1e-11)


if __name__ == "__main__":
    unittest.main()
