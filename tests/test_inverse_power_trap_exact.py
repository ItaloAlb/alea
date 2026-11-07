# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Closed-form checks for tests/references/inverse_power_trap_exact.py, the exact reference
for two particles in a 2-D harmonic trap with V = C / r^p.

Run with: py -3 -m unittest discover -s tests -p test_inverse_power_trap_exact.py
NumPy/SciPy are required.  The tolerances cover the stated discretization.
"""

import math
from pathlib import Path
import sys
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent / "references"))
warnings.filterwarnings("ignore", message="A NumPy version")
from inverse_power_trap_exact import (  # noqa: E402
    log_grid_spectrum_h, radial_spectrum_h, total_energy)


class InversePowerTrapReferenceTests(unittest.TestCase):
    def test_no_interaction_is_two_2d_oscillators(self):
        for omega in (0.5, 3.0):
            with self.subTest(omega=omega):
                self.assertAlmostEqual(total_energy(omega, 0.0, 3.0), 2.0 * omega, delta=1e-7)

    def test_coulomb_case_is_tauts_dot(self):
        # p = 1, C = 1, M = 1, w = 1: R = (1 + r) exp(-r^2/4), E = 3.
        self.assertAlmostEqual(total_energy(1.0, 1.0, 1.0), 3.0, delta=1e-7)

    def test_inverse_square_shifts_the_centrifugal_term(self):
        # p = 2: m^2 -> m^2 + 2 mu C, so eps = w (1 + sqrt(2 mu C)) at m = 0.
        omega, C, mu = 1.0, 0.75, 0.5
        exact = omega * (1.0 + math.sqrt(2.0 * mu * C))
        self.assertAlmostEqual(total_energy(omega, C, 2.0) - omega, exact, delta=1e-7)
        # The ln r grid used for p > 2 reaches the same level (Richardson h, h/2).
        e1 = log_grid_spectrum_h(0, omega, C, 2.0, mu, 4000, 1e-12, 20.0)[0]
        e2 = log_grid_spectrum_h(0, omega, C, 2.0, mu, 8001, 1e-12, 20.0)[0]
        self.assertAlmostEqual((4.0 * e2 - e1) / 3.0, exact, delta=1e-8)

    def test_dipolar_core_agrees_between_the_two_grids(self):
        # p = 3 at a weak trap, where the uniform r grid is still converged to
        # ~1e-5; the ln r grid is the production path.
        omega, C, mu = 0.5, 1.0, 0.5
        r_max = 14.0 / math.sqrt(mu * omega)
        e1 = radial_spectrum_h(0, omega, C, 3.0, mu, 40000, r_max)[0]
        e2 = radial_spectrum_h(0, omega, C, 3.0, mu, 80000, r_max)[0]
        linear = omega + (4.0 * e2 - e1) / 3.0
        self.assertAlmostEqual(total_energy(omega, C, 3.0), linear, delta=2e-5)

    def test_dipolar_core_is_converged_in_the_grid(self):
        for omega in (1.0, 20.0):
            with self.subTest(omega=omega):
                coarse = total_energy(omega, 1.0, 3.0, n_grid=10000)
                fine = total_energy(omega, 1.0, 3.0, n_grid=20000)
                self.assertLess(abs(fine - coarse), 1e-8 * fine)


if __name__ == "__main__":
    unittest.main()
