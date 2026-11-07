# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Checks of tests/references/pair_bound_state.py against exact 2-D two-body solutions
(hydrogen, harmonic oscillator) and the bilayer exciton of López Ríos et al.,
PRL 120, 177701 (2018)."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "references"))
import pair_bound_state as pbs  # noqa: E402


class ExactLimits(unittest.TestCase):
    def test_2d_hydrogen_ground_state_energy_and_size(self):
        # E_n = -mu Z^2 / (2 (n - 1/2)^2), R = e^{-kappa r} with kappa = 2 mu Z:
        # E = -1 and <r^2> = 6/(2 kappa)^2 = 3/2 at mu = 1/2.
        s = pbs.bound_state(pbs.bilayer_coulomb(0.0), 0.5)
        self.assertAlmostEqual(s.energy, -1.0, delta=1e-8)
        self.assertAlmostEqual(s.r2, 1.5, delta=1e-7)

    def test_2d_hydrogen_p_state(self):
        # n = 2, |l| = 1: E = -1/9, R = r e^{-r/3}, <r^2> = 20/(2/3)^2 = 45.
        s = pbs.bound_state(pbs.bilayer_coulomb(0.0), 0.5, ell=1, r_max=120.0)
        self.assertAlmostEqual(s.energy, -1.0 / 9.0, delta=1e-8)
        self.assertAlmostEqual(s.r2, 45.0, delta=1e-5)

    def test_reduced_mass_of_unequal_masses(self):
        # m = (1, 3): mu = 3/4, E = -2 mu = -3/2, kappa = 3/2, <r^2> = 1.5/kappa^2.
        mu = pbs.reduced_mass(1.0, 3.0)
        self.assertAlmostEqual(mu, 0.75, places=15)
        s = pbs.bound_state(pbs.bilayer_coulomb(0.0), mu)
        self.assertAlmostEqual(s.energy, -1.5, delta=1e-8)
        self.assertAlmostEqual(s.r2, 1.5 / 1.5 ** 2, delta=1e-7)

    def test_any_radial_potential_harmonic_oscillator(self):
        # V = mu w^2 r^2 / 2: E = w (|l| + 1), <r^2> = (|l| + 1)/(mu w).
        mu, w = 0.5, 0.7
        potential = lambda r: 0.5 * mu * w * w * r * r  # noqa: E731
        for ell in (0, 2):
            s = pbs.bound_state(potential, mu, ell=ell, r_max=20.0)
            self.assertAlmostEqual(s.energy, w * (ell + 1), delta=1e-8)
            self.assertAlmostEqual(s.r2, (ell + 1) / (mu * w), delta=1e-7)

    def test_orbital_is_normalized_and_positive(self):
        s = pbs.bound_state(pbs.bilayer_coulomb(0.4), 0.5)
        h = s.r[1] - s.r[0]
        self.assertAlmostEqual(float(np.sum(s.R ** 2 * s.r)) * 2 * math.pi * h, 1.0, places=12)
        self.assertTrue(np.all(s.R[: len(s.R) // 4] > 0))


class BilayerExciton(unittest.TestCase):
    def test_matches_the_papers_isolated_exciton(self):
        # López Ríos et al., data doi 10.17863/CAM.18830 (Exciton/out.2): VMC
        # E_X = -0.506519126(187) Ha* at d = 0.4.  VMC is an upper bound, and
        # an optimized nodeless trial sits within its error of the exact
        # energy; the paper quotes r_ex = 2.002 a0*.
        s = pbs.bound_state(pbs.bilayer_coulomb(0.4), 0.5)
        e_vmc, sigma = -0.506519126, 1.87e-7
        self.assertLessEqual(s.energy, e_vmc + 3 * sigma)
        self.assertLess(abs(s.energy - e_vmc), 3 * sigma)
        self.assertAlmostEqual(s.r_rms, 2.002, delta=1e-3)

    def test_grid_and_box_are_converged(self):
        coarse = pbs.bound_state(pbs.bilayer_coulomb(0.4), 0.5)
        fine = pbs.bound_state(pbs.bilayer_coulomb(0.4), 0.5, n_grid=40000, r_max=90.0)
        self.assertAlmostEqual(coarse.energy, fine.energy, delta=1e-8)
        self.assertAlmostEqual(coarse.r2, fine.r2, delta=1e-7)


class Refusals(unittest.TestCase):
    def test_repulsive_pair_has_no_bound_state(self):
        with self.assertRaisesRegex(ValueError, "no bound state"):
            pbs.radial_state_h(pbs.bilayer_coulomb(0.4, charge_product=1.0), 0.5, 0, 2000, 60.0)

    def test_non_finite_potential_is_refused(self):
        with np.errstate(invalid="ignore"), self.assertRaisesRegex(ValueError, "finite"):
            pbs.radial_state_h(lambda r: np.log(r - 1.0), 0.5, 0, 2000, 60.0)

    def test_bad_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            pbs.reduced_mass(1.0, 0.0)
        with self.assertRaises(ValueError):
            pbs.bilayer_coulomb(-0.1)
        with self.assertRaises(ValueError):
            pbs.radial_state_h(pbs.bilayer_coulomb(0.4), 0.0, 0, 2000, 60.0)


if __name__ == "__main__":
    unittest.main()
