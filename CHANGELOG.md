<!--
SPDX-FileCopyrightText: 2026 The Alea Authors
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to Alea are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html);
[docs/releasing.md](docs/releasing.md) defines what counts as its public
interface.

## [0.1.0] - 2026-11-06

### Added

- First versioned release, licensed under GPL-3.0-or-later.
- Variational Monte Carlo with Metropolis or Langevin sampling, and
  linear-method optimization of the trial wavefunction.
- Diffusion Monte Carlo: fixed node for real trials, fixed phase for complex
  trials, with population control and selectable branch-weight schemes.
- Path-integral Monte Carlo for distinguishable particles, bosons and
  fermions: permutation and joint electron–hole exchange moves, a canonical
  worm for the momentum distribution, a fixed-node restriction, and
  thermodynamic and virial energy estimators.
- Trial wavefunctions: Padé, Drummond, k-space and moiré Jastrow factors,
  real and complex Slater determinants, and pairing determinants.
- Interactions: Coulomb, Rytova–Keldysh, multilayer screened Coulomb, moiré
  potentials, inverse-power pair potentials and harmonic traps, with a
  uniform magnetic field or rotation; open or periodic cells with Ewald sums
  and twisted boundary conditions.
- Estimators: pair correlation, density, structure factor, collective
  density, electron–hole proximity, momentum distribution, Bragg and bond
  order, and one- and two-body density matrices with the pair condensate
  fraction.
- YAML input, HDF5 results, walker and path checkpoints, and analysis tools
  for reblocking, density plots, twist grids and moiré cells.

[0.1.0]: https://github.com/ItaloAlb/alea/releases/tag/v0.1.0
