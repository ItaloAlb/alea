<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Alea documentation

Alea is a quantum Monte Carlo code (VMC, DMC, wavefunction optimization and PIMC)
for electrons and holes in 2D and 3D. The [README](../README.md) gives a
one-page overview. This page suggests what to read, depending on what you want to
do.

## I want to run simulations

1. **[Getting started](getting-started.md)**: build, run a first input, read the
   output, check error bars.
2. **[Concepts](concepts.md)**: what walkers, blocks, τ, cusps, fixed node and
   time slices mean, and which YAML key controls each.
3. **[Configuration reference](configuration.md)**: every key, its default, and
   which stages read it.
4. **Example inputs**: [`configs/tests/`](../configs/tests) (excitons, trions,
   biexcitons in TMDs, lithium, moiré) and [`tests/configs/`](../tests/configs)
   (small systems with known answers).
5. **[Analysis tools](../tools/README.md)**: reblocking error bars, density
   plots, moiré cell and twist-grid generators.

## I want to change or extend the code

1. **[Architecture](architecture.md)**: the layers, what happens in a run, data
   layout, threading, and where new physics plugs in.
2. **[Extending Alea](extending.md)**: recipes for a new Hamiltonian term,
   wavefunction factor, estimator or YAML key.
3. **[Testing](testing.md)**: how to build and run the tests, what they cover,
   and the golden characterization file.
4. **[Releasing](releasing.md)**: what the version number promises and how a
   version is released.

## I want to know why something is the way it is

- **[Estimators](estimators.md)**: what each VMC, DMC and PIMC estimator
  measures (energy and optional observables, mixed versus pure averages, error
  bars), and the PIMC thermodynamic, centroid-virial and periodic-virial
  energies with the potentials each accepts.
- Header comments in `src/` give the equation each class implements and its
  reference.

## Conventions used in these pages

- Units are Hartree atomic units unless stated otherwise.
- `build/alea` means `build/alea.exe` on Windows.
- Output shown in code blocks comes from real runs of the current code.
