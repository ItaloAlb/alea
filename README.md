<!--
SPDX-FileCopyrightText: 2026 The Alea Authors
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Alea

<!-- Badges (CI, license, DOI) go here once the repository is public. -->

Alea is a quantum Monte Carlo code for condensed matter calculations in two- or
three-dimensional materials. It contains variational Monte Carlo, diffusion
Monte Carlo and path-integral Monte Carlo drivers. It builds a single C++20 executable, 
and each run is described by one YAML file.

## Features

| Method | What it computes |
|---|---|
| **VMC** (variational Monte Carlo) | $\langle\psi_T\vert H\vert\psi_T\rangle$ for a trial wavefunction $\psi_T$, by Metropolis or Langevin sampling of $\vert\psi_T\vert^2$ |
| **Linear-method optimizer** | Trial-wavefunction parameters that lower the VMC energy |
| **DMC** (diffusion Monte Carlo) | Ground-state energy by imaginary-time projection |
| **PIMC** (path-integral Monte Carlo) | Finite-temperature thermal averages |


## Installation

You need a C/C++ compiler with OpenMP (GCC 11 or newer), CMake ≥ 3.20, git, and
network access on the first configure. CMake downloads the other dependencies
with `FetchContent`:

| Library | Version |
|---|---|
| OpenMP | from the compiler |
| yaml-cpp | 0.8.0 |
| Eigen | 3.4.0 |
| HDF5 + HighFive | 1.14.5 / 2.10.0 |
| [fast-screened-coulomb](https://github.com/ItaloAlb/fast-screened-coulomb) | pinned commit |
| Catch2 | 3.5.4 |

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The first configure compiles the dependencies, HDF5 included, so it takes a few
minutes. On Windows, build in the MSYS2 UCRT64 environment (GCC, OpenMP, CMake,
Ninja); the full setup, including Windows PATH pitfalls, is in
[docs/getting-started.md](docs/getting-started.md).

## Quick start

```bash
./build/alea tests/configs/exciton_2d_cusp_m1.yaml
```

That input is a 2D exciton with an exact trial wavefunction, so the energy comes
out exactly −1 Hartree with zero variance:

```
vmc: E = -1 +/- 2.20766e-17  var = 0  acc = 0.821172
```

## Documentation

Start at **[docs/index.md](docs/index.md)**. It suggests reading orders for
people who want to *run* simulations and for people who want to *change* the
code.

| Page | Contents |
|---|---|
| [Getting started](docs/getting-started.md) | Build, run a first simulation, read the output |
| [Concepts](docs/concepts.md) | Glossary: walkers, blocks, trial wavefunction, τ, fixed node, time slices, worms, units |
| [Configuration reference](docs/configuration.md) | Every YAML key with its type, default and meaning |
| [Architecture](docs/architecture.md) | Source layout, simulation flow, data layout, threading |
| [Extending Alea](docs/extending.md) | How to add a wavefunction factor, a Hamiltonian term or an estimator |
| [Testing](docs/testing.md) | Unit, physics and golden tests, and how to run them |
| [Estimators](docs/estimators.md) | What each VMC, DMC and PIMC estimator measures, and the PIMC virial energies |
| [Releasing](docs/releasing.md) | Versioning and the release procedure |
| [Analysis tools](tools/README.md) | Reblocking, density plots, moiré cell and twist-grid generators |

## Repository layout

```
main.cpp        entry point: parse YAML → build drivers → run them in order → write results
src/
  io/           YAML parser, logger, HDF5 results, checkpoints
  builders/     config → physics objects
  qmc/          VMC, DMC, linear optimizer, and App (stage orchestration)
  pimc/         path-integral driver, actions, moves
  wavefunction/ trial-wavefunction factors (Jastrows, Slater determinants, ...)
  hamiltonian/  kinetic energy and potentials
  estimators/   optional observables (density, g(r), S(k), ...)
  population/   walkers, particle batches, distance tables, simulation cell
  numerics/     statistics, random numbers, Ewald breakup, special functions
configs/        example inputs; configs/tests/ holds small physical systems
tests/          Catch2 C++ tests, Python checks of the tools, test inputs
tools/          Python analysis tools
docs/           documentation
```

## Testing

The tests are a separate CMake project in `tests/`:

```bash
cmake -S tests -B build/tests -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/tests -j
ctest --test-dir build/tests --output-on-failure -j
python -m unittest discover -s tests -p "test_*.py" -v
```

[docs/testing.md](docs/testing.md) describes what they cover, the golden
characterization file, and what to run before a commit.

## Versioning

Alea follows [Semantic Versioning](https://semver.org). What counts as its
public interface, and how a version is released, is in
[docs/releasing.md](docs/releasing.md); the changes of each version are in
[CHANGELOG.md](CHANGELOG.md).

## Citing

If you use Alea in published work, please cite it as described in
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button reads it).

## License

Alea is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. See
[LICENSE](LICENSE).

## Acknowledgements

This study was financed in part by the Coordenação de Aperfeiçoamento de
Pessoal de Nível Superior – Brasil (CAPES) – Finance Code 001.
