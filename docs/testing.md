<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Testing

Alea has two test suites:

- **C++ tests** (Catch2 v3) in `tests/test_*.cpp`. These are unit tests of every
  component, physics regressions against analytic or published results, and a
  golden test of the config-to-object path.
- **Python checks** of the exact references in `tests/references/`:
  `tests/test_pair_bound_state.py` checks the isolated-pair solver against 2-D
  hydrogen, the 2-D harmonic oscillator and the published bilayer exciton, and
  `tests/test_inverse_power_trap_exact.py` checks the two-body reference of the
  `inverse_power` term against closed forms.

`tests/references/` holds the independent exact references whose energies the
physics tests use: `dot_two_electron_exact.py` (two electrons in a 2-D
parabolic dot in a magnetic field, for the fixed-phase tests),
`inverse_power_trap_exact.py` (two particles in a 2-D trap with a C/r^p
interaction, for the `inverse_power` term) and `pair_bound_state.py` (the
isolated pair of a bilayer, for the pairing-determinant test). Each solves a finite-volume
radial eigenproblem, extrapolated by Richardson over spacings h and h/2;
`--check` reports the remaining grid and box errors:

```bash
python tests/references/dot_two_electron_exact.py --w0 0.25 --B 0 0.2 0.4 --m 0 -1
python tests/references/inverse_power_trap_exact.py --omega 1 5 20 --strength 1 --exponent 3
python tests/references/pair_bound_state.py --d 0.4 --check
```

Full simulations compared with exact references and with the literature (the
PIMC validation battery, convergence studies, reproductions of published
results) are not a test suite you run before a commit; they are kept and run
separately from the code.

## Running the C++ tests

The tests are a **separate CMake project** in `tests/`, built into `build/tests`:

```bash
cmake -S tests -B build/tests -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/tests -j
ctest --test-dir build/tests --output-on-failure -j 6
```

- `ctest --test-dir build` (without `/tests`) finds **no tests** and still exits
  with code 0. Check the summary line.
- Building the tests does not rebuild `build/alea`, and building `build/alea`
  does not rebuild the tests. Rebuild both after changing `src/`.
- The full suite takes several minutes. To run a subset:

  ```bash
  ctest --test-dir build/tests -R pade --output-on-failure   # by test name (regex)
  ./build/tests/test_parser "[seed]"                        # one executable, by Catch2 tag
  ```

**Windows: set a workspace-local temporary directory.** Several tests (logger,
config writer, checkpoint I/O, App round-trips) create directories under the
system temp folder. On some Windows setups the test process is denied that, and
about 40 tests fail together with `Permission denied [...\AppData\Local\Temp\...]`.
This is not a code failure. Redirect the temp root before running ctest:

```bash
export TMP="$PWD/build/testtmp" TEMP="$PWD/build/testtmp" TMPDIR="$PWD/build/testtmp"
mkdir -p "$TMP"
```

Use the UCRT64-only PATH from [Getting started](getting-started.md#windows-msys2-ucrt64)
here too.

### How the test build is organized

Each test executable compiles only the `src/*.cpp` files it needs, listed in
`tests/CMakeLists.txt`. Changing one component then rebuilds only the tests that
use it. When you add a source file, add it to every test target that links
against it. A missing file shows up as an undefined-reference link error.

The Release `test_slater_determinant`, `test_density`, `test_sweep_blocking`
and `test_simulation_cell` targets also use `-march=native -fno-math-errno`,
matching production CPU arithmetic. This
exercises SIMD/FMA in the long Slater contracts and the constant-density
variance regression; generic CPU arithmetic can miss the FMA roundoff case.

The long Slater sequence, near-node recovery and rejected-nonfinite contracts
run both eager and deferred commits. Deferred sequences refresh the moved
particle before drift and all derivatives before the dense-inversion oracle,
and clone or resize populations while partner derivatives are still deferred.

`test_sweep_blocking` runs VMC (with and without drift) and two-thread DMC on
a 19-electron periodic determinant with several walker block sizes, and
requires bit-identical energies, block series, populations and coordinates.
`test_simulation_cell` requires the vectorized batch minimum image to match
the scalar fold bit for bit, including separations on the cell boundary.

`test_statistics` checks the reblocked error against the analytic error of
AR(1) series, and that independent chains (VMC threads) are never
coarse-grained across a chain boundary. `test_vmc` and `test_dmc` require the
reported energy error to be the reblocking of the block energies the run
writes; `test_vmc` also requires neighbouring seeds to give their threads
different random streams, and fixes the reliability threshold at 50 effective
blocks.

## What the C++ tests cover

| Kind | Files (examples) | What they check |
|---|---|---|
| Component contracts | `test_pade_jastrow.cpp`, `test_drummond_jastrow.cpp`, `test_slater_determinant.cpp`, `test_backflow_slater.cpp`, `test_backflow_pairing.cpp`, `test_backflow_twisted.cpp`, `test_coulomb.cpp`, `test_ewald.cpp`, `test_rytova_keldysh.cpp` | Values against direct sums; gradients and Laplacians against finite differences; ratio and accept consistency; crowd ranges; clones; parameter derivatives of ln ψ and, for the curvature term, of its gradients and Laplacians. |
| Physics regressions | `test_vmc_physics.cpp`, `test_dmc.cpp`, `test_fixed_node.cpp`, `test_fixed_phase.cpp`, `test_pimc_physics.cpp`, `test_pimc_many_body.cpp` | Full runs on small systems from `tests/configs/` compared with analytic or published energies; the forward-walking (pure) potential energy of a trapped particle against its exact lag dependence; fixed-node DMC with backflow on a two-electron dot whose node it keeps exact, and the bilayer exciton with a backflow pairing determinant; at a twist, with and without backflow, ⟨Im E_L⟩ = 0 in VMC and fixed-phase DMC invariant under θ → −θ |
| Parser and I/O | `test_parser.cpp`, `test_config_writer.cpp`, `test_walker_io.cpp`, `test_path_io.cpp` | Every key is read; removed keys are refused; snapshots and checkpoints round-trip |
| Estimator kinds | `test_estimator_builders.cpp`, `test_pair_correlation.cpp`, `test_density.cpp`, `test_electron_hole_proximity.cpp` | Each spec builds its estimator and is refused only where it cannot be measured; every result key lies in the kind's `output_namespaces()`; repeated instances get distinct keys; known answers per kind |
| Determinism | `test_app.cpp`, `test_vmc.cpp`, `test_dmc.cpp` | Same seed and threads give identical results |
| Random numbers | `test_xoshiro_random.cpp` | The VMC/DMC generator against the reference xoshiro256++ outputs; its uniforms and Gaussians against their exact distributions (χ² including the tails, moments) |
| Golden characterization | `test_build_characterization.cpp` | See below |

Example physics test: `tests/configs/exciton_2d_cusp_m1.yaml` is a 2D exciton
whose trial satisfies the cusp exactly, so VMC must return $E = -1$ with zero
variance. `tests/configs/pimc_ho_1d.yaml` is a harmonic oscillator whose exact
energy at finite $M$ is known in closed form, so PIMC is checked without
time-step bias.

### Golden build characterization

`test_build_characterization` builds **every YAML file** under `configs/`,
`configs/tests/` and `tests/configs/` through the real `App::build_drivers`, and
evaluates $\ln\vert\psi\vert$ and $E_L$ on a few deterministic walkers. The results
(or the error message, for configs meant to be refused) are compared with
`tests/golden/build_characterization.txt`. Any refactor of the builders that
changes physics fails this test.

The comparison is relative, to $10^{-10}$, with two allowances so that it
passes on any machine. $\ln\vert\psi\vert$ is compared relative to the first
walker: a constant in it is only $\psi$'s normalization, and in a periodic cell
it moves between compilers and math libraries. $E_L$ in a periodic cell is
compared to $10^{-8}$, the accuracy of the long-range breakup that the Ewald sum
and the image-summed Padé Jastrow use. The breakup's fit is ill-conditioned, so
round-off differences between builds (for example, `sin`, `cos` and `exp`
differ in the last bit between math libraries) reach about $10^{-9}$ there.

**When you add a config, or deliberately change the physics**, regenerate the
golden file and review the diff:

```bash
cp tests/golden/build_characterization.txt /tmp/golden.before
ALEA_UPDATE_GOLDEN=1 ./build/tests/test_build_characterization
git diff tests/golden/build_characterization.txt
```

For a new config, the diff should only *add* entries. A config with no golden
entry fails the test, so a forgotten regeneration is caught.

## Running the Python checks

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

They need `numpy` and `scipy`, and they do not run `build/alea`.

## Before you commit

1. Build `build/alea` and `build/tests`.
2. Run `ctest --test-dir build/tests --output-on-failure` and get a clean summary.
3. If you changed physics or added a config, regenerate the golden file and
   check its diff.
4. For a change to a sampler, Green's function or branching, show that a
   reference energy at small τ is unchanged within error bars.
