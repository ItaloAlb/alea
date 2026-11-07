<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Getting started

This page takes you from a fresh clone to a finished simulation, and shows how to
read what Alea writes. Every command and output below was run on the current code.
Your numbers will differ in the last digits if your thread count differs.

## 1. Requirements

- A C and C++ compiler with C++20 and OpenMP. The project is developed with
  GCC 15.2 (MSYS2) and CMake 4.1; any GCC with full C++20 support should work.
- CMake ≥ 3.20. Ninja is recommended.
- git, with network access the first time you configure. CMake downloads yaml-cpp,
  Eigen, HDF5, HighFive and fast-screened-coulomb.
- Python 3 with `numpy` and `h5py` for the analysis scripts in `tools/`.
  `matplotlib` and `scipy` are optional.

## 2. Build

### Windows (MSYS2 UCRT64)

Install [MSYS2](https://www.msys2.org/), open the **UCRT64** shell, and install
the toolchain:

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-ninja git
```

> **PATH pitfall.** If MSYS2's `mingw64/bin` is also on your PATH (common when
> running from Git Bash, VS Code or PowerShell), GCC can load the wrong runtime
> DLLs. It then fails to compile even an empty program, and `alea.exe` exits
> with code 127. Put only UCRT64 on the PATH, both when building and when
> running:
>
> ```bash
> export PATH="/c/msys64/ucrt64/bin:/usr/bin"
> ```
>
> If a configure already failed this way, delete the build directory before
> trying again, because CMake caches the broken compiler check.

### Linux and clusters

```bash
module load gcc cmake     # or your site's equivalent; a GCC with C++20 support
```

### Configure and compile (both platforms)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The first configure builds HDF5 from source, which takes a few minutes. Later
builds are incremental. Release mode compiles with `-O3 -march=native`, so a
binary built on one machine may not run on an older CPU. Rebuild on the machine
(or cluster node type) where you run.

The executable is `build/alea` (`build/alea.exe` on Windows).

**Debug build:** use `-DCMAKE_BUILD_TYPE=Debug` (gives `-O0 -g`) in a separate
directory such as `build-debug`.

**Local fast-screened-coulomb checkout (optional):** if you are changing the
screened-Coulomb library, point CMake at your clone instead of the pinned
download:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DFETCHCONTENT_SOURCE_DIR_FSC=/path/to/fast_screened_coulomb
```

The checkout must already provide the fitted-derivative and controlled-fit APIs.
CMake checks this and stops with a message if they are missing.

## 3. Check that it works

```bash
./build/alea tests/configs/exciton_2d_cusp_m1.yaml
```

```
vmc: E = -1 +/- 2.20766e-17  var = 0  acc = 0.821172
```

This input is a 2D electron–hole pair with an exact trial wavefunction
$\psi = e^{-r}$. Every sample has local energy exactly −1, so the variance is
zero. If you see this line, the build is correct.

## 4. Your first input file

Save this as `exciton_vmc.yaml`. It describes the same exciton with a trial
wavefunction that is deliberately *not* exact:

```yaml
project:
  name: exciton_vmc      # results go to results/exciton_vmc/<timestamp>/
  seed: 42               # same seed + same thread count → same numbers
  log: true              # also write results/.../alea.log
  output: true           # write one HDF5 file per stage (the default)

system:
  lattice: { type: open, dimension: 2, L: 1.0 }   # no boundaries; L = initial scatter box
  particles:
    - { name: e, group: electrons, type: fermions, mass: 1.0, charge: -1, up: 1, down: 0 }
    - { name: h, group: holes,     type: fermions, mass: 1.0, charge: +1, up: 1, down: 0 }

wavefunction:
  # Padé Jastrow u(r) = (a1 r + a2 r²)/(1 + b1 r), psi = exp(u).
  # a1 (the cusp) is derived from the Hamiltonian; you give a2.. and b1..
  - jastrow:
      type: pade
      pair: [electrons, holes]
      numerator: [-0.1]      # a2
      denominator: [0.3]     # b1

hamiltonian:
  terms:
    - { type: coulomb, pair: [electrons, holes], epsilon: 1.0 }

qmc:
  - method: vmc
    random_walk: metropolis
    threads: 0                # 0 = all cores
    samples: 512              # walkers, split across threads
    equilibration_blocks: 50  # discarded
    accumulation_blocks: 200  # averaged
    steps_per_block: 20       # measurements per block
    sub_steps: 5              # moves between measurements
    time_step: 0.3            # proposal scale (VMC); imaginary-time step (DMC, PIMC)
```

Run it:

```bash
./build/alea exciton_vmc.yaml
```

```
vmc: E = -0.873688 +/- 0.000488632  var = 0.129203  acc = 0.739927
```

| Field | Meaning |
|---|---|
| `E` | Mean local energy $\langle E_L\rangle$ in Hartree, with its standard error |
| `var` | Variance of $E_L$. It is zero for an exact eigenstate, so it measures trial quality |
| `acc` | Fraction of proposed moves accepted. Tune `time_step` to keep it roughly 0.3 to 0.8 for Metropolis |

The exact energy is −1. The VMC energy lies above it, as the variational principle
requires. The next step improves the trial.

### Units

Alea uses Hartree atomic units: $\hbar = e = m_e = 4\pi\varepsilon_0 = 1$.
Energies are in Hartree and lengths in Bohr. Effective-mass models enter through
each particle's `mass` (in units of $m_e$) and each interaction's `epsilon`.
For effective (excitonic) units, set the masses and ε so that the reduced mass
and dielectric constant are 1.

## 5. A multi-stage run: optimize, then VMC, then DMC

A `qmc:` list can hold several stages. Replace the `wavefunction:` and `qmc:`
blocks above with:

```yaml
wavefunction:
  - jastrow:
      type: pade
      pair: [electrons, holes]
      numerator: [-0.1]
      denominator: [0.3]
      is_fixed: [false, false]   # let the optimizer move a2 and b1

qmc:
  - method: optimizer          # linear method; omitted keys use optimizer defaults
    samples: 256
    max_iterations: 10
  - method: vmc
    random_walk: metropolis
    samples: 512
    equilibration_blocks: 50
    accumulation_blocks: 200
    steps_per_block: 20
    sub_steps: 5
    time_step: 0.3
  - method: dmc
    samples: 512
    equilibration_blocks: 50
    accumulation_blocks: 200
    steps_per_block: 20
    sub_steps: 5
    time_step: 0.01
```

```
optimize: linear  iter=4  conv=no  E=-1.00011  params=[-0.0863931, 0.0843467]  shift_final=2.68435e+07
vmc: E = -0.999982 +/- 5.01364e-06  var = 2.41899e-05  acc = 0.627869
dmc: E = -0.999981 +/- 1.13572e-05  var = 2.5797e-08  acc = 0.998056
```

After the first iteration the optimizer reaches $a_2/b_1 \approx -1$, so
$u(r)\to -r$ at large $r$: the trial is nearly exact and the variance falls by
four orders of magnitude. Later iterations could not improve it further, so the
optimizer stopped with `conv=no`. In a real study, read `alea.log`, which prints
every iteration and every candidate shift, before trusting an optimization.

How stages run:

- **All `optimizer` stages run first**, in the order written, wherever they appear
  in the list. The other stages then run in document order.
- **All stages share one trial wavefunction.** Parameters the optimizer changes are
  the ones VMC and DMC use.
- **Each sampling stage starts where the previous one ended.** DMC starts from the
  VMC walkers, which are already distributed as $\vert\psi_T\vert^2$. An explicit
  `walkers_in:` overrides this.

## 6. What Alea writes

```
results/exciton_vmc/2026-10-13_06-07-16-713/
├── 0_vmc.h5            one HDF5 file per stage: <stage index>_<method>.h5
├── alea.log            progress, timings, warnings (only with log: true)
└── exciton_vmc.yaml    snapshot of the input: resolved seed, final parameters
```

- **The snapshot reproduces the run.** It records the seed that was actually used,
  which matters when you leave `seed` at 0 and Alea picks one. It also holds the
  wavefunction parameters as they were at the end, so after an optimization it is
  an optimized input with the optimizer stage removed.
- **Checkpoints** (`walkers_out:` for DMC, `paths_out:` for PIMC) go to
  `checkpoints/<project>/<same timestamp>/`.
- `output: false` writes no HDF5 file. `log: false` (the default) writes no log.
- **A results file is complete or absent.** Each one is written as
  `<i>_<method>.h5.partial` and renamed only when complete. If writing fails, the
  partial file is removed, the error goes to stderr (and `alea.log`), the
  remaining stages still run, and `alea` exits with status 1.

Each HDF5 file stores scalars as root attributes and time series as datasets.
For the three-stage run above:

| File | Attributes (selection) | Datasets |
|---|---|---|
| `0_optimizer.h5` | `final_energy`, `final_energy_error`, `iterations`, `converged`, `last_step_checked`, `last_step_undone`, `check_energy` | `energy_history`, `shift_history`, `final_params` |
| `1_vmc.h5` | `energy`, `energy_error`, `energy_variance`, `acceptance_ratio`, `tau`, `n_walkers`, `n_threads`, `walltime_s` | `acc_block_energy` (one series per thread, concatenated) |
| `2_dmc.h5` | the VMC ones plus `etrial_final`, `node_constraint`, `population_generations`, `branch_cutoff` (scheme), `branch_cutoff_sigma`, `branch_cutoff_particles` (N), `branch_cutoff_energy`, `branch_cutoff_alpha` (universal only), `branch_cutoff_fraction`, `branch_cutoff_fraction_eq`, `reweighting`; with `suppressed` also `suppression_kappa`, `correlation_time`, `correlation_tau_int`, `correlation_sweeps`, `correlation_window_found`, `tau_eff_ratio`, `suppression_mean` | `acc_block_energy`, `acc_population`, `acc_total_weight`, `acc_etrial`, `eq_population`, `eq_etrial` |

Reading them in Python:

```python
import h5py
with h5py.File("results/exciton_vmc/<timestamp>/0_vmc.h5") as f:
    print(f.attrs["energy"], f.attrs["energy_error"])
    blocks = f["acc_block_energy"][:]
```

### Check the error bar

The printed error is already reblocked: it accounts for the correlation between
consecutive blocks. If the run was too short for that error to be trusted, the
summary ends with a `WARNING` line; run more accumulation blocks before quoting
the number. A run is long enough when the error is below the precision you
need and there is no warning. The results file keeps the plain error
(`energy_error_naive`) and the autocorrelation time (`energy_tau_int`, in
blocks). To see the whole reblocking table, or to plot it:

```bash
python tools/reblock.py results/exciton_chain/<timestamp>/2_dmc.h5 --plot reblock.png
```

See
[tools/README.md](../tools/README.md#reblockpy--decorrelation--error-bar-check-for-energy-histories)
for the method and options.

## 7. A finite-temperature (PIMC) run

```bash
./build/alea tests/configs/pimc_ho_1d.yaml
```

```
pimc: E = 0.654041 +/- 0.00423671  K = 0.32576 +/- 0.00418666  V = 0.32828 +/- 0.00235598  W2 = 0 +/- 0  acc = 0.844771  tau_int=0.232554
pimc: identity connectivity (boltzmannon); finite-tau bias is NOT included in error bars; energy estimator=thermodynamic
pimc: error diagnostics do not establish equilibration; compare independent starts and longer equilibration
```

This is one particle in a 1D harmonic trap at $\beta = 2$ with $M = 8$ time
slices. The exact energy *for this M* is 0.65238, inside the error bar. The
continuum ($M\to\infty$) value 0.65652 differs by the time-step (Trotter) bias,
which the error bar does not include, as the second line warns. PIMC results
need a time-step extrapolation: run the same input at several M and
extrapolate to τ → 0.

## 8. When something goes wrong

- **`error: ...: unknown key '...'`.** Alea refuses a key it does not read rather
  than ignoring it, so a typo stops the run before anything is computed. The
  message names the block and the key; compare against the
  [configuration reference](configuration.md). A key that was *removed* (for
  example `max_age` or `param_tol` in older inputs) is refused with a message
  saying what to do instead.
- **Energy near zero for a bound system.** The trial is probably not normalizable.
  For example, a Padé Jastrow with only a denominator, $u = a_1 r/(1+b_1 r)$, tends
  to a constant, so the particles drift apart.
- **Energy grows without bound in a trap.** A pair Jastrow depends only on
  separations, so it leaves the center of mass free. Something in $\psi$ must
  confine every coordinate, such as a determinant of localized orbitals.
- **`exit code 127` on Windows.** This is the PATH pitfall from section 2.
- **Low acceptance (< 0.2) or acceptance ≈ 1.** For VMC, adjust `time_step`.
  For DMC, acceptance near 1 is normal, and a low value means τ is too large.

## Where next

- [Concepts](concepts.md) explains the vocabulary used in configs and logs.
- [Configuration reference](configuration.md) lists every key.
- `configs/tests/` has ready inputs for excitons, trions and biexcitons in
  MoS₂, MoSe₂, WS₂ and WSe₂, for lithium, and for a moiré exciton.
