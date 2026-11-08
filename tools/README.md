<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# tools/ — analysis utilities

## `reblock.py` — decorrelation / error-bar check for energy histories

VMC, DMC and PIMC already report the reblocked energy error
(`numerics::reblock` / `reblock_chains`, the same method and plateau rule as
this tool), with `energy_error_naive`, `energy_tau_int` and
`energy_error_reliable` in the results file. `reblock.py` re-analyses an energy
history the run emits, prints the full reblocking table, plots it, and reblocks
histories from older results or other programs. It answers:

- **What block length decorrelates the data?** (`block length` = statistical
  inefficiency `s = 1 + 2·τ_int`, rounded up — average this many samples to get
  one independent sample).
- **What is the honest error bar?** (`corrected error` = the plateau standard
  error from Flyvbjerg–Petersen reblocking).

Method: H. Flyvbjerg & H. G. Petersen, *J. Chem. Phys.* **91**, 461 (1989).
The data are repeatedly coarse-grained (adjacent pairs averaged); the SEM rises
with block length and plateaus once the block exceeds the autocorrelation length.
The corrected error is the (conservative) plateau value; the decorrelation block
length follows from `(corrected/naive)²`.

### Producing the input

Run a simulation with HDF5 output enabled:

```yaml
project: { name: my_study, output: true }   # output: true → writes .h5
qmc:
  - { method: dmc, ... , accumulation_blocks: 8192, threads: 0 }
```

```sh
./build/alea.exe configs/my_study.yaml
# → results/my_study/<timestamp>/0_dmc.h5   (dataset: acc_block_energy)
```

Both VMC and DMC write the per-accumulation-block energy series as the
`acc_block_energy` dataset, with `method` / `n_threads` / `n_accumulation_blocks`
as root attributes.

### Engine shapes (handled automatically for `.h5`)

- **VMC** — `acc_block_energy` is `n_threads` *independent* chains concatenated.
  The loader reshapes to `(n_threads, n_acc)` and pools them so correlation is
  never measured across a thread boundary.
- **DMC** — one global branched chain; reblocked as-is.

### CLI

```sh
# native Alea result (auto-detects VMC vs DMC from the file)
py -3 tools/reblock.py results/my_study/<ts>/0_dmc.h5

# plain-text series (e.g. one value per line, or pick a column)
py -3 tools/reblock.py energies.txt --column 2

# treat a 1-D text series as K independent chains, and save a figure
py -3 tools/reblock.py energies.txt --chains 8 --plot reblock.png

# one-line summary, or run the built-in validation
py -3 tools/reblock.py 0_vmc.h5 --quiet
py -3 tools/reblock.py --selftest
```

Output reports the mean, naive vs corrected error, decorrelation block length,
statistical inefficiency / `τ_int`, and the full reblocking table (the plateau
onset is marked).

### Library

```python
from reblock import reblock, reblock_chains, load_series

series, meta = load_series("results/.../0_dmc.h5")   # 1-D, or (K, N) for VMC
res = reblock(series) if series.ndim == 1 else reblock_chains(series)
print(res.mean, res.error, res.block_length, res.inefficiency, res.tau_int)
```

`load_series` returns 2-D `(n_threads, n_acc)` for multi-thread VMC and 1-D
otherwise; dispatch on `series.ndim` as above.

### Dependencies

- **numpy** — required (core + text input).
- **h5py** — optional, only to read native `.h5` (`pip install h5py`). The
  plain-text path works without it.
- **matplotlib** — optional, only for `--plot`.

### Limitation

The finest series the engine emits is the per-**accumulation-block** mean, so the
tool detects residual correlation *between blocks* (it reports `block length = 1`
when the blocks are already decorrelated). Probing per-measurement correlation
would require the engine to emit a finer (per-step) series — out of scope here.

## `plot_density.py` — visualize the QMC wavefunction density

In DMC the walker population samples the mixed distribution `ψ_T·Φ` — the
walker cloud *is* the wavefunction made visible. The `{type: density}`
estimator histograms the walker positions per particle group over the whole
accumulation phase (weighted by the DMC branch weights, i.e. the mixed
estimator `⟨ψ_T|ρ̂|Φ⟩/⟨ψ_T|Φ⟩`; in VMC it gives `⟨ψ_T|ρ̂|ψ_T⟩`), and this tool
renders it: a heatmap per group, plus optional radial profile, `√ρ ~ |ψ|`
amplitude, and the pure-estimator extrapolation `ρ_pure ≈ 2ρ_DMC − ρ_VMC`
(Ceperley & Kalos 1979; Foulkes et al., *Rev. Mod. Phys.* **73**, 33 (2001)).

### Producing the input

Declare the estimator under any VMC/DMC driver and enable HDF5 output
(see `configs/tests/exciton_density.yaml` for a complete runnable example):

```yaml
project: { name: my_study, output: true }
qmc:
  - method: dmc
    # ...
    estimators:
      - { type: density, rmax: 8.0, n_bins: 64 }   # grid [-8, 8]^dim, 64 bins/axis
```

`per_particle: true` resolves one histogram per particle (`density_<group>_p<i>`,
`i` local to the group) instead of one per group. Same-spin labels are
exchangeable, so read them through channel sums/differences — e.g. lithium
1s²2s¹ (`configs/tests/lithium_density.yaml`): `ρ_↑ = p0+p1 = |1s|²+|2s|²`,
`ρ_↓ = p2 = |1s|²`, and `ρ_↑ − ρ_↓ ≈ |2s|²` isolates the excited orbital.
(In fixed-node DMC the labels cannot cross the node, so individual same-spin
densities show the nodal pockets — compact 1s vs diffuse 2s — but only the
channel combinations are label-permutation invariant.)

```sh
./build/alea configs/my_study.yaml
# → results/my_study/<timestamp>/0_dmc.h5
```

The estimator writes, per particle group `<g>`:

- `density_<g>` — the grid flattened C row-major (`cell = ix·n_bins + iy` in
  2-D); `np.reshape(a, (n_bins,)*dim)` gives `rho[ix, iy]`.
- `density_<g>_error` — per-cell block standard error, same layout.
- `density_edges` — the `n_bins+1` bin edges from `−rmax` to `+rmax` (the grid
  is a cube: one axis serves every dimension).
- root attributes `density_dim`, `density_n_bins`, `density_rmax`,
  `density_n_blocks`.

The grid is **lab-frame**, centred at the origin. For a free complex the centre
of mass diffuses and the image smears — a sharp wavefunction needs something
pinned (fixed particles, as in the coulomb_limit study, or a moiré potential).

### CLI

```sh
# heatmap per group (ρ, viridis; diverging map if the field goes negative)
py -3 tools/plot_density.py results/my_study/<ts>/0_dmc.h5

# add the azimuthally averaged profile ρ(|r|), and plot √ρ ~ |ψ| instead of ρ
py -3 tools/plot_density.py 0_dmc.h5 --radial --amplitude

# 3-D grids: z≈0 midplane slice instead of the z-integrated column density
# (preserves radial structure like shell nodes, which the integral fills in)
py -3 tools/plot_density.py 0_dmc.h5 --slice --amplitude

# pure-estimator extrapolation (needs a VMC run with the SAME rmax/n_bins)
py -3 tools/plot_density.py 1_dmc.h5 --extrapolate 0_vmc.h5 --radial

# quick-look scatter of a walker checkpoint (walkers_out) — no estimator needed
py -3 tools/plot_density.py --walkers wk.eq.h5

py -3 tools/plot_density.py --selftest
```

PNGs land next to the input (`<stem>_density_<g>.png`, `<stem>_radial.png`,
`<stem>_pure_<g>.png`, `<stem>_walkers.png`); `--out DIR` redirects. The tool
prints `∫ρ dV` per group — it should match the group's particle count when
nothing leaves the grid.

### Library

```python
from plot_density import load_density, radial_profile, extrapolate

groups, meta = load_density("results/.../1_dmc.h5")
rho, err = groups["e_grp"]                       # shaped (n_bins,)*dim
r, rho_r, err_r = radial_profile(rho, err, meta["edges"])
```

### Dependencies

- **numpy** — required.
- **h5py** — optional, only to read `.h5` (`pip install h5py`).
- **matplotlib** — optional, only for the plots.

## `moire_cells.py` — configs for electrons in a moiré superlattice

Writes one alea YAML (plus a `.meta.json` sidecar) for one competing state —
the pinned Wigner crystal or the fluid — at one point of the moiré phase
diagram. Both states of a point get the **same** periodic cell, so they differ
only in the determinant. Units are effective atomic units (`a_B*`, `Ha*`), so
`--x` is `a_M/a_B*` and `--kappa` is `V_0/Ha*`.

Before writing, it checks that the cell is a supercell of the moiré lattice and
holds exactly `M = N_e/ν` moiré cells. It also checks that the crystal sites
sit on the deepest moiré minima and form the commensurate pattern (√3×√3 at
1/3, 2×2 at 1/4, the honeycomb of √3×√3 vacancies at 2/3), and that every spin
group of the liquid closes a reciprocal shell at Γ, or does not tie at the
Fermi edge at the requested `--twist`.

- **Cells.** `--cell hex`: the hexagonal `(p, q)` supercells. `--cell rect`:
  `n × m` rectangles of `a_M × √3 a_M` (two moiré cells each), the most compact
  first, written axis-aligned (θ₀ = −30° by default) so alea takes its
  orthorhombic fast path. `--shape` picks one.
- **Spin.** `--spin polarized` (one group, `electron_group`) or `unpolarized`
  (groups `e_up` and `e_dn`, N/2 each, as backflow needs). The unpolarized
  crystal is the collinear stripe state (alternate rows of sites up and down),
  in cells the stripes tile.
- **Units.** `--x` with `--kappa` or `--lam` (λ = V₀/E_M, κ = λ/x²), or a
  material preset (`--material geier2025`, Geier et al. 2025) with
  `--epsilon` or `--inv-epsilon`.
- **Open shells.** At Γ a liquid whose shell is open needs an explicit choice
  of real orbitals, written as `orbitals.k`/`orbitals.sin`: `--pw-pick` with
  indices into the open shell, or `--pw-variants` for one file per symmetry
  class of choices (classes found with the symmetries of H in the cell).
- **Report.** `--report FILE.md` lists, per filling, spin and cell type, the
  cells that hold the crystal, the Γ occupation per spin and the twists of a
  `twist_grid.py` grid where the occupation ties (alea refuses those);
  `--detail ν:N:cell:spin` adds the open shell's orbitals, their symmetry
  classes, every twist of the grid, and the tied twists of other grids.
  `--list` is the short form for one filling.

Seeds written into the config:
- the Gaussian width, from the harmonic curvature of the moiré well plus the
  Einstein cage of the Wigner crystal;
- the one-body `moire_jastrow`, at its first-order perturbative value for the
  fluid and a tenth of that for the crystal;
- a Drummond pair Jastrow (like- and unlike-spin when unpolarized), with a
  fixed cutoff of 3 r_s within the Wigner–Seitz inradius, the same for the
  crystal and the liquid so that neither phase gets a longer-ranged Jastrow.

The optimizer stages move all of them. The DMC time steps are set from the
crystal's localization and are identical for both states, so the time-step
bias cancels in `E_fluid − E_crystal`.

```sh
py -3 tools/moire_cells.py --selftest
py -3 tools/moire_cells.py --list --nu 1/3          # valid N_e: 13, 19, 31, 37, ...
py -3 tools/moire_cells.py --nu 1/3 --ne 19 --x 15 --kappa 0.02 --phi-deg 60 \
    --state crystal --out crystal_N19.yaml
# add a DMC chain (tau*omega values, HEG2D-rs30 scale):
#   --dmc-tau-omega 0.006 0.003 0.0015
py -3 tools/moire_cells.py --nu 1/3 --ne 42 --cell rect --spin unpolarized \
    --x 9 --lam 4 --phi-deg 225 --state crystal --out crystal_N42.yaml
py -3 tools/moire_cells.py --nu 2/3 --ne 18 --material geier2025 --inv-epsilon 0.11 \
    --phi-deg 225 --state liquid --pw-variants --out liquid.yaml
py -3 tools/moire_cells.py --report shells.md --nu 1/3 2/3 --spin unpolarized polarized \
    --cells hex rect --twist-grid 4 --detail 2/3:18:hex:polarized --phi-deg 225
```

The selftest checks, among others: sites on the deep minima at φ = 225°
(−4.24 V₀); N_e = ν M for every cell; no hexagonal cell closing a Γ shell for
the unpolarized liquid at ν = 1/3, and among compact rectangles (aspect ≤ 2,
N ≥ 12) only 9 × 7 (N = 42) and 9 × 9 (N = 54); the open shell 13 + 5/6 of 18
polarized electrons in 27 cells; and, within a symmetry class of open-shell
choices, |ψ| of one choice equal to |ψ| of another at transformed coordinates.

Scope: `ν = 1/m` with `m` Loeschian (1, 3, 4, 7, ...) and `ν = 2/3`
(polarized), on single-minimum moiré shapes (for example φ = 60° or 225°).
Honeycomb shapes (φ ≈ 0°), `ν = 1/2` (stripes and the floating crystal) and
crystals on the shallow wells are not handled; they are refused with a message.
Standard library only (it imports `twist_grid.py` for the twist grid).

## `twist_grid.py` — twist-averaged boundary conditions

Makes one config per twist of a grid from a periodic base config, runs them
and averages the energies (Lin, Zong & Ceperley, PRE 64, 016702 (2001)). Each
twist is an independent run with a complex trial (fixed-phase DMC).

```powershell
py -3 tools/twist_grid.py make    BASE.yaml --grid 4 --out runs/tabc/heg
py -3 tools/twist_grid.py run     runs/tabc/heg --jobs 4 --threads 16
py -3 tools/twist_grid.py average runs/tabc/heg --method dmc
```
