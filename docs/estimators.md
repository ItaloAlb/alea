<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Estimators

An estimator turns the sampled configurations into an average and its error
bar. Every VMC, DMC and PIMC stage measures its energy. Other observables are
opt-in, listed under the stage's `estimators:`; their keys and defaults are in
the [configuration reference](configuration.md#estimators). This page says what
each estimator measures under each sampler, gives its formula and
normalization, explains its error bar, and then describes the three PIMC energy
estimators and the potentials each accepts.

- [1. What a sampler measures](#1-what-a-sampler-measures)
- [2. Energy](#2-energy)
- [3. Optional observables](#3-optional-observables)
- [4. Error bars of the observables](#4-error-bars-of-the-observables)
- [5. PIMC energy estimators](#5-pimc-energy-estimators)
- [6. Potential gradients](#6-potential-gradients)
- [7. Admission: which potentials each virial estimator accepts](#7-admission-which-potentials-each-virial-estimator-accepts)
- [8. What is actually sampled](#8-what-is-actually-sampled)
- [9. Not covered](#9-not-covered)
- [Tests](#tests)

## 1. What a sampler measures

Every driver hands its estimators a batch of configurations $R$, each with a
weight $w$. The estimator code is the same for the three drivers; the meaning of
the average comes from the distribution that was sampled:

| Driver | Configurations | Weight $w$ | Average of a diagonal observable $O(R)$ |
|---|---|---|---|
| VMC | walkers, $R \sim \vert\psi_T\vert^2$ | 1 | variational: $\langle\psi_T\vert O\vert\psi_T\rangle/\langle\psi_T\vert\psi_T\rangle$ |
| DMC | walkers, $R \sim \psi_T\Phi$ | the walker weight after the step's reweighting, before branching | mixed: $\langle\psi_T\vert O\vert\Phi\rangle/\langle\psi_T\vert\Phi\rangle$ |
| PIMC | every slice of every closed path | 1; the permutation sign $s = \pm1$ for fermions; $s\,[\text{closed}]$ with a worm move | thermal: $\mathrm{Tr}(O\,e^{-\beta H})/Z$ |

**Mixed-estimator bias (DMC).** $\Phi$ is the ground state under the node or
phase constraint. For the energy the mixed average is exact: $H$ commutes with
the projector, so $\langle\psi_T|H|\Phi\rangle/\langle\psi_T|\Phi\rangle = E_0$.
For every other observable its error is of first order in $\Phi - \psi_T$. The
extrapolated estimator

$$
\langle O\rangle_\text{pure} \approx 2\langle O\rangle_\text{DMC} - \langle O\rangle_\text{VMC}
$$

cancels the first-order term and leaves an error of second order (Ceperley &
Kalos, in *Monte Carlo Methods in Statistical Physics*, ed. K. Binder
(Springer, 1979); Foulkes *et al.*, *Rev. Mod. Phys.* **73**, 33 (2001)). It
needs a VMC and a DMC run with the same trial and the same estimator settings;
`tools/plot_density.py --extrapolate` does it for densities. A DMC stage can
instead measure the pure average itself by forward walking
([§3.1](#31-pure-dmc-estimators-forward-walking)), which removes the bias
entirely at the price of a larger error bar.

**Signs and worms (PIMC).** With fermion exchange, every PIMC output, the
energy included, is a ratio of means $\langle O s\rangle/\langle s\rangle$ with
a jackknife error, and the run reports the average sign. With a worm move,
diagonal observables are measured on closed configurations only: an open path
counts in the normalization with weight zero, and `closed_fraction` records how
often the paths were closed.

## 2. Energy

### VMC

The local energy $E_L(R) = H\psi_T(R)/\psi_T(R)$ is evaluated for every walker
at every step, after the step's `sub_steps` sweeps.

| Result | Meaning |
|---|---|
| `energy` | Mean of $E_L$ over walkers, steps and accumulation blocks; the real part for a complex trial |
| `energy_error` | Flyvbjerg–Petersen reblocking of each thread's chain of block means, pooled over threads. Each thread is an independent chain: blocks of different threads are never averaged together |
| `energy_variance` | $\langle E_L^2\rangle - \langle E_L\rangle^2$ over every accumulated sample. Zero for an exact eigenstate (zero-variance principle), so it measures the quality of the trial |
| `energy_imag`, `energy_imag_error` | Complex trial only: $\langle \mathrm{Im}\,E_L\rangle$. It vanishes for a Hermitian $H$; a value many error bars away from zero points to a wrong derivative in a complex component |
| `acc_block_energy` | Block means, thread after thread |

The reblocking diagnostics (`energy_error_naive`, `energy_tau_int`,
`energy_effective_blocks`, `energy_error_reliable`, ...) and when to trust the
error bar are described in
[Concepts](concepts.md#sampling-walkers-blocks-and-errors).

### DMC

The energy of block $b$ is the weighted mean over the whole population and the
block's steps,

$$
E_b = \frac{\sum_\text{steps}\sum_w w\,E_L(R_w)}{\sum_\text{steps}\sum_w w},
$$

with $E_L$ at the walker's position after the step and $w$ its weight after
the step's reweighting, before branching. The block energies form one chain,
correlated over several blocks through the walkers and $E_T$, and are reblocked
as one chain.

| Result | Meaning |
|---|---|
| `energy` | The mixed estimate. Exact for the constrained ground state, up to the time-step, population-control and branch-cutoff biases ([Concepts](concepts.md#dmc)) |
| `energy_error` | Reblocking of the single chain of block energies |
| `energy_variance` | **The variance of the block energies $E_b$**, not of $E_L$. It depends on the block length and the population, so it is not comparable with the VMC value and does not measure the trial |
| `acc_block_energy`, `eq_block_energy` | $E_b$ during accumulation and during equilibration. The equilibration series never enters `energy`; it is there to judge equilibration |
| `acc_etrial`, `etrial_final` | The trial energy $E_T$ of population control. It follows the energy but is not an estimate of it |

### PIMC

PIMC reports `energy`, `kinetic` and `potential`, with $E = K + V$ on every
measurement. The potential is the slice average
$\overline V = \frac1M\sum_k V(R_k)$; the kinetic energy depends on the
estimator chosen with `energy:` (§5). Each replica accumulates its own blocks;
the block means of all replicas are averaged index by index into grand blocks,
which are then reblocked. An error is reliable when the reblocking finds a
plateau and keeps at least 16 blocks.

| Result | Meaning |
|---|---|
| `energy`, `kinetic`, `potential` (and `_error`) | The estimator of §5; `acc_block_energy`, `acc_block_kinetic`, `acc_block_potential` hold the grand blocks (with exchange or a worm, the sign-weighted numerators $\langle O s\rangle_b$, and `acc_block_sign` holds $\langle s\rangle_b$) |
| `internal_energy`, `cm_energy_correction` | In an open cell with a translation-invariant Hamiltonian, the centre of mass is a free particle and contributes $d/(2\beta)$ to $E$. `internal_energy` $= E - d/(2\beta)$; the correction is 0 otherwise |
| `winding2` | Periodic cells: the squared winding number of the paths **in the lattice basis** (integer components $n_i$ of each loop's winding $W=\sum_i n_i\mathbf a_i$), summed over closed loops and divided by the number of particles and of lattice directions. In an orthogonal cell with equal sides $L$ it equals $\langle\vert W\vert^2\rangle/(L^2 N d)$, the quantity of Pollock & Ceperley, *PRB* **36**, 8343 (1987). **In a skew cell it is not**: in a hexagonal cell it is $4/3$ of that value for an isotropic fluid. Use `winding_tensor_*` for the superfluid fraction there. Zero in an open cell |
| `winding_tensor_xx`, `_xy`, `_yy` (`_xz`, `_yz`, `_zz` in 3-D), with `_error` and `acc_block_winding_tensor_*` | Periodic cells, any shape: the Cartesian winding tensor per particle, $T_{ab}=\langle\sum_c W_{c,a}W_{c,b}\rangle/N$ in bohr², in the cell's Cartesian frame. For one species of mass $m$ the superfluid-fraction tensor is $\rho_{s,ab}/\rho = m\,T_{ab}/\beta$, so $\rho_s/\rho = m\,\mathrm{tr}\,T/(d\beta)$. An isotropic (C₃ or C₄) superfluid has $T_{xx}=T_{yy}$ and $T_{xy}=0$; the winding along a unit vector $\mathbf u$ (e.g. the three lattice directions of a hexagonal cell) is $\mathbf u^{\mathsf T}T\mathbf u$. Zero in an open cell |
| `average_sign`, `sign_unreliable` | Fermion exchange: $\langle s\rangle$, flagged when $\vert\langle s\rangle\vert$ is below `sign_threshold` times its error |
| `energy_difference` | With `compare_thermodynamic: true`: the paired virial-minus-thermodynamic difference (§5) |
| `energy_estimator`, `potential_approximation` | Metadata: the estimator used, and how each potential term is represented (§8) |

PIMC error bars are statistical only: the time-step, finite-size and
equilibration biases are not included (`uncertainty_scope` in the results).

## 3. Optional observables

| `type` | Measures | Samplers | Cell | Result keys |
|---|---|---|---|---|
| `gofr` | Pair correlation $g_{ab}(r)$ of every group pair | VMC, DMC, PIMC | any; periodic: $r_\text{max} \le$ Wigner–Seitz inradius | `gofr_r`, `gofr_<a>_<b>` |
| `density` | One-body density $\rho_g(\mathbf r)$ | VMC, DMC, PIMC | any | `density_edges`, `density_<g>` |
| `collective_density` | Distribution of a linear combination of positions | VMC, DMC, PIMC | open | `cdensity_<name>` |
| `sofk` | Structure factor $S_{ab}(k)$ and $\langle\rho_g(\mathbf k)\rangle$ | VMC, DMC, PIMC | periodic 2D | `sofk_k`, `sofk_<a>_<b>`, `rhok_<g>_re/_im` |
| `electron_hole_proximity` | Fraction of electrons with no hole within $r_\text{cut}$ | VMC, DMC, PIMC | any; periodic: $r_\text{cut} \le$ inradius | `electron_hole_proximity_<e>_<h>_U_e`, `_C_e` |
| `bragg` | Crystal order on chosen reciprocal vectors | VMC, DMC, PIMC | periodic 2D | `bragg_<g>`, `bragg_<g>_directions` |
| `bond_order` | Sixfold bond order and bond-length distortion | VMC, DMC, PIMC | periodic 2D, ≥ 7 particles | `bond_<g>_global6`, `_local6`, `_distortion` |
| `density_matrix` | One- and two-body density matrices, pair condensate fraction | VMC, DMC | any | `density_matrix_r`, `density_matrix_rho1_<g>`, `density_matrix_rho2_<a>_<b>`, `density_matrix_pair_condensate_fraction_<a>_<b>` |
| `momentum_distribution` | One-body density matrix $n(r)$ and momentum distribution $n(k)$ | PIMC with a `worm` move | any; periodic: $r_\text{max} \le$ inradius | `obdm_<g>`, `nk_<g>` |

Every value comes with an `_error` series. The builder refuses a kind where its
sampler cannot measure it (`density_matrix` in PIMC, `momentum_distribution`
outside PIMC) and a cell it does not support. Several instances of one type in a
stage get distinct result keys through `name:`
([configuration](configuration.md#estimators)).

All the observables except the two density matrices are **diagonal**: functions
of $R$ alone, measured with the weights of §1. Under DMC they are mixed
estimates, and also pure ones with `forward_walking` (§3.1).

### `gofr`: pair correlation

For each unordered group pair $(a, b)$ the separations of all pairs (within a
group, $i < j$: $N_a(N_a-1)/2$ pairs; across groups, $N_aN_b$) are binned in
shells of width $\Delta = r_\text{max}/n_\text{bins}$, and

$$
g_{ab}(r) = \frac{\langle n_{ab}(r)\rangle}{N_\text{pairs}\,\Omega(r)/V},
$$

with $n_{ab}(r)$ the number of pairs in the bin of $r$ per configuration and
$\Omega(r)$ the shell area ($\pi(r_\text{hi}^2 - r_\text{lo}^2)$) or volume.
In a periodic cell $V$ is the cell volume, so $g \to 1$ at large $r$ in a
homogeneous fluid; distances are minimum-image, which is why $r_\text{max}$ may
not exceed the Wigner–Seitz inradius. An open cell has no bulk density: $V = 1$,
and $g_{ab}$ is the probability density of the pair separation per unit shell
area. A group with one particle has no same-group function.

### `density`: one-body density

Each group's particles are binned on a grid of $n_\text{bins}$ cells per axis,
and each block's estimate is

$$
\rho_g(\mathbf r) = \frac{\text{weighted counts in the cell}}{W_\text{block}\,V_\text{cell}},
\qquad W_\text{block} = \sum w,
$$

so $\int\rho_g\,dV$ is the mean number of group-$g$ particles inside the grid.
Particles outside the grid leave the numerator only. In an open cell the grid
is $[-r_\text{max}, r_\text{max}]^d$ in Cartesian coordinates. In a periodic
cell the grid is the cell itself, in fractional coordinates $[-\tfrac12,
\tfrac12)^d$ (`density_edges`), with the lattice vectors in `density_lattice`;
$r_\text{max}$ is not used and nothing falls outside.

`per_particle: true` gives one histogram per particle (`density_<g>_p<i>`).
Same-spin particles are exchangeable, so individual labels mean nothing; only
sums and differences over a spin channel are physical (e.g. in lithium the up
channel holds $|1s|^2 + |2s|^2$ and the lone down electron $|1s|^2$). The
density is in the lab frame: a free complex whose centre of mass diffuses gives
a smeared, nearly uniform density. Pin it with a trap or a fixed species, or use
`collective_density`.

### `collective_density`: distribution of a collective coordinate

One vector per configuration,

$$
\mathbf v = \sum_t c_t \sum_{p\in g_t}\mathbf r_p ,
$$

is histogrammed on $[-r_\text{max}, r_\text{max})^d$, normalized so that
$\int\rho_v\,dV = 1$ when $\mathbf v$ stays on the grid. Coefficients $+1/-1$ on
an electron and a hole group give the relative coordinate $\mathbf r_e - \mathbf
r_h$ (the internal exciton wavefunction squared); $c_g = q_g$ gives the dipole
distribution. When the coefficients sum to zero, $\mathbf v$ is invariant under
global translations and the image stays sharp while the centre of mass diffuses.
Open cells only: a sum of absolute positions changes by a lattice vector when a
particle is wrapped.

### `sofk`: static structure factor

$$
S_{ab}(k) = \frac{\langle \mathrm{Re}[\rho_a(\mathbf k)\rho_b(\mathbf k)^*]\rangle}{\sqrt{N_aN_b}},
\qquad \rho_g(\mathbf k) = \sum_{i\in g} e^{i\mathbf k\cdot\mathbf r_i},
$$

on every reciprocal-lattice vector with $0 < |\mathbf k| \le k_\text{max}$ (one
half space, since $S(-\mathbf k) = S(\mathbf k)$), averaged over each $|\mathbf
k|$ shell. $S_{aa} \to 1$ at large $k$. This is the full structure factor: for
a crystal, subtract the Bragg part $|\langle\rho_g(\mathbf k)\rangle|^2$, which
is why $\langle\rho_g(\mathbf k)\rangle$ is also written per vector (not shell
averaged: the peaks sit on particular vectors). The small-$k$ behaviour of
$S(k)$ is the input of the Chiesa finite-size correction (Chiesa, Ceperley,
Martin & Holzmann, *PRL* **97**, 076404 (2006)). Cost per measurement:
$O(n_w N n_k)$.

### `electron_hole_proximity`: unpaired-electron fraction

$$
U_e = \Big\langle \frac1{N_e}\sum_{e} \mathbf 1\big[\min_{h} r_{eh} \ge r_\text{cut}\big]\Big\rangle,
\qquad C_e = 1 - U_e,
$$

with $h$ running over `hole_group` and minimum-image distances under PBC. It is
a geometric diagnostic of local proximity. No electron is assigned to a hole,
so $C_e$ is **not** a bound-exciton fraction.

### `bragg`: crystal order on chosen wavevectors

$$
B = \frac1{n_G}\sum_{\mathbf G}\frac{\langle|\sum_{j\in g} e^{i\mathbf G\cdot\mathbf r_j}|^2\rangle}{N^2},
$$

over the listed vectors $\mathbf G$, each a nonzero reciprocal vector of the
cell (one member of each $\pm\mathbf G$ pair). $B = 1$ on a perfect lattice. It
is the full equal-time intensity, invariant under a global translation, and not
$|\langle\rho_{\mathbf G}\rangle|^2$, which vanishes for a crystal free to
slide. The directions are combined in each block before the error analysis, so
the error of $B$ keeps their covariance; `bragg_<g>_directions` gives each
direction's intensity. Crystal geometry as in Drummond & Needs, *PRL* **102**,
126402 (2009).

### `bond_order`: sixfold order

Each particle's six nearest neighbours (minimum image; a tie at the cutoff
shares the remaining weight equally, so the result does not depend on labels)
give $\psi_6(i) = \frac16\sum_j w_{ij}\,e^{6i\theta_{ij}}$, $\sum_j w_{ij} = 6$.
Three values are reported:

| Key | Value |
|---|---|
| `bond_<g>_global6` | $\langle\vert\frac1N\sum_i\psi_6(i)\vert^2\rangle$, the global orientational order |
| `bond_<g>_local6` | $\langle\frac1N\sum_i\vert\psi_6(i)\vert^2\rangle$, the local order |
| `bond_<g>_distortion` | $\langle\frac1{6N}\sum_{ij} w_{ij}(r_{ij}/a - 1)^2\rangle$, $a$ = `reference_spacing` |

The distortion measures bond lengths, not site displacements, and is not a
Lindemann criterion. Orientational order as in Nelson & Halperin, *PRB* **19**,
2457 (1979); the six-nearest convention is not a Voronoi construction.

### `density_matrix`: one- and two-body density matrices (VMC, DMC)

Off-diagonal: each measured configuration gets one virtual displacement
$\mathbf r'$, and the estimator takes ratios of $\psi_T$ (López Ríos, Perali,
Needs & Neilson, *PRL* **120**, 177701 (2018), Supplement Eqs. S6–S8;
Astrakharchik *et al.*, *PRL* **95**, 230405 (2005)):

$$
\text{rho1}_g(r) = \Big\langle\frac{\psi_T(\ldots,\mathbf r_i+\mathbf r',\ldots)}{\psi_T(R)}\Big\rangle,
\qquad
\text{rho2}_{ab}(r) = \Big\langle\frac{\psi_T(\ldots,\mathbf r_i+\mathbf r',\ldots,\mathbf r_j+\mathbf r',\ldots)}{\psi_T(R)}\Big\rangle,
$$

averaged over the configurations, the particles $i$ of $g$ (pairs $i \in a$,
$j \in b$ for rho2) and the directions of $\mathbf r'$, with $|\mathbf r'| = r$.
$\text{rho1}_g = \rho^{(1)}_g/n_g$ is 1 at $r = 0$; for bosons its large-$r$
plateau is the condensate fraction. The pair condensate fraction of a channel,

$$
n_c(r) = \frac{N_aN_b}{\min(N_a, N_b)}\left[\text{rho2}_{ab}(r) - \text{rho1}_a(r)\,\text{rho1}_b(r)\right],
$$

has a large-$r$ plateau equal to the fraction of pairs in the condensate (1 for
a single bound pair). The products are formed per block.

The length of $\mathbf r'$ is drawn uniformly within one bin, bins taken in
turn, and its direction uniformly; displaced positions are wrapped into the
cell. The random numbers of each sample come from their own hashed stream, so
the result does not depend on how walkers are split over threads. Under DMC this
is the mixed estimate, with an error of first order in $\Phi - \psi_T$: combine
it with VMC as in §1. One measurement costs about a few sweeps, so `every` skips
steps between measurements. `pairs` needs every factor of $\psi_T$ to supply
pair ratios (pairing and Slater determinants, Drummond Jastrow). PIMC refuses
this kind: it has no $\psi_T$; use `momentum_distribution`.

### `momentum_distribution`: $n(r)$ and $n(k)$ (PIMC worm)

Off-diagonal, measured on the open paths of a PIMC worm (Boninsegni,
Prokof'ev & Svistunov, *Phys. Rev. E* **74**, 036701 (2006)). At an open
measurement the head is extended by one free link to $\mathbf x$, and the
displacement $\mathbf x - \mathbf t$ to the tail is binned with weight $c$;
closed measurements give the normalization. With $N_w$ the particles of the
worm's group,

$$
n(r) = \frac1{N_w}\int\rho_1(R + \mathbf r, R)\,dR,\qquad
\tilde n(k) = \int n(r)\,e^{-i\mathbf k\cdot\mathbf r}\,d\mathbf r ,
$$

so $n(0) = 1$. In an open cell $\tilde n(k)$ is evaluated on a uniform grid of
`n_k` points up to $k_\text{max}$, with the angular average of
$e^{-i\mathbf k\cdot\mathbf r}$ ($\cos kr$, $J_0(kr)$, $\sin(kr)/kr$ for
$d = 1, 2, 3$). In a periodic cell the output is the occupation per state
$\langle n_{\mathbf k}\rangle = (N_w/V)\,\tilde n(\mathbf k)$, averaged over each
shell of reciprocal-lattice vectors with $|\mathbf k| \le k_\text{max}$
($\mathbf k = 0$ included; $\sum_{\mathbf k}\langle n_{\mathbf k}\rangle = N_w$).
`nk_<g>_number` and `nk_<g>_kinetic` are the finite-cutoff number and kinetic
sums, blocked jointly so that their errors keep the correlation between shells.
Every value is a ratio of means with a jackknife error. The statistical errors
exclude the momentum tail beyond $k_\text{max}$ and the quadrature error of the
grid.

### 3.1 Pure DMC estimators (forward walking)

With `forward_walking: {lags: [T_1, T_2, ...]}` a DMC stage measures every
estimator of its `estimators:` list a second time per lag, as a pure
estimate, together with the pure potential energy
(J. Casulleras & J. Boronat, *Phys. Rev. B* **52**, 3654 (1995)). The
configuration a walker had at time $t$ is weighted by the weight its
descendants carry at $t + T$:

$$
\langle O\rangle_T = \frac{\sum_w W_w(t+T)\,O\big(R_{a(w)}(t)\big)}{\sum_w W_w(t+T)}
= \frac{\langle\Phi|O\,e^{-T(H-E_0)}|\psi_T\rangle}{\langle\Phi|\psi_T\rangle}
\xrightarrow{T\to\infty} \frac{\langle\Phi|O|\Phi\rangle}{\langle\Phi|\Phi\rangle},
$$

with $a(w)$ the ancestor at $t$ of walker $w$ and $W$ the weights of the
mixed estimator (after the step's reweighting, before branching). $T = 0$ is
the mixed estimate itself, bit for bit. The bias of a finite $T$ is
$\sum_n c_n\langle\Phi|O|n\rangle e^{-T(E_n-E_0)}/c_0$ ($c_n = \langle n|\psi_T\rangle$):
it decays with the gaps of the excitations $O$ connects to $\Phi$. The
variance grows with $T$, because fewer and fewer ancestors carry the weight;
`fw<i>_surviving_fraction` is the mean fraction of a snapshot's walkers that
still have descendants at lag $i$. Run several lags and quote the plateau.
Under fixed-node or fixed-phase DMC, $\Phi$ is the constrained ground state.

How it is measured: every `interval` steps the population's positions are
stored once, and each walker keeps the index of its ancestor in every stored
snapshot through branching. When a snapshot is $T$ old, the current weights are
summed per ancestor and the ancestors are measured with those weights by the
ordinary estimator code. Snapshots start $T_\text{max}$ before accumulation, so
every accumulation block has the same number of pure measurements. Memory is
one copy of the positions per snapshot, $\lfloor T_\text{max}/(\text{interval}\,\tau\,n_\text{sub})\rfloor + 1$
snapshots (the run log prints the total).

| Result | Meaning |
|---|---|
| `<estimator keys with fw<i>>` | Each estimator's keys with `fw<i>` inserted after its namespace (and after its own label): `gofr_fw0_e_h`, `gofr_fine_fw1_e_h`, `density_fw2_e`, ... |
| `potential_fw<i>`, `potential_fw<i>_error` | Pure $\langle V\rangle$ at lag $i$, with the energy's reblocking diagnostics. The pure kinetic energy is $E - \langle V\rangle_\text{pure}$, since $E$ is exact |
| `acc_block_potential_fw<i>` | Its block means |
| `fw<i>_lag_time`, `fw<i>_lag_steps`, `fw<i>_measurements`, `fw<i>_surviving_fraction` | The lag, the measurements made, and the surviving fraction above |
| `fw_interval`, `fw_lag_times` | Steps between snapshots; all lags |

Only diagonal observables qualify: `density_matrix` (ratios of $\psi_T$) is
refused in a stage with `forward_walking`. The error bars of the pure
estimators follow §4: the blocks of a long lag are more correlated, so check
them with longer blocks.

## 4. Error bars of the observables

- **VMC and DMC** treat the blocks of an observable as independent samples
  (constant-memory block moments). Blocks shorter than the autocorrelation time
  then give error bars that are too small; check them with longer blocks or
  independent seeds. Only the energy is reblocked.
- **PIMC** keeps a bounded history of every bin (at most 64 MiB per replica),
  averages it over replicas into grand blocks and reblocks each bin as it does
  the energy. Each series then has `_error_naive`, `_tau_int`,
  `_reblock_plateau`, `_error_reliable` and the other reblocking diagnostics.
  A bin is reliable with a plateau and at least 16 surviving blocks.
- **Signed PIMC** (fermions, worm): each block keeps its mean sign, and every
  bin is the ratio of means $\langle O s\rangle/\langle s\rangle$ with a
  jackknife error over the reblocked blocks. Blocks are never divided one by
  one, which would bias the mean when a block's sign is small.

## 5. PIMC energy estimators

PIMC measures the energy with one of three estimators, chosen by the stage key
`energy:`. All three have the same finite-$M$ expectation value for the action
being sampled; they differ in variance and in what they require of the
Hamiltonian.

### Choosing an estimator

| `energy:` | Cell | Statistics | Requires | Variance |
|---|---|---|---|---|
| `thermodynamic` (default) | any | any | nothing | grows ∝ M |
| `centroid_virial` | open, every particle and axis confined | boltzmannons, bosons, fermions | every potential term has a gradient and open admission (§7) | roughly independent of M |
| `periodic_virial` | periodic | distinguishable only (refused when exchange is sampled) | every potential term has a gradient and periodic admission (§7) | grows slowly with M |

```yaml
energy: centroid_virial        # or periodic_virial
compare_thermodynamic: true    # optional, default false
```

`compare_thermodynamic` also accumulates the virial-minus-thermodynamic
difference **on the same paths**, block by block, and reblocks that difference
chain on its own (`acc_block_energy_difference`; a signed ratio under
exchange). The two measurements are correlated, so never compare them by
combining their separate error bars. Results record `energy_estimator`,
`energy_comparison` and, for the periodic estimator, `virial_geometry`.

Every check happens at setup, before replica paths are allocated: a missing
gradient, an unadmitted component or an unconfined axis is a named error, never
a silent fallback to `thermodynamic`. Measurements consume no random numbers
and do not move beads, so paths, acceptance and potential histories are
identical whichever estimator is selected (tested at one and four threads).

### Formulas

Primitive action, Hartree units, $\beta = M\tau$, $N$ particles in $d$
dimensions:

$$
S=S_K+\tau\sum_k V(R_k),\qquad
S_K=\sum_{p,k}\frac{\delta_{pk}^{T}m_p\,\delta_{pk}}{2\tau},
$$

with $\delta_{pk} = r_{p,k+1} - r_{pk}$ (minimum image in a periodic cell; the
next bead follows the permutation under exchange) and $m_p$ the diagonal mass
tensor.

**Thermodynamic.**

$$
E_{\rm th}=\frac{NdM}{2\beta}-\frac{S_K}{\beta}+\overline V,\qquad
\overline V=\frac1M\sum_k V(R_k).
$$

The two kinetic terms nearly cancel and each grows with $M$, hence the
variance.

**Centroid virial (open, confined).**

$$
E_{\rm cv}=\frac{Nd}{2\beta}+\overline V+
\frac{1}{2M}\sum_{p,k}(\mathbf r_{pk}-\bar{\mathbf r}_p)\cdot\nabla_p V(R_k),
\qquad K_{\rm cv}=E_{\rm cv}-\overline V,
$$

where $\bar{\mathbf r}_p$ is the arithmetic centroid of particle $p$'s ring (per
axis, also for anisotropic masses). Derivation: scale every bead's deviation
from its centroid by $\lambda$; the Jacobian gives $Nd(M-1)$ powers of
$\lambda$ and the springs scale as $\lambda^2$, so integration by parts gives
$2\langle S_K\rangle + \tau\langle \sum u\cdot\nabla V\rangle = Nd(M-1)$.
Substituting into $E_{\rm th}$ gives $E_{\rm cv}$. Reference: T. M. Yamamoto,
*J. Chem. Phys.* **123**, 104101 (2005), eqs. 6–10
([arXiv](https://arxiv.org/pdf/physics/0505109)); the estimator goes back to
Herman, Bruskin and Berne (1982).

- **Confinement is required.** The integration by parts needs zero boundary
  flux and a normalizable open trace. A bound relative coordinate does not
  confine a free centre of mass, so every particle and axis must be confined by
  a term that declares it (today a harmonic trap with $k > 0$ on that axis).
- **$M = 1$:** the fluctuation term vanishes and $K = Nd/(2\beta)$, a classical
  check.
- **Exchange:** each closed loop of $L$ particles is one ring of $LM$ beads.
  Deviations are taken from the loop's own centroid and the prefactor counts
  loops: $Nd/(2\beta) \to d\,n_\text{loops}/(2\beta)$. Identity connectivity
  reduces to the formula above.

**Periodic virial (distinguishable particles).** The open centroid depends on
the cell origin once beads wrap, and the textbook lifted winding formula does
**not** measure Alea's minimum-image action: for a free 1D ring with $L = 4$,
$m = 1$, $M = 2$, $\beta = 4$ it gives 0.125, while the sampled action's exact
energy is 0.15328 (boundary contribution $u\,e^{-u^2}/(\beta\sqrt\pi\,\mathrm{erf}\,u)$,
$u = (L/2)\sqrt{m/\tau}$). Alea instead uses a smooth periodic field and
integration by parts on the torus.

For any smooth periodic field $F$ and weight $e^{-S}$,
$\langle\nabla\cdot F - F\cdot\nabla S\rangle = 0$: opposite cell faces cancel
by periodicity, and the internal minimum-image faces cancel because the weight
is continuous there (its derivatives may jump). Hence

$$
E_{\rm pv}=E_{\rm th}+\frac{F\cdot\nabla S-\nabla\cdot F}{2\beta}
$$

has the same expectation. Alea's field, with $r = A^T f$ in reduced
coordinates, $\theta_{pak} = 2\pi f_{pak}$:

$$
z_{pa}=\frac1M\sum_k e^{i\theta_{pak}},\qquad
h_{pak}=\frac{\operatorname{Im}(e^{i\theta_{pak}}z_{pa}^{*})}{2\pi},\qquad
F_{pk}=A^{T}h_{pk},\qquad
\nabla\cdot F=\sum_{p,a}\left(M|z_{pa}|^2-1\right).
$$

It needs no unwrapping, circular mean or winding cut; it is regular at
$z = 0$; a global translation cancels. For a small non-winding ring it tends to
$r - \bar r$ and recovers the open centroid estimator; at $M = 1$ it is zero.
The contractions are

$$
Q_K=\frac1\tau\sum_{p,k}(F_{p,k+1}-F_{pk})^{T}m_p\,\delta_{pk},\qquad
Q_V=\tau\sum_{p,k}F_{pk}\cdot\nabla_pV(R_k),
$$

and the reported kinetic energy is
$NdM/(2\beta) - S_K/\beta + (Q_K - \nabla\cdot F + Q_V)/(2\beta)$, so
$E = K + V$ on every measurement. Cost $O(NdM)$ plus the gradient.

This sine field is derived for Alea; the literature supplies the method and the
cautions: Yamamoto (2005); S. Moustafa and A. J. Schultz,
[arXiv:2408.11214](https://arxiv.org/pdf/2408.11214) (coordinate mappings);
G. Spada, S. Giorgini and S. Pilati, *Condens. Matter* **7**, 30 (2022),
appendix eq. 59 ([arXiv](https://arxiv.org/pdf/2203.00010)) (a full-image
periodic expression and its endpoint term);
[arXiv:2501.17618](https://arxiv.org/pdf/2501.17618), section IV (minimum image
versus explicit winding images).

Not derived yet for permuted paths, so `periodic_virial` is refused when
exchange is sampled. Skew cells admit only isotropic mass per particle (the
Euclidean image search does not follow a mass metric); rectangular cells admit
diagonal anisotropic masses.

## 6. Potential gradients

Both virial estimators need $G = \nabla V$ of the potential **as sampled**. The
interface lives on `HamiltonianComponent` and is usable by every driver through
`ParticleBatch`:

- The returned quantity is the gradient $G = \nabla V$ (physical force $-G$), in
  energy per length, in the coordinate SoA layout `[p*dim*nw + d*nw + w]`, in
  caller-owned storage. Components accumulate; `Hamiltonian` zeros the
  requested columns, skips non-potential terms and sums in component order.
- `supports_potential_gradient()` defaults to false and requesting a gradient
  from such a term throws a named error. There is **no finite-difference
  fallback**: finite differences are test references only.
- A `PotentialGradientWorkspace` is prepared before sampling, one per replica
  or concurrent crowd (clones need their own). Evaluation allocates nothing,
  uses no RNG and changes no coordinates or distance tables.

| Component | G (pair terms: G_j = −G_i) |
|---|---|
| Harmonic trap | G_pa = m_pa ω² (r_pa − c_a) |
| Coulomb | G_i = −q_i q_j Δ / [ε (Δ² + d²)^{3/2}], Δ the minimum-image in-plane separation |
| Moiré | G_p = −2Ṽ Σ_g g sin(g·r_p + φ) |
| Rytova–Keldysh | derivative of the sampled kernel (§8), including the blend and the layer-softening chain rule; zero inside the frozen core |
| Open screened (FSC) | ½ q_i q_j V′_fit(ρ) Δ/ρ (Ry → Ha once); zero outside the fitted band |
| Ewald | real space: derivative of (actual kernel − long-range spline − cutoff correction); reciprocal: G_i = −2q_i Σ_k k Im[e^{ik·r_i} Σ_d w_{c(i)d}(k) ρ_d(k)*] with the stored half-space weights; self, background and capacitor terms have zero in-plane gradient |

Having a gradient does not admit a term to a virial estimator. A value jump in
the sampled $V$ (a kernel switch, a table join, an Ewald cutoff) adds an
interface term that pointwise gradients miss, so §8 makes every sampled
representation continuous and §7 admits terms one by one.

## 7. Admission: which potentials each virial estimator accepts

Each component opts in through `supports_open_centroid_virial()` and
`supports_periodic_virial(cell, np)`; both default to false. The driver never
tests potential names.

| Component | `centroid_virial` | `periodic_virial` | Conditions |
|---|---|---|---|
| `harmonic_trap` | yes; confines each axis with k > 0 | no | finite k ≥ 0 (inverted traps refused) |
| `coulomb` | yes | yes (minimum-image pair, not an Ewald sum) | repulsive, or attractive with layer separation d > 0 |
| `rytova_keldysh` | yes | yes | same pole rule as Coulomb; parameter scales that could overflow the value or gradient are refused |
| `moire` | yes (needs a trap for confinement) | yes | every g commensurate with the cell (g·a/2π an integer) |
| `screened_coulomb` (open FSC fit) | yes | yes | controlled fit finite in value and slope on a 1025-point scan of its band |
| `ewald` (Coulomb, RK, screened kernels) | — | yes, 2D | real-space cutoff value matches the long-range endpoint; finite weights and charges; same cell as built |
| pair-product action | no | no | experimental; thermodynamic only |
| fixed-node restriction (`nodes`) | no | no | the nodal surface carries a singular force the virials do not include; thermodynamic only |

Admission certifies the integration-by-parts identity **for the represented
potential**. It does not certify that the fitted potential equals the physical
one: fit, quadrature, clamp, cutoff, finite-τ and finite-size errors are all
outside the statistical error bar. PIMC prints each term's approximation in
`potential_approximation` (summary and result metadata). Singular attraction
in one plane stays refused by the primitive action itself, whatever the
estimator.

## 8. What is actually sampled

**Rytova–Keldysh.** V_ij = π q_i q_j K̃(x)/(2ε r₀), x = ρ/r₀, ρ = √(Δ² + d²).
K̃ is frozen below x = 0.001 (ρ_core = 0.001 r₀; continuous, slope kink, zero
force inside), a 24-coefficient stretched-coordinate Chebyshev fit up to x = 32,
a C² quintic blend s(u) = u³(10 − 15u + 6u²), u = (x − 32)/8, and above x = 40
the six-term asymptotic tail
A(x) = (2/π)/x·(1 − x⁻² + 9x⁻⁴ − 225x⁻⁶ + 11025x⁻⁸ − 893025x⁻¹⁰)
([DLMF 11.6.1](https://dlmf.nist.gov/11.6.E1)). Against an independent
quadrature of K = H₀ − Y₀ ([DLMF 11.5.3](https://dlmf.nist.gov/11.5.E3)) on 645
points over x = 0.001…10⁶: value error ≤ 3.1 × 10⁻⁹, relative force error
≤ 2.8 × 10⁻⁶ (sampled maxima, not proven bounds).

**Ewald real-space cutoff.** The optimized breakup (Natoli and Ceperley,
J. Comput. Phys. **117**, 171 (1995)) keeps its long-range spline and reciprocal
weights as an exact transform pair. To make the sampled short-range part vanish
smoothly at the cutoff R, a quintic Hermite correction P(r) on the last panel
[a, R] absorbs the endpoint residuals δ_n = V_s⁽ⁿ⁾(R) − V_long⁽ⁿ⁾(R⁻),
n = 0, 1, 2; the short-range part is V_s − V_long − P inside R and zero beyond,
continuous to second derivative. Its mean, −(2π/A)∫_a^R r P(r) dr, is added to
the k = 0 constants (omitting it would shift the energy). Construction refuses
hypot(R, d)/r₀ ≤ 0.001 (the cutoff would sit inside the RK frozen core). Force
check against an independent erfc slab sum (D. E. Parry, Surf. Sci. **49**, 433
(1975)): 3.0 × 10⁻⁸ at tolerance 10⁻⁸, 1.7 × 10⁻⁹ at 10⁻¹⁰.

**Periodic screened table.** `RadialTable` panels (degree 15) share value, slope
and curvature at every knot through a projected quintic Hermite correction, so
the table is C². Setup quadrature accumulates in `long double`; coefficients
below min(4ε_mach·max|samples|, tol/64) are dropped. Checked against an
image-charge sum for a dielectric slab (S. Ismail-Beigi,
[slab notes](https://volga.eng.yale.edu/sites/default/files/files/dielectric-slab.pdf))
for (ε, ε_out, d) = (8,1,2), (1,8,2), (14,1,6): value ≤ 7 × 10⁻¹⁴, slope
≤ 4 × 10⁻¹³, curvature ≤ 2 × 10⁻¹¹.

**Open screened (FSC controlled fit).** fast-screened-coulomb's
`fsc_fit_controlled` writes the ETM interaction (Cavalcante et al., PRB **97**,
125427 (2018)) as an analytic baseline 2/(ε_c√(ρ² + z²)) plus a smooth remainder
g(ρ) = 2∫J₀(kρ)h(k)dk, h(k) = ε_t,c(k)⁻¹ − ε_c⁻¹e^{−k|z|}, interpolated by cubic
Hermite pieces in log ρ (C¹ joins). `fit_tol` is enforced separately on value
**and** force (min(fit_tol, 5%)), each piece checked at seven off-node radii
against refined moments. Outside [ρ_min = 0.01 bohr, ρ_max] the value is frozen
and the force is zero. At a 0.01% target, measured errors against independent
image sums and a boundary-value solve are ≤ 0.0007% (value) and ≤ 0.0053%
(force). The legacy wide-band Chebyshev fit it replaced had force errors of
order 10⁶ % near ρ_min despite sub-percent value errors.

## 9. Not covered

- Pure estimates of off-diagonal observables (`density_matrix`); forward walking
  (§3.1) covers the diagonal ones and the potential energy.
- Reblocked error bars for the VMC and DMC observables (§4).
- `periodic_virial` with exchange (the torus field is not derived for cycles).
- Virial estimators for the pair-product action.
- Pressure, and any derivative with respect to the cell or the layer separation.
- A global bound on fitted-potential force error, and clamp-refinement studies.

## Tests

- `test_estimator_builders.cpp`: each kind builds where it can be measured and
  is refused elsewhere; every result key lies in the kind's namespaces;
  repeated instances get distinct keys.
- `test_pair_correlation.cpp`, `test_density.cpp`, `test_collective_density.cpp`,
  `test_structure_factor.cpp`, `test_electron_hole_proximity.cpp`,
  `test_bragg_order.cpp`, `test_bond_order.cpp`, `test_density_matrix.cpp`:
  known answers per kind.
- `test_pimc_estimators.cpp`: replica-synchronized reblocking of every PIMC
  observable, refusal of unsupported or oversized histories.
- `test_pimc_worm.cpp`: the worm and the momentum distribution.
- `test_potential_gradient.cpp`: gradient contracts, finite differences at
  three steps, Parry force reference.
- `test_pimc_centroid_virial.cpp`, `test_pimc_periodic_virial.cpp`: exact
  finite-M Gaussian traces, transfer-matrix traces on the torus,
  temperature-scaling derivatives, refusal of every unadmitted case.
- `test_pimc_rk_admission.cpp`, `test_pimc_kernel_admission.cpp`,
  `test_pimc_fsc_diagnostics.cpp`, `test_screened_table_continuity.cpp`: the
  represented kernels of §8 and their admission.
- `test_pimc_permutation.cpp`: cycle-centroid virial under exchange.
- `test_dmc.cpp` (`[forward_walking]`): descendant weights through hand-made
  branchings against hand-computed means; lag 0 equal to the mixed estimator
  bit for bit; the pure potential energy of a particle in a 2D trap with a too
  broad Gaussian trial against the exact lag dependence
  $\langle V\rangle_T = \tfrac12 + \tfrac18 e^{-2\omega T}$; refusal of lags that
  are not whole steps, exceed the equilibration or an interval that does not
  divide a block. `test_estimator_builders.cpp`: per-lag labels, refusal of
  `density_matrix` and of keys a mixed instance already writes.
