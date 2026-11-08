<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Configuration reference

An Alea run is one YAML file with up to six top-level sections. This page lists
every key the parser reads ([`src/io/parser.h`](../src/io/parser.h) and
[`parser.cpp`](../src/io/parser.cpp)), with its default and meaning. Physical
quantities are in Hartree atomic units (see [Concepts](concepts.md#system)).

```yaml
project:      { ... }   # name, seed, output switches
system:       { ... }   # cell (lattice) and particles
wavefunction: [ ... ]   # trial-wavefunction factors (VMC, DMC, optimizer)
hamiltonian:  { terms: [ ... ] }
action:       [ ... ]   # PIMC only; optional
qmc:          [ ... ]   # the stages to run, in order
```

**Validation.** Every block refuses keys it does not read, so a misspelled key
stops the run with `unknown key '<name>'` instead of silently falling back to a
default. Enumerated values are checked too: an unknown term `type`, wavefunction
item, `method`, `random_walk` or `device` is an error. Keys from features that
were removed are refused by name, with a message that says what to do instead:
`cusp:`, Padé `params:`/`num_orders:`, DMC `feedback:`, `max_age:`,
`branch_interval:` and `reconfiguration:`, optimizer `param_tol:`, and the
fixed-R keywords.

Examples: [`configs/tests/`](../configs/tests) (excitons, trions, biexcitons,
lithium, moiré), [`tests/configs/`](../tests/configs) (small reference systems),
and [`configs/`](../configs) (studies: 2D electron gas, bilayers, PIMC).

---

## `project`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | `""` | Run name. Results go to `results/<name>/<timestamp>/` |
| `seed` | integer | `0` | Base random seed. `0` means "pick a random seed"; the chosen value is written to the snapshot |
| `log` | bool | `false` | Write `alea.log` (progress, timings, warnings) in the results directory |
| `output` | bool | `true` | Write one HDF5 file per stage, plus the config snapshot |

---

## `system`

### `system.lattice`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | `open` \| `periodic` | `open` | No boundaries, or a periodic Bravais cell |
| `dimension` | int | — (required) | Spatial dimension of the motion (1, 2 or 3) |
| `L` | float or list | `0` | **open:** half-width of the box in which walkers start. **periodic:** lattice constant; a list `[Lx, Ly]` gives a rectangular cell |
| `shape` | `square` \| `rectangular` \| `hexagonal` | `square` | Periodic cell shape when built from a scalar `L` or `rs` |
| `vectors` | `[[a1x, a1y], [a2x, a2y]]` | — | Periodic: explicit primitive vectors (row per vector). Overrides `L`, `rs`, `shape` |
| `rs` | float | — | Periodic, 2D only: density parameter; the cell area is $N\pi r_s^2$ for $N$ particles |
| `ewald_tol` | float | `1e-6` | Periodic: relative accuracy of the Ewald sum |
| `capacitor` | `include` \| `exclude` | `include` | Periodic, groups in several layers: `include` keeps the parallel-plate energy of the layers' net charges (it is in $E_L$, and the log reports it) and needs a neutral cell; `exclude` drops every $k = 0$ term between layers, as if each layer were neutralized in its own plane. That is CASINO's convention, and it allows a charged multilayer cell (e.g. an extra electron in an electron–hole bilayer) |
| `truncation` | `minimum_image` | — | Periodic: use the nearest image for long-range pair terms instead of Ewald. **Not a converged interaction**; for geometry tests only |
| `twist` | list of `dimension` floats | Γ | Periodic: twisted boundary condition $\psi(r + a_i) = e^{2\pi i t_i}\psi(r)$, in fractional reciprocal coordinates. Away from Γ it requires `plane_wave` orbitals, and DMC runs fixed-phase. Not supported by PIMC |

A periodic cell needs exactly one of: `vectors`, `rs`, `L` as a list, or a scalar
`L` (with `shape`).

### `system.particles`

A list. Each entry declares one kind of particle; particle indices follow the
order of the entries.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | `""` | Label used in messages |
| `group` | string | `""` | **The name everything else refers to.** Jastrow, potential and determinant `pair`/`group` keys use group names |
| `type` | `fermions` \| `bosons` \| `boltzmannon` | `""` | Quantum statistics. Read by PIMC. VMC/DMC take the symmetry from $\psi$: a `slater` item makes the group antisymmetric, so it is refused on `bosons`, whose trial is a product of positive factors (`jastrow`, `nosanow`). Write bosons as `up: N, down: 0` |
| `mass` | float or list | `1.0` | Mass in units of $m_e$; a list gives one value per axis (anisotropic) |
| `charge` | float | `0` | Charge in units of $e$ (electron: `-1`) |
| `up`, `down` | int | `0` | Number of spin-up and spin-down particles in this entry |
| `flavours` | list of int | — | Counts per flavour (spin × valley ...). Replaces `up`/`down`; **PIMC only** |
| `z` | float | `0` | Fixed out-of-plane coordinate (layer height); pairs in different layers interact at $\sqrt{r^2 + \Delta z^2}$ |
| `initialization` | `uniform` \| `fixed` | uniform | `fixed` starts these particles at the origin (a clamped nucleus; combine with a very large mass) |

---

## `wavefunction`

A list of factors, each a map with a single key naming its kind
(`- jastrow: {...}`). The trial wavefunction is their product, and item *i*
becomes component *i*. The optimizer's parameter order and the snapshot rely on that.
PIMC ignores this section.

**Parameter flags** (Jastrow-type items): `is_fixed` has one bool per free
coefficient, and every coefficient is **fixed unless set to `false`**.
`can_flip_sign` has one bool per coefficient (default `true`); `false` keeps that
coefficient's sign during optimization.

### `- jastrow:` (pair Jastrow)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | `pade` (or `padé`) \| `drummond` \| `zero_energy` | — (required) | Functional form |
| `pair` | `[groupA, groupB]` | — (required, or `pairs`) | The groups it correlates; the same group twice for like particles |
| `pairs` | list of `[groupA, groupB]` | — | **Drummond:** one $u(r)$ shared by several group pairs, in place of `pair`. Every pair must need the same cusp, and none may repeat |
| `numerator` | list | `[]` | **Padé:** $a_2, \dots, a_N$ in $u = (a_1 r + a_2 r^2 + \dots)/(1 + b_1 r + \dots)$ |
| `denominator` | list | `[]` | **Padé:** $b_1, \dots, b_M$ |
| `params` | list | `[]` | **Drummond:** $[\alpha_0, \alpha_2, \dots, \alpha_N]$ |
| `cutoff` | float | — (required for Drummond) | **Drummond:** initial cutoff length $L > 0$; under PBC at most the Wigner–Seitz inradius. **zero_energy:** the fixed range $r_c$ where $u$ reaches 0; under PBC optional (default: the Wigner–Seitz inradius, which it may not exceed), required in an open cell |
| `cutoff_fixed` | bool | `false` | **Drummond:** keep $L$ fixed. Note that $L$ is optimized by default, unlike the other coefficients; under PBC the optimizer holds it at most at the inradius |
| `truncation` | int | `3` | **Drummond:** exponent $C \ge 2$ of $(r-L)^C$; 3 gives a continuous local energy at $L$ |
| `matching_radius` | float | — (required for zero_energy) | **zero_energy:** $R_m$, where the two-body solution is joined to the tail; $0 < R_m < r_c$ |
| `matching_radius_fixed` | bool | `true` | **zero_energy:** `false` lets the optimizer move $R_m$ |
| `is_fixed`, `can_flip_sign` | list of bool | see above | Padé order: numerator then denominator. Drummond order: `params` |

Keys belong to one form: a Padé Jastrow refuses `params`, `cutoff`,
`cutoff_fixed` and `truncation`, a Drummond Jastrow refuses `numerator` and
`denominator`, and a zero_energy Jastrow accepts only `matching_radius`,
`matching_radius_fixed` and `cutoff`. The cusp coefficient ($a_1$ for Padé, $\alpha_1$ for Drummond) is
derived from the Hamiltonian and is never an input; a `cusp:` key is an error. In a periodic
cell, a Padé Jastrow is summed over all images and needs an Ewald term. A
Drummond Jastrow needs its cutoff inside the cell.

`pairs:` gives several group pairs one set of parameters, for example every
same-spin pair of an electron–hole bilayer split into four polarized groups:

```yaml
- jastrow: { type: drummond, pairs: [[e_up, e_up], [e_dn, e_dn], [h_up, h_up], [h_dn, h_dn]],
             params: [0.0], cutoff: 5.0 }
```

The builder refuses a list whose pairs need different cusps (different masses,
charges or layers, or a determinant on only some of the groups), because one
$u(r)$ has one slope at contact. Padé and zero_energy take one `pair:` per item.

**zero_energy** is the pair factor for a repulsive core $V \approx C/r^p$ with
$p > 2$ (an `inverse_power` term; dipoles have $p = 3$), in 2D. Padé and Drummond
stay finite at contact, which leaves $\langle C/r^p\rangle$ infinite for
$p \ge 2$; this form vanishes as $\exp(-\kappa r^{-s})$ instead. Inside $R_m$ it is
the exact zero-energy two-body solution, $f = C_1 K_0(\kappa r^{-s})$ with
$s = (p-2)/2$, $\kappa = \sqrt{2\mu C}/s$, so the pair's kinetic energy cancels
$C/r^p$ there exactly. Beyond $R_m$ it is
$C_2 \exp[-C_3/r - C_3/(2r_c - r)]$, flat ($u = 0$) from $r_c$ on. $C_1, C_2, C_3$
follow from continuity of $u$ and $u'$ at $R_m$ and $u(r_c) = 0$ (Astrakharchik
*et al.*, PRL **98**, 060405 (2007)). $C$ and $p$ are read from the Hamiltonian, $\mu$ from the
group masses; the build fails if the pair has no such core. For more
variational freedom, add a `drummond` Jastrow on the same pair: it derives no
cusp from a $C/r^p$ core and does not disturb the contact behaviour.

```yaml
# open cell: an electron–hole pair
- jastrow: { type: pade, pair: [electrons, holes], numerator: [-0.1], denominator: [0.3], is_fixed: [false, false] }
# periodic cell: electrons in a 2D gas (cutoff within the Wigner–Seitz inradius)
- jastrow: { type: drummond, pair: [electrons, electrons], params: [0.1, 0.0], cutoff: 5.0, is_fixed: [false, false] }
# periodic cell: dipoles (inverse_power, exponent 3); cutoff = Wigner–Seitz inradius
- jastrow: { type: zero_energy, pair: [dipoles, dipoles], matching_radius: 0.05, matching_radius_fixed: false }
```

A pair Jastrow depends only on separations. In an open cell, something else
must make $\vert\psi\vert^2$ decay: an attractive pair factor (as for the
exciton), a determinant of localized orbitals, or a `moire_jastrow`. Otherwise
the walkers drift apart, or, in a trap, the center of mass drifts out and the
energy grows without bound.

### `- slater:` (Slater determinant)

$\det D_\uparrow \det D_\downarrow$ over one group. Spin-block sizes come from
that group's `up`/`down`.

| Key | Type | Meaning |
|---|---|---|
| `group` | string | The fermion group |
| `orbitals.type` | `gaussian` (default) \| `plane_wave` \| `fock_darwin` | Orbital family |
| `orbitals.centers` | list of points | Gaussian centers |
| `orbitals.alphas` | list | Gaussian exponents, one per center: a shared **orbital pool** (molecule; open cells only) |
| `orbitals.alpha` | float | One shared exponent. With `centers` (one per particle) or `sites`, this is a **crystal**: one site per particle, disjoint spin blocks |
| `orbitals.sites` | `triangular` | Generate crystal sites commensurate with a hexagonal periodic cell (Wigner crystal) |
| `orbitals.alpha_fixed` | bool (default `true`) | `false` lets the optimizer move the shared width (moves the nodes) |
| `orbitals.l` | list of int | **fock_darwin:** angular momenta of the occupied orbitals |
| `orbitals.beta` | float | **fock_darwin:** Gaussian exponent, $m\sqrt{\omega_0^2 + \omega_c^2/4}$ for the exact one-body state |

`plane_wave` orbitals take their k-vectors from the periodic cell and need
closed shells; the error message lists the valid particle counts. `fock_darwin`
orbitals are complex (2D only), so DMC runs fixed-phase.

**Backflow.** A `backflow:` list evaluates the orbitals at quasi-coordinates
$\mathbf x_i = \mathbf r_i + \sum_c \sum_{j \in c} \eta_c(r_{ij})\,\mathbf r_{ij}$
instead of $\mathbf r_i$ (López Ríos et al., *Phys. Rev. E* **74**, 066701 (2006)),
which moves the nodes and so the fixed-node energy; a Jastrow factor cannot.
Each channel names a source group and has its own
$\eta(r) = (1 - r/L)^C\,\Theta(L - r)\sum_k c_k r^k$:

| Key | Default | Meaning |
|---|---|---|
| `from` | — (required) | Group whose particles displace the rows: the determinant's own group (like-spin pairs; that group must hold one spin only) or any other group |
| `params` | — (required) | Free coefficients: $[c_0, c_2, c_3, \ldots]$ for the own group, where $c_1 = C c_0/L$ is derived so that $\eta'(0) = 0$ keeps the like-spin cusp; $[c_0, c_1, c_2, \ldots]$ for any other group |
| `is_fixed` | all `false` | One flag per `params` entry |
| `truncation` | 3 | $C \ge 2$; 3 keeps the local energy continuous at $L$ |
| `cutoff` | Wigner–Seitz inradius (periodic) | $L$; required in an open cell, and never beyond the inradius |
| `cutoff_fixed` | `false` | `true` keeps $L$ out of the optimization |

```yaml
- slater:
    group: electrons
    orbitals: { type: plane_wave }
    backflow:
      - { from: electrons, params: [0.0, 0.0, 0.0] }   # c0, c2, c3; L = inradius
```

Backflow is available for `gaussian` and `plane_wave` orbitals, at Γ and at
any twist, and for the `pairing` determinant; `fock_darwin` refuses it. Every
move rebuilds and inverts the determinants, $O(N^3)$ per move instead of the
$O(N^2)$ rank-one update. PIMC refuses backflow nodes.

Away from Γ the plane-wave determinant is complex, and its backflow moves the
phase as well as the nodes (DMC runs fixed-phase). Its parameters cannot be
optimized there: their log-derivatives are complex, and the optimizer refuses
any stage that would move them. They have the same names and order as at Γ, so
the workflow is to optimize the backflow (and the Jastrow) at the Γ twist, then
run every twist with those values (the Γ run's config snapshot holds them), with
no optimizer stage or one with `parameters: jastrow`.

The linear optimizer
gets the curvature term of $\bar H$ ($\partial E_L/\partial p$) for the backflow
and orbital parameters, at about one determinant refresh per parameter at each
measurement. The quasi-coordinates move with $p$, so the orbitals need third
derivatives (`plane_wave`, `gaussian` and the pair orbital have them). With
`truncation: 2` the local energy jumps at $L$, and the jump moves with $L$; such
a determinant supplies no curvature term (the log warns), and its steps are less
reliable.

### `- pairing:` (pairing determinant)

$\prod_b \det \varphi(\mathbf r_i - \mathbf r_j)$: in each block, row $i$ is a
particle of one group and column $j$ a particle of another, and one pair
orbital $\varphi$ is shared by every block. For an electron–hole bilayer split
into four polarized groups the blocks are $(e\uparrow, h\downarrow)$ and
$(e\downarrow, h\uparrow)$. The pair orbital is

$$\varphi(\mathbf r) = \sum_s p_s \sum_{\mathbf k \in s} \cos(\mathbf k\cdot\mathbf r)
+ \Big(1 - \frac{r}{L}\Big)^3 \Theta(L - r) \sum_{m=0}^{M} c_m r^m,
\qquad c_1 = 3c_0/L,$$

where a star $s$ holds every reciprocal vector of one length (both $\pm\mathbf k$;
star 0 is $\mathbf k = 0$). The constraint on $c_1$ makes $\varphi$ cuspless;
cusps belong to the Jastrow factor.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `blocks` | list | — (required) | One determinant each: `{ rows: group, columns: group, unpaired: {...} }` |
| `blocks[].unpaired` | map | none | Single-particle orbitals that complete a block with more rows than columns (or the reverse): `{ type: plane_wave, k: [[n1, n2], ...], sin: [bool, ...] }`, with $\mathbf k = n_1\mathbf b_1 + n_2\mathbf b_2$; cos unless `sin` says otherwise. Exactly $\lvert N_\text{rows} - N_\text{columns}\rvert$ of them |
| `orbital.type` | `cosine_polynomial` | `cosine_polynomial` | Form of $\varphi$ |
| `orbital.shells` | int | `0` | Number of stars $S$ in the cosine part (periodic 2D only) |
| `orbital.poly_order` | int | none | $M \ge 1$ of the cutoff polynomial; omit for no polynomial |
| `orbital.cutoff` | float | Wigner–Seitz inradius | $L$; at most the inradius under PBC, required in an open cell |
| `orbital.cutoff_fixed` | bool | `false` | Keep $L$ fixed (optimized by default and held within the inradius, like the Drummond cutoff) |
| `orbital.params` | list | $c_0 = 1$, rest 0 | $[p_0, \dots, p_{S-1}, c_0, c_2, \dots, c_M]$; required without a polynomial |
| `orbital.is_fixed` | list of bool | all fixed | Same order as `params` |

Each block group must be one spin of fermions (`up` only, or `down` only):
the determinant antisymmetrizes all of its rows, and all of its columns. That
also gives every same-group pair of a block group the P-wave cusp in the
Jastrow factor. $\psi$ is unchanged by a common scale of all $p$ and $c$, so
at least one of them must stay fixed. The parameters set the nodes and are
optimized in the `nodal` scope.

**Backflow.** A `backflow:` list, with the keys of the `slater` channels plus
`group`, evaluates the pair and unpaired orbitals at quasi-coordinates of both
the row and the column particles, $\varphi(\mathbf x_i - \mathbf y_j)$; each
channel `{ group: g, from: s, params: [...] }` displaces group $g$ (a row or
column group of some block) by the particles of $s$. A group displacing itself
has $c_1 = C c_0/L$ derived, as in the `slater` channels; any other source
lists $c_1$. The parameters follow the orbital's.

```yaml
- pairing:
    blocks: [{ rows: e, columns: h }]
    orbital: { poly_order: 2 }
    backflow:
      - { group: e, from: h, params: [0.0, 0.0] }   # c0, c1; L = inradius
      - { group: h, from: e, params: [0.0, 0.0] }
```

Walkers start with column particle $j$ of every block next to row particle
$j$, displaced by a Gaussian of width $L/8$ per axis (the inradius replaces
$L$ without a polynomial); the other particles start as usual. A compact
$\varphi$ (a tightly bound pair, $L$ far below the cell) would otherwise vanish
for nearly every row–column pair of a uniform start, and the determinant would
start at zero.

### `- kspace_jastrow:` (long-range Jastrow, periodic cells)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `pair` | `[groupA, groupB]` | — | Groups it correlates |
| `shells` | int | — (required) | Number of reciprocal-lattice shells $\vert k\vert$, one coefficient each |
| `params` | list | RPA seed | One coefficient per shell. If absent, seeded from the Gaskell RPA form with the Hamiltonian's own $\tilde V(k)$ (same-group pairs) or zero. The seed is refused next to a `zero_energy` Jastrow on the same pair, whose $u$ diverges at contact and has no Fourier transform to subtract |
| `is_fixed`, `can_flip_sign` | list of bool | fixed / `true` | Parameter flags |

### `- moire_jastrow:` (one-body moiré factor)

$\ln\psi \mathrel{+}= \sum_i\sum_s\sum_j [a_s\cos(k_{sj}\cdot r_i) + b_s\sin(k_{sj}\cdot r_i)]$ over the first `shells` moiré shells.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `group` | string | — | Group it acts on |
| `shells` | int | `1` | Number of moiré reciprocal shells |
| `params` | list | zeros | $[a_1, b_1, a_2, b_2, \dots]$ |
| `is_fixed` | list of bool | fixed | Parameter flags |
| `a_M`, `theta0` | float | from the `moire` term | Superlattice period and orientation |

### `- nosanow:` (one-body crystal factor)

$\ln\psi \mathrel{+}= -\alpha\sum_i \vert r_i - R_i\vert^2$: particle $i$ of the
group is tied to its own site $R_i$ (Nosanow, *Phys. Rev.* **146**, 120 (1966)).
It is the localizing factor of a bosonic crystal trial, used with a pair Jastrow
in place of a determinant of Gaussians.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `group` | string | — (required) | Group it acts on |
| `sites` | `triangular` | — | Generate one site per particle, a triangular lattice commensurate with a hexagonal periodic cell (the particle count must be Loeschian, $p^2 + pq + q^2$) |
| `centers` | list of points | — | Explicit sites, one per particle (any cell). Give exactly one of `sites` and `centers` |
| `alpha` | float | — (required) | Width $\alpha > 0$ |
| `alpha_fixed` | bool | `true` | `false` lets the optimizer move $\alpha$ |

Walkers start on the sites, scattered by $1/\sqrt{4\alpha}$. In a periodic cell
$r_i - R_i$ is the minimum image, and $\alpha$ is refused when
$e^{-\alpha R_{WS}^2} > 10^{-6}$ ($R_{WS}$ the Wigner–Seitz inradius): there the
folded Gaussian's gradient flips without being negligible. The factor is not
symmetric under exchange, but it is positive, so DMC has no node to fix. A
crystal trial for bosons therefore projects onto the metastable solid, not
onto the true ground state, which is a fluid below the freezing density.

```yaml
# 2D dipolar crystal: pair factor times Nosanow factor on a triangular lattice
- jastrow: { type: zero_energy, pair: [dipoles, dipoles], matching_radius: 0.04 }
- nosanow: { group: dipoles, sites: triangular, alpha: 1500.0, alpha_fixed: false }
```

### `- center_of_mass_momentum:` (Galilean boost)

`K: [Kx, Ky]`. Multiplies $\psi$ by $e^{iK\cdot R_{cm}}$, which makes the trial
complex (DMC runs fixed-phase). At most one.

---

## `hamiltonian`

`hamiltonian.terms` is a list; each term has a `type`. The kinetic energy
$-\tfrac12\sum_i \nabla_i^2/m_i$ is always included and is not listed.

In a **periodic** cell, all pair terms are replaced by a single Ewald term. They
must then all be of one family (all `coulomb`, all `rytova_keldysh`, or all
`screened_coulomb`), and Coulomb terms must share one `epsilon`. Every pair of
groups interacts under PBC, listed or not.

`inverse_power` is the exception: its sum over periodic images converges
absolutely (exponent $p$ greater than the dimension), so it is never part of the
Ewald term and only the listed pair interacts. Under PBC each pair is taken at
its minimum image, and the images outside the Wigner–Seitz cell are replaced by
a uniform continuum, $E_\text{tail} = \tfrac12 N_a n_a \int_{\text{outside WS}} C r^{-p}\,d^2r$
(or $N_a n_b\int\ldots$ for two groups), which is included in the local energy and
printed in the log. Its error, $g(r)\neq n$ outside the cell, vanishes as
$N\to\infty$ and is removed by extrapolating in $N$.

| `type` | Keys | Meaning |
|---|---|---|
| `coulomb` | `pair`, `epsilon` (1.0) | $q_a q_b/(\varepsilon r)$ |
| `rytova_keldysh` | `pair`, `epsilon` (1.0), `r0` (1.0) | 2D screened interaction with screening length $r_0$ (see [Concepts](concepts.md#cells-and-interactions)) |
| `screened_coulomb` | `pair`, `epsilons`, `interfaces`, `source_layer` (1), `target_layer` (0), `z_offset`, `fit_tol` (0.1) | Dielectric multilayer: `epsilons` lists ε per layer from bottom to top (N values); `interfaces` lists the N−1 interface positions in Bohr; `source_layer`/`target_layer` are 1-based (0 = same as source); `z_offset` overrides the out-of-plane offset derived from the particles' `z`; `fit_tol` is the fit accuracy in % |
| `inverse_power` | `pair`, `strength`, `exponent` (all required) | $C/r^p$ with $C$ = `strength`, $p$ = `exponent`; $p = 3$ is dipoles polarized perpendicular to the plane. Refuses $C<0$ with $p\ge 2$ (no ground state), $p \le d$ under PBC, and groups in different planes |
| `harmonic_trap` | `omega` (required), `groups` (all), `center` (origin) | $\tfrac12 m\omega^2\vert r - c\vert^2$ on the listed groups |
| `moire` (or `moiré`) | `group` (list of groups), `V0`, `Vtilde`, `phi`, `a_M` (1.0), `theta0` (0) | $V_0 + 2\tilde V\sum_{i=1}^3\cos(g_i\cdot r + \phi)$, triangular superlattice of period $a_M$ |
| `magnetic_field` | `B` (required) | Uniform field along $z$ (symmetric gauge about the origin; 2D open cells). At most one. Not supported by PIMC |
| `rotation` | `omega` (required) | Rotating frame $H - \Omega L_z$; adds the centrifugal $-\tfrac12 m\Omega^2 r^2$. At most one. Not supported by PIMC |

Note the plural difference: `harmonic_trap` uses `groups`, `moire` uses `group`
(both take a list).

---

## `action` (PIMC only)

Optional. It selects the pieces of the Euclidean action. Each entry is a map with
a single key.

| Entry | Meaning |
|---|---|
| `- primitive: {}` | Kinetic springs plus $\tau V$ per slice. **Default** when `action:` is absent |
| `- kinetic: primitive` | The kinetic (spring) part alone |
| `- potential: primitive` | The $\tau V$ part alone |
| `- pair_product: { extent: <float>, grid_points: 17, form: finite_grid }` | Numerical pair-product action on a finite grid (experimental). `extent` is required; `grid_points` is odd, 5 to 33. Replaces the primitive potential |
| `- nodes: wavefunction` | Fixed-node restriction: every path keeps the sign of each determinant of the `wavefunction:` block's `slater` items on all of its slices (time-independent nodes). Requires `statistics: fixed_node` in the PIMC stage, real determinants, and no `optimizer` stage. Gaussian orbitals in a periodic cell are refused when their minimum-image fold neglects more than $10^{-6}$ of the peak |

Each of kinetic, potential and nodes may be given at most once (`primitive`
counts as both kinetic and potential).

With `nodes`, the run estimates the fixed-node ground state of the
determinant's nodal surface only as $\beta \to \infty$. What the restriction
samples, and the image action on each link, are described under
[fixed-node PIMC](concepts.md#pimc).

---

## `qmc`

A list of stages; `method` selects the kind. All `optimizer` stages run first,
then the others in document order. Each stage starts from the previous stage's
final walkers unless it has its own `walkers_in`.

### Sampling keys (all methods)

| Key | Meaning | VMC / DMC default | Optimizer default | PIMC default |
|---|---|---|---|---|
| `samples` | Walkers, split across threads (PIMC: replica paths) | — (required) | 128 | 64 |
| `equilibration_blocks` | Blocks discarded before measuring | — (required) | 50 | 100 |
| `accumulation_blocks` | Blocks averaged | — (required) | 50 | 500 |
| `steps_per_block` | Measurements per block | — (required) | 10 | 50 |
| `sub_steps` | Sweeps between measurements | — (required) | 1 | 1 |
| `time_step` | τ: proposal scale (VMC); imaginary-time step (DMC, PIMC) | — (required) | 0.01 | — (required) |
| `threads` | OpenMP threads; `0` = all cores, capped at `samples` | 0 | 0 | 0 |
| `device` | Only `cpu` exists; any other value is refused | `cpu` | `cpu` | `cpu` |

"Required" means there is no useful default: the value is 0 if omitted.

### `method: vmc`

| Key | Default | Meaning |
|---|---|---|
| `random_walk` | `metropolis` | `metropolis` (symmetric Gaussian moves) or `langevin` (adds the drift $\tau\nabla\ln\vert\psi\vert/m$). Other values are refused |
| `walkers_in` | — | Start from a saved walker file (HDF5) instead of the previous stage |
| `estimators` | none | Extra observables, see [Estimators](#estimators) |

`walkers_out` is accepted but ignored, with a warning: only DMC saves its population.

### `method: dmc`

| Key | Default | Meaning |
|---|---|---|
| `population_generations` | `10` | $g$: steps over which a population deviation is corrected, $E_T = E_\text{ref} - \ln(W/N)/(g\,\tau\,n_\text{sub})$. Must be ≥ 1 |
| `branch_cutoff` | `universal` | Cutoff on the branch weight of walkers far below $E_\text{ref}$: `universal`, `drv` or `none`, or `{scheme: universal, alpha: α}` (α default 2). See [the branch cutoff](concepts.md#dmc) |
| `reweighting` | `bare` | `bare`: $f=1$ and nominal $\tau$; `effective`: $f=1$ and acceptance-weighted $\tau_\text{eff}$; `suppressed`: damp the deviation of $E_L$ from $E_\text{ref}$. The suppressed mapping accepts `correlation_time` (`auto` or a positive imaginary time), `kappa` (default 1.0967), and `correlation_sampling: {equilibration_sweeps: 256, min_sweeps: 256, max_sweeps: 8192}`. Sampling settings require automatic $\theta$ and $2\le\text{min}\le\text{max}$. Suppression settings on the other types are refused. See [reweighting](concepts.md#dmc) |
| `walkers_in` | — | Start from a saved population |
| `walkers_out` | — | Save the final population (a bare filename goes in `checkpoints/<name>/<timestamp>/`). Also writes `.eq`, `.a25`, `.a50`, `.a75` milestone files |
| `estimators` | none | Extra observables (mixed estimator) |
| `forward_walking` | off | `{lags: [T_1, T_2, ...], interval: n}`: also measure every estimator, and the potential energy, as a pure estimate at each lag $T$ (imaginary time; strictly increasing, each a whole number of steps of `time_step`·`sub_steps`, the largest within the equilibration). A snapshot is stored every `interval` steps (default 1), which must divide `steps_per_block`. Refuses `density_matrix`. See [pure estimators](estimators.md#31-pure-dmc-estimators-forward-walking) |
| `random_walk` | `drift_diffusion` | Optional. DMC always moves walkers by importance-sampled drift–diffusion; `drift_diffusion` and its synonym `langevin` are accepted, anything else is refused |

The node constraint (none, fixed-node, fixed-phase) is chosen automatically from
the trial wavefunction. Removed keys are refused with a message: `feedback` (use
`population_generations`), and `max_age`, `branch_interval` and
`reconfiguration`, which the DMC driver never read.

DMC outputs `eq_block_energy` as well as `eq_population` and `eq_etrial` for
discarded blocks. The energy uses the same weighted pre-branch estimator as
`acc_block_energy`, and is excluded from production averages. Block counts alone
do not establish equilibrium: compare this series across discards and starts.

### `method: optimizer`

The linear method (Toulouse & Umrigar 2007) with QMCPACK-style adaptive shifts.
Only parameters with `is_fixed: false` move. The method works with real
log-derivatives $\partial \ln\psi/\partial p$, so a stage that would move a
parameter of a complex factor (the backflow of a twisted determinant) is
refused; a real factor of a complex trial, such as the Jastrow at a twist, is
optimized normally. A stage that ends on an accepted
step measures that step on one more VMC sample and undoes it when the energy
rose by more than two combined standard errors (the log warns; the results
record `last_step_checked`, `last_step_undone` and `check_energy`).

| Key | Default | Meaning |
|---|---|---|
| `type` | `linear` | The only optimizer; other values are refused |
| `parameters` | `all` | `all`, `jastrow` (non-nodal factors only) or `nodal` (determinants only) |
| `max_iterations` | 20 | Iteration limit |
| `psi_step_tol` | 1e-3 | Converged when the relative wavefunction change of the undamped step is below this, **or**… |
| `energy_tol` | 1e-6 | …when the energy has settled: the least-squares slope of the energies measured on the last `convergence_window` fresh samples falls by at most max(`energy_tol`, 2σ of the slope) per iteration |
| `convergence_window` | 15 | Fresh samples (accepted iterations) the energy trend is fitted to, at least 3. A stage cannot settle in fewer iterations; a slow descent of a few 1e-4 Ha per iteration needs about 15 to be told from noise |
| `num_shifts` | 3 | Candidate shifts tried per iteration |
| `max_param_change` | 0.3 | Largest relative parameter change accepted |
| `cost_increase_tol` | 0.0 | Tolerated energy increase for a step |
| `target_shift_i` | -1 | Target diagonal shift (negative = disabled) |
| `max_failed_attempts` | 5 | Retries per iteration before giving up |
| `shift_i` | 0.1 | Starting diagonal shift of $\bar H$ |
| `shift_s` | 0.0 | Starting overlap shift |
| `rescale` | 1.0 | Step-size factor λ |
| `xi` | 0.5 | Nonlinear normalization ξ (Toulouse–Umrigar) |
| `random_walk` | `metropolis` | Sampler for each iteration's VMC: `metropolis` or `langevin`, as for `method: vmc` |

### `method: pimc`

| Key | Default | Meaning |
|---|---|---|
| `beta` | — (required) | Inverse temperature $\beta$; $M = \beta/\tau$ slices must be an integer |
| `energy` | `thermodynamic` | Energy estimator: `thermodynamic`, `centroid_virial` (open, confined systems) or `periodic_virial` (periodic cells, distinguishable particles only: refused when exchange is sampled). The virial estimators need potentials that provide gradients |
| `compare_thermodynamic` | `false` | With a virial estimator, also report the paired difference from the thermodynamic one |
| `statistics` | particles' own | `boltzmannon` treats every particle as distinguishable. `fixed_node` samples the identity permutation inside one nodal pocket of the `nodes` action entry: each fermion class of two or more particles must be exactly one determinant block, boson classes and permutation or worm moves are refused, and so are the virial energy estimators. Without `statistics`, an exchange class of more than one particle needs a `permutation` move or the run is refused |
| `sign_threshold` | `5.0` | Fermions: flag results when $\vert\langle s\rangle\vert <$ threshold × its error |
| `moves` | see below | List of moves |
| `estimators` | none | Extra observables |
| `paths_in`, `paths_out` | — | Load or save the full path configuration (HDF5) |

Not supported with PIMC: `magnetic_field`, `rotation`, a non-zero `twist`, and
`flavours` with any VMC/DMC/optimizer stage or wavefunction item.

**Moves.** Each entry is a single-key map. Without `moves:`, the default is a
bisection as deep as the ring allows (at most 3 levels) plus a centroid move with
step 0.5.

| Move | Keys (defaults) |
|---|---|
| `bisection` | `levels` (3): regrow $2^\text{levels}$ slices; `weight` (1) |
| `centroid` | `step` (0.5); `weight` (1) |
| `single_slice` | `step` (0.5); `weight` (1) |
| `permutation` | `levels`, `max_cycle` (both required); `weight` (1) |
| `joint_permutation` | `pair: [A, B]`, `levels`, `max_cycle`, `pair_radius` (all required); `weight` (1) |
| `worm` | `group`, `max_length`, `swap_length` (required), `C` (tuned during equilibration if absent); `weight` (1) |

`weight` sets the relative frequency of a move.

```yaml
moves:
  - bisection: { levels: 2, weight: 1.0 }
  - centroid:  { step: 0.5, weight: 1.0 }
```

### Estimators

Optional observables, listed under a stage's `estimators:`. The energy is always
measured. Each `type` accepts only its own keys (plus `name`); any other key, or
an unknown `type`, is an error when the config is read. Formulas, normalizations
and error bars are in [Estimators](estimators.md).

| `type` | Keys | Stages | Output |
|---|---|---|---|
| `gofr` | `rmax`, `n_bins` | VMC, DMC, PIMC | Pair-correlation $g_{ab}(r)$ for every group pair |
| `density` | `rmax`, `n_bins` (per axis), `per_particle` (false) | VMC, DMC, PIMC | One-body density on a grid $[-r_\text{max}, r_\text{max}]^d$, per group or per particle |
| `collective_density` | `name` (required), `terms: [{group, coeff}]`, `rmax`, `n_bins` | VMC, DMC, PIMC; open cells only | Histogram of $\sum_g c_g \sum_{p\in g} r_p$ (e.g. a center of mass or a relative coordinate) |
| `sofk` | `kmax` | VMC, DMC, PIMC; periodic cells | Static structure factor $S_{ab}(k)$ for $0 < \vert k\vert \le k_\text{max}$ |
| `electron_hole_proximity` | `electron_group`, `hole_group`, `r_cut` | VMC, DMC, PIMC | $U_e$: fraction of electrons with no hole of `hole_group` closer than `r_cut` (and $C_e = 1 - U_e$). A geometric diagnostic, not a bound-exciton fraction |
| `momentum_distribution` | `group`, `rmax`, `n_bins`, `kmax`, `samples`, `n_k` | PIMC with a `worm` move | One-body density matrix and $n(k)$, sampled on the worm's open paths (so PIMC only) |
| `bragg` | `group`, `wavevectors: [[kx, ky], ...]` (both required) | VMC, DMC, PIMC; periodic 2D cells | Crystal order of `group` on the listed wavevectors, $B = \frac{1}{n_G}\sum_G \langle\vert\sum_j e^{iG\cdot r_j}\vert^2\rangle/N^2$ ($B = 1$ on a perfect lattice), and each direction's intensity. Every $G$ must be a nonzero reciprocal vector of the cell; give one member of each $\pm G$ pair |
| `bond_order` | `group`, `reference_spacing` (both required) | VMC, DMC, PIMC; periodic 2D cells, at least 7 particles in `group` | Sixfold order from each particle's six nearest neighbours: global $\langle\vert\overline{\psi_6}\vert^2\rangle$, local $\langle\overline{\vert\psi_6\vert^2}\rangle$, and the bond-length distortion $\langle(r_{ij}/a - 1)^2\rangle$ with $a$ = `reference_spacing` |
| `density_matrix` | `rmax`, `n_bins` (both required), `groups: [g, ...]`, `pairs: [[a, b], ...]` (at least one of the two), `every` (1) | VMC, DMC | From ratios of $\psi_T$ at a displacement $r'$ of every configuration, $\vert r'\vert$ on $[0, r_\text{max})$: the one-body density matrix `density_matrix_rho1_<g>` $= \langle\psi(\ldots, r_i + r', \ldots)/\psi\rangle$ ($\rho^{(1)}/n_g$, 1 at $r = 0$) of each listed group and of the groups of `pairs`; for each pair of two different groups the two-body density matrix `density_matrix_rho2_<a>_<b>` (one particle of each moved by the same $r'$) and the pair condensate fraction `density_matrix_pair_condensate_fraction_<a>_<b>` $= \frac{N_a N_b}{\min(N_a, N_b)}(\text{rho2} - \text{rho1}_a\,\text{rho1}_b)$, whose large-$r$ plateau is the fraction of pairs in the condensate (López Ríos et al., PRL 120, 177701 (2018), Eqs. S6–S8). Measures every `every`-th step: one measurement costs a few sweeps. `pairs` needs a trial whose factors all supply pair ratios (pairing and Slater determinants, Drummond Jastrow) |

What a stage measures depends on its sampler: VMC gives
$\langle\psi_T|O|\psi_T\rangle$, PIMC the thermal average at $1/\beta$, and DMC the
mixed estimator $\langle\psi_T|O|\Phi\rangle/\langle\psi_T|\Phi\rangle$. The pure value is
approximately $2\langle O\rangle_\text{DMC} - \langle O\rangle_\text{VMC}$ (see
`tools/plot_density.py`), or measured directly by a DMC stage with
`forward_walking`. VMC and DMC error bars of these observables assume
independent blocks, so short blocks underestimate them.

**Several instances of one type.** A stage may list a type more than once, e.g. a
fine and a coarse $g(r)$. Every result key of an instance starts with its
namespace (`gofr`, `density`, `sofk`/`rhok`, `cdensity_<name>`,
`electron_hole_proximity_<e>_<h>`, `obdm_<group>`/`nk_<group>`, `bragg_<group>`,
`bond_<group>`, `density_matrix`):

- `name: <label>` (any type) inserts the label after the namespace:
  `{type: gofr, name: fine, ...}` writes `gofr_fine_r`, `gofr_fine_<a>_<b>`, ...
  The label must be nonempty and contain no `/`.
- An unnamed instance keeps the plain keys, unless an earlier instance of the
  stage already writes them; then it is numbered, `gofr_2_r`, `gofr_3_r`, ...
  (the run log says so).
- Two instances given the **same** `name` are refused.

`collective_density`'s `name` is already its namespace (`cdensity_<name>`), so it
is not inserted a second time.

```yaml
estimators:
  - {type: gofr, rmax: 0.5, n_bins: 25}               # gofr_r, gofr_e_h, ...
  - {type: gofr, name: coarse, rmax: 5.0, n_bins: 50}  # gofr_coarse_r, ...
  - {type: gofr, rmax: 2.0, n_bins: 10}               # gofr_2_r, ...
```

---

## Complete example: a 2D biexciton in MoS₂

[`configs/tests/biexciton_MoS2.yaml`](../configs/tests/biexciton_MoS2.yaml) is a
full DMC input: two electrons and two holes (one of each spin) interacting through
Rytova–Keldysh, with a Padé Jastrow for each of the e–h, e–e and h–h pairs.
