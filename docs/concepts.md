<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Concepts

This page defines the vocabulary that appears in configs, logs and source code.
It assumes you know quantum mechanics, but not necessarily QMC. Each entry gives
the YAML key it corresponds to, where there is one.

For textbook background: Foulkes, Mitas, Needs & Rajagopal, *Rev. Mod. Phys.* **73**,
33 (2001) for VMC and DMC; Ceperley, *Rev. Mod. Phys.* **67**, 279 (1995) for PIMC.

- [System](#system)
- [Trial wavefunction](#trial-wavefunction)
- [Sampling: walkers, blocks and errors](#sampling-walkers-blocks-and-errors)
- [VMC](#vmc)
- [Optimizer](#optimizer)
- [DMC](#dmc)
- [PIMC](#pimc)
- [Cells and interactions](#cells-and-interactions)

## System

**Units.** Hartree atomic units, $\hbar = e = m_e = 4\pi\varepsilon_0 = 1$.
Energies are in Hartree and lengths in Bohr. Effective masses are given per
particle (`mass`) and dielectric screening per interaction (`epsilon`).

**Particle entry / group.** Each item under `system.particles` declares one kind
of particle with a `name`, a `group`, a `mass`, a `charge`, and how many there
are. Everything else in the config refers to particles by **group** name, never by
`name`: a Jastrow `pair`, a potential `pair`, a determinant's `group`. Internally a
group is a contiguous range of particle indices `[first, last)`, in the order the
entries are written.

**Spin / flavour counts.** `up` and `down` give how many particles of the entry
have each spin. Within a group, spin-up particles come first. The Hamiltonians are
spin-independent, so these counts are conserved and the spin polarization is an
input. `flavours: [n0, n1, ...]` generalizes this to more flavours (spin ×
valley), but it is PIMC-only for now.

**Particle type.** `fermions`, `bosons` or `boltzmannon` (distinguishable). VMC
and DMC get antisymmetry from a Slater determinant, so for them `type` is just a
label. PIMC uses it to decide which particles can exchange.

**Mass anisotropy.** `mass: [mx, my]` gives a direction-dependent mass.

**Layer coordinate `z`.** In a 2D run, particles move in the plane. `z` places a
particle entry in a parallel plane at fixed height, and a pair across layers then
interacts at distance $\sqrt{r^2 + (z_a - z_b)^2}$. This is how indirect excitons
in bilayers are described.

## Trial wavefunction

**Trial wavefunction $\psi_T$.** An approximate ground state, written as a product
of factors ("components"): $\psi_T = \prod_c \psi_c$. The `wavefunction:` list in
the YAML holds one item per factor, in order. VMC samples it, DMC uses it for
importance sampling and to fix the nodes, and PIMC does not use it at all.

**Jastrow factor.** A positive factor $e^{U}$ that builds in correlation between
particles. Pair Jastrows have $U = \sum_{i<j} u(r_{ij})$:
- **Padé**: $u(r) = (a_1 r + a_2 r^2 + \dots)/(1 + b_1 r + \dots)$. Simple and
  good for open systems. It tends to a constant, or grows, at large $r$, so check
  that $\psi_T$ is normalizable.
- **Drummond**: $u(r) = (r - L)^C\,\Theta(L - r)\,[\alpha_0 + \alpha_1 r + \sum_{l\ge2}\alpha_l r^l]$
  (Drummond, Towler & Needs, *PRB* **70**, 235119 (2004)). It vanishes smoothly
  at the cutoff $L$, so use it in periodic cells, where $u$ must go to zero
  within the cell.
- **Zero-energy** (`zero_energy`): for a core $V \approx C/r^p$ with $p > 2$
  (2D dipoles, $p = 3$), the exact two-body solution $C_1 K_0(\kappa r^{-s})$
  at short range, joined to a tail that reaches zero at a cutoff. Padé and
  Drummond cannot cancel such a core (see the cusp condition below).

Other Jastrows: `kspace_jastrow` is a long-range term on reciprocal-lattice shells
for periodic cells, `moire_jastrow` is a one-body factor on a moiré
superlattice, and `nosanow` is a one-body factor $e^{-\alpha\sum_i|r_i - R_i|^2}$
that ties each particle to its own lattice site: the crystal trial for bosons,
which have no determinant to localize them.

**Cusp condition.** Where two charged particles meet, $V \propto 1/r$ diverges.
The kinetic energy of $\psi_T$ cancels the divergence only if $u'(0) = \Gamma$,
with $\Gamma = 2\mu R/(2\ell + d - 1)$ for reduced mass $\mu$, residue
$R = \lim_{r\to0} r\,V(r)$ and dimension $d$. Normally $\ell = 0$. It is
$\ell = 1$ for a Jastrow within one fully polarized group that has a Slater
determinant, because antisymmetry already makes $\psi$ vanish at contact. A
group with both spins has both kinds of pairs, and one $u(r)$ cannot satisfy
both cusps; Alea then uses $\ell = 0$ and logs a warning. Alea derives $\Gamma$ from the
Hamiltonian and pins it. **You never write the cusp in YAML**, and the parser
refuses a `cusp:` key. Potentials without a $1/r$ pole, such as Rytova–Keldysh,
get $\Gamma = 0$.

A core stronger than $1/r^2$ has no cusp that a finite $u(0)$ could satisfy:
the kinetic energy cancels $C/r^p$ only if $\psi$ vanishes as
$\exp(-\kappa r^{-s})$, $s = (p-2)/2$, $\kappa = \sqrt{2\mu C}/s$. The
`zero_energy` Jastrow builds this in, reading $C$ and $p$ from the Hamiltonian
just as the other forms read $R$.

**Slater determinant.** The antisymmetric factor for fermions,
$\det D_\uparrow \det D_\downarrow$ over one group's spin-up and spin-down
particles. Orbitals can be Gaussians, plane waves (periodic fluid), Gaussians on
crystal sites (Wigner crystal), or Fock–Darwin states (quantum dot). A
determinant changes sign, so it has **nodes**.

**Complex trial.** Twisted boundary conditions (with plane-wave orbitals) or a
center-of-mass boost $e^{iK\cdot R_{cm}}$ make $\psi_T$ complex. DMC then runs
fixed-phase instead of fixed-node.

**Parameters.** Every Jastrow coefficient is a variational parameter. It is
**fixed by default**. Set `is_fixed: [false, ...]` to let the optimizer move it.
`can_flip_sign: [false]` makes the optimizer keep a parameter's sign.

**Local energy.** $E_L(R) = H\psi_T(R)/\psi_T(R)$. Its average over $\vert\psi_T\vert^2$
is the VMC energy. For an exact eigenstate, $E_L$ is constant and its variance is
zero (the *zero-variance principle*), so a small `var` means a good trial.

## Sampling: walkers, blocks and errors

**Configuration $R$.** The positions of all particles, $(r_1, \dots, r_N)$.

**Walker.** One configuration that moves through configuration space. Key:
`samples` (the total number of walkers, split across threads). In PIMC, `samples`
is instead the number of independent replica paths.

**Move, sub-step, step, block.** These form a hierarchy:

| Level | Key | Meaning |
|---|---|---|
| move | none | Propose a new position for one particle and accept or reject it |
| sub-step | `sub_steps` | One sweep that moves every particle once. Sub-steps between measurements decorrelate samples |
| step | `steps_per_block` | `sub_steps` sweeps followed by one measurement |
| block | `equilibration_blocks`, `accumulation_blocks` | The average of `steps_per_block` measurements. Equilibration blocks are discarded |

**Time step τ** (`time_step`). In VMC, it sets the size of a proposed move
(standard deviation $\sqrt{\tau/m}$). In DMC and PIMC, it is the
imaginary-time step, so results carry a time-step bias that vanishes as
$\tau \to 0$.

**Acceptance ratio.** The fraction of proposed moves accepted, reported as `acc`.

**Error bar and autocorrelation.** Consecutive blocks are correlated when a
block is shorter than the autocorrelation time, and the plain standard error of
the block averages is then too small. Every driver therefore reports the
energy error from Flyvbjerg–Petersen reblocking: blocks are averaged in pairs
until the standard error stops growing, and that plateau is the error. VMC
reblocks each thread's chain separately and pools them; DMC reblocks its single
population chain. The results file also holds the plain error
(`energy_error_naive`), the integrated autocorrelation time `energy_tau_int`
(in blocks) and `energy_error_reliable`. The effective number of blocks,
$n/(1 + 2\tau_\text{int})$, counts the independent pieces of information in
one run. A VMC or DMC run whose error is not reliable (no plateau, or fewer
than 50 effective blocks; PIMC asks for 16) prints a warning: run more
accumulation blocks. The threshold is high because reblocking only sees
correlations much shorter than the run and underestimates the autocorrelation
time of a short run. In the WS2 exciton DMC test (autocorrelation time about 20
blocks), 512-block runs estimated 19–49 effective blocks and gave errors about
30 % smaller than the scatter of independent runs; 4096-block runs estimated
57–151 and gave the right error. A run is long enough when its error bar is
below the precision you need and it prints no warning. The absence of a
warning is still not a guarantee for a system with slower correlations; for a
number that matters, compare with independent seeds. `tools/reblock.py` shows
the full reblocking table. The optional estimators of VMC and DMC (g(r),
density, ...) still assume independent blocks; PIMC reblocks them too.

**Seed.** `project.seed` sets every random stream. The same seed with the same
thread count gives an identical run. Each stage and each thread hashes its own
stream from the seed, so seeds 42 and 43 give independent runs. `seed: 0` or no
seed means "pick one". The chosen value is recorded in the config snapshot so
the run can be replayed.

**Stage.** One entry of the `qmc:` list: an optimizer, VMC, DMC or PIMC run.
Stages share one trial wavefunction and pass their final walkers to the next
stage (*carry-over*).

## VMC

Variational Monte Carlo samples $\vert\psi_T\vert^2$ with the Metropolis
algorithm and averages $E_L$. The result is an upper bound on the ground-state
energy (for the right symmetry) and is exact only if $\psi_T$ is.

**`random_walk`.** `metropolis` proposes symmetric Gaussian moves. `langevin`
adds a drift $\tau\nabla\ln\vert\psi_T\vert/m$ towards larger $\vert\psi_T\vert$
(Metropolis-adjusted Langevin). Optimizer stages take the same two values for
their sampling. DMC stages always use drift–diffusion, so they accept only
`drift_diffusion` (or its synonym `langevin`).

## Optimizer

The **linear method** (Toulouse & Umrigar, *J. Chem. Phys.* **126**, 084102
(2007)) changes the free parameters to lower the energy. Each iteration samples
with VMC, builds the Hamiltonian and overlap matrices in the space of parameter
derivatives, and solves a generalized eigenproblem for the update. The
eigenproblem is solved in parameters scaled to unit overlap diagonal, an exact
change of variables that keeps the solver accurate when the derivatives span
many decades (a Drummond or pair-orbital polynomial of order 8 spans about
seven).

**Shift.** A diagonal shift added to that eigenproblem to keep steps small. Alea
tries several shifts per iteration (the QMCPACK three-shift scheme) and keeps
the best one, judged by correlated sampling on the same configurations.

**Convergence.** A stage stops when the undamped step barely changes ψ, or
when the energy has settled. To decide that, a least-squares line is fitted
to the energies of the last `convergence_window` iterations, each measured
on a new sample, and the stage stops once that line no longer falls
significantly. The step's own correlated-sampling gain is not used: the
step is fitted to the same configurations that score it, so on a plateau it
keeps predicting a gain that the next sample does not show.

**Last step.** Each accepted step is measured by the next iteration's new
sample, except the last step of a stage. A stage that ends on an accepted
step therefore takes one more sample. If the energy rose by more than two
combined standard errors from the sample the step started from, the step is
undone. Correlated sampling only sees configurations the old ψ visited, so
on rare occasions it accepts a step that the next sample shows to be clearly
worse.

**`parameters`.** `all`, `jastrow` (non-nodal factors) or `nodal` (determinants
only). Consecutive optimizer stages with different scopes alternate between
the two (`tools/moire_cells.py` writes a `nodal` stage followed by a `jastrow`
stage).

## DMC

Diffusion Monte Carlo applies $e^{-\tau(H - E_T)}$ repeatedly, which projects the
trial state onto the ground state. Each step is a drift–diffusion move followed
by a re-weighting by $e^{-\tau(E_L - E_T)}$. Walkers with large weights are
copied and walkers with small weights are removed (**branching**).

**Trial energy $E_T$ and population control.** $E_T$ is adjusted so that the
population stays near `samples`:
$E_T = E_\text{ref} - \ln(W/N)/(g\,\tau\,n_\text{sub})$, with total weight $W$
and target population $N$. The key is `population_generations` ($g$, default 10):
deviations are corrected over about $g$ steps. A finite population biases the
energy by an amount proportional to $1/(gN)$.

**Branch cutoff.** Near a node, or a cusp the trial misses, $E_L \to -\infty$, and
one walker's weight could blow up the population. So the weight of a walker
more than $e_\text{cut}$ below $E_\text{ref}$ is tapered, and switched off beyond
$1.5\,e_\text{cut}$. Any finite cutoff biases the energy, unless it lies far out
in the tail of the $E_L$ distribution. What matters is $e_\text{cut}/\sigma$,
where $\sigma$ is the spread of $E_L$. The key is `branch_cutoff`:

- `universal` (default): $e_\text{cut} = \alpha\,\sigma/\sqrt{\sigma_1\tau}$, with
  $\sigma_1 = \sigma/\sqrt{N}$ the spread per particle and $\alpha = 2$. It holds
  regardless of the unit and of $N$ (for independent copies of a system
  $\sigma_1$ is unchanged), and it recedes as $1/\sqrt\tau$, so the bias vanishes
  as $\tau \to 0$. $\sigma$ is a robust spread ($1.4826\times$ the median
  absolute deviation) of the walkers' $E_L$. It is refreshed during
  equilibration and frozen for accumulation; $N$ counts the particles that move.
  The system enters through $\sigma$, $N$ and $\tau$, so $\alpha$ is a
  convention, the same for every run:
  - It does not change the $\tau \to 0$ limit, only the finite-$\tau$ bias.
  - One rule for both phases keeps a comparison (fluid against crystal) free of
    two different cutoff biases.
  - $\alpha = 2$ comes from a calibration on eight reference cases. Trials without
    nodes and with their cusps are never cut ($\alpha \ge 1$ already
    suffices; $\alpha = 0.5$ cuts ordinary walkers). Trials whose $E_L$
    diverges (nodes, a missed cusp) stay stable, with a bias close to `drv`'s
    in hartree-scale systems.
  - Extrapolate $\tau \to 0$ either way. A study can still set its own $\alpha$
    (`{scheme: universal, alpha: …}`).
- `drv`: $e_\text{cut} = 2/\sqrt\tau$ (DePasquale, Rothstein & Vrbik,
  *JCP* **89**, 3629 (1988)), the earlier default. It assumes energies in
  hartree. In other units, or at large $N$ (Zen *et al.*, *PRB* **93**,
  241118(R) (2016)), it cuts ordinary walkers. For example, for 2-D dipoles in
  $\hbar^2/(Mr_0^2)$ units it gave $e_\text{cut} \approx 0.1\sigma$, and the
  energy came out 1% high and dependent on the trial.
- `none`: no cutoff. Use it for diagnosis only.

Each run records `branch_cutoff_sigma`, `branch_cutoff_energy` and the fraction
of walker sweeps the cutoff tapered (`branch_cutoff_fraction`). It warns when
that fraction exceeds $10^{-3}$, or when the mean population falls below 80% of
the target. Either one means the cutoff, or a divergence of $E_L$, is shaping
the projection.

**Reweighting.** The weight of one sweep is
$\exp\{\tau_\text{eff}[S(R') + S(R)]/2\}$ with
$S = E_T - E_\text{ref} + (E_\text{ref} - E_L)\,f$. The key is `reweighting`:

- `bare` (default): $f = 1$ and $\tau_\text{eff} = \tau$, the weight
  $e^{-\tau(E_L - E_T)}$ above and the paper's "naive" weight. Near a node
  $E_L \to \pm 1/d$, so a walker
  that lands there gains or loses weight out of proportion to the time it
  spends there. The time-step error is then large, negative and curved, which
  makes the $\tau \to 0$ extrapolation fragile.
- `effective`: $f=1$ with the acceptance-weighted effective time step of
  Eq. 10 below. The weight then counts only the diffusion that the accepted
  moves actually made. Comparing `bare` directly with `suppressed` changes
  both the time step in the weight and its suppression; this intermediate
  weight separates the two.
- `suppressed`: the reweighting of Anderson, Per & Umrigar, *J. Chem. Phys.*
  **160**, 104110 (2024), Eqs. 10 and 14. Here
  $f = (\sqrt\pi/2)\,\mathrm{erf}(x)/x$ and
  $x = \kappa\,\tau_\text{eff}\sqrt{u/(N\tau\theta)}$, where:
  - $u = \sum_{i,d} (\tau/m_{i,d})(\partial_{i,d}\ln|\Psi|)^2$ is the drift
    step over the diffusion step, a pure number. It grows as $1/d^2$ near a
    node, so $f \propto d$ cancels the divergence of $E_L$.
  - $\tau_\text{eff}$ is $\tau$ weighted by each particle's acceptance
    probability (their Eq. 10).
  - $\theta = \int_0^\infty \rho(t)\,dt$ is the integrated autocorrelation
    time of $E_L$ under the drift–diffusion move without weights, a property
    of the trial and not of $\tau$. With `correlation_time: auto` the stage
    measures it first, from at most 256 walkers moved with a step
    $\tau_\theta \le \tau$, as
    $\theta = \tau_\theta\,(\tfrac12 + \sum_{k\ge1}\rho(k))$ (the trapezoid
    rule; the sum alone is short by $\tau_\theta/2$). It discards 256 sweeps,
    then checks Sokal's window after 256, 512, ... sampling sweeps, up to
    8192, and the chain must span ten windows. If $\theta/\tau_\theta < 10$,
    the step is too long to resolve the dynamics, and the measurement is
    repeated at $\tau_\theta = \min(\tau_\theta/2, \theta/10)$, at most three
    times. The paper likewise counts $T_\text{corr}$ at one small step,
    $\tau = 0.01$, for every $\tau$. The three sweep budgets can be set with
    `correlation_sampling` (see the configuration reference).
    The unweighted trial must have a stationary relative distribution and
    a finite local-energy variance for this measurement to be meaningful.
  - $\kappa = 1.0967$ is their fitted constant ($k = 15.51$ for the carbon
    atom) rewritten for the measured $\theta$. Their $c\,V\tau_\text{eff}/\sqrt N$
    with $c = k/\sqrt{T_\text{corr} - 1}$ is tied to hartree units; this form
    holds in any unit, with any masses, and for independent copies of a system.

  $f \to 1$ as $\tau \to 0$, so the extrapolated energy is the same for every
  $\kappa$ and $\theta$; they only shape the finite-$\tau$ error. Each run
  records `correlation_time`, `correlation_step` ($\tau_\theta$),
  `correlation_tau_int` ($\sum_{k\ge1}\rho(k)$ in sweeps of $\tau_\theta$),
  `correlation_sweeps`, `suppression_kappa`, and the accumulation means of
  $\tau_\text{eff}/\tau$ (`tau_eff_ratio`) and of $f$ (`suppression_mean`).
  `correlation_window` records the lag window; `correlation_converged`
  requires the window, the ten-window length and $\theta/\tau_\theta \ge 10$;
  `correlation_window_found` alone is insufficient. These are zero when
  $\theta$ was supplied, because no measurement was made. `effective`
  records `tau_eff_ratio` and `suppression_mean = 1` without measuring
  $\theta$.

  The $V/\sqrt N$ scaling of Eq. 14 gives approximate size consistency;
  exact factorization for independent fragments requires the separate
  construction in Sec. IV of that paper.

  Where each weight helps. The three weights agree at small $\tau$ and in
  their $\tau \to 0$ limits; they differ in the finite-$\tau$ error:
  - near nodes (fixed-node atoms), `effective` and `suppressed` flatten
    $E(\tau)$ and keep the population where `bare` loses it at large $\tau$;
  - for nodeless trials at large $\tau$, the $\tau_\text{eff}$ weights cut
    the time-step error severalfold;
  - for a $1/r^3$ core (dipoles), the $\tau_\text{eff}$ weights make the
    positive time-step error larger: keep `bare` there;
  - with a missed cusp the three agree.

  $\theta$ needs a bound relative coordinate to converge: a trial whose
  Jastrow leaves the pair unbound in open space never converges it (the
  run warns), and its `suppressed` weight is then close to `effective`.

**Fixed-node / fixed-phase.** For fermions, a real $\psi_T$ has nodes, and DMC
rejects any move that crosses one. The energy is then an upper bound that is exact
only if the nodes are. For a complex $\psi_T$, the phase is imposed instead
(fixed-phase, Ortiz, Ceperley & Martin, *PRL* **71**, 2777 (1993)). Alea chooses
from $\psi_T$ (no nodes → none, real with nodes → fixed-node, complex →
fixed-phase) and writes the choice as `node_constraint` in the results.

**Mixed estimator.** DMC averages $E_L$ over $\psi_T\Phi_0$. For the energy this
is exact, apart from the nodes. For other observables, such as the density or
$g(r)$, it is biased towards the trial. The usual correction is the
extrapolated estimator, $2\langle O\rangle_\text{DMC} - \langle O\rangle_\text{VMC}$
(`tools/plot_density.py` does this). See [Estimators](estimators.md) for what
each observable measures under each sampler.

**Walker checkpoints.** `walkers_out` saves the final DMC population, and
`walkers_in` starts a later run from it.

## PIMC

Path-integral Monte Carlo samples the thermal density matrix at inverse
temperature $\beta$ (`beta`). It does not use a trial wavefunction.

**Time slices / beads.** The imaginary time $\beta$ is cut into
$M = \beta/\tau$ slices. Each particle becomes a closed **path** (ring polymer)
of $M$ beads joined by springs. $M$ must be an integer.

**Replica.** One independent copy of the whole path configuration, run on its own
thread. `samples` is the number of replicas.

**Action.** The weight of a path is $e^{-S}$. The **primitive** action is the
free-particle springs plus $\tau V$ on each slice, with an error of order
$\tau^2$ per slice. The **pair-product** action uses the exact two-body propagator
on a grid instead, for a smaller time-step error (experimental). The top-level
`action:` list selects them.

**Moves.** These are the ways a path changes:

| Move | What it changes |
|---|---|
| `bisection` | Regrows a segment of $2^\text{levels}$ slices of one path from the free-particle propagator |
| `centroid` | Rigidly translates a whole path |
| `single_slice` | Moves one bead |
| `permutation` | Reconnects paths of identical particles into exchange cycles |
| `joint_permutation` | Exchanges two groups together (e.g. electron and hole of two excitons), biased to keep pairs close |
| `worm` | Opens one path to sample off-diagonal quantities: the one-body density matrix and the momentum distribution $n(k)$ |

**Statistics and exchange classes.** Particles of the same group and flavour can
exchange. Such a set is an *exchange class*. A class larger than one particle
needs `statistics: boltzmannon` (treat them as distinguishable) or a
`permutation` move, otherwise the run refuses to start. For fermions, odd
permutations count negatively, so averages become $\langle Os\rangle/\langle s\rangle$
with sign $s = \pm1$. A small average sign is the **sign problem**: the run then
warns when $\vert\langle s\rangle\vert$ is not resolved (`sign_threshold`).

**Fixed-node PIMC** (`statistics: fixed_node` with the `nodes` action entry).
The restricted path integral (Ceperley, *J. Stat. Phys.* **63**, 1237 (1991);
*Rev. Mod. Phys.* **67**, 279 (1995), Sec. VI) with time-independent nodes:
every path keeps the sign of each determinant block of the `wavefunction:`
Slater items on all of its slices. A path then stays in one nodal pocket, and
only the identity permutation is sampled.
- As $\beta \to \infty$ the energy tends to the fixed-node energy of those
  determinants, the quantity fixed-node DMC computes with the same nodal
  surface. At finite $\beta$ the restricted ensemble is **not** fermion
  thermodynamics: it admits states of the wrong exchange symmetry.
- Checking the sign at the beads alone would keep a link that crosses a node
  and comes back, and bias the energy by $O(\sqrt\tau)$. Each link therefore
  also carries the image action $-\ln(1 - e^{-2dd'/\tau})$, exact for a planar
  node, with $d$ and $d'$ the distances of its two beads to the node in
  mass-scaled coordinates.
- The distance is estimated as $d = |f|/|\nabla f|$, with $f$ the determinant
  after each row is divided by its largest orbital. That keeps the nodes, but
  with localized orbitals it stops each particle's own orbital slope from
  looking like a nearby node, which placed every bead too close to one.
- The virial energy estimators are refused: the nodal surface carries a
  singular force they do not include.

**Energy estimators.** `thermodynamic` is the default; it needs nothing extra but
is noisy at small τ. `centroid_virial` (open, confined systems) and
`periodic_virial` (periodic cells, distinguishable particles only for now) have
lower variance but require potentials that provide their gradient. Both report
$E$, kinetic $K$ and potential $V$. Formulas and the potentials each accepts:
[Estimators](estimators.md#5-pimc-energy-estimators).

**Winding number `W2`.** The mean squared winding of paths around a periodic
cell. It is related to the superfluid fraction.

**Finite-τ bias.** PIMC error bars are statistical only. Converge or
extrapolate in τ with runs at several time steps.

## Cells and interactions

**Open cell** (`type: open`). There are no boundaries. `L` only sets the size of
the box in which walkers start.

**Periodic cell** (`type: periodic`). A Bravais cell repeated in every direction,
used to approach the thermodynamic limit. Give it by `shape` + `L`, by
`vectors`, or by the density parameter `rs` (2D: area $= N\pi r_s^2$).

**Ewald sum.** In a periodic cell, $1/r$ interactions summed over images
converge only conditionally. Alea replaces all pair terms of a periodic run with
one Ewald term, a short-range real-space part plus a long-range reciprocal-space
part, at relative accuracy `ewald_tol`. `truncation: minimum_image` disables it
for geometry tests only; the result is not a converged interaction.

**Twist** (`twist`). Twisted boundary conditions
$\psi(r + a_i) = e^{2\pi i t_i}\psi(r)$. Averaging over twists reduces
finite-size error in the kinetic energy. `tools/twist_grid.py` generates the grid.

**Coulomb.** $V = q_a q_b/(\varepsilon r)$.

**Rytova–Keldysh.** The interaction in a 2D layer with screening length $r_0$:
$V(r) = \dfrac{\pi q_a q_b}{2\varepsilon r_0}\left[H_0(r/r_0) - Y_0(r/r_0)\right]$,
with $H_0$ a Struve and $Y_0$ a Bessel function. It is logarithmic at short range
and tends to $q_a q_b/(\varepsilon r)$ for $r \gg r_0$.

**Screened Coulomb.** The interaction between charges in a stack of dielectric
layers (per-layer ε and interface positions), computed by the
[fast-screened-coulomb](https://github.com/ItaloAlb/fast-screened-coulomb)
library (Cavalcante *et al.*, *PRB* **97**, 125427 (2018)). Rytova–Keldysh is
its single-layer limit.

**Moiré potential.** A one-body potential
$V(r) = V_0 + 2\tilde V\sum_{i=1}^{3}\cos(g_i\cdot r + \phi)$ on a triangular
superlattice of period $a_M$. In a periodic run, the cell must be commensurate
with the superlattice; the error message prints the vectors to use.
