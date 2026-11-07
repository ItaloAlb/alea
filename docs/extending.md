<!--
SPDX-FileCopyrightText: 2026 The Alea Authors

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Extending Alea

Recipes for the three most common additions: a Hamiltonian term, a
trial-wavefunction factor and an estimator. Read [Architecture](architecture.md)
first. It explains the layers these recipes touch.

Every addition, whatever its kind, follows the same path:

```
YAML key ──► io/parser.{h,cpp} ──► builders/<family>.cpp ──► src/<area>/<class>.{h,cpp}
  (spec struct)      (parse + refuse bad input)   (spec → object)     (the physics)
```

and finishes with the same checklist:

- [ ] **Header comment = the math.** The class opens with the equation it
      implements, the reference (authors, journal, volume, page, year, equation
      number) and the physical role. `src/wavefunction/kspace_jastrow.h` is the
      model to follow.
- [ ] **Array layouts stated.** Every array declaration has a comment such as
      `// [np * dim * nw]`.
- [ ] **No allocation in hot loops.** Size scratch in the constructor or `prepare()`.
- [ ] **Bad input throws** at build time with a message that says what to write
      instead.
- [ ] **Sources registered** in `CMakeLists.txt` (`SOURCES`) and in every test
      target in `tests/CMakeLists.txt` that needs them.
- [ ] **Contract tests** (see [Testing](testing.md#what-the-c-tests-cover)), plus
      one physics test against a known result.
- [ ] **Docs updated**: the key in [configuration.md](configuration.md), and an
      example config if useful (then regenerate the golden file; see
      [Testing](testing.md#golden-build-characterization)).

---

## Add a Hamiltonian term

The simplest existing example is the harmonic trap: `src/hamiltonian/harmonic_trap.{h,cpp}`,
built in `src/builders/kinetic.cpp`. For a pair potential, copy
`src/hamiltonian/coulomb.{h,cpp}` and `src/builders/coulomb.cpp`.

**1. Spec and parser.** In `src/io/parser.h`, add a struct for the term's
parameters and a `std::vector` of it to `Config`. In `parse_config`
(`src/io/parser.cpp`), add a branch to the `hamiltonian.terms` loop:

```cpp
} else if (type == "my_term") {
    require_keys(term, "hamiltonian term 'my_term'", {"type", "groups", "strength"});
    MyTermSpec s;
    if (term["groups"]) s.groups = term["groups"].as<std::vector<std::string>>();
    if (!term["strength"]) throw std::runtime_error("my_term needs strength");
    s.strength = term["strength"].as<double>();
    c.my_terms.push_back(std::move(s));
}
```

`require_keys` lists every key the block reads; any other key is refused, so a
typo cannot leave a parameter at its default. Unknown term types already throw,
so a typo in `type` cannot drop the term.

**2. The component.** Derive from `HamiltonianComponent`
(`src/hamiltonian/hamiltonian_component.h`). Three methods are required:

| Method | Contract |
|---|---|
| `name()` | Short tag, used in messages and in the golden characterization file |
| `evaluate_potential(configs, v_out, w_begin, w_end)` | **Accumulate** (`+=`) $V(R_w)$ into `v_out[w]` for $w \in$ [w_begin, w_end). Read positions and distance tables from the `ParticleBatch` only. Stride is the full `nw` |
| `clone()` | Deep copy (`std::make_unique<MyTerm>(*this)`) |

Because the method takes a `ParticleBatch`, the term works for VMC, DMC and PIMC
at once. Override the optional methods when they apply:

| Override | When |
|---|---|
| `coulomb_cusp_residue(a, b)` | The pair potential has a $1/r$ pole: return $R = \lim r V$. Jastrow cusps are derived from it |
| `pair_fourier(a, b, k)` | You want `kspace_jastrow` to seed itself from this term's $\tilde V(k)$ |
| `delta_potential_particle(...)` | PIMC is slow with the default (full re-evaluation): return the $O(N)$ change from moving one particle |
| `supports_potential_gradient()` + `evaluate_potential_gradient(...)` | Needed by the PIMC virial energy estimators |
| `primitive_pair_admissible(a, b)` | The primitive PIMC action would not be normalizable for an attractive singularity: return `false` to refuse |

**3. The builder.** Declare `build_component(const io::MyTermSpec&, const
HamiltonianContext&)` in `src/builders/hamiltonian_builders.h`, implement it in
its own `src/builders/my_term.cpp`, and add it to `build_hamiltonian` in
`src/qmc/app.cpp`:

```cpp
for (const io::MyTermSpec& s : cfg.my_terms) h.add_component(build_component(s, ctx));
```

The builder resolves group names (`find_group`) and validates what the parser
could not, such as the term requiring a periodic cell.

**A new pair-potential family** also needs a periodic form: implement
`ewald_kernel(specs, a, b, ctx)` next to the open-boundary builder and add the
family to the dispatch in `build_ewald` (`src/builders/ewald.cpp`).

**4. Tests.** Use the "Hamiltonian component" table in the contract-test
checklist: a direct pair sum on a 2–5 particle system, asymptotic limits, the cusp
residue, and crowd ranges and clones. Then add one physics test, such as a VMC
or DMC energy against an analytic or published value, in
`tests/test_vmc_physics.cpp` or a new file.

---

## Add a trial-wavefunction factor

Examples by complexity: `center_of_mass_momentum` (no parameters, complex),
`moire_jastrow` (one-body, parameters), `pade_jastrow` (pair, with derivatives
for the optimizer), `slater_determinant` (nodal).

**1. Spec and parser.** Add a struct to `src/io/parser.h` and add it to the
`WavefunctionItem` variant:

```cpp
using WavefunctionItem = std::variant<JastrowSpec, SlaterSpec, MoireJastrowSpec,
                                      CenterOfMassMomentumSpec, KSpaceJastrowSpec,
                                      NosanowSpec, MyFactorSpec>;
```

Add a branch to the `wavefunction:` loop in `parser.cpp`, before the final
`else` that rejects unknown items, and add your item to that error message.

**2. The component.** Derive from `WavefunctionComponent`
(`src/wavefunction/wavefunction_component.h`). Every method works on all walkers
at once, in the SoA layout.

| Method | Contract |
|---|---|
| `evaluate_batch(batch)` | Full evaluation: fill the caches `log_psi_c_`, `grad_c_`, `lap_c_` and **add** the contribution to the batch totals |
| `ratio_batch(batch, p, w_begin, w_end)` | For a proposed move of particle `p`: **add** $\ln\vert\psi'_c/\psi_c\vert$ to `batch.log_ratios()` and the new gradient and Laplacian to `batch.new_grad()`/`new_lap()`. Keep the values pending |
| `apply_move_batch(batch, p, accept_mask, ...)` | Commit pending values for accepted walkers and update the batch totals by the difference |
| `parameters()`, `rebuild()` | Expose `Parameter`s; rebuild internal tables after the optimizer changes them |
| `compute_parameter_derivatives(batch, derivs)` | $\partial\ln\psi_c/\partial p_k$ for every free parameter, layout `[k * nw + w]` |
| `clone()` | Deep copy |
| `is_nodal()` / `is_complex()` | `true` if the factor changes sign / has a phase. These select fixed-node / fixed-phase DMC and the optimizer's `parameters:` scope |

Optional: `supports_parameter_gradient_derivatives()` and
`compute_parameter_gradient_derivatives(...)`. They give the linear method its
curvature term; without them the optimizer still works, but its steps for these
parameters are less reliable. Override `permute_caches` if the component keeps
extra per-walker state, so that it survives DMC branching.

Optional: `supports_shifted_pairs()` and `shifted_pair_ratios_batch(...)`, the
ratio $\psi_c(R_{ij})/\psi_c(R)$ for every pair $i \in a$, $j \in b$ moved by the
same vector, from the committed state alone. The two-body density matrix of
`density_matrix` needs it from every factor; a trial with a factor that lacks it
is refused there. Computing it through two single moves would pass through a
configuration with only $i$ moved, where a determinant can vanish. The pairing
and Slater determinants (rank-2 determinant lemma) and `RadialPairJastrow`
implement it.

A one-body factor, $\ln\psi_c = \sum_p u(\mathbf r_p)$, can call
`enable_particle_values()` in its constructor. The base class then keeps each
particle's share $u(\mathbf r_p)$ in `value_c_` (sized, permuted and cloned with
the other caches). `ratio_batch` evaluates only the proposed position, writes the
new share to `pending_value_c_` and adds `pending_value_c_ - value_c_[p]` to the
log-ratio. `apply_move_batch` then calls `commit_particle_value` instead of
evaluating anything. `MoireJastrow` is the example. A pair factor gains nothing
from this cache: every accepted move also changes its partners' shares.

A pair factor $\ln\psi_c = \sum u(r_{ij})$ with a radial $u$ can derive from
`RadialPairJastrow<Radial>` (`src/wavefunction/radial_pair_jastrow.h`) instead.
`Radial` is a value type whose `evaluate(r, u, du, d2u)` writes $u$, $u'$, $u''$
for a vector of distances. The base owns it as `radial_`, plus the walker caches
and the evaluate, ratio and commit loops. The derived class supplies its
parameters, `rebuild()` (which reassigns `radial_`) and the parameter
derivatives. Its `.cpp` includes `radial_pair_jastrow_impl.h` and instantiates
`template class RadialPairJastrow<MyFunction>;`, so the radial call inlines into
the loops. `DrummondJastrow` and `ZeroEnergyJastrow` are the examples. Under
periodic boundaries the base uses the minimum image, which is exact only if $u$
is constant beyond the Wigner–Seitz inradius; the derived class must keep it so.

Use the helpers in `WavefunctionComponent` for periodic cells (`fold_delta_`) and
for parameter defaults (`make_parameters` in `builders/wavefunction_builders.h`:
fixed unless `is_fixed` says otherwise).

**3. The builder.** Declare `build_component(const io::MyFactorSpec&, const
WavefunctionContext&)` in `src/builders/wavefunction_builders.h` and implement it
in `src/builders/my_factor.cpp`. `build_wavefunction` reaches it through
`std::visit`, so **the code does not compile until the overload exists**. That is
intended.

**4. The snapshot.** If the factor has parameters, teach
`src/io/config_writer.cpp` to write the final values back into the item, as it
does for `jastrow`, `slater`, `moire_jastrow`, `kspace_jastrow` and `nosanow`. Otherwise an
optimized run's snapshot replays the starting values.

**5. Tests.** Use the "Wavefunction component" table of the contract-test
checklist. `tests/test_pade_jastrow.cpp` is the reference implementation, so
copy its helpers. The derivative checks against finite differences (items 2, 4,
10) catch the bugs that leave $\psi$ right but bias $E_L$.

---

## Add an estimator

Examples: `src/estimators/pair_correlation.{h,cpp}` (histogram per group pair),
`density.{h,cpp}` (grid, streaming statistics).

**1. The class.** Derive from `Estimator` (`src/estimators/estimator.h`). Its
lifecycle per block is:

```
start_block() → accumulate(configs, weights) × steps_per_block → stop_block()
collect(other)   merge another thread's clone
finalize(out)    normalize and append named series to the RunResult
```

`accumulate` receives configurations and one weight per configuration (1 in VMC
and PIMC, the branching weight in DMC), so one implementation serves every driver.
Without PIMC reblocking (VMC/DMC) keep block moments and merge them in `collect`
(Chan's parallel rule), as `ElectronHoleProximity` does; `stop_block` must not
require the PIMC history.

**Result keys.** Implement `output_namespaces()`: every key `finalize` writes
(series, scalar or metadata) must equal one of these prefixes or extend one with
`_` (`PairCorrelation` returns `{"gofr"}`, `StructureFactor` `{"sofk", "rhok"}`).
That is what lets a stage hold several instances of the kind: the driver calls
`write_results`, which inserts the instance's label after the namespace. Drivers
also refuse a result with a repeated key (`RunResult::check_unique_keys`).

**For PIMC**, also implement `configure_reblocking` and
`reblocking_storage_bytes` (PIMC reblocks every observable), and
`configure_signed_weights` for fermion runs. Estimators that do not implement
them are refused by the PIMC driver at start-up, not silently mis-estimated.

**Ratios of $\psi_T$.** An estimator that evaluates the trial wavefunction at
displaced configurations (`DensityMatrix`) returns `true` from
`needs_wavefunction()`. The VMC and DMC drivers then call
`bind_wavefunction(psi, seed, n_workers)` once per run, after cloning it: it
copies $\psi_T$ (its virtual moves never touch the sampling state) and takes its
random numbers from the stream `seed`. DMC passes its thread count and
accumulates the whole population from one serial point, so such an estimator
splits the walkers among its workers itself. Its builder refuses PIMC.

**2. Registration.** Three places, all checked at compile time or parse time:

- a spec struct with the kind's keys and a `std::string name` in
  `src/io/parser.h`, added to the `io::EstimatorItem` variant;
- its reader in `parse_estimator` (`src/io/parser.cpp`), with a `require_keys`
  list of exactly its keys;
- a `build_estimator(const io::YourSpec&, const EstimatorContext&)` overload in
  `src/builders/estimators.cpp`. It holds the kind's rules: refuse a sampler
  only when the kind needs something that sampler lacks (as
  `momentum_distribution` needs the PIMC worm), refuse a cell it cannot handle,
  and pass `spec.name` on as the output label.

Then document the keys in [configuration.md](configuration.md#estimators), and
what the kind measures (formula, normalization, samplers, error bars) in
[estimators.md](estimators.md#3-optional-observables).

**3. Tests.** A tiny system with a known answer (e.g. $g(r)$ of two particles at
a fixed distance), the normalization, `collect` across clones equal to a single
pass, and a DMC run with non-unit weights. Add the kind to the key-contract
cases in `tests/test_estimator_builders.cpp`, which check that every key lies in
`output_namespaces()` on both statistics paths.

---

## Add a YAML key to an existing feature

1. Add the field, with its default, to the spec struct in `src/io/parser.h`, and
   a comment giving its meaning and units.
2. Read it in `parser.cpp`, and add it to that block's `require_keys` list.
   Every block has one, so a key missing from the list is refused as unknown.
3. Use it in the builder.
4. Add a parser test in `tests/test_parser.cpp`, and a row in
   [configuration.md](configuration.md).

If a key is **removed or renamed**, make the parser refuse the old name with a
message pointing to the new one (see how `feedback:`, `param_tol:` and the Padé
`params:` are handled). Do this *before* the `require_keys` call, so the user
gets the specific message rather than "unknown key". Old configs then fail
loudly instead of running a different model.
