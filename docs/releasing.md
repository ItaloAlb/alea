<!--
SPDX-FileCopyrightText: 2026 The Alea Authors
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Releasing Alea

How Alea is versioned and how a version is released. This page is the
reference for maintainers and for anyone preparing a release.

## Versioning

Alea follows [Semantic Versioning 2.0.0](https://semver.org). Its public
interface is:

- the YAML input: every key, its type, its default and its meaning;
- the outputs: the layout and units of the HDF5 results under
  `results/<name>/<timestamp>/`, the walker and path checkpoint formats, and
  the config snapshot written with each run;
- the command line, `alea <config.yaml>`;
- the documented meaning of each computed quantity: which estimator, in which
  units, with which error (for example the reblocked error of every driver).

The files that define it, and that a release compares against the previous
tag:

- `docs/configuration.md` and the parser, `src/io/parser.cpp` and
  `src/io/parser.h`;
- the writers of results, checkpoints and snapshots in `src/io/`;
- `main.cpp`;
- `tests/golden/build_characterization.txt`, which records what every
  shipped config builds and evaluates to.

| Change | From 1.0.0 | While 0.x |
|---|---|---|
| An input that worked fails or changes meaning; an output changes format; a command-line option disappears | MAJOR | MINOR |
| A compatible feature | MINOR | MINOR |
| A fix with no interface change | PATCH | PATCH |

A fix that changes the numbers of a documented quantity is a PATCH when the old
numbers were wrong; the changelog says which results change. A changed default
(a time step rule, a branch cutoff, an optimizer setting) that alters results
is at least a MINOR change.

A commit that breaks the interface says so in its body:
`BREAKING: <what breaks and how to migrate>`.

## Where the version lives

The single source is `project(alea VERSION X.Y.Z ...)` in `CMakeLists.txt`.
A release also updates `version` and `date-released` in `CITATION.cff` and
adds the entry to `CHANGELOG.md`.

## Release procedure

1. `main` is clean, in sync with `origin`, and the full test suite passes.
2. Compare the previous tag with `HEAD`: `BREAKING:` lines in the commit
   bodies, the interface files listed above, changed golden files. Choose the
   new version from the table.
3. Write the changelog entry for users (Added, Changed, Deprecated, Removed,
   Fixed, Security).
4. Update the version source, `CITATION.cff` and `CHANGELOG.md` in one commit,
   `Release vX.Y.Z`.
5. Create the annotated tag `vX.Y.Z` with the changelog entry as its message,
   and push the commit and the tag.
6. Publish a GitHub release from the tag; Zenodo archives it and assigns a DOI.

## Readiness checks

Run before every release, in addition to the test suite:

- A clean configure and build of `build/alea` and `build/tests` with the
  commands in the README, and `ctest --test-dir build/tests` with every test
  passing.
- `python -m unittest discover -s tests -p "test_*.py"` passes.
- The quick-start command in the README runs and prints the output shown there.
- Every key the parser accepts is in `docs/configuration.md`, with its default.
- A golden-file change in the release range is deliberate and explained in the
  changelog.
- `reuse lint` reports compliance.
