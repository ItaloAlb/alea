# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reblocking analyzer for Alea VMC/DMC energy histories.

Given the energy time-series a run already emits, find the *block length at which
the data decorrelates*, and report the honest, correlation-corrected error bar
together with the statistical inefficiency.

Alea's drivers already report this error (``numerics::reblock`` and
``numerics::reblock_chains`` implement the same rule in C++); this tool prints the
full table, plots it, and reblocks older or external histories.  Method:
Flyvbjerg–Petersen reblocking,

    H. Flyvbjerg and H. G. Petersen, "Error estimates on averages of correlated
    data", J. Chem. Phys. 91, 461 (1989).

The data are repeatedly coarse-grained by averaging adjacent pairs.  The naive
standard error of the mean (SEM) rises with the block length and then plateaus
once the block exceeds the autocorrelation length; the plateau value is the true
error.  The automatic plateau onset uses the error-on-error overlap criterion;
see also the parameter-free test of M. Jonsson, "Standard error estimation by an
automated blocking method", Phys. Rev. E 98, 043304 (2018).

Engine shapes (handled automatically when reading an Alea ``.h5``):
  * VMC ``acc_block_energy`` is ``n_threads`` INDEPENDENT chains concatenated, so
    reblocking must not cross thread boundaries (we reshape and pool).
  * DMC ``acc_block_energy`` is a single global branched chain.

Usage (manual CLI):
    py -3 tools/reblock.py results/<name>/<ts>/0_dmc.h5
    py -3 tools/reblock.py series.txt --column 2 --plot reblock.png
    py -3 tools/reblock.py --selftest

Usage (library):
    from reblock import reblock, reblock_chains, load_series
    series, meta = load_series("0_dmc.h5")
    res = reblock(series)            # or reblock_chains(series) for 2-D input
    print(res.block_length, res.error)

Depends on numpy (core) and, optionally, h5py (only to read ``.h5``) and
matplotlib (only for ``--plot``).
"""
import argparse
import math
import os
import sys
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["reblock", "reblock_chains", "load_series", "ReblockResult", "Level"]

# One row of the reblocking transformation.
#   level    : transformation number t (0 = raw data)
#   block_size: number of original samples per block = 2**t
#   n_blocks : number of blocks remaining at this level
#   sem      : standard error of the mean estimated at this level
#   sem_err  : statistical uncertainty of that SEM (error on the error)
Level = namedtuple("Level", ["level", "block_size", "n_blocks", "sem", "sem_err"])


@dataclass
class ReblockResult:
    mean: float                  # grand mean of the series
    naive_error: float           # SEM assuming independence (level 0)
    error: float                 # correlation-corrected SEM (plateau)
    block_length: int            # samples per decorrelated block = 2**opt_level
    inefficiency: float          # s = (error/naive_error)**2 ~ 1 + 2*tau_int
    tau_int: float               # integrated autocorrelation time
    opt_level: int               # plateau-onset transformation number
    n_samples: int               # total samples consumed
    n_chains: int = 1            # independent chains pooled (VMC threads)
    table: List[Level] = field(default_factory=list)


# Plateau detection only trusts levels with enough blocks for the SEM (and its
# error-on-error) to be meaningful; below this we fall back progressively.
_MIN_BLOCKS = 16


def _reblock_table(x: np.ndarray) -> List[Level]:
    """Full reblocking transformation of a single 1-D chain.

    Each step averages adjacent pairs (dropping a trailing odd sample) so the
    block length doubles and the block count halves.  Returns one Level per
    transformation, from raw data down to 2 blocks.
    """
    x = np.asarray(x, dtype=float).ravel()
    table: List[Level] = []
    cur = x
    level = 0
    while cur.size >= 2:
        n_blocks = cur.size
        var = float(np.var(cur, ddof=1))           # unbiased sample variance
        sem = math.sqrt(var / n_blocks)            # SEM = sqrt(s^2 / M)
        sem_err = sem / math.sqrt(2.0 * (n_blocks - 1)) if n_blocks > 1 else float("nan")
        table.append(Level(level, 1 << level, n_blocks, sem, sem_err))
        if cur.size % 2:                           # drop trailing odd sample
            cur = cur[:-1]
        cur = 0.5 * (cur[0::2] + cur[1::2])         # average adjacent pairs
        level += 1
    return table


def _select_plateau(table: List[Level]) -> int:
    """Return the plateau-onset level: the smallest block length at which a
    further blocking no longer increases the SEM beyond statistical noise.

    The SEM grows with block length while samples are still correlated and
    flattens once the block exceeds the autocorrelation length.  The onset is the
    first level whose increment to the next level is within the combined
    error-on-error.  Restricts attention to levels with enough blocks (falling
    back when the series is short) so the noisy tail does not drive the estimate.
    """
    cand = [L for L in table if L.n_blocks >= _MIN_BLOCKS]
    if len(cand) < 4:
        cand = [L for L in table if L.n_blocks >= 4]
    if not cand:
        return table[-1].level

    def eoe(L: Level) -> float:
        return 0.0 if math.isnan(L.sem_err) else L.sem_err

    for i in range(len(cand) - 1):
        a, b = cand[i], cand[i + 1]
        if b.sem - a.sem <= eoe(a) + eoe(b):        # growth has stopped
            return a.level
    return cand[-1].level


def _plateau_error(table: List[Level], opt_level: int) -> float:
    """Corrected SEM = the largest SEM on the plateau (from the onset level
    onward, ignoring the few-block noisy tail).  Taking the max rather than the
    onset value keeps the error bar honest — never optimistic."""
    region = [L.sem for L in table
              if L.level >= opt_level and L.n_blocks >= _MIN_BLOCKS]
    if not region:
        region = [L.sem for L in table
                  if L.level >= opt_level and L.n_blocks >= 4]
    return max(region) if region else table[opt_level].sem


def _assemble(mean: float, table: List[Level], opt_level: int,
              n_samples: int, n_chains: int) -> ReblockResult:
    naive_error = table[0].sem
    error = _plateau_error(table, opt_level)
    inefficiency = (error / naive_error) ** 2 if naive_error > 0.0 else 1.0
    tau_int = max(0.0, 0.5 * (inefficiency - 1.0))
    # The decorrelation block length is the statistical inefficiency
    # s = 1 + 2*tau_int (samples per independent sample), a more stable estimate
    # than the power-of-two plateau block size 2**opt_level (which the table marks).
    block_length = max(1, int(math.ceil(inefficiency)))
    return ReblockResult(
        mean=mean,
        naive_error=naive_error,
        error=error,
        block_length=block_length,
        inefficiency=inefficiency,
        tau_int=tau_int,
        opt_level=opt_level,
        n_samples=n_samples,
        n_chains=n_chains,
        table=table,
    )


def reblock(data: np.ndarray) -> ReblockResult:
    """Reblock a single 1-D energy chain (e.g. a DMC ``acc_block_energy``)."""
    x = np.asarray(data, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("reblock: need at least 2 samples")
    table = _reblock_table(x)
    opt = _select_plateau(table)
    return _assemble(float(x.mean()), table, opt, x.size, 1)


def reblock_chains(chains: np.ndarray) -> ReblockResult:
    """Reblock ``K`` independent, equal-length chains (e.g. VMC per-thread
    ``acc_block_energy``) and pool them.

    Each chain is reblocked separately so correlation is never measured across a
    chain boundary.  At every level the chains give independent estimates of the
    grand mean, so the pooled variance of the mean is ``(1/K^2) Σ_i v_i`` and the
    pooled ``SEM = sqrt(Σ_i sem_i^2) / K``.  K = 1 reduces to :func:`reblock`.
    """
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    K = chains.shape[0]
    if K == 1:
        return reblock(chains[0])
    per = [_reblock_table(chains[i]) for i in range(K)]
    nlev = min(len(t) for t in per)
    pooled: List[Level] = []
    for t in range(nlev):
        sem2 = sum(per[i][t].sem ** 2 for i in range(K))
        sem = math.sqrt(sem2) / K
        n_blocks = sum(per[i][t].n_blocks for i in range(K))
        # Pooled variance: each chain contributes n_i - 1 degrees of freedom.
        dof = sum(per[i][t].n_blocks - 1 for i in range(K))
        sem_err = sem / math.sqrt(2.0 * dof) if dof > 0 else float("nan")
        pooled.append(Level(t, 1 << t, n_blocks, sem, sem_err))
    opt = _select_plateau(pooled)
    return _assemble(float(chains.mean()), pooled, opt, int(chains.size), K)


# ── I/O ────────────────────────────────────────────────────────────────────

def _attr_str(node, key: str) -> Optional[str]:
    v = node.attrs.get(key)
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def load_series(path: str, column: int = 0) -> Tuple[np.ndarray, dict]:
    """Load an energy history from an Alea ``.h5`` result or a plain-text file.

    For ``.h5`` (needs ``h5py``): reads the ``acc_block_energy`` dataset and, for
    VMC with ``n_threads > 1``, reshapes into ``(n_threads, n_acc)`` independent
    chains.  For text: ``numpy.loadtxt`` with selectable ``column``.

    Returns ``(data, meta)`` where ``data`` is 1-D (single chain) or 2-D
    ``(K, N)`` (independent chains), and ``meta`` carries ``method``/``n_threads``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError:
            raise SystemExit(
                "reading .h5 requires h5py (one-time:  pip install h5py).\n"
                "Alternatively, pass a plain-text series (numpy.loadtxt works "
                "without h5py)."
            )
        with h5py.File(path, "r") as f:
            if "acc_block_energy" not in f:
                raise SystemExit(
                    f"{path}: no 'acc_block_energy' dataset — is this an Alea "
                    "result file?"
                )
            data = np.asarray(f["acc_block_energy"][...], dtype=float).ravel()
            method = _attr_str(f, "method")
            n_threads = int(f.attrs.get("n_threads", 1))
        meta = {"method": method, "n_threads": n_threads, "source": path}
        if method == "vmc" and n_threads > 1 and data.size % n_threads == 0:
            return data.reshape(n_threads, -1), meta
        return data, meta

    arr = np.loadtxt(path)
    if arr.ndim > 1:
        if column >= arr.shape[1]:
            raise SystemExit(f"{path}: column {column} out of range "
                             f"(file has {arr.shape[1]} columns)")
        arr = arr[:, column]
    return np.asarray(arr, dtype=float).ravel(), {
        "method": None, "n_threads": 1, "source": path}


# ── reporting & plotting ────────────────────────────────────────────────────

def format_report(res: ReblockResult, meta: Optional[dict] = None) -> str:
    lines = []
    src = (meta or {}).get("source")
    method = (meta or {}).get("method")
    head = "reblock"
    if method:
        head += f" [{method}]"
    if src:
        head += f"  {src}"
    lines.append(head)
    lines.append(f"  samples         : {res.n_samples}"
                 + (f"  ({res.n_chains} chains)" if res.n_chains > 1 else ""))
    lines.append(f"  mean            : {res.mean:.8g}")
    lines.append(f"  naive error     : {res.naive_error:.4g}   "
                 "(assumes independent samples)")
    lines.append(f"  corrected error : {res.error:.4g}   "
                 f"(x{res.error / res.naive_error:.2f} larger)"
                 if res.naive_error > 0 else
                 f"  corrected error : {res.error:.4g}")
    lines.append(f"  block length    : {res.block_length}   "
                 "(samples per decorrelated block)")
    lines.append(f"  inefficiency s  : {res.inefficiency:.3g}   "
                 f"tau_int = {res.tau_int:.3g}")
    lines.append("")
    lines.append("  corrected error = max SEM on the plateau (onset marked below)")
    lines.append("  level  block_size  n_blocks       sem    sem_err")
    for L in res.table:
        mark = "  <-- plateau onset" if L.level == res.opt_level else ""
        sem_err = "      nan" if math.isnan(L.sem_err) else f"{L.sem_err:9.3g}"
        lines.append(f"  {L.level:5d}  {L.block_size:10d}  {L.n_blocks:8d}  "
                     f"{L.sem:9.4g}  {sem_err}{mark}")
    return "\n".join(lines)


def plot_reblock(res: ReblockResult, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [L.level for L in res.table]
    sem = [L.sem for L in res.table]
    err = [0.0 if math.isnan(L.sem_err) else L.sem_err for L in res.table]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(levels, sem, yerr=err, marker="o", ms=4, capsize=3, lw=1)
    ax.axhline(res.error, ls="--", color="0.5", lw=1)
    ax.axvline(res.opt_level, ls=":", color="C3", lw=1,
               label=f"plateau: block={res.block_length}")
    ax.set_xlabel("reblocking level  t   (block size = $2^t$)")
    ax.set_ylabel("standard error of the mean")
    ax.set_title(f"Reblocking  (corrected error = {res.error:.4g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── self-test ───────────────────────────────────────────────────────────────

def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) process x_t = phi x_{t-1} + e_t, with known tau_int = phi/(1-phi)."""
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


def _selftest() -> int:
    rng = np.random.default_rng(12345)
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}: {detail}")

    # White noise: no correlation -> block length ~1, inefficiency ~1.
    white = rng.standard_normal(1 << 16)
    rw = reblock(white)
    check("white-noise block length", rw.block_length <= 2,
          f"block={rw.block_length}, s={rw.inefficiency:.2f}")
    check("white-noise inefficiency", 0.5 <= rw.inefficiency <= 2.0,
          f"s={rw.inefficiency:.2f}")

    # AR(1), phi=0.8 -> tau_int=4, inefficiency = 1+2*tau = 9, block ~ 8-16.
    phi = 0.8
    tau_true = phi / (1.0 - phi)
    s_true = 1.0 + 2.0 * tau_true
    ar = _ar1(1 << 16, phi, rng)
    ra = reblock(ar)
    check("AR(1) inefficiency near 1+2tau", 0.5 * s_true <= ra.inefficiency <= 2.0 * s_true,
          f"s={ra.inefficiency:.2f} (expected ~{s_true:.1f})")
    check("AR(1) block length in plausible range", 4 <= ra.block_length <= 64,
          f"block={ra.block_length}, tau_int={ra.tau_int:.2f} (expected ~{tau_true:.1f})")

    # Multi-chain pooling: K independent AR(1) chains -> combined error ~ single/sqrt(K).
    K = 8
    chains = np.stack([_ar1(1 << 13, phi, rng) for _ in range(K)])
    rc = reblock_chains(chains)
    single = reblock(chains[0])
    expected = single.error / math.sqrt(K)
    ratio = rc.error / expected if expected > 0 else float("inf")
    check("multi-chain combined error ~ single/sqrt(K)", 0.4 <= ratio <= 2.5,
          f"combined={rc.error:.4g}, expected~{expected:.4g} (ratio {ratio:.2f})")

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Flyvbjerg-Petersen reblocking of an Alea energy history: "
                    "report the block length that decorrelates the data and the "
                    "correlation-corrected error bar.")
    p.add_argument("path", nargs="?",
                   help="Alea result (.h5, dataset 'acc_block_energy') or a "
                        "plain-text series.")
    p.add_argument("--column", type=int, default=0,
                   help="column to use for a multi-column text file (default 0).")
    p.add_argument("--chains", type=int, default=0,
                   help="treat a 1-D text series as K independent chains "
                        "(reshape into K equal blocks, trimming remainder).")
    p.add_argument("--plot", metavar="OUT",
                   help="write a reblocking-curve figure (needs matplotlib).")
    p.add_argument("--quiet", action="store_true",
                   help="print only the corrected error bar.")
    p.add_argument("--selftest", action="store_true",
                   help="run the synthetic validation and exit.")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.path:
        p.error("a result/series path is required (or use --selftest)")

    data, meta = load_series(args.path, column=args.column)

    if args.chains and data.ndim == 1:
        K = args.chains
        n = (data.size // K) * K
        data = data[:n].reshape(K, -1)
        meta["n_threads"] = K

    res = reblock_chains(data) if data.ndim == 2 else reblock(data)

    if args.quiet:
        print(f"{res.mean:.8g} +/- {res.error:.4g}  "
              f"(block={res.block_length})")
    else:
        print(format_report(res, meta))

    if args.plot:
        plot_reblock(res, args.plot)
        if not args.quiet:
            print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
