# SPDX-FileCopyrightText: 2026 The Alea Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot the QMC wavefunction density from an Alea result file.

In DMC the walker population samples the mixed distribution f(r) = psi_T(r)*Phi(r),
where Phi is the projected ground state — the walker cloud *is* the wavefunction
made visible.  The ``{type: density}`` estimator histograms the walker positions
per particle group over the whole accumulation phase (weighted by the DMC branch
weights), so the ``density_<group>`` datasets in the result ``.h5`` hold:

    VMC:  rho = <psi_T| rho_hat |psi_T>           (|psi_T|^2 density)
    DMC:  rho = <psi_T| rho_hat |Phi> / <psi_T|Phi>   (mixed estimator)

and the pure density can be extrapolated in post-processing,

    rho_pure ~= 2*rho_DMC - rho_VMC + O((Phi - psi_T)^2),

D. M. Ceperley and M. H. Kalos, in *Monte Carlo Methods in Statistical Physics*
(Springer, 1979); W. M. C. Foulkes et al., Rev. Mod. Phys. 73, 33 (2001), Sec. II.F.

Dataset layout (written by src/estimators/density.cpp): ``density_<group>`` is the
C row-major flattened grid (``cell = ix*n_bins + iy`` in 2-D), ``density_edges``
the ``n_bins+1`` bin edges from -rmax to +rmax (one axis serves every dimension),
plus root attributes ``density_dim/n_bins/rmax/n_blocks``.

Usage (CLI):
    py -3 tools/plot_density.py results/<name>/<ts>/1_dmc.h5
    py -3 tools/plot_density.py 1_dmc.h5 --radial --amplitude
    py -3 tools/plot_density.py 1_dmc.h5 --extrapolate 0_vmc.h5
    py -3 tools/plot_density.py --walkers wk.eq.h5      # checkpoint quick-look
    py -3 tools/plot_density.py crystal/1_vmc.h5 --tile 2          # periodic cell
    py -3 tools/plot_density.py fluid/0_vmc.h5 --compare crystal/1_vmc.h5 --tile 2

Periodic cells: the estimator bins FRACTIONAL coordinates (edges -1/2..1/2) and
writes the lattice vectors (``density_lattice``, rows a_i); the map is drawn in
real space, r = f1*a1 + f2*a2, as the cell's true parallelogram (outlined in
white), optionally repeated ``--tile N`` times per axis.
    py -3 tools/plot_density.py --selftest

Usage (library):
    from plot_density import load_density, radial_profile
    groups, meta = load_density("1_dmc.h5")
    rho, err = groups["e"]

Depends on numpy (core) and, optionally, h5py (only to read ``.h5``) and
matplotlib (only for the plots).

Lab-frame caveat: for a free complex the centre of mass diffuses and the density
smears toward uniform — a sharp image needs something pinned (fixed particles,
a one-body potential).
"""
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["load_density", "load_collective", "load_walkers", "radial_profile",
           "extrapolate"]

# Fixed categorical order — color follows the particle group, never its rank;
# method (DMC/VMC/pure) is carried by linestyle instead.
GROUP_COLORS = ["#1f4e8c", "#c0392b", "#f0a020", "#2e7d32"]


def _require_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        raise SystemExit(
            "reading .h5 requires h5py (one-time:  pip install h5py)."
        )


def _require_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise SystemExit(
            "plotting requires matplotlib (one-time:  pip install matplotlib)."
        )


def _attr_str(node, key: str) -> Optional[str]:
    v = node.attrs.get(key)
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


# ── I/O ────────────────────────────────────────────────────────────────────

def load_density(path: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], dict]:
    """Load the density grids from an Alea result ``.h5``.

    Returns ``(groups, meta)``: ``groups[name] = (rho, err)`` with both arrays
    reshaped to ``(n_bins,)*dim`` so ``rho[ix, iy]`` follows the C row-major
    flat layout of the estimator, and ``meta`` carrying ``dim/n_bins/rmax/
    edges/n_blocks/method/source``.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        if "density_edges" not in f:
            raise SystemExit(
                f"{path}: no density_* datasets — did the run declare a "
                "{type: density} estimator under qmc: -> estimators:?"
            )
        dim = int(f.attrs.get("density_dim", 2))
        n_bins = int(f.attrs.get("density_n_bins", 0))
        meta = {
            "dim": dim,
            "n_bins": n_bins,
            "rmax": float(f.attrs.get("density_rmax", 0.0)),
            "n_blocks": int(f.attrs.get("density_n_blocks", 0)),
            "method": _attr_str(f, "method"),
            "edges": np.asarray(f["density_edges"][...], dtype=float),
            "source": path,
            # Periodic cells bin FRACTIONAL coordinates (edges -1/2..1/2);
            # density_lattice holds the primitive vectors as rows,
            # a[i*dim + d], so r = sum_i f_i a_i.
            "periodic": bool(f.attrs.get("density_periodic", 0.0)),
            "lattice": (np.asarray(f["density_lattice"][...], dtype=float)
                        .reshape(dim, dim) if "density_lattice" in f else None),
        }
        shape = (n_bins,) * dim
        groups: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in f:
            if not name.startswith("density_") \
                    or name in ("density_edges", "density_lattice") \
                    or name.endswith("_error"):
                continue
            group = name[len("density_"):]
            rho = np.asarray(f[name][...], dtype=float).reshape(shape)
            err_name = name + "_error"
            err = (np.asarray(f[err_name][...], dtype=float).reshape(shape)
                   if err_name in f else np.zeros(shape))
            groups[group] = (rho, err)
    if not groups:
        raise SystemExit(f"{path}: density_edges found but no density_<group> "
                         "datasets — corrupt file?")
    return groups, meta


def load_collective(path: str, name: Optional[str] = None
                    ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a ``{type: collective_density}`` grid from an Alea result ``.h5``.

    The estimator bins one collective coordinate v = Σ c_g·Σ_{p∈g} r_p per
    walker (src/estimators/collective_density.cpp) under keys namespaced by
    its instance name: ``cdensity_<name>`` (+ ``_error``, ``_edges``) and
    root attributes ``cdensity_<name>_{dim,n_bins,rmax,n_blocks}``.

    ``name=None`` is allowed only when the file holds exactly one instance.
    Returns ``(v, err, meta)`` with both grids reshaped to ``(n_bins,)*dim``
    and ``meta`` carrying ``name/dim/n_bins/rmax/edges/n_blocks/method/
    source``.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        instances = [k[len("cdensity_"):-len("_edges")] for k in f
                     if k.startswith("cdensity_") and k.endswith("_edges")]
        if not instances:
            raise SystemExit(
                f"{path}: no cdensity_* datasets — did the run declare a "
                "{type: collective_density, name: ...} estimator?"
            )
        if name is None:
            if len(instances) > 1:
                raise SystemExit(
                    f"{path}: several collective densities "
                    f"({', '.join(sorted(instances))}) — pass name=..."
                )
            name = instances[0]
        if name not in instances:
            raise SystemExit(
                f"{path}: no collective density '{name}' (available: "
                f"{', '.join(sorted(instances))})"
            )
        prefix = f"cdensity_{name}"
        dim = int(f.attrs.get(prefix + "_dim", 2))
        n_bins = int(f.attrs.get(prefix + "_n_bins", 0))
        meta = {
            "name": name,
            "dim": dim,
            "n_bins": n_bins,
            "rmax": float(f.attrs.get(prefix + "_rmax", 0.0)),
            "n_blocks": int(f.attrs.get(prefix + "_n_blocks", 0)),
            "method": _attr_str(f, "method"),
            "edges": np.asarray(f[prefix + "_edges"][...], dtype=float),
            "source": path,
        }
        shape = (n_bins,) * dim
        v = np.asarray(f[prefix][...], dtype=float).reshape(shape)
        err = (np.asarray(f[prefix + "_error"][...], dtype=float).reshape(shape)
               if prefix + "_error" in f else np.zeros(shape))
    return v, err, meta


def load_walkers(path: str) -> Tuple[np.ndarray, dict]:
    """Load a walker checkpoint (``walkers_out`` file, src/io/walker_io.cpp).

    Returns ``(positions, meta)`` with positions shaped ``(n_walkers,
    n_particles, dim)`` (the file stores the flat AoS ``[w][p][d]``).
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        if "positions" not in f:
            raise SystemExit(f"{path}: no 'positions' dataset — is this an "
                             "Alea walker checkpoint (walkers_out)?")
        nw = int(f.attrs["n_walkers"])
        n_particles = int(f.attrs["n_particles"])
        dim = int(f.attrs["dim"])
        pos = np.asarray(f["positions"][...], dtype=float)
        meta = {"n_walkers": nw, "n_particles": n_particles, "dim": dim,
                "tag": _attr_str(f, "tag"), "source": path}
    return pos.reshape(nw, n_particles, dim), meta


# ── analysis ────────────────────────────────────────────────────────────────

def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def cell_volume(meta: dict) -> float:
    if meta.get("periodic") and meta.get("lattice") is not None:
        # n_bins^dim equal boxes tiling the simulation cell.
        return abs(float(np.linalg.det(meta["lattice"]))) / meta["n_bins"] ** meta["dim"]
    delta = float(meta["edges"][1] - meta["edges"][0])
    return delta ** meta["dim"]


def periodic_mesh(meta: dict, tile: int = 1
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian corner coordinates of a 2-D periodic grid, repeated tile x tile.

    Returns ``(X, Y, index)``: X, Y are the (m+1, m+1) corner arrays for
    pcolormesh (m = tile*n_bins) — the cells are parallelograms, so the map is
    drawn in the cell's real shape — and ``index`` the fractional-bin index
    along each axis, so ``field[np.ix_(index, index)]`` repeats the data.
    """
    n = meta["n_bins"]
    a = meta["lattice"]
    m = tile * n
    # Fractional corners: the cell edges -1/2..1/2, continued periodically.
    f = -0.5 * tile + np.arange(m + 1) / n
    F1, F2 = np.meshgrid(f, f, indexing="ij")
    X = F1 * a[0, 0] + F2 * a[1, 0]
    Y = F1 * a[0, 1] + F2 * a[1, 1]
    return X, Y, np.arange(m) % n


def radial_profile(rho: np.ndarray, err: np.ndarray,
                   edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthal (2-D) / spherical (3-D) average of the grid about the origin.

    Cell centres are binned by |r| into ``n_bins/2`` annuli; each annulus value
    is the mean of its cells and its error the rms cell error / sqrt(n_cells)
    (cells treated as independent — a mild approximation, since neighbouring
    cells share walkers within a block).
    """
    c = _centers(edges)
    grids = np.meshgrid(*([c] * rho.ndim), indexing="ij")
    r_cell = np.sqrt(sum(g * g for g in grids))

    rmax = float(edges[-1])
    n_rad = max(len(c) // 2, 1)
    r_edges = np.linspace(0.0, rmax, n_rad + 1)
    which = np.clip(np.digitize(r_cell.ravel(), r_edges) - 1, 0, n_rad - 1)

    counts = np.bincount(which, minlength=n_rad).astype(float)
    sums = np.bincount(which, weights=rho.ravel(), minlength=n_rad)
    err2 = np.bincount(which, weights=(err ** 2).ravel(), minlength=n_rad)

    ok = counts > 0
    rho_r = np.zeros(n_rad)
    err_r = np.zeros(n_rad)
    rho_r[ok] = sums[ok] / counts[ok]
    err_r[ok] = np.sqrt(err2[ok]) / counts[ok]
    return _centers(r_edges), rho_r, err_r


def extrapolate(dmc: Tuple[np.ndarray, np.ndarray],
                vmc: Tuple[np.ndarray, np.ndarray]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-estimator extrapolation rho_pure ~= 2*rho_DMC - rho_VMC, with the
    errors combined in quadrature: err = sqrt(4*err_DMC^2 + err_VMC^2)."""
    rho = 2.0 * dmc[0] - vmc[0]
    err = np.sqrt(4.0 * dmc[1] ** 2 + vmc[1] ** 2)
    return rho, err


def _check_same_grid(a: dict, b: dict) -> None:
    for key in ("dim", "n_bins", "rmax"):
        if a[key] != b[key]:
            raise SystemExit(
                f"grid mismatch between {a['source']} and {b['source']}: "
                f"{key} = {a[key]} vs {b[key]} — the extrapolation needs both "
                "runs to declare the same density estimator (rmax, n_bins)."
            )


# ── plotting ────────────────────────────────────────────────────────────────

def _field_for_display(rho: np.ndarray, meta: dict, amplitude: bool,
                       slice_z: bool = False) -> Tuple[np.ndarray, str]:
    """Reduce to 2-D (3-D → z-integrated column density, or the z≈0 midplane
    slice with slice_z — the slice preserves radial structure like shell
    nodes that the line-of-sight integral fills in) and optionally take the
    amplitude sqrt(rho) ~ |psi| (nodeless states; for the DMC mixed density
    this is sqrt(psi_T*Phi))."""
    label = r"$\rho(\mathbf{r})$"
    field = rho
    if meta["dim"] == 3:
        if slice_z:
            field = field[:, :, field.shape[2] // 2]
            label = r"$\rho(x, y, z\!\approx\!0)$"
        else:
            delta = float(meta["edges"][1] - meta["edges"][0])
            field = field.sum(axis=2) * delta
            label = r"$\int \rho\, dz$"
    if amplitude:
        field = np.sqrt(np.clip(field, 0.0, None))
        label = r"$\sqrt{\rho} \sim |\psi|$" if meta["dim"] != 3 or not slice_z \
            else r"$\sqrt{\rho(x, y, z\!\approx\!0)}$"
    return field, label


def _draw_field(ax, field: np.ndarray, meta: dict, tile: int = 1,
                vmin: Optional[float] = None, vmax: Optional[float] = None):
    """pcolormesh of a 2-D field on its grid: the Cartesian box for open cells,
    the real (parallelogram) cell repeated tile x tile for periodic ones."""
    if field.min() < 0.0 and vmin is None:
        # Extrapolated densities can dip negative: diverging map, zero at the
        # neutral midpoint.
        lim = float(np.abs(field).max()) or 1.0
        style = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        style = dict(cmap="viridis", vmin=vmin, vmax=vmax)
    if meta.get("periodic") and meta.get("lattice") is not None:
        X, Y, idx = periodic_mesh(meta, tile)
        mesh = ax.pcolormesh(X, Y, field[np.ix_(idx, idx)], rasterized=True,
                             **style)
        # Outline the simulation cell itself.
        a = meta["lattice"]
        corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]])
        xy = corners @ a
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=0.8, alpha=0.7)
        ax.set_xlabel(r"$x$ / bohr")
        ax.set_ylabel(r"$y$ / bohr")
    else:
        edges = meta["edges"]
        mesh = ax.pcolormesh(edges, edges, field.T, rasterized=True, **style)
        ax.set_xlabel(r"$x\ /\ a_B^*$")
        ax.set_ylabel(r"$y\ /\ a_B^*$")
    ax.set_aspect("equal")
    return mesh


def plot_heatmap(rho: np.ndarray, meta: dict, group: str, out_png: str,
                 amplitude: bool = False, title_extra: str = "",
                 slice_z: bool = False, tile: int = 1) -> None:
    plt = _require_pyplot()
    field, label = _field_for_display(rho, meta, amplitude, slice_z)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    mesh = _draw_field(ax, field, meta, tile)
    method = (meta.get("method") or "?").upper()
    ax.set_title(f"{method} {title_extra}wavefunction density — group '{group}'",
                 fontsize=11)
    fig.colorbar(mesh, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_compare(panels: List[Tuple[np.ndarray, dict, str]], out_png: str,
                 amplitude: bool = False, tile: int = 1) -> None:
    """Side-by-side heatmaps on ONE shared colour scale (so a flat fluid and a
    peaked crystal are compared honestly).  panels: [(rho, meta, title)]."""
    plt = _require_pyplot()
    fields = [_field_for_display(r, m, amplitude)[0] for r, m, _ in panels]
    label = _field_for_display(panels[0][0], panels[0][1], amplitude)[1]
    vmin = min(float(f.min()) for f in fields)
    vmax = max(float(f.max()) for f in fields)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.0),
                             squeeze=False)
    mesh = None
    for ax, field, (_, meta, title) in zip(axes[0], fields, panels):
        mesh = _draw_field(ax, field, meta, tile, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
    fig.colorbar(mesh, ax=list(axes[0]), label=label, shrink=0.85)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _panel_title(meta: dict, group: str) -> str:
    """'<run name> <METHOD>' from results/<name>/<ts>/<i>_<method>.h5."""
    parts = os.path.normpath(os.path.abspath(meta["source"])).split(os.sep)
    run = parts[-3] if len(parts) >= 3 else os.path.basename(meta["source"])
    method = (meta.get("method") or "?").upper()
    return f"{run} — {method} ('{group}')"


def plot_profile_1d(rho: np.ndarray, err: np.ndarray, meta: dict, group: str,
                    out_png: str, amplitude: bool = False) -> None:
    plt = _require_pyplot()
    x = _centers(meta["edges"])
    y, label = (np.sqrt(np.clip(rho, 0.0, None)), r"$\sqrt{\rho} \sim |\psi|$") \
        if amplitude else (rho, r"$\rho(x)$")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=0)
    ax.plot(x, y, color=GROUP_COLORS[0], lw=1.3)
    if not amplitude:
        ax.fill_between(x, y - err, y + err, color=GROUP_COLORS[0], alpha=0.25,
                        lw=0)
    ax.set_xlabel(r"$x\ /\ a_B^*$")
    ax.set_ylabel(label)
    ax.set_title(f"density — group '{group}'", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_radial(curves: List[dict], out_png: str) -> None:
    """curves: [{r, rho, err, color, ls, label}] — color = group, linestyle =
    method (solid DMC, dashed VMC, dotted pure)."""
    plt = _require_pyplot()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=0)
    for cu in curves:
        ax.errorbar(cu["r"], cu["rho"], yerr=cu["err"], color=cu["color"],
                    ls=cu["ls"], lw=1.3, marker="o", ms=3.0, capsize=2,
                    ecolor="0.5", label=cu["label"])
    ax.set_xlabel(r"$|\mathbf{r}|\ /\ a_B^*$")
    ax.set_ylabel(r"$\rho(|\mathbf{r}|)$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_walkers(positions: np.ndarray, meta: dict, out_png: str) -> None:
    """Quick-look scatter of a checkpoint's walker cloud (one point per
    particle per walker, color = particle index).  The cloud samples the
    mixed distribution, so its shape previews the density estimator's image
    at far lower statistics."""
    plt = _require_pyplot()
    nw, n_particles, dim = positions.shape
    tag = f" [{meta['tag']}]" if meta.get("tag") else ""

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    if dim == 1:
        for p in range(n_particles):
            ax.hist(positions[:, p, 0], bins=50, histtype="step",
                    color=GROUP_COLORS[p % len(GROUP_COLORS)],
                    label=f"particle {p}")
        ax.set_xlabel(r"$x\ /\ a_B^*$")
        ax.set_ylabel("walkers")
    else:
        alpha = float(np.clip(2000.0 / max(nw, 1), 0.05, 0.8))
        for p in range(n_particles):
            ax.scatter(positions[:, p, 0], positions[:, p, 1], s=4.0,
                       alpha=alpha, lw=0,
                       color=GROUP_COLORS[p % len(GROUP_COLORS)],
                       label=f"particle {p}")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x\ /\ a_B^*$")
        ax.set_ylabel(r"$y\ /\ a_B^*$")
        leg = ax.legend(frameon=False, fontsize=9, markerscale=3)
        for h in leg.legend_handles:
            h.set_alpha(1.0)
    ax.set_title(f"walker cloud{tag} — {nw} walkers "
                 r"($\propto \psi_T \Phi$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ── self-test ───────────────────────────────────────────────────────────────

def _selftest() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    # Synthetic 2-D unit gaussian on the estimator's grid convention.
    n, rmax = 64, 4.0
    edges = np.linspace(-rmax, rmax, n + 1)
    c = _centers(edges)
    X, Y = np.meshgrid(c, c, indexing="ij")
    rho = np.exp(-(X ** 2 + Y ** 2) / 2.0) / (2.0 * np.pi)
    err = 0.05 * rho
    dv = (edges[1] - edges[0]) ** 2

    integral = rho.sum() * dv
    check("gaussian integrates to 1", abs(integral - 1.0) < 1e-2,
          f"integral = {integral:.4f}")

    r, rho_r, err_r = radial_profile(rho, err, edges)
    check("radial axis increasing", bool(np.all(np.diff(r) > 0)))
    check("radial profile decreasing",
          rho_r[0] > rho_r[len(r) // 2] > rho_r[-1],
          f"rho(0)={rho_r[0]:.4g}, rho(rmax)={rho_r[-1]:.4g}")
    check("radial profile matches gaussian at origin",
          abs(rho_r[0] - rho[n // 2, n // 2]) / rho[n // 2, n // 2] < 0.05)

    pure, perr = extrapolate((rho, err), (rho, err))  # identical runs
    check("extrapolation identity 2*rho - rho = rho",
          bool(np.allclose(pure, rho)))
    check("extrapolation error in quadrature",
          bool(np.allclose(perr, np.sqrt(5.0) * err)))

    # Flat-layout convention: cell = ix*n + iy reshapes to rho[ix, iy].
    flat = np.zeros(16)
    flat[2 * 4 + 3] = 7.0
    check("row-major reshape convention",
          np.reshape(flat, (4, 4))[2, 3] == 7.0)

    # Periodic: hexagonal cell a1 = (L, 0), a2 = (L/2, L*sqrt(3)/2).
    L = 2.0
    pmeta = {"dim": 2, "n_bins": 4, "periodic": True,
             "lattice": np.array([[L, 0.0], [0.5 * L, 0.5 * np.sqrt(3.0) * L]])}
    X, Y, idx = periodic_mesh(pmeta, tile=1)
    check("periodic corner f=(-1/2,-1/2) -> -(a1+a2)/2",
          np.allclose([X[0, 0], Y[0, 0]], -0.5 * pmeta["lattice"].sum(axis=0)))
    check("periodic corner f=(+1/2,-1/2) -> (a1-a2)/2",
          np.allclose([X[-1, 0], Y[-1, 0]],
                      0.5 * (pmeta["lattice"][0] - pmeta["lattice"][1])))
    check("periodic cell volume = |det A| / n^2",
          np.isclose(cell_volume(pmeta), 0.5 * np.sqrt(3.0) * L * L / 16.0))
    _, _, idx2 = periodic_mesh(pmeta, tile=2)
    check("tiling repeats the bin index", list(idx2) == [0, 1, 2, 3] * 2)

    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ── CLI ─────────────────────────────────────────────────────────────────────

def _out_path(out_dir: Optional[str], src: str, suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(src))[0]
    base = out_dir if out_dir else (os.path.dirname(os.path.abspath(src)))
    return os.path.join(base, f"{stem}_{suffix}.png")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Render the QMC wavefunction density from an Alea result "
                    "file: per-group heatmaps of the {type: density} estimator "
                    "(the walker cloud, accumulated over the whole run), plus "
                    "optional radial profile, sqrt(rho) amplitude, and the "
                    "2*DMC - VMC pure-estimator extrapolation.")
    p.add_argument("path", nargs="?",
                   help="Alea result .h5 with density_* datasets "
                        "(results/<name>/<ts>/<i>_dmc.h5).")
    p.add_argument("--out", metavar="DIR",
                   help="directory for the PNGs (default: next to the input).")
    p.add_argument("--radial", action="store_true",
                   help="also write the azimuthally averaged profile rho(|r|).")
    p.add_argument("--amplitude", action="store_true",
                   help="plot sqrt(rho) ~ |psi| instead of rho (nodeless "
                        "states; for the DMC mixed density: sqrt(psi_T*Phi)).")
    p.add_argument("--slice", action="store_true", dest="slice_z",
                   help="3-D grids: show the z~0 midplane slice instead of "
                        "the z-integrated column density (preserves radial "
                        "structure like shell nodes).")
    p.add_argument("--extrapolate", metavar="VMC_H5",
                   help="VMC result on the same grid: also render the pure "
                        "estimator 2*rho_DMC - rho_VMC.")
    p.add_argument("--walkers", metavar="WK_H5",
                   help="walker checkpoint (walkers_out): quick-look scatter "
                        "of the cloud; works without a result file.")
    p.add_argument("--tile", type=int, default=1, metavar="N",
                   help="periodic cells: repeat the cell N x N (e.g. 2 or 3) so "
                        "a crystal's lattice is visible across the boundaries.")
    p.add_argument("--compare", metavar="B_H5",
                   help="second result .h5: write side-by-side heatmaps of the "
                        "shared groups on ONE colour scale (e.g. fluid vs crystal).")
    p.add_argument("--selftest", action="store_true",
                   help="run the synthetic validation and exit.")
    args = p.parse_args(argv)
    if args.tile < 1:
        p.error("--tile must be >= 1")

    if args.selftest:
        return _selftest()
    if not args.path and not args.walkers:
        p.error("a result .h5 is required (or --walkers / --selftest)")
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    if args.walkers:
        pos, wmeta = load_walkers(args.walkers)
        png = _out_path(args.out, args.walkers, "walkers")
        plot_walkers(pos, wmeta, png)
        print(f"wrote {png}  ({wmeta['n_walkers']} walkers, "
              f"{wmeta['n_particles']} particles, dim {wmeta['dim']})")
        if not args.path:
            return 0

    groups, meta = load_density(args.path)
    dv = cell_volume(meta)
    print(f"{args.path}: method={meta['method']}, dim={meta['dim']}, "
          f"{meta['n_bins']}^{meta['dim']} cells, rmax={meta['rmax']}, "
          f"{meta['n_blocks']} blocks")

    if meta["periodic"]:
        print(f"  periodic cell, lattice vectors (bohr): "
              f"{np.array2string(meta['lattice'], precision=2)}")
        if args.radial:
            print("  --radial skipped: an average about the cell origin is "
                  "meaningless under PBC (use the gofr estimator instead)")
            args.radial = False

    if args.compare:
        b_groups, b_meta = load_density(args.compare)
        shared = sorted(set(groups) & set(b_groups))
        if not shared:
            raise SystemExit(f"--compare: no shared density groups between "
                             f"{args.path} and {args.compare}")
        for group in shared:
            png = _out_path(args.out, args.path, f"compare_{group}")
            plot_compare([(groups[group][0], meta, _panel_title(meta, group)),
                          (b_groups[group][0], b_meta, _panel_title(b_meta, group))],
                         png, args.amplitude, args.tile)
            print(f"wrote {png}")
        return 0

    vmc_groups = None
    if args.extrapolate:
        vmc_groups, vmc_meta = load_density(args.extrapolate)
        _check_same_grid(meta, vmc_meta)

    radial_curves: List[dict] = []
    for gi, (group, (rho, err)) in enumerate(sorted(groups.items())):
        color = GROUP_COLORS[gi % len(GROUP_COLORS)]
        # Sanity: the integral is the mean particle count inside the grid.
        print(f"  group '{group}': integral rho dV = {rho.sum() * dv:.4f}")

        png = _out_path(args.out, args.path, f"density_{group}")
        if meta["dim"] == 1:
            plot_profile_1d(rho, err, meta, group, png, args.amplitude)
        else:
            plot_heatmap(rho, meta, group, png, args.amplitude,
                         slice_z=args.slice_z, tile=args.tile)
        print(f"wrote {png}")

        if args.radial and meta["dim"] >= 2:
            r, rho_r, err_r = radial_profile(rho, err, meta["edges"])
            radial_curves.append(dict(r=r, rho=rho_r, err=err_r, color=color,
                                      ls="-", label=f"{group} (DMC)"))

        if vmc_groups is not None and group in vmc_groups:
            pure, perr = extrapolate((rho, err), vmc_groups[group])
            png = _out_path(args.out, args.path, f"pure_{group}")
            if meta["dim"] == 1:
                plot_profile_1d(pure, perr, meta, group, png, args.amplitude)
            else:
                plot_heatmap(pure, meta, group, png, args.amplitude,
                             title_extra="extrapolated pure ",
                             slice_z=args.slice_z, tile=args.tile)
            print(f"wrote {png}  (pure integral = {pure.sum() * dv:.4f})")

            if args.radial and meta["dim"] >= 2:
                v_rho, v_err = vmc_groups[group]
                r, y, e = radial_profile(v_rho, v_err, meta["edges"])
                radial_curves.append(dict(r=r, rho=y, err=e, color=color,
                                          ls="--", label=f"{group} (VMC)"))
                r, y, e = radial_profile(pure, perr, meta["edges"])
                radial_curves.append(dict(r=r, rho=y, err=e, color=color,
                                          ls=":", label=f"{group} (pure)"))

    if radial_curves:
        png = _out_path(args.out, args.path, "radial")
        plot_radial(radial_curves, png)
        print(f"wrote {png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
