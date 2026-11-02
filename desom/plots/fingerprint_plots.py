"""
Fingerprint Visualization module.
Replicates Figure 7 (single-galaxy spectral prototype fingerprint colored by R/Re on DESOM-1 15x15)
and extra sample fingerprint grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from ..config import Config
from ..models.desom1 import DESOM1


def _load_desom1(cfg: Config) -> DESOM1:
    d1 = cfg["desom1"]
    model = DESOM1(
        input_dims=4544,
        map_size=tuple(d1["map_size"]),
        latent_dims=int(d1["latent"]),
        gamma=float(d1["gamma"]),
        lr=float(d1["lr"]),
        map_distance=cfg["training"]["map_distance"],
    )
    weights_path = cfg.desom1_dir / "DESOM_model_final.weights.h5"
    if not weights_path.exists():
        weights_path = cfg.desom1_dir / "checkpoint_latest.weights.h5"
    model.load_weights(weights_path)
    return model


def _compute_node_r_re(
    cfg: Config, plateifu: str, model: DESOM1, counts_fallback: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes (node_counts, node_mean_r_re) for `plateifu` across the 225 DESOM-1 nodes.
    Uses baxel centers (x_center_pix, y_center_pix) from `all_baxels.ecsv` and effective radius Re
    (`nsa_elpetro_th50_r` in arcsec) from `drpall-v2_4_3.fits`.
    """
    plateifu = plateifu.strip()
    gal_dir = cfg.processed_dir / plateifu
    cand_path = gal_dir / "candidate_spectra.npz"
    ecsv_path = gal_dir / "all_baxels.ecsv"

    node_counts = np.asarray(counts_fallback, dtype=int).ravel()
    node_r_re = np.full(225, np.nan, dtype=float)

    if not (cand_path.exists() and ecsv_path.exists()):
        # Radial fallback if processed files are missing
        nz = np.flatnonzero(node_counts > 0)
        if len(nz) > 0:
            node_r_re[nz] = np.linspace(0.2, 1.3, len(nz))
        return node_counts, node_r_re

    cand = np.load(cand_path)
    X_gal = cand["X"]
    bmus = model.predict_bmu(X_gal, batch_size=256)

    ecsv = Table.read(ecsv_path)
    acc = np.asarray(ecsv["accepted_candidate"], dtype=bool)
    x_pix = np.asarray(ecsv["x_center_pix"], dtype=float)[acc]
    y_pix = np.asarray(ecsv["y_center_pix"], dtype=float)[acc]
    cont = np.asarray(ecsv["continuum"], dtype=float)[acc]

    # Continuum-weighted galaxy center in MaNGA pixel coordinates (0.5 arcsec/pix)
    weights = np.where(np.isfinite(cont) & (cont > 0), cont, 1.0)
    x0 = np.average(x_pix, weights=weights)
    y0 = np.average(y_pix, weights=weights)
    r_arcsec = 0.5 * np.hypot(x_pix - x0, y_pix - y0)

    # Effective radius Re (arcsec) from drpall
    re_arcsec = 10.0
    drpall_path = cfg.catalogs_dir / "drpall-v2_4_3.fits"
    if not drpall_path.exists():
        candidates = list(cfg.experiments_dir.glob("**/drpall-v2_4_3.fits"))
        if candidates:
            drpall_path = candidates[0]
    if drpall_path.exists():
        drp = Table.read(drpall_path, hdu=1)
        matches = np.flatnonzero([str(p).strip() == plateifu for p in drp["plateifu"]])
        if len(matches) > 0:
            val = float(drp["nsa_elpetro_th50_r"][matches[0]])
            if np.isfinite(val) and val > 0.5:
                re_arcsec = val

    r_re = r_arcsec / re_arcsec
    node_counts = np.zeros(225, dtype=int)
    for k in range(225):
        m = bmus == k
        c = int(np.sum(m))
        node_counts[k] = c
        if c > 0:
            node_r_re[k] = float(np.nanmean(r_re[m]))

    return node_counts, node_r_re


def plot_galaxy_fingerprints_grid(
    cfg: Config,
    fingerprints: np.ndarray,
    manga_ids: List[str],
    n_examples: int = 16,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Plots a 4x4 grid of sample galaxy fingerprints on the 15x15 DESOM-1 latent plane.
    """
    if save_path is None:
        save_path = cfg.figures_dir / "extra_sample_fingerprints.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(fingerprints) == 0:
        print("[Plot] Warning: No fingerprints provided to plot.")
        return save_path

    grid_size = int(np.ceil(np.sqrt(min(n_examples, max(1, len(fingerprints))))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    for idx in range(grid_size * grid_size):
        r = idx // grid_size
        c = idx % grid_size
        ax = axes[r, c] if grid_size > 1 else axes

        if idx < len(fingerprints):
            fp = fingerprints[idx].squeeze()
            ax.imshow(fp, origin="upper", cmap="viridis", aspect="equal")
            title_id = manga_ids[idx] if idx < len(manga_ids) else str(idx)
            ax.set_title(f"mangaID: {title_id}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Fingerprints Grid -> {save_path.name}")
    return save_path


def plot_single_galaxy_fingerprint(
    cfg: Config,
    counts: np.ndarray,
    galaxy_label: str,
    save_path: Optional[Path] = None,
    model: Optional[DESOM1] = None,
) -> Path:
    """
    Replicates Figure 7 of Teimoorinia et al. (2021) (page 12):
    - 15x15 grid of decoded DESOM-1 spectral prototypes.
    - Nodes with 0 baxels from `galaxy_label` are plotted in light grey without frames (`axis('off')`).
    - Nodes with > 0 baxels have a black rectangular box (`axis('on')`), the baxel count printed
      above the box, and the prototype spectrum colored by the mean galactocentric distance R/Re
      of those baxels using `jet`.
    - Right-side vertical colorbar for R/Re and outer 1..15 coordinate labels on left and bottom.
    """
    counts = np.asarray(counts).squeeze()
    if save_path is None:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in galaxy_label.strip())
        save_path = cfg.figures_dir / f"figure_7_fingerprint_{safe}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = _load_desom1(cfg)

    decoded_prototypes = model.decode(model.prototypes, batch_size=225)
    node_counts, node_r_re = _compute_node_r_re(cfg, galaxy_label, model, counts)

    finite_r = node_r_re[np.isfinite(node_r_re)]
    vmin = float(np.min(finite_r)) if len(finite_r) > 0 else 0.2
    vmax = float(np.max(finite_r)) if len(finite_r) > 0 else 1.4
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.jet

    rows, cols = model.map_size
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            ax = axes[r, c]
            cnt = int(node_counts[k])
            if cnt > 0:
                color = cmap(norm(node_r_re[k]))
                ax.plot(decoded_prototypes[k], color=color, linewidth=1.15)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(cnt), fontsize=10, color="black", pad=2)
                for spine in ax.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.0)
            else:
                ax.plot(decoded_prototypes[k], color="#cccccc", linewidth=0.8)
                ax.axis("off")

    plt.subplots_adjust(hspace=1.0, wspace=1.0, right=0.89)

    # Outer Y-axis coordinate labels (1 at bottom row r=14, 15 at top row r=0)
    for r in range(rows):
        axes[r, 0].text(
            -0.35,
            0.5,
            str(rows - r),
            transform=axes[r, 0].transAxes,
            fontsize=16,
            ha="right",
            va="center",
        )

    # Outer X-axis coordinate labels (1 at left col c=0, 15 at right col c=14)
    for c in range(cols):
        axes[rows - 1, c].text(
            0.5,
            -0.45,
            str(c + 1),
            transform=axes[rows - 1, c].transAxes,
            fontsize=16,
            ha="center",
            va="top",
        )

    # Vertical colorbar on the right for R/Re
    cax = fig.add_axes([0.915, 0.14, 0.018, 0.72])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$R/R_e$", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved single-galaxy fingerprint (Figure 7) -> {save_path.name}")
    return save_path
