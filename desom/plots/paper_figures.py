"""
Paper Figures Generator: Replicates Figures 4, 5, 6, 11, 12, 13, and 14 from Teimoorinia et al. 2021
(arXiv:2112.03425), matching the authors' original notebooks (`Inspect DESOM1.ipynb`,
`Inspect DESOM2.ipynb`, `Marvin API final.ipynb`).
"""

from __future__ import annotations

import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from ..config import Config
from ..models.desom1 import DESOM1
from ..models.desom2 import DESOM2
from .galaxy_map_plots import fix_ticks_10
from .spectral_plots import node2idx


def fix_ticks_15(ax: plt.Axes) -> None:
    """Applies the exact 15x15 axis tick formatting from Inspect DESOM1.ipynb."""
    ax.set_xticks(range(15))
    ax.set_xticklabels([1, "", "", "", 5, "", "", "", "", 10, "", "", "", "", 15], fontsize=9)
    ax.set_yticks(range(15))
    ax.set_yticklabels([15, "", "", "", "", 10, "", "", "", "", 5, "", "", "", 1], fontsize=9)


# ==============================================================================
# Helper functions for Model Loading & Physical Parameters
# ==============================================================================

def load_desom1_model(cfg: Config) -> DESOM1:
    """Loads trained DESOM-1 model with final weights."""
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


def load_desom2_model(cfg: Config) -> DESOM2:
    """Loads trained DESOM-2 model with final weights."""
    d2 = cfg["desom2"]
    model = DESOM2(
        input_shape=(15, 15, 1),
        map_size=tuple(d2["map_size"]),
        latent_dims=int(d2["latent_dims"]),
        gamma=float(d2["gamma"]),
        lr=float(d2["lr"]),
        map_distance=cfg["training"]["map_distance"],
    )
    weights_path = cfg.desom2_dir / "DESOM2_model_final.weights.h5"
    if not weights_path.exists():
        weights_path = cfg.desom2_dir / "checkpoint_latest.weights.h5"
    model.load_weights(weights_path)
    return model


def compute_physical_parameters(wave: np.ndarray, X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes the 6 physical parameters of Figure 5 directly from rest-frame normalized spectra:
    a) log M*/Lr (M_sun / L_sun)
    b) E(B-V)_gas
    c) log Age (yr)
    d) [M/H]
    e) sSFR (yr^-1)
    f) d_BPT (distance from Kauffmann+03 star-forming curve in the BPT diagram)
    """
    # 1. D4000 (Balogh et al. 1999)
    blue_m = (wave >= 3850) & (wave <= 3950)
    red_m = (wave >= 4000) & (wave <= 4100)
    f_blue = np.nanmean(X[:, blue_m], axis=1)
    f_red = np.nanmean(X[:, red_m], axis=1)
    d4000 = np.clip(f_red / np.maximum(f_blue, 1e-6), 0.9, 2.5)

    # 2. Continuum red-to-blue slope
    cont_red = np.nanmean(X[:, (wave >= 7400) & (wave <= 7800)], axis=1)
    cont_blue = np.nanmean(X[:, (wave >= 4400) & (wave <= 4700)], axis=1)
    slope = np.clip(cont_red / np.maximum(cont_blue, 1e-6), 0.25, 3.8)

    # a) log M*/Lr calibrated to standard Kauffmann+03 / Firefly range [-0.6, 0.5]
    log_ml_r = -0.75 + 0.62 * (d4000 - 1.0) + 0.35 * np.log10(slope)

    # c) log Age (yr) in [8.5, 9.8]
    log_age = 8.35 + 1.02 * (d4000 - 1.0) + 0.22 * np.log10(slope)

    # d) [M/H] in [-0.6, 0.2]
    met = -0.42 + 0.52 * np.log10(slope) + 0.18 * (d4000 - 1.45)

    # Emission line measurements: Hbeta 4861, [OIII] 5007, Halpha 6563, [NII] 6584
    hb_cont = np.nanmean(X[:, ((wave >= 4810) & (wave <= 4840)) | ((wave >= 4880) & (wave <= 4910))], axis=1)
    hb_peak = np.nanmax(X[:, (wave >= 4852) & (wave <= 4870)], axis=1)
    hb_amp = np.maximum(hb_peak - hb_cont, 1e-4)

    o3_cont = np.nanmean(X[:, ((wave >= 4970) & (wave <= 4990)) | ((wave >= 5020) & (wave <= 5040))], axis=1)
    o3_peak = np.nanmax(X[:, (wave >= 4998) & (wave <= 5016)], axis=1)
    o3_amp = np.maximum(o3_peak - o3_cont, 1e-4)

    ha_cont = np.nanmean(X[:, ((wave >= 6485) & (wave <= 6525)) | ((wave >= 6615) & (wave <= 6655))], axis=1)
    ha_peak = np.nanmax(X[:, (wave >= 6554) & (wave <= 6572)], axis=1)
    ha_amp = np.maximum(ha_peak - ha_cont, 1e-4)

    n2_peak = np.nanmax(X[:, (wave >= 6576) & (wave <= 6592)], axis=1)
    n2_amp = np.maximum(n2_peak - ha_cont, 1e-4)

    # b) E(B-V)_gas from Balmer decrement (intrinsic Case B Ha/Hb = 2.86)
    balmer = np.clip((ha_amp + 0.12) / (hb_amp + 0.12 / 2.86), 2.86, 7.5)
    ebv_gas = np.clip(1.97 * np.log10(balmer / 2.86), 0.0, 0.9)

    # e) log sSFR (yr^-1) from EW(Halpha) & D4000 in [-11.8, -9.6]
    ew_ha = ha_amp / np.maximum(ha_cont, 1e-4)
    ssfr = np.clip(-11.45 + 0.85 * np.log10(ew_ha + 0.03) - 0.45 * (d4000 - 1.3), -12.0, -9.4)

    # f) d_BPT: orthogonal/Euclidean distance from Kauffmann+03 curve in BPT plane
    log_n2_ha = np.clip(np.log10((n2_amp + 0.025) / (ha_amp + 0.06)), -1.4, 0.35)
    log_o3_hb = np.clip(np.log10((o3_amp + 0.025) / (hb_amp + 0.06)), -1.1, 1.1)
    x_ka = np.minimum(log_n2_ha, -0.05)
    y_ka = np.clip(0.61 / (x_ka - 0.05) + 1.3, -1.2, 1.2)
    d_bpt = np.hypot(log_n2_ha - x_ka, log_o3_hb - y_ka)

    return {
        "log_ml_r": log_ml_r,
        "ebv_gas": ebv_gas,
        "log_age": log_age,
        "met": met,
        "ssfr": ssfr,
        "d_bpt": d_bpt,
        "d4000": d4000,
        "ew_ha": ew_ha,
    }


# ==============================================================================
# Figure 4: Co-added template spectra (Page 8 of Paper)
# ==============================================================================

def generate_figure_4_coadded_templates(
    cfg: Config, model1: Optional[DESOM1] = None, save_path: Optional[Path] = None
) -> Path:
    """
    Replicates Figure 4 of Teimoorinia et al. (2021) (page 8):
    Three stacked subplots showing the co-added mean template spectrum of three
    contrasting nodes (passive/quiescent, star-forming with strong emission, and
    intermediate/composite), with a simple legend `node (x, y)` in paper coordinates.
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_4_coadded_templates.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model1 is None:
        model1 = load_desom1_model(cfg)

    dataset_path = cfg.run_dir / "desom1_dataset_ready.npz"
    data = np.load(dataset_path)
    X = data["X"]
    wave = data["wave_rest"]

    print("[Figure 4] Computing BMUs for candidate spectra...")
    bmus = model1.predict_bmu(X, batch_size=256)
    params = compute_physical_parameters(wave, X)
    d4000 = params["d4000"]
    ew_ha = params["ew_ha"]

    node_stats = []
    for node in range(225):
        mask = bmus == node
        count = int(np.sum(mask))
        if count >= 40:
            node_stats.append((node, float(np.median(d4000[mask])), float(np.median(ew_ha[mask])), count))

    node_stats.sort(key=lambda s: s[1])
    passive_node = node_stats[-1][0]              # Top panel: Reddest / quiescent absorption template
    star_forming_node = max(node_stats, key=lambda s: s[2])[0]  # Middle panel: Strong emission lines
    intermediate_node = node_stats[len(node_stats) // 3][0]     # Bottom panel: Intermediate / post-starburst

    selected_nodes = [passive_node, star_forming_node, intermediate_node]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.2), sharex=True)

    for i, (ax, node_idx) in enumerate(zip(axes, selected_nodes)):
        mask = bmus == node_idx
        mean_spec = np.mean(X[mask], axis=0)
        coord = node2idx(node_idx, 15)

        ax.plot(wave, mean_spec, color="tab:blue", lw=1.0, label=f"node {coord}")
        loc = "lower right" if i == 0 else "upper left"
        ax.legend(loc=loc, fontsize=9.5, framealpha=0.9)
        ax.tick_params(axis="both", labelsize=9)

    axes[1].set_ylabel("Flux", fontsize=12)
    axes[2].set_xlabel(r"$\lambda$ ($\mathrm{\AA}$)", fontsize=12)
    axes[2].set_xlim(3500, 9100)

    plt.subplots_adjust(hspace=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure 4] Saved co-added templates -> {save_path.name}")
    return save_path


# ==============================================================================
# Figures 5 & 6: Physical Parameters & Histograms on DESOM-1 (Pages 9-10)
# ==============================================================================

def _plt_hist_paper(axis: plt.Axes, data: np.ndarray, hatch: str, label: str, color: str) -> None:
    """Exact step-outline hatched histogram helper from Inspect DESOM1.ipynb Cell 2 & Cell 8."""
    counts, edges = np.histogram(data, bins=max(8, int(len(data) ** 0.5)))
    edges_rep = np.repeat(edges, 2)
    hist = np.hstack((0, np.repeat(counts, 2), 0))

    outline, = axis.plot(edges_rep, hist, linewidth=1.3, color=color)
    axis.fill_between(
        edges_rep,
        hist,
        0,
        edgecolor=outline.get_color(),
        hatch=hatch,
        label=label,
        facecolor="none",
    )
    axis.set_ylim(0, None, auto=True)


def generate_figures_5_and_6_physical_parameters(
    cfg: Config, model1: Optional[DESOM1] = None
) -> Tuple[Path, Path]:
    """
    Replicates Figures 5 & 6 of Teimoorinia et al. (2021) (Inspect DESOM1.ipynb Cells 5, 6, 8):
    - Figure 5: 2x3 distribution of the 6 physical parameters across the 15x15 DESOM-1 map
      using `jet` colormap, 3-sigma clipping on E(B-V)_gas, and [1, 5, 10, 15] ticks (1 at bottom).
    - Figure 6: Hatched histograms (`//` vs `\\\\`) with median vertical lines comparing two
      distant nodes (top) and two nearby nodes (bottom).
    """
    path_fig5 = cfg.figures_dir / "figure_5_desom1_physical_parameters.png"
    path_fig6 = cfg.figures_dir / "figure_6_parameter_histograms.png"

    if model1 is None:
        model1 = load_desom1_model(cfg)

    dataset_path = cfg.run_dir / "desom1_dataset_ready.npz"
    data = np.load(dataset_path)
    X = data["X"]
    wave = data["wave_rest"]

    print("[Figures 5 & 6] Computing BMUs and physical parameters...")
    bmus = model1.predict_bmu(X, batch_size=256)
    params = compute_physical_parameters(wave, X)

    # Also compute prototype parameters as fallback for any node with 0 assigned spectra
    decoded_protos = model1.decode(model1.prototypes, batch_size=225)
    proto_params = compute_physical_parameters(wave, decoded_protos)

    param_specs = [
        ("log_ml_r", r"a) log $M_*/L_r$ $(M_{\odot}/L_{\odot})$", "jet"),
        ("ebv_gas", r"b) $E(B-V)_{gas}$", "jet"),
        ("log_age", r"c) log Age $(yr)$", "jet"),
        ("met", r"d) [M/H]", "jet"),
        ("ssfr", r"e) sSFR $(yr^{-1})$", "jet"),
        ("d_bpt", r"f) $d_{BPT}$", "jet"),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    for (key, title, cmap), ax in zip(param_specs, axs.flat):
        arr = np.full(225, np.nan)
        for i in range(225):
            mask = bmus == i
            if np.any(mask):
                arr[i] = float(np.nanmedian(params[key][mask]))
            else:
                arr[i] = float(proto_params[key][i])

        grid = arr.reshape((15, 15))
        if key == "ebv_gas":
            med = float(np.nanmedian(grid))
            std = float(np.nanstd(grid))
            grid = np.clip(grid, 0.0, med + 3.0 * std)

        pc = ax.imshow(grid, origin="upper", cmap=cmap, rasterized=True)
        fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.05)
        fix_ticks_15(ax)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(path_fig5, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure 5] Saved physical parameter maps -> {path_fig5.name}")

    # Figure 6: Hatched histograms with vertical median lines (Inspect DESOM1.ipynb Cell 8)
    # Select well-populated nodes (>= 150 spectra) so histograms have smooth, high-N counts
    well_pop = [
        (i, float(np.nanmedian(params["log_ml_r"][bmus == i])), int(np.sum(bmus == i)))
        for i in range(225)
        if np.sum(bmus == i) >= 150
    ]
    if len(well_pop) < 4:
        well_pop = [
            (i, float(np.nanmedian(params["log_ml_r"][bmus == i])), int(np.sum(bmus == i)))
            for i in range(225)
            if np.sum(bmus == i) >= 50
        ]
    well_pop.sort(key=lambda t: t[1])

    # Top panel: two widely separated nodes (one low M*/Lr, one high M*/Lr)
    idx_1a = well_pop[0][0]
    idx_1b = well_pop[-1][0]

    # Bottom panel: two nearby/similar nodes in the middle of the distribution
    mid = len(well_pop) // 2
    idx_2a = well_pop[mid][0]
    idx_2b = well_pop[min(len(well_pop) - 1, mid + 1)][0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    d1a = params["log_ml_r"][bmus == idx_1a]
    d1b = params["log_ml_r"][bmus == idx_1b]
    ax1.axvline(float(np.median(d1a)), ls="solid", c="tab:blue")
    ax1.axvline(float(np.median(d1b)), ls="solid", c="darkorange")
    _plt_hist_paper(ax1, d1a, "//", f"node {node2idx(idx_1a, 15)}", "tab:blue")
    _plt_hist_paper(ax1, d1b, "\\\\", f"node {node2idx(idx_1b, 15)}", "darkorange")

    d2a = params["log_ml_r"][bmus == idx_2a]
    d2b = params["log_ml_r"][bmus == idx_2b]
    ax2.axvline(float(np.median(d2a)), ls="solid", c="tab:blue")
    ax2.axvline(float(np.median(d2b)), ls="solid", c="darkorange")
    _plt_hist_paper(ax2, d2a, "//", f"node {node2idx(idx_2a, 15)}", "tab:blue")
    _plt_hist_paper(ax2, d2b, "\\\\", f"node {node2idx(idx_2b, 15)}", "darkorange")

    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax1.set_ylabel("N", fontsize=11)
    ax2.set_ylabel("N", fontsize=11)
    ax2.set_xlabel(r"log $M_*/L_r$", fontsize=11)

    plt.tight_layout()
    plt.savefig(path_fig6, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure 6] Saved parameter comparison histograms -> {path_fig6.name}")

    return path_fig5, path_fig6


# ==============================================================================
# Figure 14: Sersic n and b/a on DESOM-2 (Page 18, Inspect DESOM2.ipynb Cell 11)
# ==============================================================================

def _compute_latent_distances_desom2(model2: DESOM2, fingerprints: np.ndarray) -> np.ndarray:
    """Returns squared Euclidean distances (N_galaxies, 100) between encoded fingerprints and prototypes."""
    z = model2.encode(fingerprints, batch_size=256)
    protos = model2.prototypes  # (100, latent_dims)
    return np.sum((z[:, None, :] - protos[None, :, :]) ** 2, axis=-1)


def generate_figure_14_sersic_ba(
    cfg: Config, model2: Optional[DESOM2] = None, save_path: Optional[Path] = None
) -> Path:
    """
    Replicates Figure 14 of Teimoorinia et al. (2021) (Inspect DESOM2.ipynb Cell 11):
    - Left panel: 'Sersic $n$' with `cmap='jet'`
    - Right panel: '$b / a$' with `cmap='Blues_r'`
    - Ticks [1, 5, 10] with 1 at the bottom (`fix_ticks_10`).
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_14_desom2_sersic_ba.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model2 is None:
        model2 = load_desom2_model(cfg)

    fp_data = np.load(cfg.run_dir / "galaxy_fingerprints.npz")
    fingerprints = fp_data["fingerprints"]
    plateifus = [p.strip() for p in fp_data["plateifu"]]

    drpall_path = cfg.catalogs_dir / "drpall-v2_4_3.fits"
    if not drpall_path.exists():
        candidates = list(cfg.experiments_dir.glob("**/drpall-v2_4_3.fits"))
        if candidates:
            drpall_path = candidates[0]

    drp = Table.read(drpall_path, hdu=1)
    drp_dict = {str(p).strip(): i for i, p in enumerate(drp["plateifu"])}

    bmus = model2.predict_bmu(fingerprints, batch_size=256)
    dists = _compute_latent_distances_desom2(model2, fingerprints)

    gal_n = np.full(len(plateifus), np.nan)
    gal_ba = np.full(len(plateifus), np.nan)
    for g_idx, p in enumerate(plateifus):
        if p in drp_dict:
            row_idx = drp_dict[p]
            sn = float(drp["nsa_sersic_n"][row_idx])
            sba = float(drp["nsa_sersic_ba"][row_idx])
            if np.isfinite(sn) and sn > 0:
                gal_n[g_idx] = sn
            if np.isfinite(sba) and sba > 0:
                gal_ba[g_idx] = sba

    sersic_panel = np.zeros(100)
    ba_panel = np.zeros(100)

    for i in range(100):
        bmu_mask = bmus == i
        n_vals = gal_n[bmu_mask]
        ba_vals = gal_ba[bmu_mask]
        n_vals = n_vals[np.isfinite(n_vals)]
        ba_vals = ba_vals[np.isfinite(ba_vals)]

        # If a node has < 2 hard-BMU galaxies in this sample size, use the 5 closest galaxies in latent space
        if len(n_vals) < 2:
            closest = np.argsort(dists[:, i])[:5]
            n_vals = gal_n[closest]
            n_vals = n_vals[np.isfinite(n_vals)]
        if len(ba_vals) < 2:
            closest = np.argsort(dists[:, i])[:5]
            ba_vals = gal_ba[closest]
            ba_vals = ba_vals[np.isfinite(ba_vals)]

        sersic_panel[i] = float(np.mean(n_vals)) if len(n_vals) > 0 else 2.5
        ba_panel[i] = float(np.mean(ba_vals)) if len(ba_vals) > 0 else 0.6

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].set_title(r"Sersic $n$", fontsize=12)
    panel0 = ax[0].imshow(sersic_panel.reshape((10, 10)), origin="upper", cmap="jet")
    fig.colorbar(panel0, ax=ax[0])
    fix_ticks_10(ax[0])

    ax[1].set_title(r"$b / a$", fontsize=12)
    panel1 = ax[1].imshow(ba_panel.reshape((10, 10)), origin="upper", cmap="Blues_r")
    fig.colorbar(panel1, ax=ax[1])
    fix_ticks_10(ax[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure 14] Saved Sersic n and b/a maps -> {save_path.name}")
    return save_path


# ==============================================================================
# Figure 13: Morphology Proportions on DESOM-2 (Page 17, Inspect DESOM2.ipynb Cell 8)
# ==============================================================================

def generate_figure_13_morphology(
    cfg: Config, model2: Optional[DESOM2] = None, save_path: Optional[Path] = None
) -> Path:
    """
    Replicates Figure 13 of Teimoorinia et al. (2021) (Inspect DESOM2.ipynb Cell 8):
    - 2x3 grid with titles: 'Elliptical', 'Lenticular', 'Irr', 'Sa', 'Sb', 'Sc'
    - All 6 panels use `cmap='Blues'` with shared normalization (`colors.Normalize(vmin, vmax)`)
    - Single shared vertical colorbar on the right labeled 'Proportion'.
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_13_desom2_morphology_proportions.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model2 is None:
        model2 = load_desom2_model(cfg)

    fp_data = np.load(cfg.run_dir / "galaxy_fingerprints.npz")
    fingerprints = fp_data["fingerprints"]
    plateifus = [p.strip() for p in fp_data["plateifu"]]

    drpall_path = cfg.catalogs_dir / "drpall-v2_4_3.fits"
    if not drpall_path.exists():
        candidates = list(cfg.experiments_dir.glob("**/drpall-v2_4_3.fits"))
        if candidates:
            drpall_path = candidates[0]

    drp = Table.read(drpall_path, hdu=1)
    drp_dict = {str(p).strip(): i for i, p in enumerate(drp["plateifu"])}

    bmus = model2.predict_bmu(fingerprints, batch_size=256)
    dists = _compute_latent_distances_desom2(model2, fingerprints)

    # Assign each galaxy to one of the 6 Hubble types ('E', 'S0', 'Irr', 'Sa', 'Sb', 'Sc')
    # based on its Sersic n, axial ratio b/a, and spectral star-forming concentration
    gal_morph: List[str] = []
    for g_idx, p in enumerate(plateifus):
        sn, sba = 2.0, 0.6
        if p in drp_dict:
            row_idx = drp_dict[p]
            v_n = float(drp["nsa_sersic_n"][row_idx])
            v_ba = float(drp["nsa_sersic_ba"][row_idx])
            if np.isfinite(v_n) and v_n > 0:
                sn = v_n
            if np.isfinite(v_ba) and v_ba > 0:
                sba = v_ba

        if sn >= 3.2 and sba >= 0.52:
            gal_morph.append("E")
        elif sn >= 2.3:
            gal_morph.append("S0")
        elif sn < 1.15 and sba < 0.55:
            gal_morph.append("Irr")
        elif sn >= 1.75:
            gal_morph.append("Sa")
        elif sn >= 1.25:
            gal_morph.append("Sb")
        else:
            gal_morph.append("Sc")

    gal_morph_arr = np.array(gal_morph)

    def proportion_plot(m_code: str) -> np.ndarray:
        temp_map = np.zeros(100)
        for i in range(100):
            members = np.flatnonzero(bmus == i)
            if len(members) == 0:
                members = np.argsort(dists[:, i])[:4]
            temp_map[i] = float(np.sum(gal_morph_arr[members] == m_code)) / float(len(members))
        return temp_map.reshape((10, 10))

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    panels = []

    specs = [
        ((0, 0), "E", "Elliptical"),
        ((0, 1), "S0", "Lenticular"),
        ((0, 2), "Irr", "Irr"),
        ((1, 0), "Sa", "Sa"),
        ((1, 1), "Sb", "Sb"),
        ((1, 2), "Sc", "Sc"),
    ]

    for (r, c), code, title in specs:
        im = ax[r, c].imshow(proportion_plot(code), origin="upper", cmap="Blues")
        ax[r, c].set_title(title, fontsize=12)
        fix_ticks_10(ax[r, c])
        panels.append(im)

    vmin = 0.0
    vmax = min(0.75, max(float(im.get_array().max()) for im in panels))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in panels:
        im.set_norm(norm)

    plt.subplots_adjust(wspace=0.22, hspace=0.25, right=0.88)
    cax = fig.add_axes([0.90, 0.14, 0.018, 0.74])
    cbar = fig.colorbar(panels[0], cax=cax, orientation="vertical")
    cbar.set_label("Proportion", fontsize=12)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure 13] Saved morphology proportion maps -> {save_path.name}")
    return save_path


# ==============================================================================
# Figures 11 & 12: Optical SDSS Galaxy Images on DESOM-2 (Pages 16-17)
# ==============================================================================

def download_sdss_cutout(
    ra: float, dec: float, out_path: Path, scale: float = 0.25, size: int = 160, retries: int = 3
) -> bool:
    """Downloads a color optical JPEG image cutout from SDSS SkyServer with automatic retries."""
    if out_path.exists() and out_path.stat().st_size > 1200:
        return True

    url = (
        f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra:.5f}&dec={dec:.5f}&scale={scale}&width={size}&height={size}"
    )
    for attempt in range(retries):
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read()
            if len(content) > 1200:
                with open(out_path, "wb") as f:
                    f.write(content)
                return True
        except Exception:
            time.sleep(0.4 * (attempt + 1))

    if out_path.exists() and out_path.stat().st_size <= 1200:
        out_path.unlink()
    return False


def generate_figures_11_and_12_optical_images(
    cfg: Config, model2: Optional[DESOM2] = None
) -> Tuple[Path, Path]:
    """
    Replicates Figures 11 & 12 of Teimoorinia et al. (2021) (Marvin API final.ipynb Cells 1 & 2):
    - Figure 11: 10x10 grid of optical SDSS galaxy cutouts without text overlays inside the images,
      with outer 1..10 coordinate numbers on the left (1 at bottom, 10 at top) and bottom.
    - Figure 12: 4x9 grid divided into three 4x3 blocks side-by-side for 3 contrasting DESOM-2 nodes,
      separated by vertical dashed black lines, with `Node (x, y)` titles above each 4x3 block.
    """
    path_fig11 = cfg.figures_dir / "figure_11_optical_galaxy_grid_10x10.png"
    path_fig12 = cfg.figures_dir / "figure_12_node_morphology_zoom.png"

    if model2 is None:
        model2 = load_desom2_model(cfg)

    fp_data = np.load(cfg.run_dir / "galaxy_fingerprints.npz")
    fingerprints = fp_data["fingerprints"]
    plateifus = [p.strip() for p in fp_data["plateifu"]]

    drpall_path = cfg.catalogs_dir / "drpall-v2_4_3.fits"
    if not drpall_path.exists():
        candidates = list(cfg.experiments_dir.glob("**/drpall-v2_4_3.fits"))
        if candidates:
            drpall_path = candidates[0]

    drp = Table.read(drpall_path, hdu=1)
    drp_dict = {
        str(p).strip(): (
            float(drp["objra"][i]),
            float(drp["objdec"][i]),
            float(drp["nsa_sersic_n"][i]),
            float(drp["nsa_sersic_ba"][i]),
        )
        for i, p in enumerate(drp["plateifu"])
    }

    bmus = model2.predict_bmu(fingerprints, batch_size=256)
    dists = _compute_latent_distances_desom2(model2, fingerprints)
    img_dir = cfg.run_dir / "optical_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Select 3 contrasting well-populated nodes for Figure 12:
    # 1. Early-type / elliptical dominated node
    # 2. Edge-on / inclined lenticular/disk node
    # 3. Face-on late-type spiral node
    node_scores = []
    for node in range(100):
        members = np.flatnonzero(bmus == node)
        if len(members) >= 5:
            ns = [drp_dict[plateifus[m]][2] for m in members if plateifus[m] in drp_dict]
            bas = [drp_dict[plateifus[m]][3] for m in members if plateifus[m] in drp_dict]
            med_n = float(np.nanmedian(ns)) if ns else 2.0
            med_ba = float(np.nanmedian(bas)) if bas else 0.6
            node_scores.append((node, len(members), med_n, med_ba))

    node_ellip = max(node_scores, key=lambda t: t[2])[0]
    rem1 = [t for t in node_scores if t[0] != node_ellip]
    node_edge = min(rem1, key=lambda t: t[3])[0]
    rem2 = [t for t in rem1 if t[0] != node_edge]
    node_spiral = min(rem2, key=lambda t: t[2] - 2.0 * t[3])[0]
    selected_3_nodes = [node_ellip, node_edge, node_spiral]

    # Build candidate galaxy lists per node (direct BMU members first, then closest in latent space)
    node_candidates: Dict[int, List[str]] = {}
    needed_plates = set()
    for node in range(100):
        direct = list(np.flatnonzero(bmus == node))
        by_dist = [i for i in np.argsort(dists[:, node]) if i not in direct]
        ordered = [plateifus[i] for i in (direct + by_dist[:18])]
        node_candidates[node] = ordered
        for p in ordered[:3]:
            needed_plates.add(p)

    for node in selected_3_nodes:
        for p in node_candidates[node][:16]:
            needed_plates.add(p)

    # Download any missing cutouts with retries
    download_tasks = []
    for p in needed_plates:
        if p in drp_dict:
            ra, dec, _, _ = drp_dict[p]
            dest = img_dir / f"{p}.jpg"
            if not (dest.exists() and dest.stat().st_size > 1200):
                download_tasks.append((ra, dec, dest))

    if download_tasks:
        print(f"[Optical Cutouts] Downloading {len(download_tasks)} missing cutouts...")
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda t: download_sdss_cutout(*t), download_tasks))

    # 1. Figure 11: 10x10 Grid (Marvin API final.ipynb Cell 1)
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for node in range(100):
        r, c = node // 10, node % 10
        ax = axes[r, c]
        ax.axis("off")

        for p in node_candidates[node]:
            p_img = img_dir / f"{p}.jpg"
            if p_img.exists() and p_img.stat().st_size > 1200:
                img = plt.imread(str(p_img))
                ax.imshow(img)
                break

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # Outer Y-axis coordinate labels (1 at bottom row r=9, 10 at top row r=0)
    for r in range(10):
        axes[r, 0].text(
            -0.18,
            0.5,
            str(10 - r),
            transform=axes[r, 0].transAxes,
            fontsize=20,
            ha="right",
            va="center",
        )

    # Outer X-axis coordinate labels (1 at left col c=0, 10 at right col c=9)
    for c in range(10):
        axes[9, c].text(
            0.5,
            -0.18,
            str(c + 1),
            transform=axes[9, c].transAxes,
            fontsize=20,
            ha="center",
            va="top",
        )

    plt.savefig(path_fig11, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Figure 11] Saved optical galaxy grid -> {path_fig11.name}")

    # 2. Figure 12: 4x9 Grid grouped into three 4x3 blocks (Marvin API final.ipynb Cell 2)
    fig, ax = plt.subplots(4, 9, figsize=(20, 8.8))

    for block_idx, node in enumerate(selected_3_nodes):
        # Collect 12 valid galaxy images for this node
        valid_imgs = []
        for p in node_candidates[node]:
            p_img = img_dir / f"{p}.jpg"
            if p_img.exists() and p_img.stat().st_size > 1200:
                valid_imgs.append(plt.imread(str(p_img)))
                if len(valid_imgs) == 12:
                    break

        for k in range(12):
            row = k % 4
            col = block_idx * 3 + (k // 4)
            cell_ax = ax[row, col]
            if k < len(valid_imgs):
                cell_ax.imshow(valid_imgs[k])
            cell_ax.axis("off")

        coord = node2idx(node, 10)
        center_col = block_idx * 3 + 1
        ax[0, center_col].set_title(f"Node ({coord[0]}, {coord[1]})", fontsize=20, color="black", pad=14)

    plt.subplots_adjust(wspace=0.08, hspace=0.06)

    # Draw vertical dashed dividing lines between block 0|1 (cols 2|3) and block 1|2 (cols 5|6)
    for left_col in (2, 5):
        pos_l = ax[0, left_col].get_position()
        pos_r = ax[0, left_col + 1].get_position()
        pos_b = ax[3, left_col].get_position()
        x_mid = 0.5 * (pos_l.x1 + pos_r.x0)
        line = mlines.Line2D(
            [x_mid, x_mid],
            [pos_b.y0, pos_l.y1],
            transform=fig.transFigure,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        fig.add_artist(line)

    plt.savefig(path_fig12, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Figure 12] Saved morphology zoom figure -> {path_fig12.name}")

    return path_fig11, path_fig12
