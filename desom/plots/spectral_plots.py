"""
Spectral Visualization module.
Replicates Figure 1 (sample normalized spectra) and Figure 3 (DESOM-1 15x15 prototype map)
from Teimoorinia et al. 2021 (arXiv:2112.03425).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..models.desom1 import DESOM1


def node2idx(some_node: int, mapsize: int = 15) -> tuple[int, int]:
    """Paper coordinate convention: (column 1..W, row_from_bottom 1..W)."""
    row = int(some_node // mapsize)
    column = int(some_node % mapsize)
    return (column + 1, mapsize - row)


def _compute_basic_properties(wave: np.ndarray, X: np.ndarray) -> dict[str, np.ndarray]:
    """Computes physical parameter proxies directly from normalized rest-frame spectra."""
    # D4000 (stellar age & metallicity proxy)
    f_blue = np.nanmean(X[:, (wave >= 3850) & (wave <= 3950)], axis=1)
    f_red = np.nanmean(X[:, (wave >= 4000) & (wave <= 4100)], axis=1)
    d4000 = np.clip(f_red / np.maximum(f_blue, 1e-6), 0.8, 2.8)
    log_age = np.clip(8.2 + 1.15 * (d4000 - 1.0), 8.0, 10.1)

    # Continuum slope -> [M/H] proxy
    c_red = np.nanmean(X[:, (wave >= 7400) & (wave <= 7800)], axis=1)
    c_blue = np.nanmean(X[:, (wave >= 4400) & (wave <= 4700)], axis=1)
    slope = np.clip(c_red / np.maximum(c_blue, 1e-6), 0.2, 4.0)
    met = np.clip(-0.45 + 0.45 * np.log10(slope) + 0.15 * (d4000 - 1.4), -0.8, 0.35)

    # Nebular emission lines: Hbeta 4861, [OIII] 5007, Halpha 6563, [NII] 6584
    hb_cont = np.nanmean(X[:, ((wave >= 4810) & (wave <= 4840)) | ((wave >= 4880) & (wave <= 4910))], axis=1)
    hb_line = np.maximum(np.nanmax(X[:, (wave >= 4852) & (wave <= 4870)], axis=1) - hb_cont, 1e-4)

    o3_cont = np.nanmean(X[:, ((wave >= 4970) & (wave <= 4990)) | ((wave >= 5020) & (wave <= 5040))], axis=1)
    o3_line = np.maximum(np.nanmax(X[:, (wave >= 4998) & (wave <= 5016)], axis=1) - o3_cont, 1e-4)

    ha_cont = np.nanmean(X[:, ((wave >= 6490) & (wave <= 6530)) | ((wave >= 6610) & (wave <= 6650))], axis=1)
    ha_line = np.maximum(np.nanmax(X[:, (wave >= 6554) & (wave <= 6572)], axis=1) - ha_cont, 1e-4)
    n2_line = np.maximum(np.nanmax(X[:, (wave >= 6576) & (wave <= 6592)], axis=1) - ha_cont, 1e-4)

    # E(B-V)_gas from Balmer decrement (Ha/Hb / 2.86)
    balmer = np.clip((ha_line + 0.15) / (hb_line + 0.15 / 2.86), 2.86, 8.5)
    ebv_gas = np.clip(1.97 * np.log10(balmer / 2.86), 0.0, 0.95)

    # BPT distance from Kauffmann+03 star-forming ridge
    log_n2_ha = np.clip(np.log10((n2_line + 0.02) / (ha_line + 0.05)), -1.6, 0.4)
    log_o3_hb = np.clip(np.log10((o3_line + 0.02) / (hb_line + 0.05)), -1.2, 1.2)
    x_ka = np.minimum(log_n2_ha, -0.08)
    y_ka = 0.61 / (x_ka - 0.05) + 1.3
    d_bpt = np.hypot(log_n2_ha - x_ka, log_o3_hb - y_ka)

    # Quality mask to reject spectra with sky-subtraction residual spikes at red wavelengths (>8000 A)
    red_tail = X[:, wave >= 8000]
    red_med = np.nanmedian(red_tail, axis=1)
    red_max = np.nanmax(np.abs(red_tail - red_med[:, None]), axis=1)
    clean_mask = (red_max < 0.45) & (np.nanmin(X, axis=1) > 0.05)

    return {
        "log_age": log_age,
        "met": met,
        "ebv_gas": ebv_gas,
        "d_bpt": d_bpt,
        "ha_ew": ha_line / np.maximum(ha_cont, 1e-4),
        "clean_mask": clean_mask,
    }


def pick_sample_indices(
    X: np.ndarray,
    n_samples: int = 9,
    seed: int = 42,
    wave_rest: Optional[np.ndarray] = None,
    model: Optional[DESOM1] = None,
) -> np.ndarray:
    """
    Selects 9 clean, physically representative spectra (S1..S9) following Figure 1 of the paper:
    - S1..S3: Progression of stellar age and metallicity (quiescent / absorption-dominated to younger continuum)
    - S4..S5: Dust attenuation E(B-V)_gas progression
    - S6..S9: Emission-line strength & BPT distance progression
    Also ensures that all 9 selected spectra map to 9 distinct DESOM-1 nodes when a model is available.
    """
    if len(X) <= n_samples:
        return np.arange(len(X))

    if wave_rest is None:
        output_edges = np.geomspace(3650.0, 8950.0, X.shape[1] + 1)
        wave_rest = np.sqrt(output_edges[:-1] * output_edges[1:])

    props = _compute_basic_properties(wave_rest, X)
    clean_idx = np.flatnonzero(props["clean_mask"])
    if len(clean_idx) < n_samples * 5:
        clean_idx = np.arange(len(X))

    bmus = None
    if model is not None:
        bmus = model.predict_bmu(X[clean_idx], batch_size=256)

    # Define 9 target percentile profiles across (log_age, ebv_gas, ha_ew, d_bpt)
    targets = [
        ("log_age", 95, "ha_ew", 10),   # S1: Oldest, high metallicity passive
        ("log_age", 60, "ha_ew", 25),   # S2: Intermediate age
        ("log_age", 15, "ha_ew", 50),   # S3: Youngest stellar population
        ("ebv_gas", 15, "ha_ew", 65),   # S4: Low dust extinction star-forming
        ("ebv_gas", 92, "ha_ew", 65),   # S5: High dust extinction
        ("d_bpt", 15, "ha_ew", 35),     # S6: Low BPT distance, moderate emission
        ("d_bpt", 45, "ha_ew", 75),     # S7: Moderate BPT distance, strong Halpha
        ("d_bpt", 75, "ha_ew", 90),     # S8: High BPT distance, very strong emission
        ("d_bpt", 96, "ha_ew", 97),     # S9: Extreme BPT distance / Seyfert-like emission
    ]

    chosen_global: list[int] = []
    used_nodes: set[int] = set()

    for p1_name, p1_pct, p2_name, p2_pct in targets[:n_samples]:
        v1 = props[p1_name][clean_idx]
        v2 = props[p2_name][clean_idx]
        t1 = np.percentile(v1, p1_pct)
        t2 = np.percentile(v2, p2_pct)
        s1 = np.std(v1) + 1e-6
        s2 = np.std(v2) + 1e-6
        dist = np.hypot((v1 - t1) / s1, 0.6 * (v2 - t2) / s2)
        order = np.argsort(dist)

        for local_i in order:
            g_idx = int(clean_idx[local_i])
            if g_idx in chosen_global:
                continue
            if bmus is not None:
                node = int(bmus[local_i])
                if node in used_nodes:
                    continue
                used_nodes.add(node)
            chosen_global.append(g_idx)
            break

    return np.array(chosen_global, dtype=int)


def plot_figure_1_normalized_spectra(
    cfg: Config,
    X: np.ndarray,
    wave_rest: np.ndarray,
    n_samples: int = 9,
    save_path: Optional[Path] = None,
    sample_indices: Optional[np.ndarray] = None,
    model: Optional[DESOM1] = None,
) -> Path:
    """
    Replicates Figure 1 of Teimoorinia et al. (2021):
    9 vertically stacked subplots (sharex=True), each showing one normalized rest-frame
    spectrum S1..S9 (in red text at upper-left) and a legend box at upper-right displaying
    the physical property illustrated by that spectrum.
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_1_normalized_spectra.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(X) == 0:
        print("[Plot] Warning: No spectra available in X to plot.")
        return save_path

    if sample_indices is None:
        sample_indices = pick_sample_indices(X, n_samples=n_samples, wave_rest=wave_rest, model=model)

    props = _compute_basic_properties(wave_rest, X[sample_indices])

    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(7.5, 10.5), sharex=True)
    if len(sample_indices) == 1:
        axes = [axes]

    for k, (ax, idx) in enumerate(zip(axes, sample_indices), start=1):
        spec = X[idx]
        if k in (1, 2, 3):
            label = f"log Age = {props['log_age'][k-1]:.2f}, [M/H] = {props['met'][k-1]:.2f}"
        elif k in (4, 5):
            label = f"E(B-V) gas = {props['ebv_gas'][k-1]:.2f}"
        else:
            label = f"BPT distance = {props['d_bpt'][k-1]:.2f}"

        ax.plot(wave_rest, spec, color="tab:blue", lw=0.85, label=label)
        ax.text(
            0.03,
            0.70,
            f"S{k}",
            transform=ax.transAxes,
            color="red",
            fontsize=11,
            ha="left",
            va="center",
        )
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, handlelength=1.5)
        ax.tick_params(axis="both", labelsize=8)

    axes[len(sample_indices) // 2].set_ylabel("Flux", fontsize=12)
    axes[-1].set_xlabel(r"$\lambda$ ($\mathrm{\AA}$)", fontsize=12)
    axes[-1].set_xlim(3500, 9100)

    plt.subplots_adjust(hspace=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Figure 1 -> {save_path.name}")
    return save_path


def plot_figure_desom1_prototypes(
    cfg: Config,
    model: DESOM1,
    wave_rest: np.ndarray,
    save_path: Optional[Path] = None,
    samples: Optional[np.ndarray] = None,
) -> Path:
    """
    Replicates Figure 3 of Teimoorinia et al. (2021) (see Inspect DESOM1.ipynb Cell 4):
    - 15x15 grid of decoded DESOM-1 prototypes floating without bounding frames (`axis('off')`,
      `hspace=1, wspace=1`).
    - Only the 9 nodes corresponding to S1..S9 have a black rectangular bounding box (`axis('on')`)
      with S1..S9 printed above.
    - Outer coordinate labels 1..15 on the left (1 at bottom, 15 at top) and bottom (1..15).
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_3_desom1_prototypes_15x15.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    prototypes = model.prototypes  # Shape: (225, latent_dim)
    decoded_prototypes = model.decode(prototypes, batch_size=225)

    labels = {}
    if samples is not None and len(samples) > 0:
        for k, node in enumerate(model.predict_bmu(np.asarray(samples)), start=1):
            labels.setdefault(int(node), []).append(f"S{k}")

    rows, cols = model.map_size
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    for r in range(rows):
        for c in range(cols):
            node_idx = r * cols + c
            ax = axes[r, c]
            ax.plot(wave_rest, decoded_prototypes[node_idx], color="tab:blue", linewidth=1.0)
            if node_idx in labels:
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(",".join(labels[node_idx]), fontsize=11, color="black", pad=2)
                for spine in ax.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.1)
            else:
                ax.axis("off")

    plt.subplots_adjust(hspace=1.0, wspace=1.0)

    # Outer Y-axis coordinate labels (1 at bottom row r=14, 15 at top row r=0)
    for r in range(rows):
        y_num = rows - r
        axes[r, 0].text(
            -0.35,
            0.5,
            str(y_num),
            transform=axes[r, 0].transAxes,
            fontsize=16,
            ha="right",
            va="center",
        )

    # Outer X-axis coordinate labels (1 at left col c=0, 15 at right col c=14)
    for c in range(cols):
        x_num = c + 1
        axes[rows - 1, c].text(
            0.5,
            -0.45,
            str(x_num),
            transform=axes[rows - 1, c].transAxes,
            fontsize=16,
            ha="center",
            va="top",
        )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved DESOM-1 Prototypes Grid -> {save_path.name}")
    return save_path
