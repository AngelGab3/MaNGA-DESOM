"""
Spectral Visualization module.
Replicates Figure 1 (sample normalized spectra) and Figure 2/4 (DESOM-1 prototype map).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..models.desom1 import DESOM1


def pick_sample_indices(X: np.ndarray, n_samples: int = 9, seed: int = 42) -> np.ndarray:
    """Deterministic random sample of spectra, shared by Figure 1 (S1..Sn) and Figure 3."""
    rng = np.random.RandomState(seed)
    return rng.choice(len(X), size=min(n_samples, len(X)), replace=False)


def plot_figure_1_normalized_spectra(
    cfg: Config,
    X: np.ndarray,
    wave_rest: np.ndarray,
    n_samples: int = 9,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Replicates Figure 1: Sample of normalized baxel spectra in rest-frame.
    Highlights prominent emission lines: [OII] 3727, Hbeta 4861, [OIII] 5007,
    Halpha 6563, [NII] 6584, [SII] 6717.
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_1_normalized_spectra.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(X) == 0:
        print("[Plot] Warning: No spectra available in X to plot.")
        return save_path

    fig, ax = plt.subplots(figsize=(12, 6))
    sample_indices = pick_sample_indices(X, n_samples)

    lines = {
        "[OII] 3727": 3727.0,
        r"$H\beta$ 4861": 4861.0,
        "[OIII] 5007": 5007.0,
        r"$H\alpha$ 6563": 6563.0,
        "[NII] 6584": 6584.0,
        "[SII] 6717": 6717.0,
    }

    offset = 0.0
    for k, idx in enumerate(sample_indices, start=1):
        ax.plot(wave_rest, X[idx] + offset, lw=0.9, alpha=0.9)
        ax.text(wave_rest.min() + 30, offset + 0.2, f"S{k}", fontsize=9, fontweight="bold")
        offset += 2.0

    for name, w in lines.items():
        if wave_rest.min() <= w <= wave_rest.max():
            ax.axvline(w, color="gray", linestyle="--", lw=0.8, alpha=0.6)
            ax.text(w, offset + 0.2, name, rotation=90, fontsize=8, ha="center", va="bottom")

    ax.set_xlabel(r"Rest-frame Wavelength [$\AA$]", fontsize=12)
    ax.set_ylabel(r"Normalized Flux + Offset", fontsize=12)
    ax.set_title("Figure 1 Replicant: Normalized Baxel Spectra (MaNGA)", fontsize=14)
    ax.set_xlim(wave_rest.min(), wave_rest.max())
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
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
    Replicates the paper's Figure 3: the 15x15 map of decoded DESOM-1 prototypes.
    Row 0 is drawn at the TOP (node index = row * width + col), consistent with Utils.node2idx
    of the original code and with every other map in this package.

    samples: optional array (n, n_wave) of spectra (e.g. the S1..S9 of Figure 1). Each one is
    assigned to its BMU and the node is labelled S1..Sn, as in the paper.
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
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20), sharex=True, sharey=True)

    for r in range(rows):
        for c in range(cols):
            node_idx = r * cols + c
            ax = axes[r, c]
            ax.plot(wave_rest, decoded_prototypes[node_idx], color="navy", lw=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            if node_idx in labels:
                ax.set_title(",".join(labels[node_idx]), fontsize=9, color="crimson", pad=2)
                for spine in ax.spines.values():
                    spine.set_edgecolor("crimson")
                    spine.set_linewidth(1.5)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle("Figure 3 Replicant: 15x15 Spectral Prototypes (DESOM-1)", fontsize=16, y=0.92)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved DESOM-1 Prototypes Grid -> {save_path.name}")
    return save_path
