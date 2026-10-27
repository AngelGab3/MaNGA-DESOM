"""
Fingerprint Visualization module.
Galaxy fingerprints on the DESOM-1 15x15 map (Figure 7 style single-galaxy plot + sample grid).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import Config


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
            im = ax.imshow(fp, origin="upper", cmap="magma", aspect="equal")
            title_id = manga_ids[idx] if idx < len(manga_ids) else str(idx)
            ax.set_title(f"mangaID: {title_id}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    plt.suptitle("Sample galaxy fingerprints (15x15, normalized for DESOM-2)", fontsize=14, y=0.94)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Fingerprints Grid -> {save_path.name}")
    return save_path


def plot_single_galaxy_fingerprint(
    cfg: Config,
    counts: np.ndarray,
    galaxy_label: str,
    save_path: Optional[Path] = None,
) -> Path:
    """
    Replicates the style of the paper's Figure 7: raw number of baxels of ONE galaxy that land on
    every node of the 15x15 DESOM-1 map (counts written on the nodes; empty nodes shown as 0).
    The R/Re colouring of the paper needs the NSA effective radius and is not reproduced here.
    """
    counts = np.asarray(counts).squeeze()
    if save_path is None:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in galaxy_label)
        save_path = cfg.figures_dir / f"figure_7_fingerprint_{safe}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = counts.shape
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(counts, origin="upper", cmap="viridis", aspect="equal")
    for r in range(h):
        for c in range(w):
            v = int(counts[r, c])
            ax.text(c, r, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v < counts.max() * 0.6 else "black")
    ax.set_xticks(range(w)); ax.set_xticklabels(range(1, w + 1), fontsize=6)
    ax.set_yticks(range(h)); ax.set_yticklabels(range(1, h + 1), fontsize=6)
    ax.set_title(f"Figure 7 Replicant: fingerprint of {galaxy_label} ({int(counts.sum())} baxels)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved single-galaxy fingerprint -> {save_path.name}")
    return save_path
