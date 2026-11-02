"""
Galaxy Map Visualization module (DESOM-2).

* Figure 9  : number of galaxies per node of the 10x10 map (clean viridis heatmap, ticks [1, 5, 10], 1 at bottom).
* Figure 10 : decoded DESOM-2 prototype fingerprints with outer 1..10 coordinates and 0.0-1.0 colorbar.
* extra     : mean fingerprint of the galaxies actually assigned to every node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..models.desom2 import DESOM2


def fix_ticks_10(ax: plt.Axes) -> None:
    """Applies the exact 10x10 axis tick formatting from Inspect DESOM2.ipynb."""
    ax.set_xticks(range(10))
    ax.set_xticklabels([1, "", "", "", 5, "", "", "", "", 10], fontsize=10)
    ax.set_yticks(range(10))
    ax.set_yticklabels([10, "", "", "", "", 5, "", "", "", 1], fontsize=10)


def assign_galaxies(model: DESOM2, fingerprints: np.ndarray) -> Dict[int, List[int]]:
    """Maps node index -> list of galaxy indices whose BMU is that node."""
    bmus = model.predict_bmu(fingerprints, batch_size=256)
    rows, cols = model.map_size
    node_to_galaxies: Dict[int, List[int]] = {i: [] for i in range(rows * cols)}
    for g_idx, bmu in enumerate(bmus):
        node_to_galaxies[int(bmu)].append(g_idx)
    return node_to_galaxies


def plot_desom2_node_counts(
    cfg: Config, model: DESOM2, node_to_galaxies: Dict[int, List[int]], save_path: Optional[Path] = None
) -> Path:
    """
    Replicates Figure 9 of Teimoorinia et al. (2021) (Inspect DESOM2.ipynb Cell 4):
    Clean 10x10 viridis heatmap of galaxy counts per DESOM-2 node, without cell text overlays,
    with Y-axis running from 1 (bottom) to 10 (top) and major tick labels at [1, 5, 10].
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_9_desom2_node_counts.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = model.map_size
    counts = np.array([len(node_to_galaxies[i]) for i in range(rows * cols)]).reshape(rows, cols)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(counts, origin="upper", cmap="viridis")
    fix_ticks_10(ax)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved DESOM-2 node counts (Figure 9) -> {save_path.name}")
    return save_path


def _grid_of_images(
    images: Dict[int, Optional[np.ndarray]],
    map_size: tuple[int, int],
    save_path: Path,
    cmap: str = "viridis",
) -> Path:
    """
    Replicates Figure 10 of Teimoorinia et al. (2021) (Inspect DESOM2.ipynb Cell 5):
    - 10x10 grid of 15x15 fingerprints (`hspace=0.02, wspace=0.02`), `axis('off')`.
    - Outer coordinate labels 1..10 on left (1 at bottom, 10 at top) and bottom (1..10).
    - Vertical viridis colorbar on the right labeled 0.0 (bottom) to 1.0 (top).
    """
    rows, cols = map_size
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for r in range(rows):
        for c in range(cols):
            node_idx = r * cols + c
            ax = axes[r, c]
            img = images.get(node_idx)
            if img is not None:
                ax.imshow(img, origin="upper", cmap=cmap, aspect="equal")
            ax.axis("off")

    plt.subplots_adjust(hspace=0.02, wspace=0.02, right=0.89)

    # Outer Y-axis coordinate labels (1 at bottom row r=9, 10 at top row r=0)
    for r in range(rows):
        axes[r, 0].text(
            -0.18,
            0.5,
            str(rows - r),
            transform=axes[r, 0].transAxes,
            fontsize=20,
            ha="right",
            va="center",
        )

    # Outer X-axis coordinate labels (1 at left col c=0, 10 at right col c=9)
    for c in range(cols):
        axes[rows - 1, c].text(
            0.5,
            -0.18,
            str(c + 1),
            transform=axes[rows - 1, c].transAxes,
            fontsize=20,
            ha="center",
            va="top",
        )

    # Right-side vertical colorbar (0.0 at bottom, 1.0 at top) as in Inspect DESOM2.ipynb Cell 5
    cax = fig.add_axes([0.915, 0.16, 0.015, 0.68])
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient)).T
    cax.imshow(gradient, cmap="viridis_r", aspect="auto")
    cax.axis("off")
    cax.text(0.5, -0.02, "0.0", transform=cax.transAxes, ha="center", va="top", fontsize=16)
    cax.text(0.5, 1.02, "1.0", transform=cax.transAxes, ha="center", va="bottom", fontsize=16)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {save_path.name}")
    return save_path


def plot_desom2_galaxy_grid(
    cfg: Config,
    model: DESOM2,
    fingerprints: np.ndarray,
    plateifus: List[str],
    manga_ids: List[str],
    save_path: Optional[Path] = None,
) -> Path:
    """
    Figure 10: decoded prototype fingerprint of every DESOM-2 node.
    Also writes Figure 9 (node occupancy) and the mean-fingerprint variant.
    """
    if save_path is None:
        save_path = cfg.figures_dir / "figure_10_desom2_galaxy_grid_10x10.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(fingerprints) == 0:
        print("[Plot] Warning: No fingerprints provided to map.")
        return save_path

    rows, cols = model.map_size
    node_to_galaxies = assign_galaxies(model, fingerprints)

    # Figure 10: decoded prototypes (what the paper shows)
    decoded = model.decode(model.prototypes, batch_size=rows * cols)
    protos = {i: decoded[i].reshape(decoded.shape[1], decoded.shape[2]) for i in range(rows * cols)}
    _grid_of_images(protos, model.map_size, save_path, cmap="viridis")

    # Variant: average of the real fingerprints in each node
    means = {
        i: np.mean(fingerprints[m], axis=0).squeeze() if m else protos[i]
        for i, m in node_to_galaxies.items()
    }
    _grid_of_images(
        means,
        model.map_size,
        cfg.figures_dir / "extra_desom2_node_mean_fingerprints.png",
        cmap="viridis",
    )

    plot_desom2_node_counts(cfg, model, node_to_galaxies)
    return save_path
