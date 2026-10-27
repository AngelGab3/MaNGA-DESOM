"""
Galaxy Map Visualization module (DESOM-2).

* Figure 9  : number of galaxies per node of the 10x10 map.
* Figure 10 : decoded DESOM-2 prototype fingerprints (as in the paper).
* extra     : mean fingerprint of the galaxies actually assigned to every node.

All maps draw node (0, 0) at the TOP-left corner (node index = row * width + col).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..models.desom2 import DESOM2


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
    """Figure 9: galaxies per node, to check that the map is well utilised."""
    if save_path is None:
        save_path = cfg.figures_dir / "figure_9_desom2_node_counts.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = model.map_size
    counts = np.array([len(node_to_galaxies[i]) for i in range(rows * cols)]).reshape(rows, cols)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(counts, origin="upper", cmap="viridis")
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, str(counts[r, c]), ha="center", va="center", fontsize=7,
                    color="white" if counts[r, c] < counts.max() * 0.6 else "black")
    ax.set_xticks(range(cols)); ax.set_xticklabels(range(1, cols + 1), fontsize=7)
    ax.set_yticks(range(rows)); ax.set_yticklabels(range(1, rows + 1), fontsize=7)
    fig.colorbar(im, ax=ax, label="galaxies per node")
    ax.set_title("Figure 9 Replicant: galaxies per DESOM-2 node")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved DESOM-2 node counts -> {save_path.name}")
    return save_path


def _grid_of_images(
    cfg: Config, images: Dict[int, Optional[np.ndarray]], map_size, title: str,
    save_path: Path, cmap: str, counts: Dict[int, List[int]],
) -> Path:
    rows, cols = map_size
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    for r in range(rows):
        for c in range(cols):
            node_idx = r * cols + c
            ax = axes[r, c]
            img = images.get(node_idx)
            if img is not None:
                ax.imshow(img, origin="upper", cmap=cmap, aspect="auto")
                ax.text(0.05, 0.90, f"N={len(counts[node_idx])}", transform=ax.transAxes,
                        color="white", fontsize=7, fontweight="bold")
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle(title, fontsize=16, y=0.92)
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
    Figure 10: decoded prototype fingerprint of every DESOM-2 node, N = galaxies assigned to it.
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
    _grid_of_images(cfg, protos, model.map_size,
                    "Figure 10 Replicant: decoded DESOM-2 prototype fingerprints",
                    save_path, "viridis", node_to_galaxies)

    # Variant: average of the real fingerprints in each node
    means = {i: np.mean(fingerprints[m], axis=0).squeeze() if m else None
             for i, m in node_to_galaxies.items()}
    _grid_of_images(cfg, means, model.map_size,
                    "Mean fingerprint of the galaxies assigned to each DESOM-2 node",
                    cfg.figures_dir / "extra_desom2_node_mean_fingerprints.png", "viridis", node_to_galaxies)

    plot_desom2_node_counts(cfg, model, node_to_galaxies)
    return save_path
