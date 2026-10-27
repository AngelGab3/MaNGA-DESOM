"""
Held-out evaluation of a trained DESOM (counterpart of the diagnostics of the original
DESOM_train.py / DESOM2_series.py): test MSE, reconstruction bias, quantization / topographic
errors, and node occupancy (the paper asks for enough spectra in every node).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from ..io_utils import atomic_save_json
from ..models.base import BaseDESOM
from .metrics import latent_errors, original_space_errors


def evaluate_model(
    model: BaseDESOM,
    X_test: np.ndarray,
    out_dir: Path,
    tag: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Evaluates on X_test, writes `<tag>_test_metrics.json` and `<tag>_node_occupancy.png`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recon, d = model.model.predict(X_test, batch_size=batch_size, verbose=0)
    bmu = np.argmin(d, axis=1)
    rows, cols = model.map_size

    diff = (recon - X_test).reshape(len(X_test), -1)
    occupancy = np.bincount(bmu, minlength=rows * cols)

    metrics: Dict[str, Any] = {
        "n_test": int(len(X_test)),
        "map_distance": model.map_distance,
        "test_mse": float(np.mean(diff ** 2)),
        "delta_mean": float(diff.mean()),
        "delta_median": float(np.median(diff)),
        "nodes_used": int((occupancy > 0).sum()),
        "n_nodes": int(rows * cols),
        "min_nonzero_occupancy": int(occupancy[occupancy > 0].min()) if (occupancy > 0).any() else 0,
        "median_occupancy": float(np.median(occupancy)),
    }
    metrics.update(latent_errors(d, model.map_size))
    sample = X_test[: min(len(X_test), 2048)]
    metrics.update(original_space_errors(sample, model.decode(model.prototypes), model.map_size))
    atomic_save_json(out_dir / f"{tag}_test_metrics.json", metrics)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(occupancy.reshape(rows, cols), origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax[0], label="samples in node")
    ax[0].set_title(f"{tag}: test-set occupancy")
    ax[1].hist(occupancy, bins=20)
    ax[1].set_xlabel("samples in node")
    ax[1].set_ylabel("number of nodes")
    ax[1].set_title("occupancy distribution")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_node_occupancy.png", dpi=150)
    plt.close()

    print(f"[Eval {tag}] test MSE={metrics['test_mse']:.5f} | topographic err={metrics['latent_topographic_err']:.3f} "
          f"| nodes used {metrics['nodes_used']}/{metrics['n_nodes']}")
    return metrics
