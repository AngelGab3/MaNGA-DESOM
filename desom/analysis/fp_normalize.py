"""
Normalization of galaxy fingerprints before DESOM-2, replicating the original DESOM2_series.py:

    1. clip every node (column) at its 99.9th percentile ACROSS galaxies,
    2. MinMax-scale every node across galaxies (this weights nodes by their observed variation),
    3. MinMax-scale every fingerprint (row) to [0, 1].

The per-node statistics are returned so the identical transform can be applied to new galaxies.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _safe_minmax(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(x - lo) / (hi - lo); a zero range maps to 0 (same as sklearn's MinMaxScaler)."""
    rng = hi - lo
    rng = np.where(rng == 0, 1.0, rng)
    return (x - lo) / rng


def fit_node_stats(counts: np.ndarray, percentile: float = 99.9) -> Dict[str, np.ndarray]:
    """Per-node clip value and post-clip min / max, from raw counts (N_galaxies, n_nodes)."""
    clip_val = np.percentile(counts, percentile, axis=0)
    clipped = np.clip(counts, 0.0, clip_val)
    return {
        "clip": clip_val.astype(np.float64),
        "min": clipped.min(axis=0).astype(np.float64),
        "max": clipped.max(axis=0).astype(np.float64),
    }


def apply_normalization(counts: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Applies steps 1-3 with given node statistics. Returns (N, n_nodes) in [0, 1]."""
    clipped = np.clip(counts, 0.0, stats["clip"])
    z2 = _safe_minmax(clipped, stats["min"], stats["max"])
    row_min = z2.min(axis=1, keepdims=True)
    row_max = z2.max(axis=1, keepdims=True)
    return _safe_minmax(z2, row_min, row_max)


def normalize_fingerprints(
    counts: np.ndarray,
    map_size: Tuple[int, int],
    percentile: Optional[float] = 99.9,
    stats: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    counts: raw baxel counts per node, shape (N, n_nodes).
    Returns (fingerprints (N, h, w, 1) float32, stats).
    """
    counts = np.asarray(counts, dtype=np.float64)
    if stats is None:
        stats = fit_node_stats(counts, 99.9 if percentile is None else percentile)
    fp = apply_normalization(counts, stats)
    return fp.reshape(-1, map_size[0], map_size[1], 1).astype(np.float32), stats
