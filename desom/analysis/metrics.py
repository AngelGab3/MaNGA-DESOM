"""
SOM quality metrics (as logged by the original DESOM code and used in the paper to judge map smoothness).

* quantization error:  mean distance between each input and its best matching prototype.
* topographic error:   fraction of inputs whose two best matching units are NOT adjacent on the map.
  (The paper: "fraction of input spectra in which the winning node is not adjacent to the runner-up".)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def quantization_error(d: np.ndarray) -> float:
    """d: squared distances (N, n_prototypes)."""
    return float(d.min(axis=1).mean())


def topographic_error(d: np.ndarray, map_size: Tuple[int, int]) -> float:
    """Fraction of samples whose best two prototypes are not 4-neighbours on the grid."""
    w = int(map_size[1])
    if d.shape[1] < 2:
        return 0.0
    best2 = np.argsort(d, axis=1)[:, :2]
    r0, c0 = best2[:, 0] // w, best2[:, 0] % w
    r1, c1 = best2[:, 1] // w, best2[:, 1] % w
    adjacent = (np.abs(r0 - r1) + np.abs(c0 - c1)) == 1
    return float(1.0 - adjacent.mean())


def latent_errors(d: np.ndarray, map_size: Tuple[int, int]) -> Dict[str, float]:
    return {
        "latent_quantization_err": quantization_error(d),
        "latent_topographic_err": topographic_error(d, map_size),
    }


def original_space_errors(
    X: np.ndarray, decoded_prototypes: np.ndarray, map_size: Tuple[int, int], chunk: int = 64
) -> Dict[str, float]:
    """
    Errors measured in the input space between data and DECODED prototypes.
    X and decoded_prototypes are flattened to (N, D) and (n_prototypes, D).
    """
    Xf = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    P = np.asarray(decoded_prototypes, dtype=np.float32).reshape(len(decoded_prototypes), -1)
    p2 = (P**2).sum(axis=1)
    qe_sum = 0.0
    n_nonadj = 0
    w = int(map_size[1])
    for i in range(0, len(Xf), chunk):
        xb = Xf[i : i + chunk]
        # ||x - p||^2 = ||x||^2 - 2 x.p + ||p||^2  (avoids a (N, n_proto, D) tensor)
        d = (xb**2).sum(axis=1)[:, None] - 2.0 * xb @ P.T + p2[None, :]
        d = np.maximum(d, 0.0)
        qe_sum += d.min(axis=1).sum()
        best2 = np.argsort(d, axis=1)[:, :2]
        r0, c0 = best2[:, 0] // w, best2[:, 0] % w
        r1, c1 = best2[:, 1] // w, best2[:, 1] % w
        n_nonadj += int(((np.abs(r0 - r1) + np.abs(c0 - c1)) != 1).sum())
    return {
        "quantization_err": float(qe_sum / len(Xf)),
        "topographic_err": float(n_nonadj / len(Xf)),
    }
