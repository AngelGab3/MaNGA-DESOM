"""
Training entry point for DESOM-2 (galaxy fingerprints -> 10x10 SOM). See `engine.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..config import Config
from ..models.desom2 import DESOM2
from .engine import train_desom_model
from .trainer_desom1 import seed_everything


def train_desom2(
    cfg: Config,
    X_train_fp: np.ndarray,
    X_val_fp: Optional[np.ndarray] = None,
    resume: bool = True,
) -> Tuple[DESOM2, Path]:
    """
    Trains DESOM-2 on (already normalized) galaxy fingerprints of shape (N, 15, 15, 1).
    Returns (model, final_weights_path).
    """
    if len(X_train_fp) == 0:
        raise ValueError("X_train_fp is empty. Cannot train DESOM-2 without fingerprints!")

    d2 = cfg["desom2"]
    seed = int(cfg["seed"])
    seed_everything(seed)

    model = DESOM2(
        input_shape=X_train_fp.shape[1:],
        map_size=tuple(d2["map_size"]),
        latent_dims=int(d2["latent_dims"]),
        gamma=float(d2["gamma"]),
        lr=float(d2["lr"]),
        map_distance=cfg["training"]["map_distance"],
    )
    final = train_desom_model(
        model, d2, X_train_fp, X_val_fp,
        out_dir=cfg.desom2_dir,
        final_name="DESOM2_model_final.h5",
        log_name="desom2_log.csv",
        desc="DESOM-2",
        seed=seed,
        resume=resume,
    )
    return model, final
