"""
Training entry point for DESOM-1 (spectra -> 15x15 SOM). See `engine.py` for the procedure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..config import Config
from ..models.desom1 import DESOM1
from .engine import parse_loss, train_desom_model  # noqa: F401  (parse_loss re-exported)


def seed_everything(seed: int) -> None:
    """Seeds numpy and TensorFlow so that weight init and shuffling are reproducible."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_desom1(
    cfg: Config,
    X_train: np.ndarray,
    X_val: np.ndarray,
    resume: bool = True,
) -> Tuple[DESOM1, Path]:
    """Trains DESOM-1 jointly (reconstruction + SOM loss). Returns (model, final_weights_path)."""
    d1 = cfg["desom1"]
    seed = int(cfg["seed"])
    seed_everything(seed)

    model = DESOM1(
        input_dims=X_train.shape[1],
        map_size=tuple(d1["map_size"]),
        latent_dims=int(d1["latent"]),
        gamma=float(d1["gamma"]),
        lr=float(d1["lr"]),
        map_distance=cfg["training"]["map_distance"],
    )
    final = train_desom_model(
        model, d1, X_train, X_val,
        out_dir=cfg.desom1_dir,
        final_name="DESOM_model_final.h5",
        log_name="desom1_log.csv",
        desc="DESOM-1",
        seed=seed,
        resume=resume,
    )
    return model, final
