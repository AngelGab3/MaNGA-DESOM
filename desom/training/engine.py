"""
Generic training engine shared by DESOM-1 and DESOM-2.

Implements the original DESOM procedure (Forest et al. 2019; Teimoorinia et al. 2021):
  for every iteration:
      1. find the BMU of each sample in the batch,
      2. weight every prototype with a Gaussian of the grid distance to the BMU,
         exp(-d^2 / T^2), where T decays exponentially from Tmax to Tmin,
      3. take one gradient step on  L = L_recon + gamma * L_SOM.

Additions with respect to the original code (kept from this project):
  * checkpointing / resume (weights + optimizer state + iteration counter),
  * per-epoch shuffling, validation tracking,
  * quantization / topographic error logging (as in the original logs).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..analysis.metrics import latent_errors, original_space_errors
from ..io_utils import atomic_save_json, tqdm
from ..models.base import BaseDESOM

LOG_FIELDS: List[str] = [
    "iter", "T",
    "loss_total", "loss_recon", "loss_som",
    "val_loss_total", "val_loss_recon", "val_loss_som",
    "latent_quantization_err", "latent_topographic_err",
    "quantization_err", "topographic_err",
]


def parse_loss(loss_val: Any) -> Tuple[float, float, float]:
    """
    Unpacks the value returned by train_on_batch / test_on_batch for the two-output DESOM model.
    Expected layout: [total, reconstruction, som]. Raises if the layout is unexpected, so that a
    Keras change cannot silently corrupt the logs.
    """
    arr = np.asarray(loss_val, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(
            f"Expected 3 loss values [total, recon, som] from Keras, got {arr.size}: {arr.tolist()}"
        )
    return float(arr[0]), float(arr[1]), float(arr[2])


def _open_log(log_csv: Path, resume_from: int) -> Tuple[Any, csv.DictWriter]:
    """Opens the CSV log. If resuming with an older header, the old log is rotated, not corrupted."""
    mode = "w"
    if resume_from > 0 and log_csv.exists():
        with open(log_csv, "r", newline="") as f:
            header = f.readline().strip().split(",")
        if header == LOG_FIELDS:
            mode = "a"
        else:
            log_csv.replace(log_csv.with_name(log_csv.stem + ".old.csv"))
    f = open(log_csv, mode, newline="")
    writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
    if mode == "w":
        writer.writeheader()
    return f, writer


def train_desom_model(
    model: BaseDESOM,
    section: Dict[str, Any],
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    out_dir: Path,
    final_name: str,
    log_name: str,
    desc: str,
    seed: int = 0,
    resume: bool = True,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Trains `model` in place and returns the path of the final weights file.

    section: the "desom1" / "desom2" block of the config. Keys used:
        batch_size, iterations, som_iterations, Tmax, Tmin, checkpoint_interval, log_interval.
    """
    from ..models.som_layer import get_temperature  # local import keeps module import light

    n_samples = len(X_train)
    if n_samples == 0:
        raise ValueError(f"[{desc}] Empty training set.")

    batch_size = min(int(section["batch_size"]), n_samples)
    total_iters = int(section["iterations"])
    som_iters = int(section["som_iterations"])
    Tmax, Tmin = float(section["Tmax"]), float(section["Tmin"])
    ckpt_interval = int(section.get("checkpoint_interval", 1000))
    log_interval = int(section.get("log_interval", 100))

    out_dir.mkdir(parents=True, exist_ok=True)
    final_weights = model._resolve_weights_path(out_dir / final_name)
    latest_ckpt = out_dir / "checkpoint_latest.weights.h5"
    opt_ckpt = out_dir / "checkpoint_optimizer.npz"
    state_file = out_dir / "train_state.json"
    log_csv = out_dir / log_name

    # ---- resume -----------------------------------------------------------------------
    start_iter = 0
    if resume and state_file.exists() and latest_ckpt.exists():
        try:
            with open(state_file, "r") as f:
                start_iter = int(json.load(f).get("iteration", 0))
            model.load_weights(latest_ckpt)
            restored = model.load_optimizer_state(opt_ckpt)
            print(f"[{desc}] Resuming from iteration {start_iter}/{total_iters} "
                  f"(optimizer state {'restored' if restored else 'NOT restored: Adam moments restart'})")
        except Exception as e:
            print(f"[{desc}] Could not load checkpoint, starting fresh: {e}")
            start_iter = 0

    if start_iter == 0:
        print(f"[{desc}] Initializing SOM prototypes from a random sample of the training set...")
        model.init_som_weights(X_train)

    log_file, writer = _open_log(log_csv, start_iter)

    # Different stream per resume point so a resumed run does not replay the same batch order.
    rng = np.random.RandomState(seed + start_iter)
    indices = rng.permutation(n_samples)
    curr_idx = 0

    val_source = X_val if (X_val is not None and len(X_val) > 0) else X_train
    val_sample = val_source[: min(1024, len(val_source))]

    print(f"[{desc}] Training {start_iter}->{total_iters} iterations | batch={batch_size} | "
          f"samples={n_samples:,} | map distance='{model.map_distance}'")
    pbar = tqdm(range(start_iter, total_iters), desc=desc, initial=start_iter, total=total_iters)

    for ite in pbar:
        # 1. batch (reshuffle after every pass over the data)
        if curr_idx + batch_size > n_samples:
            indices = rng.permutation(n_samples)
            curr_idx = 0
        X_batch = X_train[indices[curr_idx: curr_idx + batch_size]]
        curr_idx += batch_size

        # 2. temperature + neighbourhood weights (one compiled graph call)
        T = get_temperature(ite, som_iters, Tmax=Tmax, Tmin=Tmin)
        d_batch, w_batch = model.bmu_weights(X_batch, T)

        # 3. gradient step
        loss = model.model.train_on_batch(X_batch, [X_batch, w_batch])

        is_ckpt = (ite + 1) % ckpt_interval == 0 or ite == total_iters - 1
        is_log = (ite + 1) % log_interval == 0 or is_ckpt
        if not is_log:
            continue

        row: Dict[str, Any] = {"iter": ite + 1, "T": f"{T:.4f}"}
        l_tot, l_rec, l_som = parse_loss(loss)
        row.update(loss_total=f"{l_tot:.6g}", loss_recon=f"{l_rec:.6g}", loss_som=f"{l_som:.6g}")

        d_val, w_val = model.bmu_weights(val_sample, T)
        vl_tot, vl_rec, vl_som = parse_loss(model.model.test_on_batch(val_sample, [val_sample, w_val]))
        row.update(val_loss_total=f"{vl_tot:.6g}", val_loss_recon=f"{vl_rec:.6g}", val_loss_som=f"{vl_som:.6g}")

        for k, v in latent_errors(d_val, model.map_size).items():
            row[k] = f"{v:.6g}"

        if is_ckpt:
            decoded = model.decode(model.prototypes)
            for k, v in original_space_errors(val_sample, decoded, model.map_size).items():
                row[k] = f"{v:.6g}"

        writer.writerow(row)
        log_file.flush()
        pbar.set_postfix({"L_rec": f"{l_rec:.4f}", "val_rec": f"{vl_rec:.4f}", "T": f"{T:.2f}"})

        if is_ckpt:
            model.save_weights(latest_ckpt)
            model.save_optimizer_state(opt_ckpt)
            state = {"iteration": ite + 1, "T": T}
            if extra_state:
                state.update(extra_state)
            atomic_save_json(state_file, state)

    log_file.close()
    model.save_weights(final_weights)
    print(f"[{desc}] Finished successfully -> {final_weights.name}")
    return final_weights
