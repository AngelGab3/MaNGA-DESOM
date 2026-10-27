"""
Fingerprint Generation module.

A galaxy "fingerprint" is the histogram of the DESOM-1 best matching units of all its baxels
(a 2D histogram on the 15x15 map). Raw counts are stored, and a normalized version following the
original DESOM2_series.py is produced for DESOM-2 (see `fp_normalize.py`).

Only spectra that survived every quality cut (those in desom1_dataset_ready.npz, i.e. train + test
sets in the original code; its 'unused' spectra are excluded) contribute to fingerprints.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..io_utils import atomic_save_npz, tqdm
from ..models.desom1 import DESOM1
from .fp_normalize import normalize_fingerprints


def _find_weights(cfg: Config) -> Path:
    for name in ("DESOM_model_final.weights.h5", "DESOM_model_final.h5"):
        p = cfg.desom1_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing DESOM-1 weights in {cfg.desom1_dir}")


def generate_all_fingerprints(
    cfg: Config,
    model_desom1: Optional[DESOM1] = None,
    weights_path: Optional[Path] = None,
) -> Path:
    """Computes the BMU of every kept spectrum and builds raw + normalized fingerprints."""
    if model_desom1 is None:
        weights_path = weights_path or _find_weights(cfg)
        d1_cfg = cfg["desom1"]
        model_desom1 = DESOM1(
            input_dims=cfg["preprocessing"]["n_wave"],
            map_size=tuple(d1_cfg["map_size"]),
            latent_dims=int(d1_cfg["latent"]),
            map_distance=cfg["training"]["map_distance"],
        )
        model_desom1.load_weights(weights_path)

    ready_npz = cfg.run_dir / "desom1_dataset_ready.npz"
    if not ready_npz.exists():
        raise FileNotFoundError(f"Missing {ready_npz}. Run Stage 2 first!")
    selection = pd.read_csv(cfg.run_dir / "selection.csv")
    meta = {str(r.plateifu): (str(r.mangaID), float(r.z), str(r.split)) for r in selection.itertuples()}

    map_size = tuple(cfg["desom1"]["map_size"])
    n_nodes = map_size[0] * map_size[1]

    with np.load(ready_npz, allow_pickle=False) as data:
        X = data["X"]
        plateifu_all = data["plateifu"].astype(str)

    print(f"[Fingerprints] Assigning {len(X):,} spectra to the {map_size[0]}x{map_size[1]} map...")
    bmus = model_desom1.predict_bmu(X, batch_size=512)

    counts: List[np.ndarray] = []
    manga, plateifu, zs, splits, n_baxels = [], [], [], [], []
    for pid in tqdm(sorted(set(plateifu_all.tolist())), desc="Fingerprints"):
        if pid not in meta:
            continue
        sel = plateifu_all == pid
        counts.append(np.bincount(bmus[sel], minlength=n_nodes).astype(np.float64))
        m, z, sp = meta[pid]
        manga.append(m); plateifu.append(pid); zs.append(z); splits.append(sp)
        n_baxels.append(int(sel.sum()))

    if not counts:
        raise RuntimeError("No fingerprints could be generated (no spectra in the dataset).")

    counts_arr = np.stack(counts)
    clip_pct = cfg["desom2"].get("clipping_percentile", 99.9)
    fp_norm, stats = normalize_fingerprints(counts_arr, map_size, percentile=clip_pct)

    output_path = cfg.run_dir / "galaxy_fingerprints.npz"
    atomic_save_npz(
        output_path,
        fingerprints=fp_norm,                                # (N, h, w, 1) normalized, DESOM-2 input
        counts=counts_arr.reshape(-1, map_size[0], map_size[1]).astype(np.float32),  # raw histograms
        n_baxels=np.array(n_baxels, dtype=np.int32),
        node_clip=stats["clip"], node_min=stats["min"], node_max=stats["max"],
        mangaID=np.array(manga),
        plateifu=np.array(plateifu),
        z=np.array(zs, dtype=np.float32),
        split=np.array(splits),
    )
    print(f"[Fingerprints] Saved {len(fp_norm)} fingerprints -> {output_path.name}")
    return output_path
