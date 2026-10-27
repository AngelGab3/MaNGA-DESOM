"""
Dataset Consolidation and Audit module.
Consolidates candidate spectra from all processed galaxies and builds
the verified train/val/test dataset for DESOM-1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..io_utils import atomic_save_json, atomic_save_npz, sha256_file, tqdm


def consolidate_stage1_spectra(cfg: Config) -> Path:
    """
    Scans processed/<plateifu>/candidate_spectra.npz, concatenates spectra,
    and saves consolidated manga_spectra.npz.
    """
    selection_csv = cfg.run_dir / "selection.csv"
    if not selection_csv.exists():
        raise FileNotFoundError(f"Missing selection.csv in {cfg.run_dir}")

    selection = pd.read_csv(selection_csv)
    split_by_manga = dict(zip(selection["mangaID"].astype(str), selection["split"]))

    all_X: List[np.ndarray] = []
    all_wave: np.ndarray | None = None
    all_manga: List[str] = []
    all_plateifu: List[str] = []
    all_baxel: List[int] = []
    all_sn: List[float] = []
    all_splits: List[str] = []

    galaxies_processed = 0
    galaxies_with_spectra = 0

    print(f"[Consolidate] Scanning processed directories for {len(selection)} galaxies...")
    for _, row in tqdm(selection.iterrows(), total=len(selection), desc="Consolidating"):
        plateifu = str(row["plateifu"])
        manga_id = str(row["mangaID"])
        cand_path = cfg.processed_dir / plateifu / "candidate_spectra.npz"

        if not cand_path.exists():
            continue

        galaxies_processed += 1
        with np.load(cand_path, allow_pickle=False) as data:
            X = data["X"]
            if X.shape[0] == 0:
                continue

            if all_wave is None:
                all_wave = data["wave_rest"]

            n_spec = X.shape[0]
            galaxies_with_spectra += 1
            all_X.append(X)
            all_manga.extend([manga_id] * n_spec)
            all_plateifu.extend([plateifu] * n_spec)
            all_baxel.extend(data["baxel_id"].tolist())
            all_sn.extend(data["sn_cont_corrected"].tolist())
            all_splits.extend([split_by_manga.get(manga_id, "train")] * n_spec)

    if not all_X:
        raise RuntimeError("No candidate spectra found! Did you run Stage 1 processing?")

    X_full = np.concatenate(all_X, axis=0).astype(np.float32)
    output_path = cfg.run_dir / f"manga_{cfg['n_target']}_spectra.npz"

    atomic_save_npz(
        output_path,
        X=X_full,
        wave_rest=all_wave,
        mangaID=np.array(all_manga),
        plateifu=np.array(all_plateifu),
        baxel_id=np.array(all_baxel, dtype=np.int32),
        sn_cont_corrected=np.array(all_sn, dtype=np.float32),
        split=np.array(all_splits),
    )

    print(f"[Consolidate] Done! {len(X_full):,} spectra across {galaxies_with_spectra}/{galaxies_processed} galaxies -> {output_path.name}")
    return output_path


def flux_outlier_masks(
    X: np.ndarray,
    wave_rest: np.ndarray,
    em_lines: List[float],
    low_pct: float = 0.5,
    high_pct: float = 0.5,
    width_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Global flux outlier screening of Paper Section 2 (same logic as the original Split_data.py):

    * low-flux outliers: spectra whose minimum flux is within the lowest `low_pct` percentile of
      all minima (strongly negative flux caused by sky residuals in the red).
    * high-flux outliers: spectra whose maximum exceeds the (100 - high_pct) percentile of all
      maxima AND where the excess is NOT inside a known emission line (+/- width_bins).

    Returns (low_flux_outlier, high_flux_outlier, thresholds). Boolean arrays are True = remove.
    """
    x_mins = X.min(axis=1)
    min_thr = float(np.percentile(x_mins, low_pct))
    low_out = x_mins <= min_thr

    x_maxes = X.max(axis=1)
    max_thr = float(np.percentile(x_maxes, 100.0 - high_pct))

    above = X > max_thr
    for line in em_lines:
        k = int(np.argmin(np.abs(wave_rest - line)))
        lo, hi = max(0, k - width_bins), min(X.shape[1], k + width_bins)
        above[:, lo:hi] = False
    high_out = above.any(axis=1)

    return low_out, high_out, {"min_threshold": min_thr, "max_threshold": max_thr}


def make_split_indices(
    splits_by_galaxy: np.ndarray,
    n_spectra: int,
    mode: str,
    ratios: List[float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds (idx_train, idx_val, idx_test).

    mode "galaxy":   uses the galaxy-level label of every baxel (no leakage between splits).
    mode "spectrum": shuffles baxels individually, as in the original paper.
    """
    if mode == "spectrum":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_spectra)
        n_train = int(np.floor(ratios[0] * n_spectra))
        n_val = int(np.floor(ratios[1] * n_spectra))
        return (
            np.sort(perm[:n_train]),
            np.sort(perm[n_train: n_train + n_val]),
            np.sort(perm[n_train + n_val:]),
        )
    if mode != "galaxy":
        raise ValueError(f"Unknown split_mode '{mode}'. Use 'galaxy' or 'spectrum'.")

    idx_train = np.flatnonzero(splits_by_galaxy == "train")
    idx_val = np.flatnonzero(splits_by_galaxy == "validation")
    idx_test = np.flatnonzero(splits_by_galaxy == "test")
    return idx_train, idx_val, idx_test


def audit_and_split_dataset(cfg: Config) -> Path:
    """
    Audits consolidated spectra, applies the global flux-outlier cuts of the paper, splits the
    spectra into train/val/test and writes desom1_dataset_ready.npz.
    """
    input_path = cfg.run_dir / f"manga_{cfg['n_target']}_spectra.npz"
    output_path = cfg.run_dir / "desom1_dataset_ready.npz"

    if not input_path.exists():
        input_path = consolidate_stage1_spectra(cfg)

    print(f"[Audit] Loading {input_path.name}...")
    with np.load(input_path, allow_pickle=False) as data:
        X = data["X"]
        wave_rest = data["wave_rest"]
        manga_id = data["mangaID"]
        plateifu = data["plateifu"]
        baxel_id = data["baxel_id"]
        sn = data["sn_cont_corrected"]
        splits = data["split"]

    # 1. Checks
    assert np.isfinite(X).all(), "Audit error: NaN or Inf found in spectra!"
    assert np.all(np.diff(wave_rest) > 0), "Audit error: wave_rest is not strictly monotonic!"
    assert X.shape[1] == cfg["preprocessing"]["n_wave"], f"Dimension mismatch: {X.shape[1]} vs {cfg['preprocessing']['n_wave']}"

    # 2. Global flux outlier cuts (Paper Section 2)
    pre = cfg["preprocessing"]
    audit: Dict[str, Any] = {"n_input": int(len(X))}
    if pre.get("apply_flux_outlier_cuts", True) and len(X) > 0:
        low_out, high_out, thr = flux_outlier_masks(
            X, wave_rest, pre["em_lines"],
            low_pct=float(pre["low_flux_percentile"]),
            high_pct=float(pre["high_flux_percentile"]),
            width_bins=int(pre.get("em_line_half_width_bins", 3)),
        )
        keep = ~(low_out | high_out)
        audit.update(
            thresholds=thr,
            n_low_flux_outliers=int(low_out.sum()),
            n_high_flux_outliers=int(high_out.sum()),
            n_removed_total=int((~keep).sum()),
        )
        print(f"[Audit] Flux outliers removed: low={low_out.sum():,} high(non-emission)={high_out.sum():,} "
              f"(total {(~keep).sum():,} of {len(X):,})")
        X, manga_id, plateifu = X[keep], manga_id[keep], plateifu[keep]
        baxel_id, sn, splits = baxel_id[keep], sn[keep], splits[keep]
    audit["n_kept"] = int(len(X))

    # 3. Split (galaxy-level or spectrum-level; selectable from config / dashboard)
    mode = cfg["sample"].get("split_mode", "galaxy")
    idx_train, idx_val, idx_test = make_split_indices(
        splits, len(X), mode, cfg["sample"]["split_ratios"], int(cfg["seed"])
    )
    if len(idx_train) == 0:
        print("[Audit] WARNING: empty train split; using every spectrum for training.")
        idx_train = np.arange(len(X))
    if len(idx_val) == 0:
        print("[Audit] WARNING: empty validation split; borrowing from train.")
        idx_val = idx_train[: max(1, int(0.15 * len(idx_train)))]
    if len(idx_test) == 0:
        print("[Audit] WARNING: empty test split; borrowing from train.")
        idx_test = idx_train[: max(1, int(0.15 * len(idx_train)))]

    audit.update(split_mode=mode, n_train=int(len(idx_train)), n_val=int(len(idx_val)), n_test=int(len(idx_test)))
    n_tot = max(len(X), 1)
    print(f"[Audit] Split mode: {mode}")
    print(f"[Audit] Total spectra: {len(X):,}")
    print(f"[Audit] Train set:      {len(idx_train):,} ({len(idx_train)/n_tot:.1%})")
    print(f"[Audit] Validation set: {len(idx_val):,} ({len(idx_val)/n_tot:.1%})")
    print(f"[Audit] Test set:       {len(idx_test):,} ({len(idx_test)/n_tot:.1%})")

    atomic_save_npz(
        output_path,
        X=X,
        wave_rest=wave_rest,
        mangaID=manga_id,
        plateifu=plateifu,
        baxel_id=baxel_id,
        sn_cont_corrected=sn,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    atomic_save_json(cfg.run_dir / "audit_report.json", audit)

    print(f"[Audit] Stage 2 Complete -> {output_path.name}")
    return output_path
