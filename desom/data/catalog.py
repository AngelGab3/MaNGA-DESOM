"""
Catalog and Galaxy Selection module.
Downloads DRPall catalog and selects primary/secondary MaNGA galaxies.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from astropy.table import Table

from ..config import Config
from ..io_utils import atomic_save_csv, sha256_file


def download_file(url: str, destination: Path, timeout: int = 180, verbose: bool = True) -> None:
    """Downloads a file using curl or python urllib with resume support."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        if verbose:
            print(f"[Catalog] File already exists: {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.part")

    if verbose:
        print(f"[Catalog] Downloading {url} -> {destination.name}...")

    # Try curl first
    try:
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(part), "--connect-timeout", "30", "--max-time", str(timeout), url],
            check=True,
            stdout=subprocess.DEVNULL if not verbose else None,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to urllib
        import urllib.request
        urllib.request.urlretrieve(url, str(part))

    part.replace(destination)
    if verbose:
        print(f"[Catalog] Download complete: {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")


def get_drpall_catalog(cfg: Config, force_download: bool = False) -> Tuple[Table, Path]:
    """Retrieves or downloads the MaNGA DRPall FITS catalog."""
    catalog_path = cfg.catalogs_dir / "drpall-v2_4_3.fits"
    if force_download or not catalog_path.exists():
        download_file(cfg["paths"]["drpall_url"], catalog_path)

    print(f"[Catalog] Reading catalog: {catalog_path.name}...")
    catalog = Table.read(catalog_path, hdu=1)
    return catalog, catalog_path


def select_galaxies(cfg: Config, force_reselect: bool = False) -> pd.DataFrame:
    """
    Selects sample of galaxies matching paper criteria:
    - Survey mode: MaNGA dither
    - Pipeline version: v2_4_3 (DR15)
    - Valid redshift in NSA: 0.0 < z < 0.15
    - DRP3QUAL without critical errors (bit 30)
    - Primary/secondary sample (mngtarg1 >= 1)
    """
    selection_csv = cfg.run_dir / "selection.csv"
    if not force_reselect and selection_csv.exists():
        df = pd.read_csv(selection_csv)
        if len(df) == cfg["n_target"]:
            print(f"[Catalog] Using existing selection of {len(df)} galaxies from {selection_csv.name}")
            return df

    catalog, catalog_path = get_drpall_catalog(cfg)
    cols = {c.lower(): c for c in catalog.colnames}

    def col(name: str) -> np.ndarray:
        return np.asarray(catalog[cols[name]])

    def strings(name: str) -> np.ndarray:
        return np.char.strip(col(name).astype(str))

    sample_cfg = cfg["sample"]
    mngtarg1 = col("mngtarg1").astype(np.int64)
    z = col("nsa_z").astype(float)
    srvymode = strings("srvymode")
    versdrp3 = strings("versdrp3")
    quality = col("drp3qual").astype(np.int64)

    # Filtering according to Paper Section 2
    eligible = (
        (mngtarg1 >= int(sample_cfg["mngtarg1_min"]))
        & (srvymode == sample_cfg["srvymode"])
        & (versdrp3 == sample_cfg["versdrp3"])
        & ((quality & (1 << int(sample_cfg["drp3qual_exclude_bit"]))) == 0)
        & np.isfinite(z)
        & (z > float(sample_cfg["min_redshift"]))
        & (z < float(sample_cfg["max_redshift"]))
    )

    pool = pd.DataFrame({
        "plateifu": strings("plateifu")[eligible],
        "mangaID": strings("mangaid")[eligible],
        "z": z[eligible],
    })

    # Drop duplicates by mangaID (56 galaxies observed multiple times; keep first unique)
    pool = pool.sort_values("plateifu").drop_duplicates("mangaID").reset_index(drop=True)
    n_eligible = len(pool)
    print(f"[Catalog] Total eligible unique galaxies: {n_eligible}")

    n_target = min(int(cfg["n_target"]), n_eligible)
    rng = np.random.RandomState(int(cfg["seed"]))
    chosen_idx = rng.choice(n_eligible, size=n_target, replace=False)
    selected = pool.iloc[chosen_idx].copy().reset_index(drop=True)

    # Assign split at galaxy level (70% train, 15% validation, 15% test)
    ratios = cfg["sample"]["split_ratios"]
    n_train = int(np.floor(ratios[0] * n_target))
    n_val = int(np.floor(ratios[1] * n_target))
    splits = np.array(["train"] * n_train + ["validation"] * n_val + ["test"] * (n_target - n_train - n_val))
    rng.shuffle(splits)
    selected["split"] = splits

    atomic_save_csv(selected, selection_csv, index=False)
    print(f"[Catalog] Saved selection: {n_target} galaxies ({sum(splits=='train')} train, {sum(splits=='validation')} val, {sum(splits=='test')} test) -> {selection_csv.name}")
    return selected
