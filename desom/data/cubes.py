"""
MaNGA Data Cube downloader and reader.
Downloads 3D spectroscopic LOGCUBES from SDSS SAS archive.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from astropy.io import fits

from ..config import Config
from .catalog import download_file


def get_cube_path(cfg: Config, plateifu: str) -> Path:
    """Returns local path where LOGCUBE should reside."""
    return cfg.cubes_dir / f"manga-{plateifu}-LOGCUBE.fits.gz"


def download_galaxy_cube(cfg: Config, plateifu: str, verbose: bool = False) -> Path:
    """
    Downloads LOGCUBE for a galaxy if not already present.
    Verifies FITS integrity upon download.
    """
    destination = get_cube_path(cfg, plateifu)
    if destination.exists() and destination.stat().st_size > 1000:
        return destination

    plate = str(plateifu).split("-")[0]
    base_url = cfg["paths"]["manga_base_url"]
    url = f"{base_url}/{plate}/stack/manga-{plateifu}-LOGCUBE.fits.gz"

    download_file(url, destination, verbose=verbose)

    # Verify FITS integrity
    try:
        with fits.open(destination, memmap=False) as hdul:
            if len(hdul) < 2:
                raise ValueError("Corrupt FITS: less than 2 HDUs")
    except Exception as e:
        if destination.exists():
            destination.unlink()
        raise IOError(f"Downloaded cube for {plateifu} is invalid: {e}")

    return destination
