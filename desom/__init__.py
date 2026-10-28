"""
DESOM: Deep Embedded Self-Organizing Maps for Galaxy Spectra
Unified package for local and cloud execution.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

__version__ = "2.0.0"

from .config import Config, get_config, setup_hardware
from .io_utils import (
    atomic_save_json,
    atomic_save_npz,
    atomic_save_csv,
    sha256_file,
    load_json,
)

__all__ = [
    "Config",
    "get_config",
    "setup_hardware",
    "atomic_save_json",
    "atomic_save_npz",
    "atomic_save_csv",
    "sha256_file",
    "load_json",
]
