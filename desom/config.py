"""
Configuration and Hardware Setup for DESOM Pipeline.
Supports dual storage (Local vs Google Drive) and adaptive GPU/CPU hardware.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================================
# Hardware Setup (Adaptive to GTX 1650, RTX series, Apple Silicon, CPU)
# ============================================================================
def setup_hardware(
    enable_memory_growth: bool = True,
    mixed_precision: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Detects and configures available compute devices (NVIDIA CUDA, Apple MPS, or CPU).
    Safely enables dynamic memory growth to prevent Out-Of-Memory (OOM) errors on GPUs
    with limited VRAM (e.g., GTX 1650 4GB).
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "backend": "cpu",
        "devices": [],
        "memory_growth": False,
        "mixed_precision": False,
    }

    try:
        import tensorflow as tf

        # 1. Check physical GPUs (NVIDIA / ROCm)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["backend"] = "gpu"
            info["devices"] = [gpu.name for gpu in gpus]
            if enable_memory_growth:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        if verbose:
                            print(f"[Hardware] Warning setting memory growth on {gpu.name}: {e}")
                info["memory_growth"] = True

            if mixed_precision:
                try:
                    from tensorflow.keras import mixed_precision as mp
                    mp.set_global_policy("mixed_float16")
                    info["mixed_precision"] = True
                except Exception as e:
                    if verbose:
                        print(f"[Hardware] Warning enabling mixed precision: {e}")

            if verbose:
                print(f"[Hardware] GPU detected ({len(gpus)} device(s)): {[g.name for g in gpus]}")
                print(f"[Hardware] Memory growth: {info['memory_growth']} | Mixed precision: {info['mixed_precision']}")
            return info

        # 2. Check Apple Silicon MPS
        try:
            # Apple Silicon / macOS check
            if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
                # TensorFlow Metal check
                info["backend"] = "apple_silicon"
                info["devices"] = ["Apple Silicon Metal"]
                if verbose:
                    print("[Hardware] Running on Apple Silicon (MPS / CPU backend).")
                return info
        except Exception:
            pass

        # 3. CPU fallback
        if verbose:
            print("[Hardware] No GPU detected. Falling back to multi-threaded CPU.")
        info["backend"] = "cpu"
        info["devices"] = ["CPU"]
        return info

    except ImportError:
        if verbose:
            print("[Hardware] TensorFlow not imported yet. Proceeding with CPU-safe defaults.")
        return info


# ============================================================================
# Master Configuration with Paper Defaults
# ============================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline_version": "2.0.0",
    "seed": 20261002,
    "n_target": 500,  # 500 for validation sample, 4609 for full DR15 paper dataset

    "storage": {
        "backend": "drive",  # "local" or "drive" (nube)
        "local_root": "./workspace_data",
        "drive_root": "/content/drive/MyDrive/GalaxyAE_colab",
        "experiments_dir": "experiments",
    },

    "paths": {
        "manga_base_url": "https://data.sdss.org/sas/dr15/manga/spectro/redux/v2_4_3",
        "drpall_url": "https://data.sdss.org/sas/dr15/manga/spectro/redux/v2_4_3/drpall-v2_4_3.fits",
    },

    "sample": {
        "split_ratios": [0.70, 0.15, 0.15],
        # "galaxy": every baxel of a galaxy goes to the same split (no leakage between splits).
        # "spectrum": baxels are shuffled and split individually (as in the original paper).
        "split_mode": "galaxy",
        "min_redshift": 0.0,
        "max_redshift": 0.15,
        "mngtarg1_min": 1,
        "srvymode": "MaNGA dither",
        "versdrp3": "v2_4_3",
        "drp3qual_exclude_bit": 30,  # CRITICAL flag
    },

    "preprocessing": {
        # CRITICAL FIX FROM PAPER:
        # Paper explicitly sets 3650 to 8950 Angstroms to guarantee coverage across all redshifts (0 < z < 0.15)
        "rest_limits": [3650.0, 8950.0],
        "n_wave": 4544,
        "sn_window_rest": [4050.0, 4150.0],
        "continuum_rest": [4050.0, 4150.0],
        "target_sn": 20.0,
        "min_spaxel_sn": 2.0,  # Paper Section 2: masked individual spaxels having S/N < 2
        # Paper Section 2: baxels with S/N < 10 are removed from the training set.
        # (The paper also reports that using S/N < 5 does not change results qualitatively.)
        "min_baxel_sn": 10.0,
        "covariance_in_binning": True,
        "cvt": True,
        "wvt": True,
        "min_continuum_coverage": 0.8,
        "bad_drp_pixel_bits": [0, 1, 2, 3, 10],
        "sky_observed": [5577.0, 5895.0, 6300.0],
        "sky_half_width": 10.0,
        "min_spatial_coverage": 0.8,
        "max_interpolation_gap_A": 50.0,
        "min_good_rebinned": 3900,
        "low_flux_percentile": 0.5,
        "high_flux_percentile": 0.5,
        # Global (cross-galaxy) flux outlier cuts from Paper Section 2. Applied in consolidate.py.
        "apply_flux_outlier_cuts": True,
        "em_line_half_width_bins": 3,
        # Emission lines centers (at rest) to protect from high-flux cut
        "em_lines": [3727.0, 4861.0, 4959.0, 5007.0, 6563.0, 6584.0, 6717.0, 6731.0],
    },

    "training": {
        # Distance between SOM nodes used by the neighbourhood function.
        # "manhattan": exact |dr| + |dc| on the grid (default).
        # "paper": replicates the original code, |a - b| // width + |dc| (row term is approximate).
        "map_distance": "manhattan",
    },

    "desom1": {
        "map_size": [15, 15],
        "latent": 256,
        "gamma": 5e-4,
        "lr": 5e-5,
        "batch_size": 128,      # Safe for 4GB GPUs (GTX 1650); use 64 if running background desktop tasks
        "iterations": 65000,
        "som_iterations": 50000,
        "Tmax": 15.0,
        "Tmin": 0.7,
        "checkpoint_interval": 1000,
        "log_interval": 100,
    },

    "desom2": {
        "map_size": [10, 10],
        "latent_dims": 128,
        "gamma": 8e-6,
        "lr": 1e-4,
        "batch_size": 1024,     # Input is only 15x15x1; 1024 fits easily in any GPU
        "iterations": 40000,
        "som_iterations": 35000,
        "Tmax": 15.0,
        "Tmin": 0.5,
        "checkpoint_interval": 1000,
        "log_interval": 100,
        "clipping_percentile": 99.9,
    },

    "monitoring": {
        "progress_every_n_galaxies": 10,
        "log_every_n_galaxies": 5,
    },
}


# ============================================================================
# Config Object with Dynamic Path Resolvers
# ============================================================================
class Config:
    """
    Manages active configuration, experiment paths, and storage abstraction.
    """

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        storage_backend: Optional[str] = None,
        run_id: Optional[str] = None,
        n_target: Optional[int] = None,
    ):
        import copy
        self.raw = copy.deepcopy(DEFAULT_CONFIG)
        if config_dict:
            self._deep_update(self.raw, config_dict)

        if storage_backend:
            self.raw["storage"]["backend"] = storage_backend
        if n_target is not None:
            self.raw["n_target"] = int(n_target)

        self.run_id = run_id or f"run_{self.raw['n_target']}g"
        self._init_paths()

    def _deep_update(self, base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def _init_paths(self) -> None:
        backend = self.raw["storage"]["backend"].lower()
        if backend in ("drive", "colab", "cloud"):
            drive_path = Path(self.raw["storage"]["drive_root"]).expanduser()
            # If running on macOS/Windows/Linux outside Colab and drive_root is /content/drive/...
            if str(drive_path).startswith("/content/") and not Path("/content/drive").exists():
                # Check for Google Drive for Desktop on macOS
                cloud_storage = Path.home() / "Library" / "CloudStorage"
                gdrive_candidates = []
                try:
                    if cloud_storage.exists():
                        for p in cloud_storage.glob("GoogleDrive*"):
                            for sub in ("My Drive", "Mi unidad"):
                                if (p / sub).exists():
                                    gdrive_candidates.append(p / sub / "GalaxyAE_colab")
                except OSError:
                    pass
                if gdrive_candidates:
                    root = gdrive_candidates[0]
                else:
                    # Safe local cloud-cache directory when /content/drive is not mounted
                    root = Path(self.raw["storage"]["local_root"]).resolve()
            else:
                root = drive_path
        else:
            root = Path(self.raw["storage"]["local_root"]).expanduser().resolve()

        self.root_dir = root
        self.experiments_dir = root / self.raw["storage"]["experiments_dir"]
        self.run_dir = self.experiments_dir / self.run_id

        # Subdirectories for clean pipeline stages
        self.catalogs_dir = self.run_dir / "catalogs"
        self.cubes_dir = self.run_dir / "cubes"
        self.processed_dir = self.run_dir / "processed"
        self.desom1_dir = self.run_dir / "desom1"
        self.desom2_dir = self.run_dir / "desom2"
        self.figures_dir = self.run_dir / "figures"
        self.logs_dir = self.run_dir / "logs"

    def ensure_dirs(self) -> None:
        """Creates all necessary directories on disk."""
        for d in [
            self.root_dir,
            self.experiments_dir,
            self.run_dir,
            self.catalogs_dir,
            self.cubes_dir,
            self.processed_dir,
            self.desom1_dir,
            self.desom2_dir,
            self.figures_dir,
            self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Saves current config to JSON."""
        target = Path(path) if path else (self.run_dir / "config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.raw, f, indent=2)
        return target

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def get_config(
    storage_backend: Optional[str] = None,
    n_target: Optional[int] = None,
    run_id: Optional[str] = None,
    custom_config_path: Optional[str] = None,
) -> Config:
    """Helper to instantiate a Config instance, preserving saved run config.json if present."""
    cfg_dict = None
    if custom_config_path and Path(custom_config_path).exists():
        with open(custom_config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

    cfg = Config(
        config_dict=cfg_dict,
        storage_backend=storage_backend,
        run_id=run_id,
        n_target=n_target,
    )

    saved_cfg_path = cfg.run_dir / "config.json"
    if custom_config_path is None and saved_cfg_path.exists():
        try:
            with open(saved_cfg_path, "r", encoding="utf-8") as f:
                saved_dict = json.load(f)
            cfg._deep_update(cfg.raw, saved_dict)
            if storage_backend is not None:
                cfg.raw["storage"]["backend"] = storage_backend
            if n_target is not None:
                cfg.raw["n_target"] = int(n_target)
            cfg._init_paths()
        except Exception:
            pass

    cfg.ensure_dirs()
    if not saved_cfg_path.exists():
        cfg.save()
    return cfg
