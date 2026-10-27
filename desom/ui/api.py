"""
DESOM UI API Handlers.
Calculates dataset demographic statistics, pipeline status, diagnostics,
and model inspection data for the web dashboard.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

from ..config import Config, get_config, setup_hardware
from ..io_utils import StateManager, load_json, sha256_file


class DashboardAPI:
    """Provides backend data and control operations for the UI."""

    def __init__(self, storage_backend: Optional[str] = None, n_target: Optional[int] = None, run_id: Optional[str] = None):
        self.cfg = get_config(storage_backend=storage_backend, n_target=n_target, run_id=run_id)
        self.running_process: Optional[subprocess.Popen] = None
        self.process_lock = threading.Lock()
        self.last_logs: List[str] = []

    def get_hardware_info(self) -> Dict[str, Any]:
        """Returns detected compute hardware."""
        return setup_hardware(verbose=False)

    def get_active_config(self) -> Dict[str, Any]:
        """Returns current full configuration dictionary."""
        return self.cfg.raw

    def save_active_config(self, new_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Updates and saves active configuration."""
        self.cfg._deep_update(self.cfg.raw, new_cfg)
        self.cfg._init_paths()
        self.cfg.ensure_dirs()
        saved_path = self.cfg.save()
        return {"status": "saved", "path": str(saved_path), "config": self.cfg.raw}

    def reset_to_paper_defaults(self) -> Dict[str, Any]:
        """Resets active config to paper defaults."""
        import copy
        from ..config import DEFAULT_CONFIG
        self.cfg.raw = copy.deepcopy(DEFAULT_CONFIG)
        saved_path = self.cfg.save()
        return {"status": "reset", "path": str(saved_path), "config": self.cfg.raw}

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Lists available experiment runs in the experiments directory."""
        exp_dir = self.cfg.experiments_dir
        if not exp_dir.exists():
            return []

        runs = []
        for d in sorted(exp_dir.iterdir(), reverse=True):
            if d.is_dir() and not d.name.startswith("."):
                state = load_json(d / "pipeline_state.json", default={})
                cfg_data = load_json(d / "config.json", default={})
                selection_path = d / "selection.csv"
                n_galaxies = 0
                if selection_path.exists():
                    try:
                        import csv
                        with open(selection_path, "r", encoding="utf-8") as f:
                            n_galaxies = max(0, sum(1 for _ in f) - 1)
                    except Exception:
                        pass

                target_val = cfg_data.get("n_target")
                if target_val is None:
                    import re
                    m = re.search(r"(\d+)", d.name)
                    target_val = int(m.group(1)) if m else self.cfg["n_target"]

                runs.append({
                    "run_id": d.name,
                    "path": str(d),
                    "created_at": state.get("created_at", "N/A"),
                    "updated_at": state.get("updated_at", "N/A"),
                    "stages": state.get("stages", {}),
                    "n_target": target_val,
                    "n_galaxies": n_galaxies,
                    "is_active": (d.name == self.cfg.run_id),
                })
        return runs

    def switch_run(self, run_id: str) -> None:
        """Switches active run, reloading its saved config and target sample size."""
        self.cfg = get_config(
            storage_backend=self.cfg.raw["storage"]["backend"],
            n_target=None,
            run_id=run_id,
        )

    def create_run(
        self,
        run_id: str,
        n_target: int = 500,
        clone_current: bool = True,
    ) -> Dict[str, Any]:
        """Creates a new experiment directory with fresh state and config."""
        clean_id = "".join(c for c in run_id.strip() if c.isalnum() or c in ("_", "-"))
        if not clean_id:
            return {"status": "error", "message": "El nombre de la corrida no es válido. Usa letras, números o guiones."}

        target_dir = self.cfg.experiments_dir / clean_id
        if target_dir.exists():
            return {"status": "error", "message": f"La corrida '{clean_id}' ya existe."}

        import copy
        if clone_current:
            base_config = copy.deepcopy(self.cfg.raw)
        else:
            from ..config import DEFAULT_CONFIG
            base_config = copy.deepcopy(DEFAULT_CONFIG)

        base_config["n_target"] = int(n_target)

        self.cfg = Config(
            config_dict=base_config,
            storage_backend=base_config["storage"]["backend"],
            run_id=clean_id,
            n_target=int(n_target),
        )
        self.cfg.ensure_dirs()
        self.cfg.save()

        # Initialize fresh state
        state_mgr = StateManager(self.cfg.run_dir / "pipeline_state.json")
        state_mgr.save()

        return {
            "status": "created",
            "run_id": clean_id,
            "n_target": int(n_target),
            "path": str(self.cfg.run_dir),
        }

    def get_pipeline_state(self) -> Dict[str, Any]:
        """Returns current state and progress for active run."""
        state_file = self.cfg.run_dir / "pipeline_state.json"
        return load_json(state_file, default={
            "run_id": self.cfg.run_id,
            "stages": {
                "stage_01_spectra": {"status": "pending"},
                "stage_02_dataset": {"status": "pending"},
                "stage_03_desom1": {"status": "pending"},
                "stage_04_fingerprints": {"status": "pending"},
                "stage_05_desom2": {"status": "pending"},
            }
        })

    def get_dataset_demographics(self) -> Dict[str, Any]:
        """
        Extracts comprehensive demographic statistics of the selected sample:
        - Redshift min, max, mean, median, std, percentiles
        - Splits distribution (Train / Val / Test)
        - Baxels per galaxy (if processed)
        - Histogram data ready for Plotly
        """
        import csv
        import math

        selection_csv = self.cfg.run_dir / "selection.csv"
        if not selection_csv.exists():
            return {"error": "selection.csv no encontrado. Ejecuta la Etapa 1 o selecciona una muestra."}

        rows = []
        try:
            with open(selection_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception as e:
            return {"error": f"Error leyendo selection.csv: {e}"}

        n_total = len(rows)
        if n_total == 0:
            return {"error": "selection.csv está vacío."}

        z_values = []
        splits_counts: Dict[str, int] = {}
        sample_rows = []

        for idx, r in enumerate(rows):
            try:
                zv = float(r.get("z", 0.0))
                if math.isfinite(zv):
                    z_values.append(zv)
            except ValueError:
                pass

            sp = r.get("split", "unknown")
            splits_counts[sp] = splits_counts.get(sp, 0) + 1

            if idx < 100:
                sample_rows.append({
                    "plateifu": r.get("plateifu", ""),
                    "mangaID": r.get("mangaID", ""),
                    "z": float(r.get("z", 0.0)),
                    "split": sp,
                })

        z_values.sort()
        z_min = min(z_values) if z_values else 0.0
        z_max = max(z_values) if z_values else 0.0
        z_mean = (sum(z_values) / len(z_values)) if z_values else 0.0
        z_median = z_values[len(z_values) // 2] if z_values else 0.0
        variance = sum((x - z_mean) ** 2 for x in z_values) / max(len(z_values), 1)
        z_std = math.sqrt(variance)

        # 30 Bins histogram
        n_bins = 30
        hist_counts = [0] * n_bins
        bin_centers = []
        bin_width = (z_max - z_min) / max(n_bins, 1) if z_max > z_min else 1.0

        for i in range(n_bins):
            bin_centers.append(round(z_min + (i + 0.5) * bin_width, 5))

        if z_max > z_min:
            for x in z_values:
                b_idx = min(int((x - z_min) / bin_width), n_bins - 1)
                hist_counts[b_idx] += 1

        # Check for processed baxels
        baxels_per_gal: List[int] = []
        for r in rows:
            cand_file = self.cfg.processed_dir / str(r.get("plateifu")) / "candidate_spectra.npz"
            if cand_file.exists():
                try:
                    if np is not None:
                        with np.load(cand_file, allow_pickle=False) as d:
                            baxels_per_gal.append(len(d["baxel_id"]))
                except Exception:
                    pass

        baxel_stats = {}
        if baxels_per_gal:
            baxel_stats = {
                "mean": float(sum(baxels_per_gal) / len(baxels_per_gal)),
                "median": float(sorted(baxels_per_gal)[len(baxels_per_gal) // 2]),
                "min": int(min(baxels_per_gal)),
                "max": int(max(baxels_per_gal)),
                "total_spectra": int(sum(baxels_per_gal)),
                "galaxies_processed": len(baxels_per_gal),
            }

        return {
            "n_total": n_total,
            "redshift": {
                "min": round(z_min, 5),
                "max": round(z_max, 5),
                "mean": round(z_mean, 5),
                "median": round(z_median, 5),
                "std": round(z_std, 5),
                "hist_counts": hist_counts,
                "hist_bins": bin_centers,
            },
            "splits": splits_counts,
            "baxels": baxel_stats,
            "sample_galaxies": sample_rows,
        }

    def get_preprocessing_diagnostics(self) -> Dict[str, Any]:
        """
        Inspects processed baxels and rejections across all processed galaxies.
        Diagnoses why any baxels/galaxies were accepted or rejected.
        """
        processed_dir = self.cfg.processed_dir
        if not processed_dir.exists():
            return {"error": "Directorio processed no existe aún."}

        reasons_counter: Dict[str, int] = {}
        total_baxels = 0
        total_accepted = 0
        galaxies_checked = 0
        galaxies_with_accepted = 0

        for pdir in processed_dir.iterdir():
            if not pdir.is_dir() or pdir.name.startswith("."):
                continue

            manifest_file = pdir / "manifest.json"
            if manifest_file.exists():
                galaxies_checked += 1
                try:
                    with open(manifest_file, "r") as f:
                        m = json.load(f)
                    summary = m.get("summary", {})
                    bax_tot = summary.get("n_baxels_total", 0)
                    bax_acc = summary.get("n_candidates_accepted", 0)
                    total_baxels += bax_tot
                    total_accepted += bax_acc
                    if bax_acc > 0:
                        galaxies_with_accepted += 1
                except Exception:
                    pass

        retention_rate = (galaxies_with_accepted / galaxies_checked) if galaxies_checked > 0 else 0.0
        baxel_rate = (total_accepted / total_baxels) if total_baxels > 0 else 0.0

        return {
            "galaxies_checked": galaxies_checked,
            "galaxies_retained": galaxies_with_accepted,
            "galaxy_retention_rate": round(retention_rate * 100.0, 2),
            "total_baxels": total_baxels,
            "total_accepted_baxels": total_accepted,
            "baxel_acceptance_rate": round(baxel_rate * 100.0, 2),
            "rest_limits": self.cfg["preprocessing"]["rest_limits"],
        }

    def get_sample_spectra(self, n_samples: int = 5) -> Dict[str, Any]:
        """
        Returns a few sample normalized spectra for interactive plotting.
        """
        ready_npz = self.cfg.run_dir / "desom1_dataset_ready.npz"
        cand_npz = self.cfg.run_dir / f"manga_{self.cfg['n_target']}_spectra.npz"
        target_file = ready_npz if ready_npz.exists() else cand_npz

        if not target_file.exists():
            return {"error": "No se encontraron espectros consolidados (Etapa 2 pendiente)."}

        if np is None:
            py_exec = self._resolve_python_executable()
            script = (
                f"import json, numpy as np; "
                f"d = np.load(r'{target_file}'); "
                f"X = d['X']; w = d['wave_rest']; "
                f"n = min({n_samples}, len(X)); "
                f"rng = np.random.RandomState(42); "
                f"idx = rng.choice(len(X), size=n, replace=False); "
                f"specs = [{{'index': int(i), 'flux': X[i].tolist()[::4]}} for i in idx]; "
                f"print(json.dumps({{'wavelength': w.tolist()[::4], 'spectra': specs, 'n_wave_total': len(w)}}))"
            )
            try:
                res = subprocess.run([py_exec, "-c", script], capture_output=True, text=True, timeout=10)
                if res.returncode == 0:
                    return json.loads(res.stdout.strip())
                return {"error": f"Error leyendo espectros: {res.stderr}"}
            except Exception as e:
                return {"error": f"Error ejecutando lector de espectros: {e}"}

        with np.load(target_file, allow_pickle=False) as data:
            X = data["X"]
            wave_rest = data["wave_rest"]
            if len(X) == 0:
                return {"error": "El dataset no contiene espectros aún."}
            n = min(n_samples, len(X))
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), size=n, replace=False)

            spectra_list = []
            for i in idx:
                spectra_list.append({
                    "index": int(i),
                    "flux": X[i].tolist()[::4],  # Subsample by 4 for lightweight JSON transmission
                })

            return {
                "wavelength": wave_rest.tolist()[::4],
                "spectra": spectra_list,
                "n_wave_total": len(wave_rest),
            }

    def get_generated_figures(self) -> List[Dict[str, Any]]:
        """Lists generated figures in the figures directory."""
        fig_dir = self.cfg.figures_dir
        if not fig_dir.exists():
            return []

        figs = []
        for f in sorted(fig_dir.iterdir()):
            if f.suffix.lower() in (".png", ".pdf", ".svg"):
                figs.append({
                    "name": f.name,
                    "size_kb": round(f.stat().st_size / 1024.0, 1),
                    "url": f"/files/figures/{f.name}",
                    "is_image": f.suffix.lower() in (".png", ".svg"),
                })
        return figs

    def _resolve_python_executable(self) -> str:
        """Finds a Python interpreter with scientific packages (numpy/astropy) installed, falling back to sys.executable."""
        candidates = [sys.executable]
        home = Path.home()
        for p in [
            "/opt/miniconda3/bin/python",
            "/opt/conda/bin/python",
            str(home / "miniconda3" / "bin" / "python"),
            str(home / "anaconda3" / "bin" / "python"),
            "/usr/bin/python3",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
        ]:
            if p not in candidates and Path(p).exists():
                candidates.append(p)

        # Check conda envs in standard Linux and Mac locations
        for env_root in [
            Path("/opt/miniconda3/envs"),
            Path("/opt/conda/envs"),
            home / "miniconda3" / "envs",
            home / "anaconda3" / "envs",
        ]:
            try:
                if env_root.exists():
                    for env_py in env_root.glob("*/bin/python"):
                        candidates.insert(1, str(env_py))
            except OSError:
                pass

        for py in candidates:
            try:
                res = subprocess.run(
                    [py, "-c", "import numpy, astropy"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                if res.returncode == 0:
                    return py
            except Exception:
                continue
        return sys.executable

    def run_pipeline_stage(self, stage: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Launches a pipeline stage asynchronously in a subprocess."""
        with self.process_lock:
            if self.running_process is not None and self.running_process.poll() is None:
                return {"status": "error", "message": "Ya hay un proceso del pipeline en ejecución."}

            sample = sample_size or self.cfg["n_target"]
            py_exec = self._resolve_python_executable()
            cmd = [
                py_exec,
                "-u",
                "-m",
                f"desom.cli.run_stage{stage}",
                "--storage",
                self.cfg.raw["storage"]["backend"],
                "--sample",
                str(sample),
                "--run-id",
                self.cfg.run_id,
            ]
            if stage == "1":
                workers_val = self.cfg["preprocessing"].get("max_workers", 0)
                if workers_val > 0:
                    cmd.extend(["--workers", str(workers_val)])

            log_file = self.cfg.logs_dir / f"stage_{stage}.log"
            self.cfg.logs_dir.mkdir(parents=True, exist_ok=True)
            f_out = open(log_file, "w", buffering=1)

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"

            self.running_process = subprocess.Popen(
                cmd,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path.cwd()),
                env=env,
            )

            return {
                "status": "started",
                "stage": stage,
                "pid": self.running_process.pid,
                "cmd": " ".join(cmd),
                "log_file": str(log_file),
            }

    def get_process_status(self) -> Dict[str, Any]:
        """Checks if a background pipeline run is still active."""
        with self.process_lock:
            if self.running_process is None:
                return {"is_running": False, "exit_code": None}
            ret = self.running_process.poll()
            return {
                "is_running": (ret is None),
                "exit_code": ret,
            }
