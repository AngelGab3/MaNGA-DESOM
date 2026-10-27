"""
Safe I/O Utilities: Atomic file operations, SHA256 checksums, and JSON/NPZ helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def now_utc_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Computes SHA-256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_save_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically writes JSON by saving to .tmp and renaming."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp_{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        # Flush to disk
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_json(path: Union[str, Path], default: Any = None) -> Any:
    """Loads JSON file or returns default if not found."""
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_npz(path: Union[str, Path], **arrays: Any) -> None:
    """Atomically writes compressed .npz archive."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp_{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_save_csv(df: Any, path: Union[str, Path], index: bool = False) -> None:
    """Atomically writes pandas DataFrame to CSV."""
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp_{os.getpid()}")
    try:
        df.to_csv(tmp, index=index)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class StateManager:
    """Tracks pipeline stages, progress, and execution states."""

    def __init__(self, state_file: Union[str, Path]):
        self.state_file = Path(state_file)
        self.state: Dict[str, Any] = load_json(self.state_file, default={})
        if not self.state:
            self.state = {
                "created_at": now_utc_iso(),
                "updated_at": now_utc_iso(),
                "stages": {
                    "stage_01_spectra": {"status": "pending"},
                    "stage_02_dataset": {"status": "pending"},
                    "stage_03_desom1": {"status": "pending"},
                    "stage_04_fingerprints": {"status": "pending"},
                    "stage_05_desom2": {"status": "pending"},
                },
            }
            self.save()

    def update_stage(self, stage_name: str, status: str, **kwargs: Any) -> None:
        """Updates status of a pipeline stage."""
        if stage_name not in self.state["stages"]:
            self.state["stages"][stage_name] = {}
        self.state["stages"][stage_name].update({
            "status": status,
            "updated_at": now_utc_iso(),
            **kwargs,
        })
        self.state["updated_at"] = now_utc_iso()
        self.save()

    def is_complete(self, stage_name: str) -> bool:
        """Checks if a stage is already marked as complete."""
        stage = self.state.get("stages", {}).get(stage_name, {})
        return stage.get("status") == "complete"

    def save(self) -> None:
        atomic_save_json(self.state_file, self.state)


class _SimpleProgressBar:
    """Clean progress bar fallback when tqdm is not installed (ideal for web log streaming)."""

    def __init__(self, iterable=None, total: Optional[int] = None, desc: str = "", initial: int = 0, **kwargs):
        self.iterable = iterable
        self.desc = desc or "Progress"
        self.initial = initial
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        self.total = total
        self.n = initial
        self.postfix = ""
        self._start_time = time.time()
        self._last_print_time = 0.0

    def __iter__(self):
        if self.iterable is None:
            return
        step = max(1, (self.total // 20)) if self.total and self.total > 20 else 1
        for idx, item in enumerate(self.iterable):
            self.n = self.initial + idx + 1
            now = time.time()
            if idx == 0 or self.n == self.total or (idx + 1) % step == 0 or (now - self._last_print_time) >= 2.0:
                self._last_print_time = now
                tot_str = f"/{self.total}" if self.total else ""
                pct_str = f" ({100.0 * self.n / self.total:.0f}%)" if self.total else ""
                post_str = f" | {self.postfix}" if self.postfix else ""
                print(f"[{self.desc}] {self.n}{tot_str}{pct_str}{post_str}", flush=True)
            yield item

    def set_postfix(self, ordered_dict=None, **kwargs):
        d = {}
        if ordered_dict:
            d.update(ordered_dict)
        d.update(kwargs)
        self.postfix = ", ".join(f"{k}={v}" for k, v in d.items())


try:
    from tqdm import tqdm as _real_tqdm
    tqdm = _real_tqdm
except ImportError:
    tqdm = _SimpleProgressBar

