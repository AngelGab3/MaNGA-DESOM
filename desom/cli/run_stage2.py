"""
Stage 2 CLI: Dataset consolidation, rigorous audit, train/val/test splits, and Figure 1.
"""

from __future__ import annotations

import argparse
import numpy as np

from ..config import get_config, setup_hardware
from ..data.consolidate import audit_and_split_dataset, consolidate_stage1_spectra
from ..io_utils import StateManager
from ..plots.spectral_plots import plot_figure_1_normalized_spectra


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Consolidate and Audit Spectroscopic Dataset")
    parser.add_argument("--storage", choices=["local", "drive"], default=None, help="Storage backend")
    parser.add_argument("--sample", type=int, default=None, help="Number of target galaxies")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment run ID")
    parser.add_argument("--split-mode", choices=["galaxy", "spectrum"], default=None,
                        help="Train/val/test split unit (default: value saved in the run config)")
    args = parser.parse_args()

    cfg = get_config(storage_backend=args.storage, n_target=args.sample, run_id=args.run_id)
    if args.split_mode:
        cfg.raw["sample"]["split_mode"] = args.split_mode
        cfg.save()
    setup_hardware(verbose=False)
    state = StateManager(cfg.run_dir / "pipeline_state.json")

    print(f"\n========================================================")
    print(f" DESOM Pipeline — Stage 2: Consolidation & Audit")
    print(f" Run ID:   {cfg.run_id}")
    print(f"========================================================\n")

    # 1. Consolidate & Audit
    state.update_stage("stage_02_dataset", "running")
    ready_npz = audit_and_split_dataset(cfg)

    # 2. Plot Figure 1 (Normalized Spectra)
    with np.load(ready_npz, allow_pickle=False) as data:
        X = data["X"]
        wave_rest = data["wave_rest"]
        plot_figure_1_normalized_spectra(cfg, X, wave_rest)

    state.update_stage("stage_02_dataset", "complete", path=str(ready_npz))
    print("[Stage 2 Complete] Ready for DESOM-1 training.\n")


if __name__ == "__main__":
    main()
