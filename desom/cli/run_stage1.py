"""
Stage 1 CLI: Catalog download, galaxy selection, and Voronoi spectroscopic processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import get_config, setup_hardware
from ..data.catalog import select_galaxies
from ..data.processor import process_single_galaxy
from ..io_utils import StateManager, tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Process MaNGA Spectroscopic Data")
    parser.add_argument("--storage", choices=["local", "drive"], default=None, help="Storage backend")
    parser.add_argument("--sample", type=int, default=None, help="Number of target galaxies (e.g. 500 or 4609)")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment run ID")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputing existing galaxies")
    args = parser.parse_args()

    cfg = get_config(storage_backend=args.storage, n_target=args.sample, run_id=args.run_id)
    setup_hardware(verbose=True)
    state = StateManager(cfg.run_dir / "pipeline_state.json")

    print(f"\n========================================================")
    print(f" DESOM Pipeline — Stage 1: Data Acquisition & Preprocessing")
    print(f" Run ID:   {cfg.run_id}")
    print(f" Storage:  {cfg.raw['storage']['backend']} ({cfg.run_dir})")
    print(f" Target:   {cfg['n_target']} galaxies")
    print(f" Limits:   {cfg['preprocessing']['rest_limits']} Angstroms")
    print(f"========================================================\n")

    # 1. Select galaxies
    selection = select_galaxies(cfg)
    state.update_stage("stage_01_spectra", "running", n_selected=len(selection))

    # 2. Process each galaxy
    success_count = 0
    total_candidates = 0

    for idx, row in tqdm(selection.iterrows(), total=len(selection), desc="Stage 1 Processing"):
        plateifu = str(row["plateifu"])
        manga_id = str(row["mangaID"])
        z = float(row["z"])

        try:
            res = process_single_galaxy(
                cfg,
                plateifu=plateifu,
                redshift=z,
                manga_id=manga_id,
                force_recompute=args.force_recompute,
            )
            if res.get("n_candidates_accepted", 0) > 0:
                success_count += 1
                total_candidates += res["n_candidates_accepted"]
        except Exception as e:
            print(f"\n[Warning] Error processing galaxy {plateifu}: {e}")

    retention_rate = (success_count / len(selection)) if len(selection) > 0 else 0.0
    print(f"\n[Stage 1 Complete]")
    print(f"Galaxies processed with valid spectra: {success_count}/{len(selection)} ({retention_rate:.1%})")
    print(f"Total candidate spectra extracted:    {total_candidates:,}")

    state.update_stage(
        "stage_01_spectra",
        "complete",
        galaxies_processed=len(selection),
        galaxies_with_spectra=success_count,
        total_candidates=total_candidates,
    )


if __name__ == "__main__":
    main()
