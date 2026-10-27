"""
Stage 4 CLI: DESOM-2 Training and 10x10 Galaxy Diversity Map (Fig 10).
"""

from __future__ import annotations

import argparse
import numpy as np

from ..analysis.evaluation import evaluate_model
from ..config import get_config, setup_hardware
from ..io_utils import StateManager
from ..plots.galaxy_map_plots import plot_desom2_galaxy_grid
from ..training.trainer_desom2 import train_desom2


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: Train DESOM-2 and Map Galaxy Diversity")
    parser.add_argument("--storage", choices=["local", "drive"], default=None, help="Storage backend")
    parser.add_argument("--sample", type=int, default=None, help="Number of target galaxies")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment run ID")
    parser.add_argument("--epochs", type=int, default=None, help="Override DESOM-2 iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--no-resume", action="store_true", help="Start training from scratch")
    parser.add_argument("--map-distance", choices=["manhattan", "paper"], default=None,
                        help="Grid distance used by the SOM neighbourhood (default: value saved in the run config)")
    args = parser.parse_args()

    cfg = get_config(storage_backend=args.storage, n_target=args.sample, run_id=args.run_id)
    if args.map_distance:
        cfg.raw["training"]["map_distance"] = args.map_distance
        cfg.save()
    if args.epochs:
        cfg.raw["desom2"]["iterations"] = args.epochs
    if args.batch_size:
        cfg.raw["desom2"]["batch_size"] = args.batch_size

    setup_hardware(verbose=True)
    state = StateManager(cfg.run_dir / "pipeline_state.json")

    print(f"\n========================================================")
    print(f" DESOM Pipeline — Stage 4: DESOM-2 Training & Galaxy Map")
    print(f" Run ID:      {cfg.run_id}")
    print(f" Iterations:  {cfg['desom2']['iterations']}")
    print(f"========================================================\n")

    fp_path = cfg.run_dir / "galaxy_fingerprints.npz"
    if not fp_path.exists():
        raise FileNotFoundError(f"Missing {fp_path}. Run Stage 3 first!")

    with np.load(fp_path, allow_pickle=False) as data:
        fps = data["fingerprints"]
        splits = data["split"]
        manga_ids = data["mangaID"].tolist()
        plateifus = data["plateifu"].tolist()

    if len(fps) == 0:
        raise ValueError(f"No fingerprints found in {fp_path}. Run Stage 3 first!")

    idx_train = np.flatnonzero(splits == "train")
    if len(idx_train) == 0:
        idx_train = np.arange(len(fps))

    idx_val = np.flatnonzero(splits == "validation")
    if len(idx_val) == 0:
        idx_val = idx_train[: max(1, int(0.15 * len(idx_train)))]

    # 1. Train DESOM-2
    state.update_stage("stage_05_desom2", "running")
    model_d2, weights_path = train_desom2(
        cfg,
        X_train_fp=fps[idx_train],
        X_val_fp=fps[idx_val],
        resume=not args.no_resume,
    )
    state.update_stage("stage_05_desom2", "complete", weights=str(weights_path))

    # 2. Held-out evaluation on the test galaxies
    idx_test = np.flatnonzero(splits == "test")
    if len(idx_test) > 0:
        evaluate_model(model_d2, fps[idx_test], cfg.desom2_dir, tag="desom2", batch_size=256)

    # 3. Plot Figures 9 and 10: 10x10 Galaxy Diversity Map
    plot_desom2_galaxy_grid(
        cfg,
        model=model_d2,
        fingerprints=fps,
        plateifus=plateifus,
        manga_ids=manga_ids,
    )

    print("[Stage 4 Complete] Pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
