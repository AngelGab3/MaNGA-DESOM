"""
Stage 3 CLI: DESOM-1 training, prototype grid (paper Fig. 3), evaluation and fingerprints.
"""

from __future__ import annotations

import argparse
import numpy as np

from ..analysis.evaluation import evaluate_model
from ..analysis.fingerprints import generate_all_fingerprints
from ..config import get_config, setup_hardware
from ..io_utils import StateManager
from ..plots.fingerprint_plots import plot_galaxy_fingerprints_grid, plot_single_galaxy_fingerprint
from ..plots.spectral_plots import pick_sample_indices, plot_figure_desom1_prototypes
from ..training.trainer_desom1 import train_desom1


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Train DESOM-1 and Generate Galaxy Fingerprints")
    parser.add_argument("--storage", choices=["local", "drive"], default=None, help="Storage backend")
    parser.add_argument("--sample", type=int, default=None, help="Number of target galaxies")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment run ID")
    parser.add_argument("--epochs", type=int, default=None, help="Override DESOM-1 iterations")
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
        cfg.raw["desom1"]["iterations"] = args.epochs
    if args.batch_size:
        cfg.raw["desom1"]["batch_size"] = args.batch_size

    setup_hardware(verbose=True)
    state = StateManager(cfg.run_dir / "pipeline_state.json")

    print(f"\n========================================================")
    print(f" DESOM Pipeline — Stage 3: DESOM-1 Training & Fingerprints")
    print(f" Run ID:      {cfg.run_id}")
    print(f" Iterations:  {cfg['desom1']['iterations']}")
    print(f" Batch Size:  {cfg['desom1']['batch_size']}")
    print(f"========================================================\n")

    ready_npz = cfg.run_dir / "desom1_dataset_ready.npz"
    if not ready_npz.exists():
        raise FileNotFoundError(f"Missing {ready_npz}. Run Stage 2 first!")

    with np.load(ready_npz, allow_pickle=False) as data:
        X = data["X"]
        idx_train = data["idx_train"]
        idx_val = data["idx_val"]
        idx_test = data["idx_test"]
        wave_rest = data["wave_rest"]

    # 1. Train DESOM-1
    state.update_stage("stage_03_desom1", "running")
    model_d1, weights_path = train_desom1(
        cfg,
        X_train=X[idx_train],
        X_val=X[idx_val],
        resume=not args.no_resume,
    )
    state.update_stage("stage_03_desom1", "complete", weights=str(weights_path))

    # 2. Held-out evaluation (test MSE, topographic error, node occupancy)
    evaluate_model(model_d1, X[idx_test], cfg.desom1_dir, tag="desom1")

    # 3. Figure 3: 15x15 spectral prototypes with the S1..S9 spectra of Figure 1 located on it
    plot_figure_desom1_prototypes(cfg, model_d1, wave_rest, samples=X[pick_sample_indices(X)])

    # 4. Generate Fingerprints for all galaxies
    state.update_stage("stage_04_fingerprints", "running")
    fp_path = generate_all_fingerprints(cfg, model_desom1=model_d1)

    # 5. Fingerprint figures
    with np.load(fp_path, allow_pickle=False) as fp_data:
        fps = fp_data["fingerprints"]
        manga_ids = fp_data["mangaID"].tolist()
        plot_galaxy_fingerprints_grid(cfg, fps, manga_ids)
        # Figure 7 style: raw counts of the galaxy with the most baxels
        biggest = int(np.argmax(fp_data["n_baxels"]))
        plot_single_galaxy_fingerprint(cfg, fp_data["counts"][biggest], str(fp_data["plateifu"][biggest]))

    state.update_stage("stage_04_fingerprints", "complete", path=str(fp_path))
    print("[Stage 3 Complete] Fingerprints ready for DESOM-2.\n")


if __name__ == "__main__":
    main()
