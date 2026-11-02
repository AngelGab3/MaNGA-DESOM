"""
CLI entry point to generate all scientific data figures from Teimoorinia et al. 2021 (arXiv:2112.03425).
Figures: 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14.
"""

from __future__ import annotations

import argparse
import numpy as np

from ..config import get_config
from ..plots.fingerprint_plots import plot_galaxy_fingerprints_grid, plot_single_galaxy_fingerprint
from ..plots.galaxy_map_plots import plot_desom2_galaxy_grid
from ..plots.paper_figures import (
    generate_figure_4_coadded_templates,
    generate_figure_13_morphology,
    generate_figure_14_sersic_ba,
    generate_figures_5_and_6_physical_parameters,
    generate_figures_11_and_12_optical_images,
    load_desom1_model,
    load_desom2_model,
)
from ..plots.spectral_plots import (
    pick_sample_indices,
    plot_figure_1_normalized_spectra,
    plot_figure_desom1_prototypes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all scientific paper figures")
    parser.add_argument("--run-id", type=str, default="prueba500", help="Experiment run ID")
    parser.add_argument("--storage", choices=["local", "drive"], default=None, help="Storage backend")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        help="List of figures to generate (e.g. 1 3 4 5 6 7 9 10 11 12 13 14 or 'all')",
    )
    args = parser.parse_args()

    cfg = get_config(storage_backend=args.storage, run_id=args.run_id)

    print("\n========================================================")
    print(" DESOM Pipeline — Generating Scientific Paper Figures")
    print(f" Run ID:      {cfg.run_id}")
    print(f" Target Dir:  {cfg.figures_dir}")
    print(f" Figures:     {args.figures}")
    print("========================================================\n")

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    figs_to_run = set(args.figures)
    run_all = "all" in figs_to_run

    model1 = load_desom1_model(cfg)
    model2 = load_desom2_model(cfg)

    # Figures 1 & 3 (shared S1..S9 sample indices across 9 distinct DESOM-1 nodes)
    if run_all or "1" in figs_to_run or "3" in figs_to_run:
        print("\n--- Generating Figures 1 & 3 (Normalized Spectra S1..S9 & 15x15 Prototypes) ---")
        with np.load(cfg.run_dir / "desom1_dataset_ready.npz", allow_pickle=False) as d1_data:
            X = d1_data["X"]
            wave_rest = d1_data["wave_rest"]
            sample_idx = pick_sample_indices(X, n_samples=9, wave_rest=wave_rest, model=model1)
            if run_all or "1" in figs_to_run:
                plot_figure_1_normalized_spectra(
                    cfg, X, wave_rest, sample_indices=sample_idx, model=model1
                )
            if run_all or "3" in figs_to_run:
                plot_figure_desom1_prototypes(
                    cfg, model1, wave_rest, samples=X[sample_idx]
                )

    # Figure 4
    if run_all or "4" in figs_to_run:
        print("\n--- Generating Figure 4 (Co-added Templates) ---")
        generate_figure_4_coadded_templates(cfg, model1=model1)

    # Figures 5 & 6
    if run_all or "5" in figs_to_run or "6" in figs_to_run:
        print("\n--- Generating Figures 5 & 6 (Physical Parameters on SOM-1 & Histograms) ---")
        generate_figures_5_and_6_physical_parameters(cfg, model1=model1)

    # Figure 7
    if run_all or "7" in figs_to_run:
        print("\n--- Generating Figure 7 (Single Galaxy Prototype Fingerprint colored by R/Re) ---")
        with np.load(cfg.run_dir / "galaxy_fingerprints.npz", allow_pickle=False) as fp_data:
            biggest = int(np.argmax(fp_data["n_baxels"]))
            plot_single_galaxy_fingerprint(
                cfg,
                fp_data["counts"][biggest],
                str(fp_data["plateifu"][biggest]),
                model=model1,
            )
            plot_galaxy_fingerprints_grid(cfg, fp_data["fingerprints"], fp_data["mangaID"].tolist())

    # Figures 9 & 10
    if run_all or "9" in figs_to_run or "10" in figs_to_run:
        print("\n--- Generating Figures 9 & 10 (DESOM-2 Occupancy & 10x10 Prototype Fingerprints) ---")
        with np.load(cfg.run_dir / "galaxy_fingerprints.npz", allow_pickle=False) as fp_data:
            plot_desom2_galaxy_grid(
                cfg,
                model=model2,
                fingerprints=fp_data["fingerprints"],
                plateifus=fp_data["plateifu"].tolist(),
                manga_ids=fp_data["mangaID"].tolist(),
            )

    # Figures 11 & 12
    if run_all or "11" in figs_to_run or "12" in figs_to_run:
        print("\n--- Generating Figures 11 & 12 (Optical Galaxy Cutouts 10x10 & 4x9 Node Zoom) ---")
        generate_figures_11_and_12_optical_images(cfg, model2=model2)

    # Figure 13
    if run_all or "13" in figs_to_run:
        print("\n--- Generating Figure 13 (Morphology Proportions on DESOM-2) ---")
        generate_figure_13_morphology(cfg, model2=model2)

    # Figure 14
    if run_all or "14" in figs_to_run:
        print("\n--- Generating Figure 14 (Sersic n & b/a on DESOM-2) ---")
        generate_figure_14_sersic_ba(cfg, model2=model2)

    print("\n========================================================")
    print(" All requested paper figures generation finished!")
    print(f" Check output in: {cfg.figures_dir}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
