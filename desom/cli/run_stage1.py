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
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel CPU worker processes (0 = Auto)")
    parser.add_argument("--download-workers", type=int, default=5, help="Simultaneous cube downloads (network-bound)")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment run ID")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputing existing galaxies")
    args = parser.parse_args()

    cfg = get_config(storage_backend=args.storage, n_target=args.sample, run_id=args.run_id)
    state = StateManager(cfg.run_dir / "pipeline_state.json")

    # Resolve parallel workers
    cfg_workers = cfg["preprocessing"].get("max_workers", 0)
    if args.workers is not None and args.workers > 0:
        n_workers = int(args.workers)
    elif cfg_workers > 0:
        n_workers = int(cfg_workers)
    else:
        import os
        n_workers = max(1, min(16, (os.cpu_count() or 2) - 1))

    print(f"\n========================================================")
    print(f" DESOM Pipeline — Stage 1: Data Acquisition & Preprocessing")
    print(f" Run ID:   {cfg.run_id}")
    print(f" Storage:  {cfg.raw['storage']['backend']} ({cfg.run_dir})")
    print(f" Target:   {cfg['n_target']} galaxies")
    print(f" Workers:  {n_workers} CPU parallel processes")
    print(f" Limits:   {cfg['preprocessing']['rest_limits']} Angstroms")
    print(f"========================================================\n")

    # 1. Select galaxies
    selection = select_galaxies(cfg)
    state.update_stage("stage_01_spectra", "running", n_selected=len(selection))

    # 2. Process each galaxy (Parallel or Sequential)
    success_count = 0
    total_candidates = 0

    if n_workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from ..data.processor import _worker_process_galaxy_task

        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from ..data.cubes import download_galaxy_cube, get_cube_path

        tasks = [
            (cfg.raw, str(row["plateifu"]), str(row["mangaID"]), float(row["z"]), args.force_recompute, cfg.run_id)
            for _, row in selection.iterrows()
        ]

        def _is_ready(plateifu: str) -> bool:
            """Galaxy needs no download: already processed or cube already on disk."""
            if (cfg.processed_dir / plateifu / "manifest.json").exists() and not args.force_recompute:
                return True
            cube = get_cube_path(cfg, plateifu)
            return cube.exists() and cube.stat().st_size > 1000

        ready = [t for t in tasks if _is_ready(t[1])]
        to_download = [t for t in tasks if not _is_ready(t[1])]
        dl_workers = max(1, min(int(args.download_workers), max(1, len(to_download))))
        print(
            f"[Stage 1] {len(tasks)} galaxias: {len(ready)} listas en disco, {len(to_download)} por descargar "
            f"({dl_workers} descargas simultaneas, {n_workers} procesos CPU)."
        )

        results_q: "queue.Queue" = queue.Queue()
        dl_state = {"done": len(ready)}
        dl_lock = threading.Lock()
        ctx = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor, \
                    ThreadPoolExecutor(max_workers=dl_workers) as dl_pool:

                def _on_processed(fut, plateifu):
                    try:
                        results_q.put(fut.result())
                    except Exception as exc:  # worker crashed
                        results_q.put((plateifu, None, str(exc)))

                def _submit_processing(t):
                    executor.submit(_worker_process_galaxy_task, t).add_done_callback(
                        lambda f, p=t[1]: _on_processed(f, p)
                    )

                def _download_then_process(t):
                    try:
                        download_galaxy_cube(cfg, t[1], verbose=False)
                    except Exception as exc:
                        with dl_lock:
                            dl_state["done"] += 1
                        results_q.put((t[1], None, f"descarga fallida: {exc}"))
                        return
                    with dl_lock:
                        dl_state["done"] += 1
                    _submit_processing(t)

                for t in ready:
                    _submit_processing(t)
                for t in to_download:
                    dl_pool.submit(_download_then_process, t)

                pbar = tqdm(total=len(tasks), desc=f"Stage 1 ({n_workers} CPUs)")
                for _ in range(len(tasks)):
                    plateifu, res, err = results_q.get()
                    if err:
                        print(f"\n[Warning] Error procesando galaxia {plateifu}: {err}")
                    elif res and res.get("n_candidates_accepted", 0) > 0:
                        success_count += 1
                        total_candidates += res["n_candidates_accepted"]
                    if hasattr(pbar, "set_description"):
                        pbar.set_description(
                            f"Stage 1 ({n_workers} CPUs | descargas {dl_state['done']}/{len(tasks)})"
                        )
                    pbar.update(1)
                pbar.close()
        except Exception as pool_err:
            print(f"\n[Multiproceso] Interrupción en pool ({pool_err}). Continuando secuencialmente...")
            n_workers = 1

    if n_workers <= 1 and success_count == 0:
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
