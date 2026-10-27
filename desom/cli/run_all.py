"""
Master CLI Runner: Executes the entire DESOM pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import get_config, setup_hardware
from .run_stage1 import main as stage1_main
from .run_stage2 import main as stage2_main
from .run_stage3 import main as stage3_main
from .run_stage4 import main as stage4_main


def main() -> None:
    parser = argparse.ArgumentParser(description="DESOM: End-to-End Pipeline Runner")
    parser.add_argument("--storage", choices=["local", "drive"], default=None, help="Storage backend (local laptop vs Google Drive)")
    parser.add_argument("--sample", type=int, default=None, help="Number of target galaxies (e.g. 500 or 4609)")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment identifier")
    parser.add_argument("--stages", nargs="+", default=["1", "2", "3", "4"], help="Stages to execute: 1 2 3 4")
    parser.add_argument("--epochs-desom1", type=int, default=None, help="Override DESOM-1 iterations")
    parser.add_argument("--epochs-desom2", type=int, default=None, help="Override DESOM-2 iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--split-mode", choices=["galaxy", "spectrum"], default=None, help="Split unit for stage 2")
    parser.add_argument("--map-distance", choices=["manhattan", "paper"], default=None, help="SOM grid distance for stages 3-4")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("      DESOM: DEEP EMBEDDED SELF-ORGANIZING MAPS")
    print("      Mapping Galaxy Spectra & Spatially Resolved Diversity")
    print("=" * 65)
    print(f" Storage backend: {args.storage}")
    print(f" Target sample:   {args.sample} galaxies")
    print(f" Stages to run:   {args.stages}")
    print("=" * 65 + "\n")

    # Configure hardware dynamically
    setup_hardware(verbose=True)

    # Backup sys.argv for delegated mains
    orig_argv = list(sys.argv)

    try:
        if "1" in args.stages:
            sys.argv = ["run_stage1", "--storage", args.storage, "--sample", str(args.sample)]
            if args.run_id:
                sys.argv.extend(["--run-id", args.run_id])
            stage1_main()

        if "2" in args.stages:
            sys.argv = ["run_stage2", "--storage", args.storage, "--sample", str(args.sample)]
            if args.split_mode:
                sys.argv.extend(["--split-mode", args.split_mode])
            if args.run_id:
                sys.argv.extend(["--run-id", args.run_id])
            stage2_main()

        if "3" in args.stages:
            sys.argv = ["run_stage3", "--storage", args.storage, "--sample", str(args.sample)]
            if args.map_distance:
                sys.argv.extend(["--map-distance", args.map_distance])
            if args.run_id:
                sys.argv.extend(["--run-id", args.run_id])
            if args.epochs_desom1:
                sys.argv.extend(["--epochs", str(args.epochs_desom1)])
            if args.batch_size:
                sys.argv.extend(["--batch-size", str(args.batch_size)])
            stage3_main()

        if "4" in args.stages:
            sys.argv = ["run_stage4", "--storage", args.storage, "--sample", str(args.sample)]
            if args.map_distance:
                sys.argv.extend(["--map-distance", args.map_distance])
            if args.run_id:
                sys.argv.extend(["--run-id", args.run_id])
            if args.epochs_desom2:
                sys.argv.extend(["--epochs", str(args.epochs_desom2)])
            if args.batch_size:
                sys.argv.extend(["--batch-size", str(args.batch_size)])
            stage4_main()

        print("\n" + "=" * 65)
        print(" Full pipeline execution finished successfully!")
        print(" Check your figures and models in the experiment directory.")
        print("=" * 65 + "\n")

    finally:
        sys.argv = orig_argv


if __name__ == "__main__":
    main()
