"""
DESOM Dashboard Launcher.
Launches the interactive web dashboard for monitoring, sample demographics,
pipeline execution, and results inspection.
"""

from __future__ import annotations

import argparse
import os
import sys

from desom.ui.server import start_dashboard


def main() -> None:
    # Ensure app runs in an environment with numpy/astropy installed
    try:
        import numpy, astropy
    except ImportError:
        from desom.ui.api import DashboardAPI
        api = DashboardAPI()
        py_exec = api._resolve_python_executable()
        if py_exec != sys.executable:
            print(f"[DESOM Launcher] Cambiando intérprete a {py_exec}...")
            os.execv(py_exec, [py_exec] + sys.argv)

    parser = argparse.ArgumentParser(description="DESOM Interactive Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (e.g. 127.0.0.1 or 0.0.0.0 for remote server)")
    parser.add_argument("--port", type=int, default=8050, help="Local port (default: 8050)")
    parser.add_argument("--storage", choices=["local", "drive"], default=None, help="Storage backend (local vs drive)")
    parser.add_argument("--sample", type=int, default=None, help="Number of target galaxies (default: 500)")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment identifier")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    start_dashboard(
        host=args.host,
        port=args.port,
        storage=args.storage,
        sample=args.sample,
        run_id=args.run_id,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
