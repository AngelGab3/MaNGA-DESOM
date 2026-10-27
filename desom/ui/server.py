"""
Multi-threaded HTTP Server and REST API for the DESOM Dashboard.
Zero-dependency implementation using Python standard library.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

from .api import DashboardAPI


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DashboardHandler(BaseHTTPRequestHandler):
    """Handles REST API requests and static web files."""

    api: DashboardAPI = None  # Injected on server startup
    static_dir = Path(__file__).parent / "static"

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP logs, keep errors
        if args and str(args[1]) in ("500", "404"):
            sys.stderr.write(f"[UI Server] {format % args}\n")

    def send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, text: str, status: int = 200) -> None:
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    # =========================================================================
    # GET Requests
    # =========================================================================
    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            # 1. Static Web Files
            if path == "/" or path == "/index.html":
                self.serve_file(self.static_dir / "index.html")
                return

            if path.startswith("/static/"):
                rel_path = path[len("/static/") :]
                self.serve_file(self.static_dir / rel_path)
                return

            # 2. Generated Figures / Artifacts
            if path.startswith("/files/figures/"):
                fig_name = path[len("/files/figures/") :]
                target = self.api.cfg.figures_dir / fig_name
                self.serve_file(target)
                return

            # 3. REST API Routes
            if path == "/api/hardware":
                self.send_json(self.api.get_hardware_info())
                return

            if path == "/api/config":
                self.send_json(self.api.get_active_config())
                return

            if path == "/api/experiments":
                self.send_json(self.api.get_experiments())
                return

            if path == "/api/pipeline/state":
                self.send_json(self.api.get_pipeline_state())
                return

            if path == "/api/dataset/demographics":
                self.send_json(self.api.get_dataset_demographics())
                return

            if path == "/api/preprocessing/diagnostics":
                self.send_json(self.api.get_preprocessing_diagnostics())
                return

            if path == "/api/spectra/sample":
                self.send_json(self.api.get_sample_spectra())
                return

            if path == "/api/figures":
                self.send_json(self.api.get_generated_figures())
                return

            if path == "/api/pipeline/logs":
                stage = query.get("stage", ["1"])[0]
                log_path = self.api.cfg.logs_dir / f"stage_{stage}.log"
                if log_path.exists():
                    try:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            text = "".join(lines[-300:])  # Last 300 lines
                            self.send_text(text)
                            return
                    except Exception as e:
                        self.send_text(f"Error leyendo log: {e}")
                        return
                self.send_text("Esperando salida del proceso...")
                return

            if path == "/api/pipeline/status":
                self.send_json(self.api.get_process_status())
                return

            self.send_error(404, f"API endpoint not found: {path}")
        except Exception as e:
            sys.stderr.write(f"[UI Server] Exception in GET {self.path}: {e}\n")
            self.send_json({"error": str(e)}, status=500)

    # =========================================================================
    # POST Requests
    # =========================================================================
    def do_POST(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"

            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                payload = {}

            if path == "/api/switch_run":
                run_id = query.get("run_id", [""])[0] or payload.get("run_id", "")
                if run_id:
                    self.api.switch_run(run_id)
                    self.send_json({"status": "switched", "run_id": run_id})
                else:
                    self.send_json({"status": "error", "message": "Falta run_id"}, status=400)
                return

            if path == "/api/runs/create":
                run_id = payload.get("run_id", "")
                n_target = int(payload.get("n_target", 500))
                clone_current = bool(payload.get("clone_current", True))
                res = self.api.create_run(run_id=run_id, n_target=n_target, clone_current=clone_current)
                status_code = 200 if res.get("status") == "created" else 400
                self.send_json(res, status=status_code)
                return

            if path == "/api/config":
                res = self.api.save_active_config(payload)
                self.send_json(res)
                return

            if path == "/api/config/reset":
                res = self.api.reset_to_paper_defaults()
                self.send_json(res)
                return

            if path == "/api/pipeline/run":
                stage = str(payload.get("stage", "1"))
                sample_size = payload.get("sample_size")
                res = self.api.run_pipeline_stage(stage=stage, sample_size=sample_size)
                self.send_json(res)
                return

            self.send_json({"status": "error", "message": f"Ruta POST no encontrada: {path}"}, status=404)
        except Exception as e:
            self.send_json({"status": "error", "message": str(e)}, status=500)


def start_dashboard(
    host: str = "127.0.0.1",
    port: int = 8050,
    storage: str | None = None,
    sample: int | None = None,
    run_id: str | None = None,
    open_browser: bool = True,
) -> None:
    """Launches the dashboard web server."""
    api = DashboardAPI(storage_backend=storage, n_target=sample, run_id=run_id)
    DashboardHandler.api = api

    server = ThreadedHTTPServer((host, port), DashboardHandler)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"

    active_storage = api.cfg.raw["storage"]["backend"]
    active_sample = api.cfg.raw["n_target"]

    print("\n" + "=" * 65)
    print("      🌌 DESOM DASHBOARD & GALAXY VISUALIZER 🌌")
    print("=" * 65)
    print(f" URL local:        {url}")
    print(f" Almacenamiento:   {active_storage} ({api.cfg.run_dir})")
    print(f" Muestra activa:   {active_sample} galaxias (Run ID: {api.cfg.run_id})")
    print(f" Presiona Ctrl + C para detener el servidor.")
    print("=" * 65 + "\n")

    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[UI Server] Apagando servidor...")
        server.server_close()
