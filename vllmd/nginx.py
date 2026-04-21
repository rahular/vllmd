"""nginx config generation and subprocess management."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple


class NginxManager:
    """Manages a single nginx process for a session."""

    def __init__(self, session_dir: Path, lb_port: int, nginx_bin: Optional[str] = None):
        self.session_dir = session_dir
        self.lb_port = lb_port
        self.conf_path = session_dir / "nginx.conf"
        self.pid_path = session_dir / "nginx.pid"
        self.log_dir = session_dir / "logs"
        self.tmp_dir = session_dir / "nginx_tmp"
        self.nginx_bin = nginx_bin or _find_nginx()
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, backends: List[Tuple[str, int]]) -> None:
        """Write config and start nginx. backends = [(host, port), ...]"""
        self._write_config(backends)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [self.nginx_bin, "-c", str(self.conf_path.resolve())]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Brief wait — nginx daemonizes by default; pid file is used for reload/stop
        time.sleep(0.5)

    def reload(self, backends: List[Tuple[str, int]]) -> None:
        """Rewrite config and send SIGHUP (graceful reload — no dropped connections)."""
        self._write_config(backends)
        pid = self._read_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGHUP)
            except ProcessLookupError:
                pass

    def stop(self) -> None:
        """Send SIGQUIT (graceful drain) to nginx master process."""
        pid = self._read_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGQUIT)
            except ProcessLookupError:
                pass
        if self._proc:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def is_running(self) -> bool:
        pid = self._read_pid()
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _write_config(self, backends: List[Tuple[str, int]]) -> None:
        self.conf_path.parent.mkdir(parents=True, exist_ok=True)
        self.conf_path.write_text(self._render_config(backends))

    def _render_config(self, backends: List[Tuple[str, int]]) -> str:
        upstream_block = ""
        if backends:
            servers = "\n        ".join(
                f"server {host}:{port};" for host, port in backends
            )
            upstream_block = f"""
    upstream vllm_backends {{
        least_conn;
        {servers}
    }}
"""
        else:
            # No backends yet — nginx starts but returns 502 until workers register.
            upstream_block = """
    upstream vllm_backends {
        # no workers yet
        server 127.0.0.1:1 down;
    }
"""

        tmp = self.tmp_dir.resolve()
        logs = self.log_dir.resolve()
        pid_file = self.pid_path.resolve()

        return f"""\
worker_processes auto;
error_log {logs}/nginx_error.log warn;
pid {pid_file};
daemon on;

events {{
    worker_connections 4096;
}}

http {{
    access_log {logs}/nginx_access.log;

    client_body_temp_path  {tmp}/client_body;
    proxy_temp_path        {tmp}/proxy;
    fastcgi_temp_path      {tmp}/fastcgi;
    uwsgi_temp_path        {tmp}/uwsgi;
    scgi_temp_path         {tmp}/scgi;
{upstream_block}
    server {{
        listen {self.lb_port};

        location / {{
            proxy_pass         http://vllm_backends;
            proxy_http_version 1.1;
            proxy_set_header   Connection "";
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;

            proxy_connect_timeout 60s;
            proxy_send_timeout    600s;
            proxy_read_timeout    600s;

            proxy_buffering    off;
            proxy_cache        off;
        }}
    }}
}}
"""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_pid(self) -> Optional[int]:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text().strip())
        except (ValueError, OSError):
            return None


def _find_nginx() -> str:
    """Find nginx binary: check PATH, then common conda-on-scratch locations."""
    import shutil
    found = shutil.which("nginx")
    if found:
        return found

    # LUMI convention: conda prefix on scratch
    scratch = os.environ.get("SCRATCH", "/scratch/project_462000963")
    user = os.environ.get("USER", "")
    candidates = [
        f"{scratch}/users/{user}/sft-nginx/bin/nginx",
        f"{scratch}/users/{user}/vllmd-nginx/bin/nginx",
        "/opt/conda/bin/nginx",
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    raise FileNotFoundError(
        "nginx not found. Install it with:\n"
        "  conda create -p /scratch/.../vllmd-nginx -c conda-forge nginx -y\n"
        "or set NGINX_BIN environment variable."
    )
