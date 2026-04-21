"""Worker-side logic: startup polling, registration, heartbeat loop.

This module is used by scripts/worker_launch.sh (via `vllmd-worker`).
It runs inside each sbatch job after vLLM has started.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
import logging

import httpx

logger = logging.getLogger(__name__)


def wait_for_vllm(port: int, timeout: int = 900) -> bool:
    """Poll http://localhost:{port}/health until 200 or timeout. Returns True on success."""
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def register(session_url: str, worker_id: str, port: int) -> None:
    hostname = socket.gethostname()
    url = f"{session_url.rstrip('/')}/register"
    r = httpx.post(url, json={"worker_id": worker_id, "hostname": hostname, "port": port}, timeout=30)
    r.raise_for_status()
    logger.info("Registered as %s at %s:%d", worker_id, hostname, port)


def heartbeat_loop(session_url: str, worker_id: str, interval: int = 30) -> None:
    """Send heartbeats forever. Returns only on signal."""
    url = f"{session_url.rstrip('/')}/heartbeat/{worker_id}"
    while True:
        try:
            httpx.post(url, timeout=10)
        except Exception as exc:
            logger.warning("Heartbeat failed: %s", exc)
        time.sleep(interval)


def main() -> None:
    """Entry point: called by worker_launch.sh after vLLM is healthy."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    session_url = os.environ["VLLMD_SESSION_URL"]
    worker_id = os.environ["VLLMD_WORKER_ID"]
    port = int(os.environ["VLLMD_PORT"])
    timeout = int(os.environ.get("VLLMD_STARTUP_TIMEOUT", "900"))

    logger.info("Waiting for vLLM on port %d (timeout %ds)", port, timeout)
    if not wait_for_vllm(port, timeout):
        logger.error("vLLM did not become healthy within %ds — exiting", timeout)
        sys.exit(1)

    register(session_url, worker_id, port)
    heartbeat_loop(session_url, worker_id)


if __name__ == "__main__":
    main()
