"""FastAPI session manager daemon.

Runs on the node where `vllmd start` was called.  Workers (sbatch jobs)
connect back to this process to register themselves and send heartbeats.
The CLI also talks to this daemon for status / add / remove / stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel

from .nginx import NginxManager
from .session import (
    SessionState,
    WorkerState,
    WorkerStatus,
    load_state,
    save_state,
    session_dir,
    write_manager_pid,
    write_manager_url,
)
from .slurm import scancel, sbatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated in main() before app starts)
# ---------------------------------------------------------------------------

_state: Optional[SessionState] = None
_nginx: Optional[NginxManager] = None
_lock = threading.Lock()

HEARTBEAT_DEAD_AFTER = 90   # seconds without heartbeat → mark dead


# ---------------------------------------------------------------------------
# Background: heartbeat watchdog
# ---------------------------------------------------------------------------

def _heartbeat_watchdog() -> None:
    while True:
        time.sleep(15)
        with _lock:
            if _state is None:
                continue
            now = datetime.now(timezone.utc)
            changed = False
            for w in list(_state.workers.values()):
                if w.status != WorkerStatus.HEALTHY:
                    continue
                if w.last_heartbeat is None:
                    continue
                age = (now - w.last_heartbeat).total_seconds()
                if age > HEARTBEAT_DEAD_AFTER:
                    logger.warning("Worker %s missed heartbeat (%.0fs ago) — marking dead", w.worker_id, age)
                    w.status = WorkerStatus.DEAD
                    changed = True
            if changed:
                _reload_nginx()
                save_state(_state)


def _reload_nginx() -> None:
    """Rewrite nginx config with current healthy workers and signal reload."""
    if _nginx is None or _state is None:
        return
    backends = [
        (w.hostname, w.port)
        for w in _state.workers.values()
        if w.status == WorkerStatus.HEALTHY and w.hostname
    ]
    _nginx.reload(backends)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_heartbeat_watchdog, daemon=True).start()
    yield
    # On shutdown: stop nginx cleanly
    if _nginx:
        _nginx.stop()


app = FastAPI(title="vllmd session manager", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    worker_id: str
    hostname: str
    port: int


class RemoveRequest(BaseModel):
    worker_ids: Optional[List[str]] = None


class AddWorkersResponse(BaseModel):
    workers: List[Dict[str, Any]]


class RemoveWorkersResponse(BaseModel):
    removed: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def get_status():
    with _lock:
        if _state is None:
            raise HTTPException(500, "Session state not initialised")
        workers_out = []
        now = datetime.now(timezone.utc)
        for w in _state.workers.values():
            age_s = None
            if w.submitted_at:
                age_s = int((now - w.submitted_at).total_seconds())
            workers_out.append({
                "worker_id": w.worker_id,
                "slurm_job_id": w.slurm_job_id,
                "hostname": w.hostname,
                "port": w.port,
                "status": w.status,
                "age_s": age_s,
            })
        return {
            "session_id": _state.config.session_id,
            "lb_endpoint": _state.lb_endpoint,
            "manager_url": _state.manager_url,
            "healthy": len(_state.healthy_workers()),
            "pending": len(_state.pending_workers()),
            "dead": len(_state.dead_workers()),
            "workers": workers_out,
            "model": _state.config.model,
            "tensor_parallel_size": _state.config.tensor_parallel_size,
            "data_parallel_size": _state.config.data_parallel_size,
            "enable_expert_parallel": _state.config.enable_expert_parallel,
        }


@app.post("/register")
def register(req: RegisterRequest):
    with _lock:
        if _state is None:
            raise HTTPException(500, "Session state not initialised")
        w = _state.workers.get(req.worker_id)
        if w is None:
            raise HTTPException(404, f"Unknown worker_id: {req.worker_id}")
        w.hostname = req.hostname
        w.port = req.port
        w.status = WorkerStatus.HEALTHY
        w.registered_at = datetime.now(timezone.utc)
        w.last_heartbeat = w.registered_at
        _reload_nginx()
        save_state(_state)
        logger.info("Worker %s registered at %s:%d", req.worker_id, req.hostname, req.port)
    return {"status": "registered"}


@app.post("/heartbeat/{worker_id}")
def heartbeat(worker_id: str):
    with _lock:
        if _state is None:
            raise HTTPException(500, "Session state not initialised")
        w = _state.workers.get(worker_id)
        if w is None:
            raise HTTPException(404, f"Unknown worker_id: {worker_id}")
        w.last_heartbeat = datetime.now(timezone.utc)
        if w.status == WorkerStatus.DEAD:
            # Re-register a zombie that came back
            w.status = WorkerStatus.HEALTHY
            _reload_nginx()
        save_state(_state)
    return {"status": "ok"}


@app.post("/workers/add", response_model=AddWorkersResponse)
def add_workers(count: int = Query(1, ge=1)):
    with _lock:
        if _state is None:
            raise HTTPException(500, "Session state not initialised")
        added = []
        for _ in range(count):
            worker_id = _state.next_worker_id()
            port = _state.next_port()
            _state.next_worker_seq += 1

            w = WorkerState(worker_id=worker_id, port=port)
            _state.workers[worker_id] = w

            # Submit the sbatch job
            try:
                job_id = _submit_worker_job(worker_id, port)
                w.slurm_job_id = job_id
                logger.info("Submitted worker %s as Slurm job %s (port %d)", worker_id, job_id, port)
            except Exception as exc:
                logger.error("Failed to submit worker %s: %s", worker_id, exc)
                w.status = WorkerStatus.DEAD
                w.slurm_job_id = None

            added.append({"worker_id": worker_id, "slurm_job_id": w.slurm_job_id, "port": port})

        save_state(_state)
    return {"workers": added}


@app.post("/workers/remove", response_model=RemoveWorkersResponse)
def remove_workers(
    count: Optional[int] = Query(None, ge=1),
    dead: bool = Query(False),
    req: RemoveRequest = Body(RemoveRequest()),
):
    with _lock:
        if _state is None:
            raise HTTPException(500, "Session state not initialised")

        if dead:
            to_remove = [w.worker_id for w in _state.workers.values() if w.status == WorkerStatus.DEAD]
        elif req.worker_ids:
            to_remove = req.worker_ids
        elif count:
            # Pick the `count` oldest workers (by submission time)
            sorted_workers = sorted(
                _state.workers.values(),
                key=lambda w: w.submitted_at,
            )
            to_remove = [w.worker_id for w in sorted_workers[:count]]
        else:
            raise HTTPException(400, "Provide count, worker_ids, or dead=true")

        removed = []
        job_ids_to_cancel = []
        for wid in to_remove:
            w = _state.workers.pop(wid, None)
            if w is None:
                continue
            removed.append(wid)
            if w.slurm_job_id:
                job_ids_to_cancel.append(w.slurm_job_id)

        if job_ids_to_cancel:
            scancel(job_ids_to_cancel)
            logger.info("Cancelled Slurm jobs: %s", job_ids_to_cancel)

        _reload_nginx()
        save_state(_state)

    return {"removed": removed}


@app.post("/shutdown")
def shutdown():
    """Graceful shutdown: cancel all workers, stop nginx, exit process."""
    with _lock:
        if _state:
            job_ids = [w.slurm_job_id for w in _state.workers.values() if w.slurm_job_id]
            if job_ids:
                scancel(job_ids)
            _reload_nginx()
    if _nginx:
        _nginx.stop()

    def _do_exit():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_do_exit, daemon=True).start()
    return {"status": "shutting down"}


# ---------------------------------------------------------------------------
# Worker job submission
# ---------------------------------------------------------------------------

def _submit_worker_job(worker_id: str, port: int) -> str:
    """Build and submit the sbatch script for one worker."""
    assert _state is not None
    cfg = _state.config
    scripts_dir = Path(__file__).parent.parent / "scripts"
    log_dir = session_dir(cfg.session_id) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build the sbatch header. We use --export=NONE + explicit env so that the
    # login-node environment doesn't leak into the compute node.
    header_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=vllmd-{worker_id}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks=1",
        f"#SBATCH --gpus-per-node={cfg.gpus_per_worker}",
        f"#SBATCH --cpus-per-task=14",
        f"#SBATCH --mem=0",          # use all available memory on the node
        f"#SBATCH --time={cfg.time_limit}",
        f"#SBATCH --partition={cfg.partition}",
        f"#SBATCH --output={log_dir}/{worker_id}_%j.log",
        f"#SBATCH --error={log_dir}/{worker_id}_%j.log",
    ]
    if cfg.account:
        header_lines.append(f"#SBATCH --account={cfg.account}")

    env_block = f"""\
export VLLMD_SESSION_URL="{_state.manager_url}"
export VLLMD_WORKER_ID="{worker_id}"
export VLLMD_MODEL="{cfg.model}"
export VLLMD_PORT="{port}"
export VLLMD_TENSOR_PARALLEL_SIZE="{cfg.tensor_parallel_size}"
export VLLMD_DATA_PARALLEL_SIZE="{cfg.data_parallel_size}"
export VLLMD_ENABLE_EXPERT_PARALLEL="{int(cfg.enable_expert_parallel)}"
export VLLMD_GPU_MEM_UTIL="{cfg.gpu_mem_util}"
export VLLMD_STARTUP_TIMEOUT="{cfg.startup_timeout}"
export VLLMD_VLLM_EXTRA_ARGS={shlex.quote(cfg.vllm_extra_args or "")}
export HF_HOME="{cfg.effective_hf_home}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
"""
    if cfg.max_model_len:
        env_block += f'export VLLMD_MAX_MODEL_LEN="{cfg.max_model_len}"\n'
    if cfg.sif:
        env_block += f'export VLLMD_SIF="{cfg.sif}"\n'
        env_block += f'export VLLMD_SCRATCH="{cfg.scratch}"\n'

    script = "\n".join(header_lines) + "\n\nset -euo pipefail\n\n" + env_block + \
        f'\nbash "{scripts_dir}/worker_launch.sh"\n'

    return sbatch(script)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(state: SessionState, host: str, port: int, nginx: NginxManager) -> None:
    """Called by `vllmd start` after daemonisation."""
    global _state, _nginx
    _state = state
    _nginx = nginx

    write_manager_url(state.config.session_id, host, port)
    write_manager_pid(state.config.session_id, os.getpid())

    # Start nginx with no backends initially
    nginx.start([])

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )


def main():
    """Entry point when the daemon is re-launched as a subprocess."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--session-id", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--nginx-bin", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    state = load_state(args.session_id)
    sdir = session_dir(args.session_id)
    nginx = NginxManager(sdir, state.config.lb_port, nginx_bin=args.nginx_bin)

    state.manager_host = socket.gethostname()
    save_state(state)

    serve(state, args.host, state.config.manager_port, nginx)


if __name__ == "__main__":
    main()
