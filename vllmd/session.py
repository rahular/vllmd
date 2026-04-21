"""Session config, worker state, and persistent file I/O."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Port layout constants
# ---------------------------------------------------------------------------

WORKER_PORT_BASE = 10000        # slot 0 → workers 10000-10999
WORKER_PORTS_PER_SESSION = 1000 # 1000 worker ports reserved per session
LB_PORT_BASE = 9000             # slot 0 → nginx lb :9000, slot 1 → :9001, ...
MANAGER_PORT_BASE = 9500        # slot 0 → manager :9500, slot 1 → :9501, ...


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def sessions_dir() -> Path:
    """Root directory for all vllmd session state."""
    base = Path(os.environ.get("VLLMD_HOME", Path.home() / ".vllmd"))
    return base / "sessions"


def session_dir(session_id: str) -> Path:
    return sessions_dir() / session_id


def current_path() -> Path:
    """Path to the file storing the current session name."""
    base = Path(os.environ.get("VLLMD_HOME", Path.home() / ".vllmd"))
    return base / "current"


def read_current() -> Optional[str]:
    """Return the current session name, or None if unset / session no longer exists."""
    p = current_path()
    if not p.exists():
        return None
    name = p.read_text().strip()
    if name and _state_path(name).exists():
        return name
    return None


def write_current(name: str) -> None:
    p = current_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(name)


def clear_current() -> None:
    p = current_path()
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WorkerStatus(str, Enum):
    PENDING   = "pending"    # sbatch submitted, not yet registered
    HEALTHY   = "healthy"    # registered + heartbeating
    DEAD      = "dead"       # missed heartbeats; removed from nginx


class WorkerState(BaseModel):
    worker_id: str
    slurm_job_id: Optional[str] = None
    hostname: Optional[str] = None
    port: int
    status: WorkerStatus = WorkerStatus.PENDING
    registered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionConfig(BaseModel):
    """Immutable configuration set at `vllmd start`."""
    session_id: str                     # human-readable name chosen by the user
    slot: int = 0                       # port slot: determines all port assignments
    model: str
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    sif: Optional[str] = None           # Singularity SIF path; None = native python
    max_model_len: Optional[int] = None
    gpu_mem_util: float = 0.92
    account: Optional[str] = None       # Slurm account
    partition: str = "standard-g"       # Slurm partition
    time_limit: str = "2-00:00:00"      # Slurm --time per worker job
    scratch: str = "/scratch/project_462000963"
    hf_home: Optional[str] = None       # HF_HOME inside worker; defaults to scratch/hf_cache
    vllm_extra_args: Optional[str] = None  # space-separated extra vLLM flags (passed verbatim)
    startup_timeout: int = 900          # seconds to wait for vLLM /health

    # ── Computed port assignments (derived from slot) ──────────────────────
    @property
    def port_base(self) -> int:
        """First worker port for this session."""
        return WORKER_PORT_BASE + self.slot * WORKER_PORTS_PER_SESSION

    @property
    def lb_port(self) -> int:
        """nginx load-balancer listen port."""
        return LB_PORT_BASE + self.slot

    @property
    def manager_port(self) -> int:
        """Session manager FastAPI port."""
        return MANAGER_PORT_BASE + self.slot

    # ── Other computed fields ──────────────────────────────────────────────
    @property
    def gpus_per_worker(self) -> int:
        """Total GPUs needed per worker: TP × DP."""
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def effective_hf_home(self) -> str:
        return self.hf_home or f"{self.scratch}/hf_cache"


class SessionState(BaseModel):
    """Mutable runtime state — written to disk on every change."""
    config: SessionConfig
    manager_host: str = ""
    workers: Dict[str, WorkerState] = Field(default_factory=dict)
    next_worker_seq: int = 0            # monotonically increasing; used for port assignment

    @property
    def manager_port(self) -> int:
        return self.config.manager_port

    @property
    def lb_endpoint(self) -> str:
        return f"http://{self.manager_host}:{self.config.lb_port}/v1"

    @property
    def manager_url(self) -> str:
        return f"http://{self.manager_host}:{self.config.manager_port}"

    def healthy_workers(self) -> List[WorkerState]:
        return [w for w in self.workers.values() if w.status == WorkerStatus.HEALTHY]

    def pending_workers(self) -> List[WorkerState]:
        return [w for w in self.workers.values() if w.status == WorkerStatus.PENDING]

    def dead_workers(self) -> List[WorkerState]:
        return [w for w in self.workers.values() if w.status == WorkerStatus.DEAD]

    def next_port(self) -> int:
        """Return the next worker port based on the monotonic sequence counter."""
        return self.config.port_base + self.next_worker_seq

    def next_worker_id(self) -> str:
        return f"worker-{self.next_worker_seq}"


# ---------------------------------------------------------------------------
# Slot assignment
# ---------------------------------------------------------------------------

def next_slot() -> int:
    """Return the smallest slot number not currently used by any active session."""
    used: set[int] = set()
    for sid in list_sessions():
        try:
            state = load_state(sid)
            used.add(state.config.slot)
        except Exception:
            pass
    slot = 0
    while slot in used:
        slot += 1
    return slot


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _state_path(session_id: str) -> Path:
    return session_dir(session_id) / "state.json"


def save_state(state: SessionState) -> None:
    path = _state_path(state.config.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    tmp.replace(path)


def load_state(session_id: str) -> SessionState:
    path = _state_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"No session found: {session_id} (looked in {path})")
    return SessionState.model_validate_json(path.read_text())


def list_sessions() -> List[str]:
    base = sessions_dir()
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and _state_path(d.name).exists()
    )


def delete_session(session_id: str) -> None:
    import shutil
    d = session_dir(session_id)
    if d.exists():
        shutil.rmtree(d)


def write_manager_url(session_id: str, host: str, port: int) -> None:
    path = session_dir(session_id) / "manager.url"
    path.write_text(f"http://{host}:{port}")


def read_manager_url(session_id: str) -> str:
    path = session_dir(session_id) / "manager.url"
    if not path.exists():
        raise FileNotFoundError(
            f"Session manager URL not found for session '{session_id}'. "
            "Is the session running? Try `vllmd list`."
        )
    return path.read_text().strip()


def write_manager_pid(session_id: str, pid: int) -> None:
    path = session_dir(session_id) / "manager.pid"
    path.write_text(str(pid))


def read_manager_pid(session_id: str) -> Optional[int]:
    path = session_dir(session_id) / "manager.pid"
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None
