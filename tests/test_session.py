"""Tests for session config, state, and persistence."""

from pathlib import Path
from datetime import datetime, timezone

from vllmd.session import (
    LB_PORT_BASE,
    MANAGER_PORT_BASE,
    WORKER_PORT_BASE,
    WORKER_PORTS_PER_SESSION,
    SessionConfig,
    SessionState,
    WorkerState,
    WorkerStatus,
    clear_current,
    delete_session,
    list_sessions,
    load_state,
    next_slot,
    read_current,
    save_state,
    write_current,
)


# ---------------------------------------------------------------------------
# SessionConfig.gpus_per_worker
# ---------------------------------------------------------------------------

class TestGpusPerWorker:
    def test_single_gpu(self):
        cfg = SessionConfig(session_id="x", model="gpt2", tensor_parallel_size=1, data_parallel_size=1)
        assert cfg.gpus_per_worker == 1

    def test_tp_only(self):
        cfg = SessionConfig(session_id="x", model="gpt2", tensor_parallel_size=4, data_parallel_size=1)
        assert cfg.gpus_per_worker == 4

    def test_dp_only(self):
        cfg = SessionConfig(session_id="x", model="m", tensor_parallel_size=1, data_parallel_size=8)
        assert cfg.gpus_per_worker == 8

    def test_tp_and_dp(self):
        cfg = SessionConfig(session_id="x", model="m", tensor_parallel_size=2, data_parallel_size=4)
        assert cfg.gpus_per_worker == 8


# ---------------------------------------------------------------------------
# SessionConfig port assignment
# ---------------------------------------------------------------------------

class TestPortAssignment:
    def test_slot0_ports(self):
        cfg = SessionConfig(session_id="x", model="m", slot=0)
        assert cfg.port_base   == WORKER_PORT_BASE
        assert cfg.lb_port     == LB_PORT_BASE
        assert cfg.manager_port == MANAGER_PORT_BASE

    def test_slot1_ports(self):
        cfg = SessionConfig(session_id="x", model="m", slot=1)
        assert cfg.port_base   == WORKER_PORT_BASE + WORKER_PORTS_PER_SESSION
        assert cfg.lb_port     == LB_PORT_BASE + 1
        assert cfg.manager_port == MANAGER_PORT_BASE + 1

    def test_slots_dont_overlap(self):
        """Worker port range for slot N must not overlap slot N+1's lb/manager ports."""
        for slot in range(5):
            cfg = SessionConfig(session_id="x", model="m", slot=slot)
            next_cfg = SessionConfig(session_id="y", model="m", slot=slot + 1)
            # Last worker port of this slot < first worker port of next slot
            assert cfg.port_base + WORKER_PORTS_PER_SESSION - 1 < next_cfg.port_base
            # LB and manager ports are outside the worker range
            assert not (cfg.port_base <= cfg.lb_port < cfg.port_base + WORKER_PORTS_PER_SESSION)
            assert not (cfg.port_base <= cfg.manager_port < cfg.port_base + WORKER_PORTS_PER_SESSION)


# ---------------------------------------------------------------------------
# SessionConfig.effective_hf_home
# ---------------------------------------------------------------------------

class TestEffectiveHfHome:
    def test_uses_hf_home_when_set(self):
        cfg = SessionConfig(session_id="x", model="m", hf_home="/custom/hf")
        assert cfg.effective_hf_home == "/custom/hf"

    def test_defaults_to_scratch(self):
        cfg = SessionConfig(session_id="x", model="m", scratch="/scratch/proj")
        assert cfg.effective_hf_home == "/scratch/proj/hf_cache"


# ---------------------------------------------------------------------------
# SessionState helpers
# ---------------------------------------------------------------------------

class TestSessionState:
    def _make_state(self, slot: int = 0) -> SessionState:
        cfg = SessionConfig(session_id="test-session", model="gpt2", slot=slot)
        return SessionState(config=cfg, manager_host="node1")

    def test_lb_endpoint(self):
        s = self._make_state(slot=0)
        s.manager_host = "nid007"
        assert s.lb_endpoint == f"http://nid007:{LB_PORT_BASE}/v1"

    def test_manager_url(self):
        s = self._make_state(slot=2)
        s.manager_host = "nid007"
        assert s.manager_url == f"http://nid007:{MANAGER_PORT_BASE + 2}"

    def test_manager_port_delegates_to_config(self):
        s = self._make_state(slot=3)
        assert s.manager_port == s.config.manager_port == MANAGER_PORT_BASE + 3

    def test_next_port_increments_with_sequence(self):
        s = self._make_state(slot=0)
        assert s.next_port() == WORKER_PORT_BASE
        s.next_worker_seq = 3
        assert s.next_port() == WORKER_PORT_BASE + 3

    def test_next_worker_id(self):
        s = self._make_state()
        s.next_worker_seq = 5
        assert s.next_worker_id() == "worker-5"

    def test_worker_status_filtering(self):
        s = self._make_state()
        now = datetime.now(timezone.utc)
        s.workers = {
            "w0": WorkerState(worker_id="w0", port=10000, status=WorkerStatus.HEALTHY,
                              submitted_at=now),
            "w1": WorkerState(worker_id="w1", port=10001, status=WorkerStatus.PENDING,
                              submitted_at=now),
            "w2": WorkerState(worker_id="w2", port=10002, status=WorkerStatus.DEAD,
                              submitted_at=now),
        }
        assert len(s.healthy_workers()) == 1
        assert len(s.pending_workers()) == 1
        assert len(s.dead_workers()) == 1


# ---------------------------------------------------------------------------
# next_slot
# ---------------------------------------------------------------------------

class TestNextSlot:
    def test_no_sessions_returns_0(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))
        assert next_slot() == 0

    def test_returns_next_free_slot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))
        for slot, name in enumerate(["a", "b", "c"]):
            cfg = SessionConfig(session_id=name, model="m", slot=slot)
            save_state(SessionState(config=cfg))
        assert next_slot() == 3

    def test_reuses_freed_slot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))
        # slots 0 and 2 in use, slot 1 free
        for slot, name in [(0, "a"), (2, "c")]:
            cfg = SessionConfig(session_id=name, model="m", slot=slot)
            save_state(SessionState(config=cfg))
        assert next_slot() == 1


# ---------------------------------------------------------------------------
# current session pointer
# ---------------------------------------------------------------------------

class TestCurrentSession:
    def test_read_current_none_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))
        assert read_current() is None

    def test_write_and_read_current(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))
        cfg = SessionConfig(session_id="mymodel", model="m")
        save_state(SessionState(config=cfg))
        write_current("mymodel")
        assert read_current() == "mymodel"

    def test_read_current_returns_none_if_session_deleted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))
        cfg = SessionConfig(session_id="gone", model="m")
        save_state(SessionState(config=cfg))
        write_current("gone")
        delete_session("gone")
        assert read_current() is None

    def test_clear_current(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))
        cfg = SessionConfig(session_id="s", model="m")
        save_state(SessionState(config=cfg))
        write_current("s")
        clear_current()
        assert read_current() is None


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))

        cfg = SessionConfig(session_id="persist-test", model="gpt2", tensor_parallel_size=2, data_parallel_size=4, slot=1)
        state = SessionState(config=cfg, manager_host="node42")

        now = datetime.now(timezone.utc)
        state.workers["worker-0"] = WorkerState(
            worker_id="worker-0",
            port=11000,
            hostname="node42",
            status=WorkerStatus.HEALTHY,
            submitted_at=now,
            registered_at=now,
            last_heartbeat=now,
            slurm_job_id="123456",
        )

        save_state(state)
        loaded = load_state("persist-test")

        assert loaded.config.model == "gpt2"
        assert loaded.config.tensor_parallel_size == 2
        assert loaded.config.data_parallel_size == 4
        assert loaded.config.slot == 1
        assert loaded.config.port_base == WORKER_PORT_BASE + WORKER_PORTS_PER_SESSION
        assert loaded.manager_host == "node42"
        assert "worker-0" in loaded.workers
        assert loaded.workers["worker-0"].hostname == "node42"
        assert loaded.workers["worker-0"].status == WorkerStatus.HEALTHY

    def test_list_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))

        for sid in ["alpha", "beta", "gamma"]:
            cfg = SessionConfig(session_id=sid, model="m")
            save_state(SessionState(config=cfg))

        sessions = list_sessions()
        assert sorted(sessions) == ["alpha", "beta", "gamma"]

    def test_delete_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VLLMD_HOME", str(tmp_path))

        cfg = SessionConfig(session_id="del-me", model="m")
        save_state(SessionState(config=cfg))
        assert "del-me" in list_sessions()

        delete_session("del-me")
        assert "del-me" not in list_sessions()
