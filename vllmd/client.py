"""HTTP client for CLI → session manager communication."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .session import SessionState


class SessionClient:
    """Thin httpx wrapper around the session manager API."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.base = url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self.base}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        r = httpx.get(f"{self.base}/status", timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def add_workers(self, count: int) -> List[Dict[str, Any]]:
        """Ask the session manager to submit `count` new sbatch jobs.
        Returns list of {worker_id, slurm_job_id, port}."""
        r = httpx.post(
            f"{self.base}/workers/add",
            params={"count": count},
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()["workers"]

    def remove_workers(
        self,
        count: Optional[int] = None,
        worker_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Remove workers by count (oldest first) or explicit IDs.
        Returns list of removed worker_ids."""
        payload: Dict[str, Any] = {}
        params: Dict[str, Any] = {}
        if count is not None:
            params["count"] = count
        if worker_ids:
            payload["worker_ids"] = worker_ids
        r = httpx.post(
            f"{self.base}/workers/remove",
            params=params,
            json=payload or None,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()["removed"]

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        try:
            httpx.post(f"{self.base}/shutdown", timeout=10.0)
        except Exception:
            pass  # server may exit before responding


# ------------------------------------------------------------------
# OpenAI-compatible check helper
# ------------------------------------------------------------------

def check_endpoint(
    lb_endpoint: str,
    model: str,
    prompt: str = "Hello, what is 1+1?",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Send a single completion request to the load balancer and return the result."""
    import time

    url = lb_endpoint.rstrip("/")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    r = httpx.post(f"{url}/chat/completions", json=payload, timeout=timeout)
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return {
        "model": data.get("model", model),
        "response": text,
        "latency_s": round(elapsed, 2),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
    }
