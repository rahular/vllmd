"""Slurm utilities: sbatch, squeue, scancel."""

from __future__ import annotations

import re
import subprocess
from typing import Dict, List, Optional


class SlurmError(RuntimeError):
    pass


def sbatch(script: str, env: Optional[Dict[str, str]] = None) -> str:
    """Submit a job script string via sbatch. Returns the Slurm job ID."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", prefix="vllmd_", delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        merged_env = {**os.environ, **(env or {})}
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            env=merged_env,
        )
    finally:
        os.unlink(script_path)

    if result.returncode != 0:
        raise SlurmError(f"sbatch failed:\n{result.stderr.strip()}")

    # "Submitted batch job 1234567"
    match = re.search(r"(\d+)", result.stdout)
    if not match:
        raise SlurmError(f"Could not parse job ID from sbatch output: {result.stdout!r}")
    return match.group(1)


def squeue(job_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Return a list of dicts with keys: job_id, state, reason, node.
    If job_ids is provided, only those jobs are queried.
    """
    cmd = [
        "squeue",
        "--noheader",
        "--format=%i|%T|%R|%B",  # jobid, state, reason, exec_host
    ]
    if job_ids:
        cmd += ["--jobs", ",".join(job_ids)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # squeue returns non-zero when job IDs are not found (already finished)
    # so we tolerate non-zero exit when all jobs are gone
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        rows.append({
            "job_id": parts[0].strip(),
            "state": parts[1].strip(),
            "reason": parts[2].strip(),
            "node": parts[3].strip(),
        })
    return rows


def scancel(job_ids: List[str]) -> None:
    """Cancel one or more Slurm jobs."""
    if not job_ids:
        return
    result = subprocess.run(
        ["scancel"] + job_ids,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # scancel exits non-zero if some jobs are already done — that's fine
        pass


def slurm_available() -> bool:
    """Return True if sbatch is on PATH."""
    result = subprocess.run(
        ["which", "sbatch"], capture_output=True, text=True
    )
    return result.returncode == 0
