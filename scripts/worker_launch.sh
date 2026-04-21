#!/bin/bash
# =============================================================================
# vllmd worker launch script — submitted as an independent sbatch job per worker.
#
# Environment variables set by the session manager (server.py/_submit_worker_job):
#   VLLMD_SESSION_URL     Session manager URL (e.g. http://nid001:9999)
#   VLLMD_WORKER_ID       Unique worker ID (e.g. worker-3)
#   VLLMD_MODEL           HuggingFace model ID or local path
#   VLLMD_PORT            Port this worker listens on
#   VLLMD_TENSOR_PARALLEL_SIZE    Tensor-parallel size
#   VLLMD_DATA_PARALLEL_SIZE      Data-parallel size (1 = disabled)
#   VLLMD_ENABLE_EXPERT_PARALLEL  Expert-parallel flag (0 = disabled)
#   VLLMD_GPU_MEM_UTIL    GPU memory utilisation fraction
#   VLLMD_STARTUP_TIMEOUT Seconds to wait for vLLM /health
#   VLLMD_VLLM_EXTRA_ARGS Extra vLLM flags (space-separated, verbatim)
#   VLLMD_SIF             Singularity SIF path (unset = native Python)
#   VLLMD_SCRATCH         Scratch root for bind mounts
#   HF_HOME               HuggingFace cache directory
# =============================================================================

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] [${VLLMD_WORKER_ID:-worker}] $*"; }
err() { echo "[$(date '+%H:%M:%S')] [${VLLMD_WORKER_ID:-worker}] ✗ $*" >&2; }

# Validate required vars
: "${VLLMD_SESSION_URL:?VLLMD_SESSION_URL is required}"
: "${VLLMD_WORKER_ID:?VLLMD_WORKER_ID is required}"
: "${VLLMD_MODEL:?VLLMD_MODEL is required}"
: "${VLLMD_PORT:?VLLMD_PORT is required}"

log "Starting on $(hostname), port ${VLLMD_PORT}, Slurm job ${SLURM_JOB_ID:-?}"

# ── GPU visibility ────────────────────────────────────────────────────────────
# When Singularity uses --rocm it bypasses Slurm's cgroup GPU restriction and
# exposes ALL GCDs on the node.  Restrict vLLM to the allocated GPUs using
# ROCR_VISIBLE_DEVICES (takes precedence over cgroups inside the container).
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export ROCR_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
    log "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}"
elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
    export ROCR_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
    log "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} (from SLURM_STEP_GPUS)"
fi

# ── Locate serve.sh ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_SH="${SCRIPT_DIR}/serve.sh"
if [ ! -f "${SERVE_SH}" ]; then
    err "serve.sh not found at ${SERVE_SH}"
    exit 1
fi

# ── Start vLLM in background ─────────────────────────────────────────────────
log "Launching vLLM ..."
bash "${SERVE_SH}" &
VLLM_PID=$!

# ── Wait for vLLM to become healthy, then register + heartbeat ───────────────
# vllmd-worker: polls /health, calls POST /register, then loops POST /heartbeat
log "Waiting for vLLM to be healthy (timeout ${VLLMD_STARTUP_TIMEOUT:-900}s) ..."
vllmd-worker

# ── If the heartbeat loop exits (session stopped), kill vLLM ─────────────────
log "Heartbeat loop exited — stopping vLLM"
kill "${VLLM_PID}" 2>/dev/null || true
wait "${VLLM_PID}" 2>/dev/null || true
log "Worker exiting."
