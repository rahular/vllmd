#!/bin/bash
# =============================================================================
# Start a single vLLM OpenAI-compatible server (inside Singularity or natively).
#
# Called by worker_launch.sh for each sbatch worker job.
#
# Key env vars (set by vllmd server before sbatch submission):
#   VLLMD_MODEL           HuggingFace model ID or path
#   VLLMD_PORT            Port for this worker
#   VLLMD_TENSOR_PARALLEL_SIZE    Tensor-parallel size
#   VLLMD_DATA_PARALLEL_SIZE      Data-parallel size (1 = disabled)
#   VLLMD_ENABLE_EXPERT_PARALLEL  Expert-parallel flag (0 = disabled)
#   VLLMD_GPU_MEM_UTIL    GPU memory utilisation fraction (default 0.92)
#   VLLMD_VLLM_EXTRA_ARGS Extra vLLM CLI flags (space-separated, verbatim)
#   VLLMD_SIF             Singularity SIF path (unset = native Python)
#   VLLMD_SCRATCH         Scratch root (used for bind mounts + caches)
#   HF_HOME               HuggingFace cache directory
#   ROCR_VISIBLE_DEVICES  GPU(s) to expose (set from SLURM_JOB_GPUS)
#
# Optional overrides:
#   ROCM_COMPAT  1 (default) = --rocm + LD_LIBRARY_PATH strip (required on LUMI)
#                0           = --bind /dev/kfd --bind /dev/dri
#   NIC          Force NIC name for NCCL/RCCL binding (auto-detected if unset)
#   BIND_HOST    vLLM bind address (default 0.0.0.0; never use HOST on Cray nodes)
# =============================================================================

set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

# ── Required env vars ────────────────────────────────────────────────────────
: "${VLLMD_MODEL:?VLLMD_MODEL is required}"
: "${VLLMD_PORT:?VLLMD_PORT is required}"
: "${VLLMD_TENSOR_PARALLEL_SIZE:=1}"
: "${VLLMD_DATA_PARALLEL_SIZE:=1}"
: "${VLLMD_ENABLE_EXPERT_PARALLEL:=0}"
: "${VLLMD_GPU_MEM_UTIL:=0.92}"
: "${VLLMD_VLLM_EXTRA_ARGS:=}"
: "${BIND_HOST:=0.0.0.0}"
: "${ROCM_COMPAT:=1}"
: "${NIC:=}"
SCRATCH="${VLLMD_SCRATCH:-/scratch/project_462000963}"

log "Model            : ${VLLMD_MODEL}"
log "Port             : ${VLLMD_PORT}"
log "tensor-parallel-size=${VLLMD_TENSOR_PARALLEL_SIZE}  data-parallel-size=${VLLMD_DATA_PARALLEL_SIZE}  enable-expert-parallel=${VLLMD_ENABLE_EXPERT_PARALLEL}"
log "GPU mem util     : ${VLLMD_GPU_MEM_UTIL}"
log "HF_HOME          : ${HF_HOME:-<not set>}"
[ -n "${VLLMD_VLLM_EXTRA_ARGS}" ] && log "Extra vLLM args  : ${VLLMD_VLLM_EXTRA_ARGS}"

# ── Build vLLM argument array ────────────────────────────────────────────────
VLLM_ARGS=(
    python -m vllm.entrypoints.openai.api_server
    --model                  "${VLLMD_MODEL}"
    --tensor-parallel-size   "${VLLMD_TENSOR_PARALLEL_SIZE}"
    --dtype                  float16
    --gpu-memory-utilization "${VLLMD_GPU_MEM_UTIL}"
    --host                   "${BIND_HOST}"
    --port                   "${VLLMD_PORT}"
    --trust-remote-code
)

# Data parallelism
if [ "${VLLMD_DATA_PARALLEL_SIZE}" -gt 1 ]; then
    VLLM_ARGS+=(--data-parallel-size "${VLLMD_DATA_PARALLEL_SIZE}")
fi

# Expert parallelism (MoE)
if [ "${VLLMD_ENABLE_EXPERT_PARALLEL}" -eq 1 ]; then
    VLLM_ARGS+=(--enable-expert-parallel)
fi

# Max model length
if [ -n "${VLLMD_MAX_MODEL_LEN:-}" ]; then
    VLLM_ARGS+=(--max-model-len "${VLLMD_MAX_MODEL_LEN}")
fi

# Append extra args verbatim (user-controlled; split on whitespace)
if [ -n "${VLLMD_VLLM_EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2086
    read -r -a _EXTRA <<< "${VLLMD_VLLM_EXTRA_ARGS}"
    VLLM_ARGS+=("${_EXTRA[@]}")
fi

# ── Native Python path (no SIF) ──────────────────────────────────────────────
if [ -z "${VLLMD_SIF:-}" ]; then
    log "No SIF specified — running with native Python"
    log "Endpoint: http://$(hostname -s):${VLLMD_PORT}/v1"
    exec "${VLLM_ARGS[@]}"
fi

# ── Singularity / Apptainer path ─────────────────────────────────────────────
if [ ! -f "${VLLMD_SIF}" ]; then
    err "SIF not found: ${VLLMD_SIF}"
    exit 1
fi

if   command -v apptainer   >/dev/null 2>&1; then SING_CMD=apptainer
elif command -v singularity >/dev/null 2>&1; then SING_CMD=singularity
else
    err "Neither 'singularity' nor 'apptainer' found in PATH."
    exit 1
fi

# ── GFX architecture guard ────────────────────────────────────────────────────
HOST_GFX=""
if [ -x /opt/rocm/bin/rocminfo ]; then
    HOST_GFX="$(/opt/rocm/bin/rocminfo 2>/dev/null \
                 | awk '/^\s+Name:/ && /gfx/ {gsub(/^[[:space:]]+Name:[[:space:]]+/,""); print $0; exit}')" || true
fi
log "Host GPU arch    : ${HOST_GFX:-unknown}"

SIF_TAG="${VLLM_TAG:-}"
if [ -n "${HOST_GFX}" ] && echo "${SIF_TAG}" | grep -q 'gfx'; then
    TAG_GFX="$(echo "${SIF_TAG}" | grep -o 'gfx[0-9a-z]*' | head -1)"
    if [ "${TAG_GFX}" != "${HOST_GFX}" ]; then
        err "GFX ARCHITECTURE MISMATCH — host: ${HOST_GFX}, SIF tag: ${TAG_GFX}"
        err "Rebuild SIF with a tag matching ${HOST_GFX}."
        exit 1
    fi
    log "GFX arch check   : ✓ ${TAG_GFX} matches host"
fi

# ── Network interface detection ───────────────────────────────────────────────
if [ -z "${NIC}" ]; then
    NIC="$(ip route show default 2>/dev/null | awk '/dev/ {print $5; exit}')" || true
fi
if [ -n "${NIC}" ]; then
    NODE_IP="$(ip -4 addr show dev "${NIC}" 2>/dev/null \
               | awk '/inet / {split($2,a,"/"); print a[1]; exit}')" || true
else
    NODE_IP=""
fi
[ -n "${NODE_IP}" ] && log "Primary NIC      : ${NIC} (${NODE_IP})"

# ── Cache dirs ────────────────────────────────────────────────────────────────
mkdir -p "${HF_HOME:-${SCRATCH}/hf_cache}"
mkdir -p "${SCRATCH}/triton_cache"

# ── Launch via Singularity ────────────────────────────────────────────────────
log "Starting vLLM server (ROCM_COMPAT=${ROCM_COMPAT}) ..."
log "Endpoint: http://$(hostname -s):${VLLMD_PORT}/v1"

SING_ENV_ARGS=(
    --env HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
    --env TRITON_CACHE_DIR="${SCRATCH}/triton_cache"
    --env ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"
    --env VLLM_HOST_IP="${NODE_IP:-}"
    --env NCCL_SOCKET_IFNAME="${NIC:-}"
    --env GLOO_SOCKET_IFNAME="${NIC:-}"
    --env RAY_DISABLE_DASHBOARD=1
)

VLLM_CMD_STR="${VLLM_ARGS[*]}"

if [ "${ROCM_COMPAT}" = "1" ]; then
    # --rocm: lets Singularity handle AMD KFD device delegation (required on LUMI
    # for both interactive srun sessions and sbatch jobs — cgroups v2 blocks
    # /dev/kfd even when --bind /dev/kfd is used).
    # Strip /.singularity.d/libs from LD_LIBRARY_PATH: --rocm injects host ROCm
    # libs (need glibc 2.38+) but the vLLM container is Ubuntu 22.04 (glibc 2.35).
    "${SING_CMD}" exec \
        --rocm \
        --bind "${SCRATCH}" \
        "${SING_ENV_ARGS[@]}" \
        "${VLLMD_SIF}" \
        bash -c '
            export LD_LIBRARY_PATH="$(
                printf "%s" "${LD_LIBRARY_PATH:-}" \
                | tr ":" "\n" | grep -v "^/.singularity.d" \
                | tr "\n" ":" | sed "s/:$//"
            )"
            exec '"${VLLM_CMD_STR}"'
        '
else
    # ROCM_COMPAT=0: only for clusters that propagate device cgroups natively.
    "${SING_CMD}" exec \
        --bind /dev/kfd \
        --bind /dev/dri \
        --bind "${SCRATCH}" \
        "${SING_ENV_ARGS[@]}" \
        "${VLLMD_SIF}" \
        "${VLLM_ARGS[@]}"
fi
