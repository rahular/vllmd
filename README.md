# vllmd — vLLM Daemon

Dynamically manage vLLM workers on a Slurm cluster behind an nginx load balancer. Start with zero workers and scale up or down with single commands. Run multiple named sessions side-by-side (one per model) and switch between them with `vllmd use`.

```
vllmd start --name qwen3 --model Qwen/Qwen3-30B-A3B-Thinking-2507 --tensor-parallel-size 2
vllmd add 8
vllmd status
vllmd check
vllmd stop
```

---

## How It Works

```
submit node
┌──────────────────────────────────────────────────────┐
│  vllmd CLI                                           │
│      │  HTTP                                         │
│  Session Manager (FastAPI daemon)                    │
│      │  sbatch (one job per worker)                  │
│  nginx (load balancer, :9000)                        │
└──────────────────────────────────────────────────────┘
            │  registered workers POST /heartbeat every 30s
            ▼
    compute node A          compute node B
  ┌──────────────┐        ┌──────────────┐
  │ vLLM :10000  │        │ vLLM :10001  │
  └──────────────┘        └──────────────┘
```

- **Session manager** — FastAPI daemon running on the submit node. Tracks worker state, manages nginx config, monitors heartbeats.
- **Workers** — Each `vllmd add N` submits N independent `sbatch` jobs. Each job runs `scripts/serve.sh` (starts vLLM) + `vllmd-worker` (registers + heartbeats).
- **nginx** — Proxies requests to all healthy workers using `least_conn`. Config is rewritten and reloaded gracefully (`nginx -s reload`) every time a worker joins or leaves.
- **Dead detection** — If a worker misses heartbeats for 90s it is removed from nginx automatically.
- **State** — Persisted atomically to `~/.vllmd/sessions/<name>/state.json` so `vllmd status` works across terminals.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10 and nginx somewhere in PATH (or specify `--nginx-bin`).

For development:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Quick Start

### Single model

```bash
# Start session manager + nginx (no workers yet)
vllmd start \
    --name qwen3 \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 2 \
    --partition standard-g \
    --account <your-account>

# Add 4 workers (submits 4 independent sbatch jobs)
vllmd add 4

# Watch them come online
vllmd status

# Send a test prompt through the load balancer
vllmd check

# API endpoint (same as a normal OpenAI-compatible server)
curl http://<submit-node>:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
       "messages": [{"role": "user", "content": "Hello"}]}'
```

### Multiple models

```bash
# Start two sessions — ports are assigned automatically, no conflicts
vllmd start --name qwen3  --model Qwen/Qwen3-30B-A3B-Thinking-2507 --tensor-parallel-size 2
vllmd start --name fast   --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1

# Switch between sessions
vllmd use qwen3
vllmd add 4         # adds workers to qwen3
vllmd status        # shows qwen3 status

vllmd use fast
vllmd add 2         # adds workers to fast
vllmd check         # checks fast's endpoint

# See all sessions (* = current)
vllmd list
```

Port layout per session (1000 worker ports reserved each):

| Slot | Workers | LB port | Manager port |
|------|---------|---------|--------------|
| 0 | 10000–10999 | :9000 | :9500 |
| 1 | 11000–11999 | :9001 | :9501 |
| 2 | 12000–12999 | :9002 | :9502 |

Slots are assigned automatically. When a session is stopped its slot is freed and reused by the next `vllmd start`.

### MoE model with Expert Parallelism

```bash
# Qwen3-235B-A22B: TP=2, DP=4 → 8 GPUs per worker, experts sharded across all TP×DP GPUs
vllmd start \
    --name moe \
    --model Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --partition standard-g

vllmd add 2
```

### Singularity / ROCm (LUMI)

```bash
# Build SIF once (on a login node)
singularity pull docker://rocm/vllm:latest
# or build your own: see scripts/serve.sh

vllmd start \
    --name qwen3 \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 2 \
    --sif /scratch/project_462000963/vllm.sif \
    --scratch /scratch/project_462000963 \
    --partition standard-g \
    --account <your-account>
```

When `--sif` is provided, workers run inside the container. `ROCM_COMPAT=1` is the default: Singularity's `--rocm` flag handles AMD KFD device delegation (required on LUMI due to cgroups v2), and `/.singularity.d/libs` is stripped from `LD_LIBRARY_PATH` so the container's own ROCm copy is used.

---

## CLI Reference

### `vllmd start`

Start a new session: create state, launch the session manager daemon and nginx.

| Flag | Default | Description |
|------|---------|-------------|
| `--name`, `-n` | *(prompted)* | Session name (e.g. `qwen3`, `fast`). Must be unique |
| `--model`, `-m` | *(prompted)* | HuggingFace model ID or local path |
| `--tensor-parallel-size` | `1` | Tensor parallel size |
| `--data-parallel-size` | `1` | Data parallel size |
| `--enable-expert-parallel` | `false` | Enable expert parallelism (MoE) |
| `--sif` | *(none)* | Singularity SIF path. Omit for native Python |
| `--max-model-len` | *(vLLM default)* | Max sequence length |
| `--gpu-mem-util` | `0.92` | GPU memory utilisation fraction |
| `--account` | *(none)* | Slurm account |
| `--partition` | `standard-g` | Slurm partition |
| `--time` | `2-00:00:00` | Slurm `--time` per worker job |
| `--scratch` | `/scratch/project_462000963` | Scratch filesystem root |
| `--hf-home` | `<scratch>/hf_cache` | `HF_HOME` inside workers |
| `--vllm-args` | *(none)* | Extra vLLM server flags (quoted string) |
| `--startup-timeout` | `900` | Seconds to wait for each worker's `/health` |
| `--nginx-bin` | *(auto-detected)* | Path to nginx binary |
| `--foreground` | `false` | Run manager in foreground (for debugging) |

Ports (LB, manager, worker base) are assigned automatically from the next free slot.

### `vllmd use <name>`

Set the current session. All commands (`add`, `remove`, `status`, `check`, `stop`) operate on the current session by default.

```bash
vllmd use qwen3
```

When only one session is active, it is used automatically without needing `vllmd use`.

### `vllmd add N`

Submit N independent sbatch jobs. Workers register automatically when vLLM is healthy.

```bash
vllmd add 4
```

### `vllmd remove`

Cancel specific workers or the N oldest.

```bash
vllmd remove 2                     # remove 2 oldest workers
vllmd remove --worker-id worker-3  # remove a specific worker
```

### `vllmd status`

Show all workers: node, port, Slurm job ID, status (pending / healthy / dead), and age.

```bash
vllmd status
```

```
Session: qwen3
LB endpoint: http://nid001:9000/v1
Model: Qwen/Qwen3-30B-A3B-Thinking-2507  (tp=2, ep=1)
Workers: 4 healthy, 1 pending, 0 dead

 ID        Node      Port    Slurm Job  Status    Age
 worker-0  nid002    10000   12345      healthy   15m
 worker-1  nid003    10001   12346      healthy   15m
 worker-2  nid004    10002   12347      healthy   14m
 worker-3  nid005    10003   12348      healthy   14m
 worker-4  (pending) 10004   12349      pending   2m
```

### `vllmd list`

List all active sessions. The current session is marked with `*`.

```bash
vllmd list
```

```
  *  qwen3   0   http://nid001:9000/v1   Qwen/Qwen3-30B   4   running
     fast    1   http://nid001:9001/v1   Qwen2.5-7B       2   running
```

### `vllmd check`

Send a test prompt through the load balancer and print the response + latency.

```bash
vllmd check
vllmd check --prompt "Explain gradient descent in one sentence."
```

### `vllmd stop`

Cancel all workers, stop nginx, stop the session manager, and clean up state. If other sessions exist, the next one becomes current automatically.

```bash
vllmd stop
vllmd stop --yes   # skip confirmation
```

---

## Extra vLLM Args

Pass any vLLM server flag via `--vllm-args`:

```bash
vllmd start \
    --name qwen3 \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 2 \
    --vllm-args "--quantization awq --enable-chunked-prefill --max-num-seqs 256"
```

The string is word-split and appended verbatim to the vLLM command line inside each worker job.

---

## Architecture Details

### Session Manager

The manager is a FastAPI process daemonised by `vllmd start` (re-exec with `start_new_session=True`). Its endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/status` | Full session state |
| `POST` | `/register` | Worker calls this when vLLM is healthy |
| `POST` | `/heartbeat/{id}` | Worker calls every 30s |
| `POST` | `/workers/add` | Submit N new sbatch jobs |
| `POST` | `/workers/remove` | Cancel workers by ID or count |
| `POST` | `/shutdown` | Stop everything |

A background thread checks heartbeats every 15s. Workers that have not sent a heartbeat in 90s are marked dead and removed from nginx.

### Worker Lifecycle

```
vllmd add N
  └─► POST /workers/add  (N times)
        └─► sbatch worker_launch.sh
              ├─► bash serve.sh &   (starts vLLM)
              └─► vllmd-worker      (polls /health, POST /register, loops POST /heartbeat)
                    └─► heartbeat exits when session stops → kills vLLM PID
```

### nginx Config

nginx runs as a daemon in the session directory. On every worker add/remove the config is rewritten and `nginx -s reload` is called (graceful reload — no in-flight requests are dropped). When there are no healthy workers, a `down` placeholder backend is used so nginx starts cleanly and returns 502 until the first worker is ready.

### Port Assignment

Each session occupies a slot. Slots are the smallest non-negative integer not currently in use. Ports within a slot:

- **Workers**: `10000 + slot × 1000` to `10000 + slot × 1000 + 999`
- **LB (nginx)**: `9000 + slot`
- **Manager**: `9500 + slot`

Worker ports within a session are assigned monotonically (`port_base + seq`). The sequence counter never resets, so ports are never reused even after workers are removed.

### Current Session Pointer

`~/.vllmd/current` holds the name of the current session. `vllmd start` sets it automatically. `vllmd use <name>` switches it. When only one session exists, it is used automatically even without an explicit `vllmd use`. When a session is stopped, the pointer advances to the next active session if one exists.

### State Persistence

```
~/.vllmd/
├── current                  ← current session name
└── sessions/
    ├── qwen3/
    │   ├── state.json        ← full session + worker state (atomic write)
    │   ├── manager.url       ← http://nid001:9500
    │   ├── manager.pid       ← PID of session manager process
    │   ├── manager.log       ← session manager stdout/stderr
    │   └── nginx.conf        ← current nginx config
    └── fast/
        └── ...
```

Set `VLLMD_HOME` to override `~/.vllmd`.

---

## ROCm / LUMI Notes

- **`ROCM_COMPAT=1` (default)**: Uses `singularity --rocm` for AMD KFD device delegation. Required on LUMI because cgroups v2 blocks `/dev/kfd` even with `--bind /dev/kfd`. After `--rocm` injects host ROCm libs, `/.singularity.d/libs` is stripped from `LD_LIBRARY_PATH` to avoid glibc version mismatches with the container's Ubuntu 22.04.
- **`ROCR_VISIBLE_DEVICES`**: Set from `SLURM_JOB_GPUS` in `worker_launch.sh` to restrict vLLM to its allocated GPUs (needed because `--rocm` bypasses Slurm's cgroup GPU restriction).
- **SIF GFX arch**: The SIF tag must match the host GPU architecture. `serve.sh` includes a guard that fails with a clear error on mismatch. For MI250X use a tag containing `gfx90a` or a multi-arch tag.
- **`HOST` variable on Cray**: `HOST` is pre-set by the OS to the node hostname. `serve.sh` uses `BIND_HOST` (default `0.0.0.0`) to avoid accidentally binding to the node's hostname.

---

## Troubleshooting

**`vllmd status` shows workers stuck in `pending`**

Check the worker log: `squeue -j <slurm_job_id>` to see if the job is queued or running. If running, check the job's stdout for errors (often a model download or GPU visibility issue).

**`vllmd check` returns 502**

No healthy workers are registered yet. Run `vllmd status` and wait for workers to reach `healthy`, or check logs.

**Session manager did not start within 30s**

Check `~/.vllmd/sessions/<name>/manager.log`. Common causes: port conflict (another process on the auto-assigned port), missing nginx binary. Try `--foreground` to see errors directly.

**`GLIBC_2.38 not found` inside container**

`ROCM_COMPAT=1` is not stripping the injected libs correctly. Verify the `/.singularity.d/libs` strip in `serve.sh` is running. Alternatively set `ROCM_COMPAT=0` if your cluster propagates device cgroups natively.

**Multiple sessions, wrong one selected**

Run `vllmd list` to see which is current (`*`), then `vllmd use <name>` to switch.
