"""vllmd CLI — vLLM dispatcher command line interface."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .client import SessionClient, check_endpoint
from .session import (
    SessionConfig,
    SessionState,
    WorkerStatus,
    clear_current,
    delete_session,
    list_sessions,
    load_state,
    next_slot,
    read_current,
    read_manager_url,
    save_state,
    session_dir,
    write_current,
    write_manager_pid,
    write_manager_url,
)

app = typer.Typer(
    name="vllmd",
    help="Dynamic vLLM dispatcher — start, scale, and manage vLLM workers on Slurm clusters.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

@app.command()
def start(
    name: Optional[str]          = typer.Option(None, "--name", "-n", help="Session name (e.g. 'qwen3', 'fast'). Must be unique."),
    model: Optional[str]         = typer.Option(None, "--model", "-m", help="HuggingFace model ID or local path."),
    tensor_parallel_size: int    = typer.Option(1,     "--tensor-parallel-size",    help="Tensor parallel size."),
    data_parallel_size: int      = typer.Option(1,     "--data-parallel-size",      help="Data parallel size."),
    enable_expert_parallel: bool = typer.Option(False, "--enable-expert-parallel",  help="Enable expert parallelism (MoE).", is_flag=True),
    sif: Optional[str]           = typer.Option(None, "--sif",              help="Singularity SIF path. Omit to run native Python."),
    max_model_len: Optional[int] = typer.Option(None, "--max-model-len",    help="Max sequence length (passed to vLLM)."),
    gpu_mem_util: float          = typer.Option(0.92, "--gpu-mem-util",     help="GPU memory utilisation fraction for vLLM KV cache."),
    account: Optional[str]       = typer.Option(None, "--account",          help="Slurm account."),
    partition: str               = typer.Option("standard-g", "--partition", help="Slurm partition."),
    time_limit: str              = typer.Option("2-00:00:00", "--time",     help="Slurm --time per worker job."),
    scratch: str                 = typer.Option("/scratch/project_462000963", "--scratch", help="Scratch filesystem root."),
    hf_home: Optional[str]       = typer.Option(None, "--hf-home",         help="HF_HOME inside workers. Defaults to <scratch>/hf_cache."),
    vllm_extra_args: Optional[str] = typer.Option(None, "--vllm-args",     help="Extra vLLM server args (quoted string, e.g. '--trust-remote-code --quantization awq')."),
    startup_timeout: int         = typer.Option(900,  "--startup-timeout",  help="Seconds to wait for each vLLM worker to become healthy."),
    nginx_bin: Optional[str]     = typer.Option(None, "--nginx-bin",        help="Path to nginx binary. Auto-detected if omitted."),
    foreground: bool             = typer.Option(False, "--foreground",       help="Run session manager in the foreground (for debugging)."),
):
    """Start a new vllmd session: configure the model and launch the session manager + nginx."""

    # Prompt for required values not given on CLI
    if not name:
        name = typer.prompt("Session name")
    if not model:
        model = typer.prompt("Model (HuggingFace ID or local path)")

    # Check name is not already in use
    existing = list_sessions()
    if name in existing:
        rprint(f"[red]Error:[/red] A session named '[bold]{name}[/bold]' already exists.")
        rprint("Run [bold]vllmd stop[/bold] to stop it, or choose a different name.")
        raise typer.Exit(1)

    # Assign the next free port slot
    slot = next_slot()

    cfg = SessionConfig(
        session_id=name,
        slot=slot,
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        sif=sif,
        max_model_len=max_model_len,
        gpu_mem_util=gpu_mem_util,
        account=account,
        partition=partition,
        time_limit=time_limit,
        scratch=scratch,
        hf_home=hf_home,
        vllm_extra_args=vllm_extra_args,
        startup_timeout=startup_timeout,
    )

    host = socket.gethostname()
    state = SessionState(config=cfg, manager_host=host)
    sdir = session_dir(name)
    sdir.mkdir(parents=True, exist_ok=True)
    save_state(state)

    rprint(f"[bold]Session:[/bold]     {name}  (slot {slot})")
    rprint(f"[bold]Model:[/bold]       {model}")
    rprint(f"[bold]Parallelism:[/bold] tensor-parallel-size={tensor_parallel_size}  data-parallel-size={data_parallel_size}  enable-expert-parallel={enable_expert_parallel}  →  {cfg.gpus_per_worker} GPU(s) per worker")
    rprint(f"[bold]LB endpoint:[/bold] http://{host}:{cfg.lb_port}/v1")
    rprint(f"[bold]Manager:[/bold]     http://{host}:{cfg.manager_port}")

    # Find nginx upfront so we fail early with a helpful message
    from .nginx import NginxManager, _find_nginx
    try:
        resolved_nginx = nginx_bin or _find_nginx()
    except FileNotFoundError as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if foreground:
        nginx_mgr = NginxManager(sdir, cfg.lb_port, nginx_bin=resolved_nginx)
        from .server import serve
        serve(state, "0.0.0.0", cfg.manager_port, nginx_mgr)
    else:
        cmd = [
            sys.executable, "-m", "vllmd.server",
            "--session-id", name,
            "--host", "0.0.0.0",
            "--nginx-bin", resolved_nginx,
        ]
        log_path = sdir / "manager.log"
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=lf,
                start_new_session=True,
            )
        write_manager_pid(name, proc.pid)

        # Wait until the manager is accepting connections
        url = f"http://localhost:{cfg.manager_port}"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            client = SessionClient(url)
            if client.health():
                break
            time.sleep(0.5)
        else:
            rprint("[red]Session manager did not start within 30s. Check:[/red]")
            rprint(f"  {log_path}")
            raise typer.Exit(1)

        write_manager_url(name, host, cfg.manager_port)
        write_current(name)
        rprint(f"\n[green]Session '{name}' started.[/green]  (current session set to '{name}')")
        rprint(f"  Manager log: {log_path}")
        rprint(f"  vllmd add 4    # start 4 workers")
        rprint(f"  vllmd status   # check progress")


# ---------------------------------------------------------------------------
# use
# ---------------------------------------------------------------------------

@app.command()
def use(name: str = typer.Argument(..., help="Session name to set as current.")):
    """Set the current session (used by default by all other commands)."""
    sessions = list_sessions()
    if name not in sessions:
        rprint(f"[red]Error:[/red] No session named '[bold]{name}[/bold]'.")
        rprint(f"Active sessions: {', '.join(sessions) if sessions else '(none)'}")
        raise typer.Exit(1)
    write_current(name)
    rprint(f"[green]Current session:[/green] {name}")


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

@app.command()
def add(
    count: int = typer.Argument(..., help="Number of workers to add."),
):
    """Add N vLLM workers to the current session (submits N independent sbatch jobs)."""
    sid = _require_session()
    client = _require_client(sid)

    with console.status(f"Submitting {count} worker job(s)..."):
        workers = client.add_workers(count)

    table = Table(title=f"Submitted {len(workers)} worker(s)")
    table.add_column("Worker ID")
    table.add_column("Slurm Job ID")
    table.add_column("Port")
    for w in workers:
        table.add_row(w["worker_id"], str(w.get("slurm_job_id") or "—"), str(w["port"]))
    console.print(table)
    rprint("\nWorkers will register automatically when vLLM is healthy. Run [bold]vllmd status[/bold] to monitor.")


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

@app.command()
def remove(
    count: Optional[int]     = typer.Argument(None, help="Number of workers to remove (oldest first)."),
    worker_id: Optional[str] = typer.Option(None, "--worker-id", help="Specific worker ID to remove."),
    dead: bool               = typer.Option(False, "--dead", help="Remove all dead workers.", is_flag=True),
):
    """Remove N workers (or a specific worker) from the current session."""
    if count is None and worker_id is None and not dead:
        rprint("[red]Error:[/red] Provide a count, --worker-id, or --dead.")
        raise typer.Exit(1)

    sid = _require_session()
    client = _require_client(sid)

    removed = client.remove_workers(
        count=count,
        worker_ids=[worker_id] if worker_id else None,
        dead=dead,
    )
    rprint(f"[green]Removed:[/green] {', '.join(removed) if removed else 'none'}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status():
    """Show status of the current session: workers, nodes, ports, and health."""
    sid = _require_session()
    client = _require_client(sid)

    data = client.status()

    log_path = session_dir(sid) / "manager.log"
    rprint(f"\n[bold]Session:[/bold] {data['session_id']}")
    rprint(f"[bold]LB endpoint:[/bold] {data['lb_endpoint']}")
    rprint(f"[bold]Manager:[/bold]     {data['manager_url']}  (log: {log_path})")
    rprint(f"[bold]Model:[/bold] {data['model']}  (tensor-parallel-size={data['tensor_parallel_size']}, data-parallel-size={data['data_parallel_size']}, enable-expert-parallel={data['enable_expert_parallel']})")
    rprint(
        f"[bold]Workers:[/bold] "
        f"[green]{data['healthy']} healthy[/green], "
        f"[yellow]{data['pending']} pending[/yellow], "
        f"[red]{data['dead']} dead[/red]"
    )

    workers = data.get("workers", [])
    if not workers:
        rprint("\n  No workers yet. Run [bold]vllmd add N[/bold] to start some.")
        return

    table = Table()
    table.add_column("ID",         style="bold")
    table.add_column("Node")
    table.add_column("Port")
    table.add_column("Slurm Job")
    table.add_column("Status")
    table.add_column("Age")

    status_color = {
        WorkerStatus.HEALTHY: "green",
        WorkerStatus.PENDING: "yellow",
        WorkerStatus.DEAD:    "red",
    }

    for w in sorted(workers, key=lambda x: x["worker_id"]):
        age = f"{w['age_s'] // 60}m" if w.get("age_s") else "—"
        st = w["status"]
        colour = status_color.get(st, "white")
        table.add_row(
            w["worker_id"],
            w["hostname"] or "(pending)",
            str(w["port"]),
            w.get("slurm_job_id") or "—",
            f"[{colour}]{st}[/{colour}]",
            age,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

@app.command()
def check(
    prompt: str = typer.Option("What is 1+1?", "--prompt", "-p", help="Test prompt to send."),
):
    """Send a test prompt through the load balancer and print the response."""
    sid = _require_session()
    state = load_state(sid)

    with console.status("Sending test request..."):
        try:
            result = check_endpoint(state.lb_endpoint, state.config.model, prompt)
        except Exception as e:
            rprint(f"[red]Request failed:[/red] {e}")
            raise typer.Exit(1)

    rprint(f"\n[bold]Model:[/bold]    {result['model']}")
    rprint(f"[bold]Prompt:[/bold]   {prompt}")
    rprint(f"[bold]Response:[/bold] {result['response']}")
    rprint(
        f"[bold]Latency:[/bold]  {result['latency_s']}s  "
        f"({result['prompt_tokens']} prompt tokens, {result['completion_tokens']} completion tokens)"
    )


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

@app.command()
def stop(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Stop the current session: cancel all workers, stop nginx, stop session manager."""
    sid = _require_session()

    if not yes:
        typer.confirm(f"Stop session '{sid}' and cancel all workers?", abort=True)

    client = _require_client(sid)

    with console.status("Shutting down session..."):
        client.shutdown()
        time.sleep(1)

    delete_session(sid)
    if read_current() == sid:
        clear_current()
    rprint(f"[green]Session '{sid}' stopped.[/green]")

    # If other sessions exist, set current to the first one
    remaining = list_sessions()
    if remaining:
        write_current(remaining[0])
        rprint(f"Current session is now: [bold]{remaining[0]}[/bold]")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_cmd():
    """List all active vllmd sessions."""
    sessions = list_sessions()
    if not sessions:
        rprint("No active sessions.")
        return

    current = read_current()

    table = Table(title="vllmd sessions")
    table.add_column("")               # * marker
    table.add_column("Name",      style="bold")
    table.add_column("Slot")
    table.add_column("LB Endpoint")
    table.add_column("Model")
    table.add_column("Healthy")
    table.add_column("Manager")

    for sid in sessions:
        marker = "[green]*[/green]" if sid == current else " "
        try:
            state = load_state(sid)
            manager_url = read_manager_url(sid)
            client = SessionClient(manager_url, timeout=5)
            if client.health():
                try:
                    d = client.status()
                    healthy = str(d["healthy"])
                except Exception:
                    healthy = "?"
                manager_status = "[green]running[/green]"
            else:
                healthy = "?"
                manager_status = "[red]unreachable[/red]"
            table.add_row(
                marker,
                sid,
                str(state.config.slot),
                state.lb_endpoint,
                state.config.model,
                healthy,
                manager_status,
            )
        except Exception as e:
            table.add_row(marker, sid, "—", "—", "—", "—", f"[red]error: {e}[/red]")

    console.print(table)
    if current:
        rprint(f"\n[dim]* = current session  (change with [bold]vllmd use <name>[/bold])[/dim]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_session() -> str:
    current = read_current()
    if current:
        return current
    sessions = list_sessions()
    if not sessions:
        rprint("[red]Error:[/red] No active session. Run [bold]vllmd start[/bold] first.")
        raise typer.Exit(1)
    if len(sessions) == 1:
        return sessions[0]
    rprint(
        f"[red]Error:[/red] Multiple sessions active: {', '.join(sessions)}.\n"
        "Run [bold]vllmd use <name>[/bold] to select one."
    )
    raise typer.Exit(1)


def _require_client(session_id: str) -> SessionClient:
    try:
        url = read_manager_url(session_id)
    except FileNotFoundError as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    client = SessionClient(url)
    if not client.health():
        rprint(
            f"[red]Error:[/red] Session manager at {url} is not responding.\n"
            "Check the manager log or run [bold]vllmd start[/bold] again."
        )
        raise typer.Exit(1)
    return client
