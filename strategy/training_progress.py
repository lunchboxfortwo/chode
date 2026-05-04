"""
Training progress tracking module for chode poker.
NN-only: reports neural preflop and postflop training status.
"""
import json
import os
import subprocess
import time
from pathlib import Path
import shutil


def _data_root() -> Path:
    return Path(__file__).parent.parent / "data"


def _is_trainer_running(script: str) -> bool:
    """Check if a trainer process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", script],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _nn_job(kind: str, progress_path: Path) -> dict:
    """Build a job dict for an NN trainer from its progress file."""
    prog = None
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                prog = json.load(f)
        except Exception:
            pass

    is_running = _is_trainer_running(f"train_{kind}_nn.py")

    if prog:
        iteration = prog.get("iteration", 0)
        total_iters = prog.get("target_games", 0)
        step = prog.get("step", 0)
        phase = prog.get("phase", "distill")
        games_done = prog.get("games_played", 0)
        elapsed = prog.get("elapsed_seconds", 0)
        rate = prog.get("games_per_second")
        ckpt_time = prog.get("last_checkpoint_time")
        progress_fresh = prog.get("last_update", 0) and (time.time() - prog.get("last_update", 0)) < 600

        # iteration/target_games is the actual progress metric
        # (iteration = distill round or online iter number)
        pct = min(100.0, (iteration / total_iters * 100.0) if total_iters > 0 else 0.0)

        if pct >= 100.0:
            status, eta = "done", None
        elif is_running and progress_fresh:
            status = "running"
            iters_left = total_iters - iteration
            # Use iteration rate, not games rate, for ETA
            if elapsed > 0 and iteration > 0:
                iters_per_sec = iteration / elapsed
                eta_secs = int(iters_left / iters_per_sec) if iters_per_sec > 0 else None
            else:
                eta_secs = None
            if eta_secs is not None:
                if eta_secs < 3600:
                    eta = f"{eta_secs // 60}m {eta_secs % 60}s"
                else:
                    eta = f"{eta_secs // 3600}h {(eta_secs % 3600) // 60}m"
            else:
                eta = None
        elif iteration > 0 and not is_running:
            status, eta = "halted", None
        else:
            status, eta = "pending", None

        return {
            "name": f"NN {kind.title()}",
            "phase": phase,
            "step": step,
            "pct_done": round(pct, 1),
            "status": status,
            "rate": round(rate, 0) if rate else None,
            "eta": eta,
            "games_played": games_done,
            "last_checkpoint": ckpt_time,
        }
    else:
        return {
            "name": f"NN {kind.title()}",
            "phase": "pending",
            "step": 0,
            "pct_done": 0.0,
            "status": "running" if is_running else "pending",
            "rate": None,
            "eta": None,
            "games_played": 0,
            "last_checkpoint": None,
        }


def all_jobs() -> list[dict]:
    """Return NN training jobs only."""
    jobs = []

    # NN Preflop
    preflop_prog_path = _data_root() / "preflop_nn" / "training_progress.json"
    jobs.append(_nn_job("preflop", preflop_prog_path))

    # NN Postflop
    postflop_prog_path = _data_root() / "postflop_nn" / "training_progress.json"
    jobs.append(_nn_job("postflop", postflop_prog_path))

    return jobs


def system_state() -> dict:
    """Return a system state snapshot: RAM, swap, disk, load, trainer count."""
    meminfo = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, val = line.split(":")
                meminfo[key.strip()] = int(val.split()[0])
    except Exception:
        pass

    ram_total_kb = meminfo.get("MemTotal", 0)
    ram_available_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
    swap_total_kb = meminfo.get("SwapTotal", 0)
    swap_free_kb = meminfo.get("SwapFree", 0)

    try:
        disk = shutil.disk_usage("/home/do/chode/data")
        disk_used_pct = disk.used / disk.total * 100 if disk.total > 0 else 0
        disk_free_gb = disk.free / 1_000_000_000
    except Exception:
        disk_used_pct, disk_free_gb = 0, 0

    load = os.getloadavg()

    n_trainers = 0
    for script in ("train_preflop_nn.py", "train_postflop_nn.py"):
        try:
            result = subprocess.run(
                ["pgrep", "-af", script],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                n_trainers += len(result.stdout.decode().splitlines())
        except Exception:
            pass

    return {
        "ram_used_gb": (ram_total_kb - ram_available_kb) / 1024 / 1024,
        "free_ram_gb": ram_available_kb / 1024 / 1024,
        "swap_used_gb": (swap_total_kb - swap_free_kb) / 1024 / 1024,
        "disk_used_pct": disk_used_pct,
        "disk_free_gb": disk_free_gb,
        "load_avg_1m": load[0],
        "load_avg_5m": load[1],
        "load_avg_15m": load[2],
        "active_trainers": n_trainers,
        "cpu_pct": round(load[0] / os.cpu_count() * 100, 0) if os.cpu_count() else 0,
    }
