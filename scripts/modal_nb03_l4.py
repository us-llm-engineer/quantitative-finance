"""Run the immutable notebook-03 training/replay/figure pipeline on one Modal L4.

Usage:
  modal run scripts/modal_nb03_l4.py --run-id l4-20261003-full-01 --workers 4
  modal volume get rough-hedge-nb03-results runs/l4-20261003-full-01 ./cloud-results

The cloud function never alters the local tree. It copies only its new run-scoped
outputs into a persistent Modal volume; a later transfer can publish that compact
bundle to OneDrive without uploading raw market inputs.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/rough-hedge"
VOLUME_NAME = "rough-hedge-nb03-results"


def _ignore(path: Path) -> bool:
    # Modal may probe a project-relative candidate rather than an absolute path
    # while walking the upload tree.  Treat that as relative to ROOT, and leave
    # any unrelated absolute candidate alone.
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        rel = path if not path.is_absolute() else Path(path.name)
    blocked = {".git", ".pytest_cache", "__pycache__", "log", "logs", "plans", "research", "traces", "state", "nlm", "nlm-b", "exec", "review", "lookahead"}
    return any(part in blocked for part in rel.parts) or path.suffix in {".pyc", ".sqlite", ".wal", ".shm"}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.1.3", "scipy==1.14.1", "pandas==2.2.3", "matplotlib==3.10.0",
        "pillow==11.3.0", "psutil==6.1.0",
    )
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .add_local_dir(ROOT, REMOTE_ROOT, copy=True, ignore=_ignore)
)
app = modal.App("rough-hedge-nb03-l4")
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, gpu="L4", cpu=4, memory=8192, timeout=1800,
              volumes={"/outputs": results_volume})
def run(run_id: str, workers: int = 4) -> dict:
    if workers < 1 or workers > 4:
        raise ValueError("workers must be in [1, 4]")
    out_root = Path("/outputs/runs") / run_id
    if out_root.exists():
        raise FileExistsError(f"immutable cloud result already exists: {out_root}")
    # Sandbox containers can be reused after a failed invocation.  A unique
    # workspace prevents a stale *ephemeral* failed run from tripping the
    # immutable artifact guard for a later official run.
    work = Path(tempfile.mkdtemp(prefix="rough-hedge-nb03-"))
    shutil.copytree(REMOTE_ROOT, work)
    env = {**dict(__import__("os").environ), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONDONTWRITEBYTECODE": "1"}
    started = time.time()
    commands = [
        ["python", "scripts/exp_nb03_hedging.py", "train", "--run-id", run_id, "--workers", str(workers)],
        ["python", "scripts/exp_nb03_hedging.py", "replay", "--run-id", run_id],
        ["python", "scripts/exp_nb03_hedging.py", "figures", "--run-id", run_id],
    ]
    logs = work / "results" / "nb03" / "runs" / run_id / "cloud-command-logs"
    logs.mkdir(parents=True)
    for idx, command in enumerate(commands, start=1):
        log_path = logs / f"{idx:02d}-{command[2]}.log"
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            stream.flush()
            try:
                subprocess.run(command, cwd=work, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True,
                               timeout=1700 - int(time.time() - started))
            except subprocess.CalledProcessError as exc:
                # Surface the child-process diagnosis in the Modal task log.  The
                # run remains uncommitted and therefore safe to retry under its
                # immutable identifier.
                stream.flush()
                tail = log_path.read_text(encoding="utf-8")[-12000:]
                raise RuntimeError(f"cloud command failed: {' '.join(command)}\\n{tail}") from exc
    local_run = work / "results" / "nb03" / "runs" / run_id
    figure_names = [f"fig-4.5-{run_id}.csv", f"fig-4.5-{run_id}.png", f"fig-4.6-{run_id}.csv",
                    f"fig-4.6-{run_id}.png", f"fig-4.6-{run_id}-failure_paths.csv"]
    figures = work / "figures"
    missing = [name for name in figure_names if not (figures / name).is_file()]
    if missing:
        raise RuntimeError(f"expected new figures missing: {missing}")
    shutil.copytree(local_run, out_root / "results")
    (out_root / "figures").mkdir(parents=True)
    for name in figure_names:
        shutil.copy2(figures / name, out_root / "figures" / name)
    manifest = {
        "run_id": run_id,
        "hardware": __import__("torch").cuda.get_device_name(0),
        "workers": workers,
        "wall_s": time.time() - started,
        "figure_files": figure_names,
        "result_dir": f"runs/{run_id}/results",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results_volume.commit()
    return manifest


@app.local_entrypoint()
def main(run_id: str, workers: int = 4):
    print(json.dumps(run.remote(run_id, workers), indent=2))
