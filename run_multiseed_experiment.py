#!/usr/bin/env python3
"""Execute the k=5 training/evaluation notebooks for 10 independent seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from aggregate_multiseed_results import aggregate_results


DEFAULT_ROOT = Path(
    os.environ.get("MULTISEED_ROOT", "coco2014_multiseed_k5")
)
LC_CHECKPOINT = (
    "lc_protonets_52way_5shot_dim2048_finetuneTrue_euclidean_bias14.7.pth"
)
STEPS = (
    (
        "step3_lc_protonet.ipynb",
        Path("lc_protonets") / "checkpoints" / LC_CHECKPOINT,
    ),
    (
        "step4_comc_text_only_training.ipynb",
        Path("comc") / "comc_classifier_best.pth",
    ),
    (
        "step5_ensemble_evaluation0.1.ipynb",
        Path("evaluation") / "ensemble_evaluation_results.json",
    ),
)


def execute_notebook(
    notebook: Path,
    executed_notebook: Path,
    env: dict[str, str],
    kernel_name: str,
    progress_file: Path,
    seed: int,
) -> None:
    executed_notebook.parent.mkdir(parents=True, exist_ok=True)
    log_file = executed_notebook.with_suffix(".log")
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output",
        executed_notebook.name,
        "--output-dir",
        str(executed_notebook.parent),
        "--ExecutePreprocessor.timeout=-1",
        f"--ExecutePreprocessor.kernel_name={kernel_name}",
    ]
    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=notebook.parent,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        monitor_notebook_process(
            process,
            progress_file,
            seed,
            notebook.name,
        )

    if process.returncode != 0:
        tail = read_log_tail(log_file, 40)
        raise RuntimeError(
            f"Notebook failed with exit code {process.returncode}: {notebook}\n"
            f"Log: {log_file}\n\nLast log lines:\n{tail}"
        )


def read_log_tail(log_file: Path, line_count: int) -> str:
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<log unavailable>"
    return "\n".join(lines[-line_count:])


def read_gpu_status() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "GPU n/a"

    first_gpu = result.stdout.strip().splitlines()[0]
    used, total, utilization = [value.strip() for value in first_gpu.split(",")]
    return f"GPU {used}/{total} MiB, util {utilization}%"


def format_progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def monitor_notebook_process(
    process: subprocess.Popen,
    progress_file: Path,
    seed: int,
    notebook_name: str,
) -> None:
    started_at = time.monotonic()
    last_gpu_check = 0.0
    gpu_status = "GPU pending"
    spinner = "|/-\\"
    spinner_index = 0

    try:
        while process.poll() is None:
            now = time.monotonic()
            if now - last_gpu_check >= 10:
                gpu_status = read_gpu_status()
                last_gpu_check = now

            payload = None
            try:
                with progress_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                pass

            elapsed = int(now - started_at)
            if payload:
                phase = str(payload.get("phase", notebook_name))
                current = int(payload.get("current", 0))
                total = int(payload.get("total", 0))
                percent = (100.0 * current / total) if total else 0.0
                bar = format_progress_bar(current, total)
                details = payload.get("details", {})
                loss_text = ""
                if "loss" in details:
                    loss_text = f" | loss {float(details['loss']):.4f}"
                status = (
                    f"seed {seed} | {phase:<26} {bar} "
                    f"{current:>4}/{total:<4} {percent:6.2f}% | "
                    f"{elapsed:>5}s | {gpu_status}{loss_text}"
                )
            else:
                marker = spinner[spinner_index % len(spinner)]
                spinner_index += 1
                status = (
                    f"seed {seed} | {notebook_name:<26} {marker} | "
                    f"{elapsed:>5}s | {gpu_status}"
                )

            print("\r" + status[:180].ljust(180), end="", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    print()


def install_current_environment_kernel(kernel_name: str) -> None:
    """Register a kernelspec that points to the current Python executable."""
    command = [
        sys.executable,
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        kernel_name,
        "--display-name",
        f"Python ({kernel_name})",
    ]
    print(f"Registering Jupyter kernel '{kernel_name}' with {sys.executable}")
    subprocess.run(command, check=True)


def write_manifest(
    run_dir: Path,
    seed: int,
    output_root: Path,
    completed_steps: list[str],
    status: str,
) -> None:
    manifest = {
        "experiment_seed": seed,
        "episode_seed": seed,
        "episodes": 200,
        # Paths in the manifest are relative to its containing seed directory.
        "output_root": os.path.relpath(output_root, run_dir).replace(os.sep, "/"),
        "completed_steps": completed_steps,
        "status": status,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def run_seed(
    script_dir: Path,
    output_root: Path,
    seed: int,
    force: bool,
    kernel_name: str,
) -> None:
    run_dir = output_root / f"seed_{seed}"
    completed_steps = []
    env = os.environ.copy()
    env["EXPERIMENT_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    env["MULTISEED_ROOT"] = str(output_root)
    env.setdefault("STRICT_DETERMINISM", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    progress_file = run_dir / "live_progress.json"
    env["RUN_PROGRESS_FILE"] = str(progress_file)

    write_manifest(run_dir, seed, output_root, completed_steps, "running")
    try:
        for notebook_name, relative_artifact in STEPS:
            expected_artifact = run_dir / relative_artifact
            if expected_artifact.exists() and not force:
                print(
                    f"[seed {seed}] resume: {notebook_name} already produced "
                    f"{expected_artifact}"
                )
                completed_steps.append(notebook_name)
                continue

            notebook = script_dir / notebook_name
            if not notebook.exists():
                raise FileNotFoundError(notebook)

            executed_notebook = (
                run_dir / "executed_notebooks" / f"executed_{notebook_name}"
            )
            progress_file.unlink(missing_ok=True)
            print(f"[seed {seed}] executing {notebook_name}")
            execute_notebook(
                notebook,
                executed_notebook,
                env,
                kernel_name,
                progress_file,
                seed,
            )
            if not expected_artifact.exists():
                raise FileNotFoundError(
                    f"Notebook completed but expected artifact is missing: "
                    f"{expected_artifact}"
                )
            completed_steps.append(notebook_name)
            write_manifest(run_dir, seed, output_root, completed_steps, "running")

        write_manifest(run_dir, seed, output_root, completed_steps, "complete")
        print(f"[seed {seed}] complete")
    except Exception:
        write_manifest(run_dir, seed, output_root, completed_steps, "failed")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42, 52)),
        help="Exactly 10 independent seeds by default: 42 through 51.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-execute completed notebooks instead of resuming.",
    )
    parser.add_argument(
        "--kernel-name",
        default="multiseed-k5",
        help=(
            "Jupyter kernel registered for the current Python environment; "
            "default is multiseed-k5."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 10:
        raise ValueError(
            f"This protocol requires exactly 10 seeds; received {len(args.seeds)}"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")

    script_dir = Path(__file__).resolve().parent
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    install_current_environment_kernel(args.kernel_name)

    for seed in args.seeds:
        run_seed(
            script_dir,
            output_root,
            seed,
            args.force,
            args.kernel_name,
        )

    payload = aggregate_results(output_root, args.seeds)
    print("All 10 seeds completed; mean +/- SD artifacts:")
    for name, path in payload["artifacts"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
