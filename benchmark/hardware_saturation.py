"""Concurrent CPU/GPU saturation test for a Linux PyTorch environment."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - exercised on minimal environments
    psutil = None  # type: ignore[assignment]


def _cpu_worker(stop: Any, threads: int, size: int) -> None:
    del threads, size
    value = 0x9E3779B97F4A7C15
    while not stop.is_set():
        for _ in range(100_000):
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
            value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
            value ^= value >> 31


def _gpu_loop(stop: Any, batch: int, result: dict[str, int]) -> None:
    import torch

    device = torch.device("cuda")
    # FP16 GEMMs are representative of the network's CUDA workload while
    # keeping memory bounded on 12 GB and larger cards.
    width = max(1024, min(4096, batch))
    left = torch.randn((width, width), device=device, dtype=torch.float16)
    right = torch.randn((width, width), device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    operations = 0
    while not stop.is_set():
        torch.mm(left, right)
        operations += 1
    torch.cuda.synchronize()
    result["operations"] = operations


def _nvidia_snapshot() -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        gpu, used, total = (
            float(item.strip()) for item in result.stdout.splitlines()[0].split(",")
        )
        return {
            "gpu_utilization_percent": gpu,
            "memory_used_mib": used,
            "memory_total_mib": total,
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run(
    duration: float, *, device: str, cpu_workers: int, cpu_threads: int, gpu_batch: int
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment"
        ) from error
    if psutil is None:
        raise RuntimeError("psutil is required for CPU utilization sampling")
    if duration <= 0 or cpu_workers <= 0 or cpu_threads <= 0 or gpu_batch <= 0:
        raise ValueError("duration and worker parameters must be positive")
    cuda_available = bool(torch.cuda.is_available())
    use_gpu = device == "cuda" or (device == "auto" and cuda_available)
    if use_gpu and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    context = mp.get_context("spawn")
    stop = context.Event()
    processes = [
        context.Process(target=_cpu_worker, args=(stop, cpu_threads, 512), daemon=True)
        for _ in range(cpu_workers)
    ]
    for process in processes:
        process.start()
    started = time.perf_counter()
    cpu_samples: list[float] = []
    gpu_samples: list[dict[str, float]] = []
    gpu_result = {"operations": 0}
    gpu_thread: threading.Thread | None = None
    if use_gpu:
        gpu_thread = threading.Thread(
            target=_gpu_loop,
            args=(stop, gpu_batch, gpu_result),
            name="hexadeca-gpu-saturation",
            daemon=True,
        )
        gpu_thread.start()
    try:
        while time.perf_counter() - started < duration:
            cpu_samples.append(float(psutil.cpu_percent(interval=0.5)))
            snapshot = _nvidia_snapshot()
            if snapshot is not None:
                gpu_samples.append(snapshot)
    finally:
        stop.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=10)
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        if use_gpu:
            torch.cuda.empty_cache()
    elapsed = max(1e-9, time.perf_counter() - started)
    gpu_operations = gpu_result["operations"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "duration_seconds": elapsed,
        "cpu_workers": cpu_workers,
        "cpu_threads_per_worker": cpu_threads,
        "cpu_utilization_percent_mean": sum(cpu_samples) / len(cpu_samples)
        if cpu_samples
        else 0.0,
        "cpu_utilization_percent_max": max(cpu_samples) if cpu_samples else 0.0,
        "gpu_operations": gpu_operations,
        "gpu_operations_per_second": gpu_operations / elapsed,
        "nvidia_samples": gpu_samples,
        "gpu_utilization_percent_mean": (
            sum(item["gpu_utilization_percent"] for item in gpu_samples)
            / len(gpu_samples)
            if gpu_samples
            else None
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--gpu-batch", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(
        args.duration,
        device=args.device,
        cpu_workers=args.cpu_workers,
        cpu_threads=args.cpu_threads,
        gpu_batch=args.gpu_batch,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
