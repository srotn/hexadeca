"""Reproducible Stage 6 CPU and CUDA batch-inference benchmark."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from benchmark.timing import measure_operation
from config import load_config
from network import NetworkSpecification, PolicyValueNetwork


def run_network_benchmark(
    specification: NetworkSpecification,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    cpu_batch_size: int,
    gpu_batch_sizes: tuple[int, ...],
    random_seed: int,
    include_cuda: bool = True,
) -> dict[str, object]:
    """Measure inference latency and position throughput by device and dtype."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    if cpu_batch_size <= 0 or any(batch_size <= 0 for batch_size in gpu_batch_sizes):
        raise ValueError("Benchmark batch sizes must be positive")

    torch.manual_seed(random_seed)
    model = PolicyValueNetwork(specification).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    results: dict[str, object] = {
        "cpu_float32": _measure_inference(
            model,
            specification,
            device=torch.device("cpu"),
            batch_size=cpu_batch_size,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            use_amp=False,
        )
    }

    cuda_available = torch.cuda.is_available()
    if include_cuda and cuda_available:
        cuda_device = torch.device("cuda")
        cuda_model = model.to(cuda_device)
        for batch_size in gpu_batch_sizes:
            results[f"cuda_float32_batch_{batch_size}"] = _measure_inference(
                cuda_model,
                specification,
                device=cuda_device,
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                use_amp=False,
            )
            results[f"cuda_amp_batch_{batch_size}"] = _measure_inference(
                cuda_model,
                specification,
                device=cuda_device,
                batch_size=batch_size,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                use_amp=True,
            )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "system": platform.platform(),
            "processor": platform.processor() or "unavailable",
            "cpu_threads": torch.get_num_threads(),
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if cuda_available else "unavailable"
            ),
        },
        "network": {
            **specification.to_dict(),
            "parameter_count": parameter_count,
        },
        "parameters": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "cpu_batch_size": cpu_batch_size,
            "gpu_batch_sizes": list(gpu_batch_sizes),
            "random_seed": random_seed,
        },
        "results": results,
    }


def _measure_inference(
    model: PolicyValueNetwork,
    specification: NetworkSpecification,
    *,
    device: torch.device,
    batch_size: int,
    warmup_iterations: int,
    measurement_iterations: int,
    use_amp: bool,
) -> dict[str, int | float | str]:
    inputs = torch.randn(
        batch_size,
        specification.input_planes,
        specification.board_size,
        specification.board_size,
        device=device,
    )

    def inference() -> object:
        amp_context: AbstractContextManager[Any]
        if use_amp:
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            amp_context = nullcontext()
        with torch.inference_mode(), amp_context:
            output = model(inputs)
            win_probability = output.win_probability
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return output, win_probability

    timing = measure_operation(inference, warmup_iterations, measurement_iterations)
    throughput = float(timing["throughput_ops_per_second"])
    return {
        **timing,
        "batch_size": batch_size,
        "positions_per_second": throughput * batch_size,
        "precision": "amp_float16" if use_amp else "float32",
    }


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional TOML profile layered over config/default.toml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults to the configured benchmark directory.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip CUDA measurements even when a CUDA device is available.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run configured inference benchmarks and persist their JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    specification = NetworkSpecification.from_config(config.rules, config.network)
    report = run_network_benchmark(
        specification,
        warmup_iterations=config.benchmark.network_warmup_iterations,
        measurement_iterations=config.benchmark.network_measurement_iterations,
        cpu_batch_size=config.benchmark.network_cpu_batch_size,
        gpu_batch_sizes=config.benchmark.network_gpu_batch_sizes,
        random_seed=config.benchmark.random_seed,
        include_cuda=not parsed.cpu_only,
    )
    destination = output_path or (
        config.paths.benchmark_directory / "stage6-network.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
