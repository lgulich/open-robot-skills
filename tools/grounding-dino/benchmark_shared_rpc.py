#!/usr/bin/env python3
"""Bounded live POC for shared serialized vs dynamically batched DINO RPC.

Run each mode in a fresh process.  The harness is deliberately offline-only
and starts spawned scalar client processes before the parent initializes CUDA.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

_TOOL_NAME = "grounding-dino.detect"
_MODEL_ID = "IDEA-Research/grounding-dino-base"


class _CountingModel:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls = 0

    def __call__(self, **inputs: Any) -> Any:
        self.calls += 1
        return self.model(**inputs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def _fixed_image(index: int, size: int) -> np.ndarray:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    offset = 24 + (index % 4) * 12
    width = max(32, size // 3)
    image[offset : offset + width, size // 3 : 2 * size // 3] = (200, 30, 30)
    return image


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _equivalence_errors(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    expected_detections = expected["detections"]
    actual_detections = actual["detections"]
    errors: list[str] = []
    if len(expected_detections) != len(actual_detections):
        return [
            f"detection count: expected {len(expected_detections)}, got {len(actual_detections)}"
        ]
    expected_labels = [item["label"] for item in expected_detections]
    actual_labels = [item["label"] for item in actual_detections]
    if expected_labels != actual_labels:
        errors.append(f"labels: expected {expected_labels!r}, got {actual_labels!r}")
    keys = ("x1", "y1", "x2", "y2")
    for index, (expected_item, actual_item) in enumerate(
        zip(expected_detections, actual_detections, strict=True)
    ):
        expected_box = np.asarray([expected_item["box"][key] for key in keys])
        actual_box = np.asarray([actual_item["box"][key] for key in keys])
        if not np.allclose(expected_box, actual_box, rtol=rtol, atol=atol):
            errors.append(
                f"detection {index} box: expected {expected_box.tolist()}, "
                f"got {actual_box.tolist()}"
            )
        if not np.isclose(expected_item["score"], actual_item["score"], rtol=rtol, atol=atol):
            errors.append(
                f"detection {index} score: expected {expected_item['score']}, "
                f"got {actual_item['score']}"
            )
    return errors


def _load_bundle_registry() -> tuple[Any, Any]:
    from gap_core.tools import ToolRegistry

    tools_path = Path(__file__).with_name("tools.py")
    module_name = "gap_batch_poc_grounding_dino_tools"
    spec = importlib.util.spec_from_file_location(module_name, tools_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {tools_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    registry = ToolRegistry()
    registry.discover_pending()
    return module, registry


def _cached_snapshot() -> str:
    from huggingface_hub import try_to_load_from_cache

    required = ("config.json", "model.safetensors", "preprocessor_config.json")
    paths = [try_to_load_from_cache(_MODEL_ID, filename) for filename in required]
    missing = [
        name for name, path in zip(required, paths, strict=True) if not isinstance(path, str)
    ]
    if missing:
        raise RuntimeError(f"offline DINO cache is incomplete; missing {missing!r} for {_MODEL_ID}")
    return Path(paths[1]).parent.name


def _client_worker(
    client_index: int,
    socket_path: str,
    rounds: int,
    image_size: int,
    query_text: str,
    start_event: Any,
    barrier: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    from gap_core.rpc.shared import SharedToolClient

    client = None
    try:
        client = SharedToolClient("grounding-dino", socket_path)
        image = _fixed_image(client_index, image_size)
        ready_queue.put({"client": client_index, "ready": True})
        if not start_event.wait(timeout=600):
            raise TimeoutError("timed out waiting for benchmark start")
        timings: list[float] = []
        results: list[dict[str, Any]] = []
        for _ in range(rounds):
            barrier.wait(timeout=600)
            started = time.perf_counter()
            results.append(client.call(_TOOL_NAME, image=image, query=query_text))
            timings.append(time.perf_counter() - started)
        result_queue.put(
            {
                "client": client_index,
                "pid": os.getpid(),
                "timings_s": timings,
                "results": results,
                "error": None,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "client": client_index,
                "pid": os.getpid(),
                "timings_s": [],
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        if client is not None:
            client.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serialized", "batched"), required=True)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--query", default="red square")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.clients < 1 or args.rounds < 1 or args.image_size < 128:
        raise SystemExit("clients/rounds must be positive and image-size must be >= 128")

    snapshot = _cached_snapshot()
    bundle, registry = _load_bundle_registry()
    batching = registry.batching(_TOOL_NAME)
    if batching is None:
        raise RuntimeError(f"{_TOOL_NAME} did not register a batch handler")

    batch_sizes: list[int] = []
    batch_sizes_lock = threading.Lock()
    if args.mode == "serialized":
        registry._batching.pop(_TOOL_NAME)  # POC: exercise scalar fallback
    else:
        original_handler = batching.handler

        def _recording_handler(requests: list[dict[str, Any]]) -> list[Any]:
            with batch_sizes_lock:
                batch_sizes.append(len(requests))
            return original_handler(requests)

        registry._batching[_TOOL_NAME] = dataclasses.replace(batching, handler=_recording_handler)

    from gap_core.rpc.shared import SharedToolServer

    context = mp.get_context("spawn")
    start_event = context.Event()
    barrier = context.Barrier(args.clients)
    ready_queue = context.Queue()
    result_queue = context.Queue()

    with tempfile.TemporaryDirectory(prefix="gap-dino-batch-") as temp_dir:
        socket_path = Path(temp_dir) / "tools.sock"
        server = SharedToolServer(
            registry, socket_path, mode=args.mode, bundle_name="grounding-dino"
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        deadline = time.monotonic() + args.timeout_s
        while not socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not socket_path.exists():
            raise TimeoutError("shared tool server did not create its socket")

        processes = [
            context.Process(
                target=_client_worker,
                args=(
                    index,
                    str(socket_path),
                    args.rounds,
                    args.image_size,
                    args.query,
                    start_event,
                    barrier,
                    ready_queue,
                    result_queue,
                ),
            )
            for index in range(args.clients)
        ]
        try:
            # Spawn before the parent imports/initializes torch CUDA.
            for process in processes:
                process.start()
            for _ in processes:
                remaining = max(0.01, deadline - time.monotonic())
                ready_queue.get(timeout=remaining)

            import torch

            scalar_references = [
                registry.invoke(
                    _TOOL_NAME,
                    None,
                    image=_fixed_image(index, args.image_size),
                    query=args.query,
                )
                for index in range(args.clients)
            ]
            model, processor = bundle._get_model()
            counting_model = _CountingModel(model)
            bundle._model = counting_model
            bundle._processor = processor
            torch.cuda.synchronize()
            model_memory = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            start_event.set()

            child_results = []
            for _ in processes:
                remaining = max(0.01, deadline - time.monotonic())
                child_results.append(result_queue.get(timeout=remaining))
            for process in processes:
                remaining = max(0.01, deadline - time.monotonic())
                process.join(timeout=remaining)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            gpu_memory = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        except (queue.Empty, threading.BrokenBarrierError) as exc:
            raise TimeoutError(f"benchmark did not complete before {args.timeout_s}s") from exc
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            server.close()
            server_thread.join(timeout=5)

    child_results.sort(key=lambda item: item["client"])
    child_errors = [item for item in child_results if item["error"]]
    equivalence_errors: list[str] = []
    timings = []
    for child in child_results:
        timings.extend(child["timings_s"])
        if child["error"]:
            continue
        expected = scalar_references[child["client"]]
        for round_index, actual in enumerate(child["results"]):
            for error in _equivalence_errors(expected, actual, rtol=args.rtol, atol=args.atol):
                equivalence_errors.append(f"client {child['client']} round {round_index}: {error}")

    request_count = args.clients * args.rounds
    report = {
        "mode": args.mode,
        "clients": args.clients,
        "rounds": args.rounds,
        "request_count": request_count,
        "elapsed_s": elapsed,
        "requests_per_s": request_count / elapsed,
        "latency_s": {
            "mean": statistics.fmean(timings) if timings else 0.0,
            "p50": _percentile(timings, 0.50),
            "p95": _percentile(timings, 0.95),
        },
        "model_forward_count": counting_model.calls,
        "batch_sizes": batch_sizes if args.mode == "batched" else [1] * request_count,
        "model_memory": model_memory,
        "measured_gpu_memory": gpu_memory,
        "equivalent": not equivalence_errors and not child_errors,
        "equivalence_errors": equivalence_errors,
        "child_errors": child_errors,
        "client_pids": [item["pid"] for item in child_results],
        "model": _MODEL_ID,
        "cached_revision": snapshot,
        "offline": True,
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    expected_forwards = request_count if args.mode == "serialized" else None
    forward_ok = expected_forwards is None or counting_model.calls == expected_forwards
    batch_ok = args.mode == "serialized" or (
        sum(batch_sizes) == request_count and any(size > 1 for size in batch_sizes)
    )
    return 0 if report["equivalent"] and forward_ok and batch_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
