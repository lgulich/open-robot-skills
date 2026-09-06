#!/usr/bin/env python3
"""Offline 1280x720 shared-RPC benchmark for scalar versus batched SAM3."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
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

_TOOL_NAME = "sam3.segment_text"
_MODEL_ID = "facebook/sam3"
_SCORE_ATOL = 5e-3
_BOX_ATOL_PX = 8.0
_MASK_IOU_MIN = 0.98
_TOP1_BOX_ATOL_PX = 0.25
_TOP1_PIXEL_AGREEMENT_MIN = 0.999


class _CountingModel:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.grounding_forwards = 0

    def forward_grounding(self, *args: Any, **kwargs: Any) -> Any:
        self.grounding_forwards += 1
        return self.model.forward_grounding(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def _fixed_image(index: int) -> np.ndarray:
    image = np.full((720, 1280, 3), 220, dtype=np.uint8)
    offset = 100 + (index % 4) * 180
    image[220:520, offset : offset + 260] = (190, 25, 25)
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


def _load_image(path: Path | None, index: int) -> np.ndarray:
    if path is None:
        return _fixed_image(index)
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((1280, 720)), dtype=np.uint8)


def _load_bundle_registry() -> tuple[Any, Any]:
    from gap_core.tools import ToolRegistry

    tools_path = Path(__file__).with_name("tools.py")
    module_name = "gap_batch_poc_sam3_tools"
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

    config = try_to_load_from_cache(_MODEL_ID, "config.json")
    if not isinstance(config, str):
        raise RuntimeError(f"offline SAM3 cache is incomplete for {_MODEL_ID}")
    return Path(config).parent.name


def _equivalence_errors(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("masks", "scores", "boxes"):
        if len(expected[key]) != len(actual[key]):
            return [f"{key} count: expected {len(expected[key])}, got {len(actual[key])}"]
    if not np.allclose(expected["scores"], actual["scores"], rtol=1e-3, atol=_SCORE_ATOL):
        errors.append(f"scores differ beyond rtol=1e-3, atol={_SCORE_ATOL}")
    box_keys = ("x1", "y1", "x2", "y2")
    expected_boxes = [[box[key] for key in box_keys] for box in expected["boxes"]]
    actual_boxes = [[box[key] for key in box_keys] for box in actual["boxes"]]
    if not np.allclose(expected_boxes, actual_boxes, rtol=1e-3, atol=_BOX_ATOL_PX):
        errors.append(f"boxes differ beyond rtol=1e-3, atol={_BOX_ATOL_PX} px")
    for index, (expected_mask, actual_mask) in enumerate(
        zip(expected["masks"], actual["masks"], strict=True)
    ):
        intersection = np.logical_and(expected_mask > 0, actual_mask > 0).sum()
        union = np.logical_or(expected_mask > 0, actual_mask > 0).sum()
        iou = float(intersection / max(1, union))
        if iou < _MASK_IOU_MIN:
            errors.append(f"mask {index} IoU is {iou:.6f}")
    if expected["scores"]:
        top1_box_delta = max(
            abs(expected["boxes"][0][key] - actual["boxes"][0][key]) for key in box_keys
        )
        if top1_box_delta > _TOP1_BOX_ATOL_PX:
            errors.append(f"top-1 box max delta is {top1_box_delta:.6f} px")
        top1_agreement = float(np.mean(expected["masks"][0] == actual["masks"][0]))
        if top1_agreement < _TOP1_PIXEL_AGREEMENT_MIN:
            errors.append(f"top-1 mask pixel agreement is {top1_agreement:.6f}")
    return errors


def _run_phase(
    *,
    mode: str,
    client_count: int,
    rounds: int,
    requests: list[dict[str, Any]],
    bundle: Any,
    registry: Any,
    timeout_s: float,
) -> dict[str, Any]:
    import torch
    from gap_core.rpc.shared import SharedToolClient, SharedToolServer

    model, processor = bundle._get_model()
    if isinstance(model, _CountingModel):
        model = model.model
    counting_model = _CountingModel(model)
    bundle._image_model = counting_model
    processor.model = counting_model

    batch_sizes: list[int] = []
    batching = registry.batching(_TOOL_NAME)
    if mode == "batched":
        original_handler = batching.handler

        def _recording_handler(items: list[dict[str, Any]]) -> list[Any]:
            batch_sizes.append(len(items))
            return original_handler(items)

        registry._batching[_TOOL_NAME] = dataclasses.replace(batching, handler=_recording_handler)

    results: list[list[dict[str, Any]] | None] = [None] * client_count
    timings: list[list[float] | None] = [None] * client_count
    errors: list[str] = []
    barrier = threading.Barrier(client_count)

    with tempfile.TemporaryDirectory(prefix=f"gap-sam3-{mode}-") as temp_dir:
        socket_path = Path(temp_dir) / "sam3.sock"
        server = SharedToolServer(
            registry,
            socket_path,
            mode=mode,
            max_batch_size=client_count,
            bundle_name="sam3",
        )
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        deadline = time.monotonic() + timeout_s
        while not socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not socket_path.exists():
            raise TimeoutError("shared SAM3 server did not create its socket")

        def _client(index: int) -> None:
            client = None
            try:
                client = SharedToolClient(
                    "sam3",
                    socket_path,
                    call_timeout_s=timeout_s,
                    session_id=f"benchmark-env-{index}",
                    episode_id=f"{mode}-0",
                )
                barrier.wait(timeout=timeout_s)
                local_results = []
                local_timings = []
                for _ in range(rounds):
                    barrier.wait(timeout=timeout_s)
                    started = time.perf_counter()
                    local_results.append(client.call(_TOOL_NAME, **requests[index]))
                    local_timings.append(time.perf_counter() - started)
                results[index] = local_results
                timings[index] = local_timings
            except BaseException as exc:
                errors.append(f"client {index}: {type(exc).__name__}: {exc}")
                try:
                    barrier.abort()
                except BaseException:
                    pass
            finally:
                if client is not None:
                    client.close()

        threads = [
            threading.Thread(target=_client, args=(index,), daemon=True)
            for index in range(client_count)
        ]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics_client = SharedToolClient(
            "sam3",
            socket_path,
            session_id="benchmark-metrics",
            episode_id=f"{mode}-0",
        )
        metrics = metrics_client.metrics()
        metrics_client.close()
        memory = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        server.close()
        server_thread.join(timeout=5)

    processor.model = model
    bundle._image_model = model
    if mode == "batched":
        registry._batching[_TOOL_NAME] = batching
    flat_timings = [value for per_client in timings if per_client for value in per_client]
    return {
        "mode": mode,
        "elapsed_s": elapsed,
        "requests_per_s": client_count * rounds / elapsed,
        "latency_s": {
            "mean": statistics.fmean(flat_timings) if flat_timings else 0.0,
            "p50": _percentile(flat_timings, 0.50),
            "p95": _percentile(flat_timings, 0.95),
        },
        "grounding_forward_count": counting_model.grounding_forwards,
        "batch_sizes": batch_sizes if mode == "batched" else [1] * (client_count * rounds),
        "memory": memory,
        "metrics": metrics,
        "results": results,
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--query", default="red object")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",")]
    if not batch_sizes or any(size < 1 or size > 8 for size in batch_sizes):
        raise SystemExit("batch sizes must be between 1 and 8")
    if args.rounds < 1:
        raise SystemExit("rounds must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 benchmark requires CUDA")
    snapshot = _cached_snapshot()
    bundle, registry = _load_bundle_registry()
    cases = []
    stopped_before_oom = False
    for client_count in batch_sizes:
        if cases:
            previous = cases[-1]["batched"]["memory"]
            previous_size = cases[-1]["batch_size"]
            _free_bytes, total_bytes = torch.cuda.mem_get_info()
            projected_peak = int(previous["peak_allocated_bytes"] * client_count / previous_size)
            if projected_peak > int(total_bytes * 0.90):
                cases.append(
                    {
                        "batch_size": client_count,
                        "skipped": "projected peak exceeds 90 percent of device memory",
                        "projected_peak_allocated_bytes": projected_peak,
                    }
                )
                stopped_before_oom = True
                break
        requests = [
            {"image": _load_image(args.image, index), "query": args.query}
            for index in range(client_count)
        ]
        serialized = _run_phase(
            mode="serialized",
            client_count=client_count,
            rounds=args.rounds,
            requests=requests,
            bundle=bundle,
            registry=registry,
            timeout_s=args.timeout_s,
        )
        batched = _run_phase(
            mode="batched",
            client_count=client_count,
            rounds=args.rounds,
            requests=requests,
            bundle=bundle,
            registry=registry,
            timeout_s=args.timeout_s,
        )
        equivalence_errors = []
        for client_index in range(client_count):
            if (
                serialized["results"][client_index] is None
                or batched["results"][client_index] is None
            ):
                continue
            for round_index, (expected, actual) in enumerate(
                zip(
                    serialized["results"][client_index],
                    batched["results"][client_index],
                    strict=True,
                )
            ):
                for error in _equivalence_errors(expected, actual):
                    equivalence_errors.append(f"client {client_index} round {round_index}: {error}")
        case = {
            "batch_size": client_count,
            "serialized": {key: value for key, value in serialized.items() if key != "results"},
            "batched": {key: value for key, value in batched.items() if key != "results"},
            "equivalent": not equivalence_errors
            and not serialized["errors"]
            and not batched["errors"],
            "equivalence_errors": equivalence_errors,
        }
        cases.append(case)
        if any(
            "out of memory" in error.lower() for error in serialized["errors"] + batched["errors"]
        ):
            stopped_before_oom = True
            break
        torch.cuda.empty_cache()

    report = {
        "model": _MODEL_ID,
        "cached_revision": snapshot,
        "resolution": [1280, 720],
        "rounds": args.rounds,
        "cases": cases,
        "stopped_before_or_after_oom_signal": stopped_before_oom,
        "offline": True,
        "equivalence_tolerance": {
            "score_atol": _SCORE_ATOL,
            "box_atol_px": _BOX_ATOL_PX,
            "mask_iou_min": _MASK_IOU_MIN,
            "top1_box_atol_px": _TOP1_BOX_ATOL_PX,
            "top1_pixel_agreement_min": _TOP1_PIXEL_AGREEMENT_MIN,
        },
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    completed = [case for case in cases if "equivalent" in case]
    return 0 if completed and all(case["equivalent"] for case in completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
