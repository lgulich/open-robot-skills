"""grounding-dino tool bundle — zero-shot object detection.

Extracted from the original Grounding DINO gRPC servicer in the dev
tree. The transformers pipeline
(processor → model → post_process_grounded_object_detection) is verbatim;
the proto byte decode/encode is replaced by numpy arrays + gap.types dicts.

The model loads lazily on first call (module-level singleton); importing
this module never pulls torch/transformers. Knobs via env:

- ``GAP_DINO_DEVICE`` — torch device (default ``cuda``).
- ``GAP_DINO_MODEL``  — HuggingFace model name
  (default ``IDEA-Research/grounding-dino-base``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool, tool_batch
from gap_core.types import BoundingBox2D

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "IDEA-Research/grounding-dino-base"
_DEFAULT_BOX_THRESHOLD = 0.20
_DEFAULT_TEXT_THRESHOLD = 0.20

_DEVICE = os.environ.get("GAP_DINO_DEVICE", "cuda")
_MODEL_NAME = os.environ.get("GAP_DINO_MODEL", _DEFAULT_MODEL_NAME)

_load_lock = threading.Lock()
_model: Any = None
_processor: Any = None


class Detection(TypedDict):
    box: BoundingBox2D  # [x1, y1, x2, y2] in pixels
    label: str  # matched text label
    score: float  # confidence [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


def _effective_threshold(value: float, default: float) -> float:
    return value if value > 0 else default


def _batch_compatibility_key(request: dict[str, Any]) -> tuple[float, float]:
    return (
        _effective_threshold(
            float(request.get("box_threshold", _DEFAULT_BOX_THRESHOLD)),
            _DEFAULT_BOX_THRESHOLD,
        ),
        _effective_threshold(
            float(request.get("text_threshold", _DEFAULT_TEXT_THRESHOLD)),
            _DEFAULT_TEXT_THRESHOLD,
        ),
    )


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check``.

    Checks the Hugging Face cache for the configured model's config.json
    (the canonical presence marker) — never downloads, never imports
    torch. ``None`` when huggingface_hub is unavailable ("unknown").
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        result = try_to_load_from_cache(_MODEL_NAME, "config.json")
    except Exception:
        return None
    return isinstance(result, str)


def prefetch() -> None:
    """Snapshot-download the configured GDINO weights into the HF cache.

    Called by ``gap skills check --download``. Idempotent: re-running
    against an already-cached snapshot is a near-no-op (HF revision
    check + symlink refresh). Logs the bytes added so the user sees
    progress; raises on network / auth / disk errors so ``gap skills
    check --download`` exits non-zero.

    Uses ``snapshot_download`` rather than ``AutoModel.from_pretrained``
    so we never load torch / instantiate the model at prefetch time —
    important for CI lanes and for the bare-engine venv.
    """
    from huggingface_hub import snapshot_download

    logger.info("[grounding-dino] prefetching weights for %s ...", _MODEL_NAME)
    snapshot_download(repo_id=_MODEL_NAME, repo_type="model")
    logger.info("[grounding-dino] prefetch complete (cached at HF default)")


def _get_model() -> tuple[Any, Any]:
    """Load the Grounding DINO model + processor once (lazy singleton)."""
    global _model, _processor
    with _load_lock:
        if _model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info("Loading Grounding DINO model: %s on %s ...", _MODEL_NAME, _DEVICE)
            _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_NAME)
            model = model.to(_DEVICE)
            model.eval()
            _model = model
            logger.info("Grounding DINO model loaded successfully on %s.", _DEVICE)
        return _model, _processor


@tool(
    name="grounding-dino.detect",
    summary="Zero-shot object detection from a text prompt; returns labeled boxes with confidence scores.",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    query: str,
    box_threshold: float = _DEFAULT_BOX_THRESHOLD,
    text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
) -> DetectResult:
    """Detect objects matching ``query`` in an RGB uint8 [H, W, 3] image.

    Grounding DINO expects period-separated phrases (``"red cube. green
    cube."``); a missing trailing period is appended automatically. Returns
    an empty detections list when nothing clears the thresholds — select the
    best box downstream (e.g. closest to a pointing-model pixel) and feed it
    to ``sam3.segment_box`` for a pixel-accurate mask.
    """
    prepared = _prepare_request(image, query)
    box_threshold = _effective_threshold(box_threshold, _DEFAULT_BOX_THRESHOLD)
    text_threshold = _effective_threshold(text_threshold, _DEFAULT_TEXT_THRESHOLD)
    results = _infer_prepared([prepared], box_threshold, text_threshold)[0]
    return _format_result(results)


def _prepare_request(image: np.ndarray, query: str) -> tuple[Any, str]:
    from PIL import Image

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ToolError(
            "grounding-dino.detect",
            f"expected RGB uint8 [H, W, 3] image, got shape {arr.shape}",
        )
    prompt = query if query.endswith(".") else query + "."
    return Image.fromarray(arr, "RGB"), prompt


def _infer_prepared(
    prepared: list[tuple[Any, str]],
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    import torch

    model, processor = _get_model()
    images = [item[0] for item in prepared]
    prompts = [item[1] for item in prepared]
    try:
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(
            _DEVICE
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1] for image in images],
        )
    except Exception as e:
        raise PerceptionFailed(f"Grounding DINO detection failed: {e}") from e


def _format_result(results: dict[str, Any]) -> DetectResult:
    detections: list[Detection] = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    for box, score, label in zip(boxes, scores, labels, strict=False):
        detections.append(
            {
                "box": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                },
                "label": str(label),
                "score": float(score),
            }
        )

    logger.info("grounding-dino.detect returning %d detections.", len(detections))
    return {"detections": detections}


@tool_batch(
    name="grounding-dino.detect",
    max_batch_size=8,
    max_wait_ms=10,
    compatibility_key=_batch_compatibility_key,
)
def detect_batch(requests: list[dict[str, Any]]) -> list[DetectResult | BaseException]:
    """Private serving handler; public callers continue to use ``detect``."""
    results: list[DetectResult | BaseException | None] = [None] * len(requests)
    prepared: list[tuple[Any, str]] = []
    indices: list[int] = []
    for index, request in enumerate(requests):
        try:
            prepared.append(_prepare_request(request["image"], str(request["query"])))
            indices.append(index)
        except BaseException as exc:
            results[index] = exc

    if prepared:
        box_threshold, text_threshold = _batch_compatibility_key(requests[indices[0]])
        try:
            raw_results = _infer_prepared(prepared, box_threshold, text_threshold)
            for index, raw_result in zip(indices, raw_results, strict=True):
                results[index] = _format_result(raw_result)
        except BaseException as exc:
            for index in indices:
                results[index] = exc

    return [
        result if result is not None else RuntimeError("missing batch result") for result in results
    ]
