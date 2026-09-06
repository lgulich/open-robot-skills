"""SAM3 text-segmentation batch declaration and deferred live equivalence gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _CountingSam3Model:
    def __init__(self, model):
        self.model = model
        self.grounding_forwards = 0

    def forward_grounding(self, *args, **kwargs):
        self.grounding_forwards += 1
        return self.model.forward_grounding(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


@pytest.fixture(scope="module")
def sam3_bundle():
    from gap_core.tools import ToolRegistry

    module_name = "poc_sam3_tools"
    tools_path = Path(__file__).parents[1] / "tools" / "sam3" / "tools.py"
    spec = importlib.util.spec_from_file_location(module_name, tools_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    registry = ToolRegistry()
    registry.discover_pending()
    return module, registry


def test_segment_text_batch_declaration(sam3_bundle):
    _module, registry = sam3_bundle
    batching = registry.batching("sam3.segment_text")
    assert batching is not None
    assert batching.max_batch_size == 2
    assert batching.max_wait_ms == 10


def test_segment_text_batch_uses_inference_mode_for_entire_impl(
    sam3_bundle,
    monkeypatch,
):
    module, _registry = sam3_bundle
    active = False

    class _InferenceMode:
        def __enter__(self):
            nonlocal active
            active = True

        def __exit__(self, *_args):
            nonlocal active
            active = False

    sentinel = [{"masks": [], "scores": [], "boxes": []}]

    def _impl(_prepared):
        assert active
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=lambda: _InferenceMode()),
    )
    monkeypatch.setattr(module, "_run_segment_text_batch_impl", _impl)
    assert module._run_segment_text_batch([]) is sentinel
    assert active is False


def test_segment_text_batch_preserves_order_and_isolates_bad_image(
    sam3_bundle,
    monkeypatch,
):
    module, registry = sam3_bundle
    calls = []

    def _fake_batch(prepared):
        calls.append(prepared)
        return [
            {"masks": [], "scores": [float(index)], "boxes": []}
            for index, _item in enumerate(prepared)
        ]

    monkeypatch.setattr(module, "_run_segment_text_batch", _fake_batch)
    results = registry.invoke_batch(
        "sam3.segment_text",
        [
            {"image": np.zeros((8, 10, 3), dtype=np.uint8), "query": "first"},
            {"image": np.zeros((8, 10), dtype=np.uint8), "query": "bad"},
            {
                "image": np.zeros((12, 14, 3), dtype=np.uint8),
                "query": "third",
                "max_results": 2,
            },
        ],
    )

    assert len(calls) == 1
    assert [item[1:] for item in calls[0]] == [("first", 5), ("third", 2)]
    assert results[0]["scores"] == [0.0]
    assert isinstance(results[1], BaseException)
    assert results[2]["scores"] == [1.0]


@pytest.mark.gpu
def test_segment_text_gpu_batch_matches_paired_scalar_results(
    sam3_bundle,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    module, registry = sam3_bundle
    model, _processor = module._get_model()
    images = []
    for offset in (140, 420):
        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        image[220:520, offset : offset + 320] = (220, 30, 30)
        images.append(image)
    requests = [{"image": image, "query": "red rectangle"} for image in images]
    scalar = [registry.invoke("sam3.segment_text", None, **request) for request in requests]

    counting_model = _CountingSam3Model(model)
    monkeypatch.setattr(module, "_image_model", counting_model)
    batched = registry.invoke_batch("sam3.segment_text", requests)

    assert counting_model.grounding_forwards == 1
    for scalar_result, batch_result in zip(scalar, batched, strict=True):
        assert not isinstance(batch_result, BaseException)
        assert len(batch_result["masks"]) == len(scalar_result["masks"])
        # BF16 kernels are batch-shape dependent. The live RTX 6000 Ada gate
        # found identical result counts/order, <=0.005 score drift, <=8 px box
        # drift, and >=0.98 mask IoU across all five returned candidates.
        np.testing.assert_allclose(
            batch_result["scores"], scalar_result["scores"], rtol=1e-3, atol=5e-3
        )
        np.testing.assert_allclose(
            [[box[key] for key in ("x1", "y1", "x2", "y2")] for box in batch_result["boxes"]],
            [[box[key] for key in ("x1", "y1", "x2", "y2")] for box in scalar_result["boxes"]],
            rtol=1e-3,
            atol=8.0,
        )
        for scalar_mask, batch_mask in zip(
            scalar_result["masks"], batch_result["masks"], strict=True
        ):
            intersection = np.logical_and(scalar_mask > 0, batch_mask > 0).sum()
            union = np.logical_or(scalar_mask > 0, batch_mask > 0).sum()
            assert intersection / max(1, union) >= 0.98

        # The top candidate is what the syringe graph consumes; pin the much
        # tighter observed equivalence separately from low-confidence tails.
        assert batch_result["scores"][0] == pytest.approx(scalar_result["scores"][0], abs=5e-3)
        scalar_box = scalar_result["boxes"][0]
        batch_box = batch_result["boxes"][0]
        assert (
            max(abs(scalar_box[key] - batch_box[key]) for key in ("x1", "y1", "x2", "y2")) <= 0.25
        )
        scalar_mask = scalar_result["masks"][0]
        batch_mask = batch_result["masks"][0]
        assert np.mean(scalar_mask == batch_mask) >= 0.999
