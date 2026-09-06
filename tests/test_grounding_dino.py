"""grounding-dino bundle: signature/schema units + GPU smoke."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


class _Inputs(dict):
    input_ids = [[1], [2]]

    def to(self, _device):
        return self


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, **_inputs):
        self.calls += 1
        return object()


class _CountingModel:
    """Transparent model proxy used only by the live GPU equivalence gate."""

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __call__(self, **inputs):
        self.calls += 1
        return self.model(**inputs)

    def __getattr__(self, name):
        return getattr(self.model, name)


class _Processor:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, *, images, text, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        assert len(images) == len(text)
        self.batch_sizes.append(len(images))
        inputs = _Inputs()
        inputs.input_ids = [[index] for index in range(len(images))]
        return inputs

    def post_process_grounded_object_detection(self, _outputs, _ids, **kwargs):
        return [
            {
                "boxes": _Array([[index, index + 1, index + 2, index + 3]]),
                "scores": _Array([0.9 - index * 0.1]),
                "labels": [f"object-{index}"],
            }
            for index, _ in enumerate(kwargs["target_sizes"])
        ]


@pytest.fixture(scope="module")
def dino_registry():
    """Load only this bundle so the batch POC has no optional-bundle deps."""
    module_name = "poc_grounding_dino_tools"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).parents[1] / "tools" / "grounding-dino" / "tools.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from gap_core.tools import ToolRegistry

    registry = ToolRegistry()
    registry.discover_pending()
    return registry


def test_detect_registered(dino_registry):
    assert "grounding-dino.detect" in dino_registry
    desc = dino_registry.get("grounding-dino.detect")
    assert desc.tags == ("perception",)


def test_detect_schema(dino_registry):
    schema = dino_registry.get("grounding-dino.detect").schema
    assert set(schema.inputs) == {"image", "query", "box_threshold", "text_threshold"}
    assert schema.inputs["image"].required
    assert schema.inputs["query"].required
    assert schema.inputs["box_threshold"].default == pytest.approx(0.20)
    assert schema.inputs["text_threshold"].default == pytest.approx(0.20)
    assert set(schema.outputs) == {"detections"}


def test_public_scalar_detect_path_still_works(dino_registry, monkeypatch):
    batching = dino_registry.batching("grounding-dino.detect")
    model = _Model()
    processor = _Processor()
    monkeypatch.setitem(batching.handler.__globals__, "_model", model)
    monkeypatch.setitem(batching.handler.__globals__, "_processor", processor)

    result = dino_registry.invoke(
        "grounding-dino.detect",
        None,
        image=np.zeros((8, 10, 3), dtype=np.uint8),
        query="object",
    )

    assert result["detections"][0]["label"] == "object-0"
    assert model.calls == 1
    assert processor.batch_sizes == [1]


def test_detect_private_batch_handler_uses_one_model_forward(dino_registry, monkeypatch):
    batching = dino_registry.batching("grounding-dino.detect")
    assert batching is not None
    model = _Model()
    processor = _Processor()
    monkeypatch.setitem(batching.handler.__globals__, "_model", model)
    monkeypatch.setitem(batching.handler.__globals__, "_processor", processor)
    images = [np.zeros((8, 10, 3), dtype=np.uint8) for _ in range(2)]

    results = dino_registry.invoke_batch(
        "grounding-dino.detect",
        [
            {"image": images[0], "query": "first"},
            {"image": images[1], "query": "second"},
        ],
    )

    assert model.calls == 1
    assert processor.batch_sizes == [2]
    assert [result["detections"][0]["label"] for result in results] == ["object-0", "object-1"]
    assert batching.compatibility_key({"image": images[0], "query": "x"}) == (0.2, 0.2)


def test_detect_batch_isolates_invalid_request(dino_registry, monkeypatch):
    batching = dino_registry.batching("grounding-dino.detect")
    model = _Model()
    processor = _Processor()
    monkeypatch.setitem(batching.handler.__globals__, "_model", model)
    monkeypatch.setitem(batching.handler.__globals__, "_processor", processor)

    results = dino_registry.invoke_batch(
        "grounding-dino.detect",
        [
            {"image": np.zeros((8, 10), dtype=np.uint8), "query": "bad"},
            {"image": np.zeros((8, 10, 3), dtype=np.uint8), "query": "good"},
        ],
    )

    assert isinstance(results[0], BaseException)
    assert results[1]["detections"][0]["label"] == "object-0"
    assert model.calls == 1


@pytest.mark.gpu
def test_detect_gpu_smoke(dino_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    img[40:120, 40:120] = (200, 30, 30)
    out = dino_registry.invoke("grounding-dino.detect", None, image=img, query="red square")
    assert set(out) == {"detections"}
    for det in out["detections"]:
        assert set(det) == {"box", "label", "score"}
        assert 0.0 <= det["score"] <= 1.0
        box = det["box"]
        assert box["x2"] >= box["x1"] and box["y2"] >= box["y1"]


@pytest.mark.gpu
def test_detect_gpu_batch_matches_paired_scalar_results(dino_registry, monkeypatch):
    """Pending live gate: batching changes throughput, not scalar semantics."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    batching = dino_registry.batching("grounding-dino.detect")
    module_globals = batching.handler.__globals__
    get_model = module_globals["_get_model"]
    model, processor = get_model()
    images = []
    for offset in (32, 56):
        image = np.full((192, 192, 3), 255, dtype=np.uint8)
        image[offset : offset + 72, 48:144] = (200, 30, 30)
        images.append(image)
    requests = [{"image": image, "query": "red square"} for image in images]

    scalar_results = [
        dino_registry.invoke("grounding-dino.detect", None, **request) for request in requests
    ]
    counting_model = _CountingModel(model)
    monkeypatch.setitem(module_globals, "_model", counting_model)
    monkeypatch.setitem(module_globals, "_processor", processor)
    batch_results = dino_registry.invoke_batch("grounding-dino.detect", requests)

    assert counting_model.calls == 1
    for scalar, batched in zip(scalar_results, batch_results, strict=True):
        assert not isinstance(batched, BaseException)
        scalar_detections = scalar["detections"]
        batch_detections = batched["detections"]
        assert [item["label"] for item in batch_detections] == [
            item["label"] for item in scalar_detections
        ]
        assert len(batch_detections) == len(scalar_detections)
        for scalar_detection, batch_detection in zip(
            scalar_detections, batch_detections, strict=True
        ):
            np.testing.assert_allclose(
                [scalar_detection["box"][key] for key in ("x1", "y1", "x2", "y2")],
                [batch_detection["box"][key] for key in ("x1", "y1", "x2", "y2")],
                rtol=1e-3,
                atol=1e-3,
            )
            assert batch_detection["score"] == pytest.approx(
                scalar_detection["score"], rel=1e-3, abs=1e-3
            )
