from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark_module():
    path = Path(__file__).parents[1] / "tools" / "grounding-dino" / "benchmark_shared_rpc.py"
    spec = importlib.util.spec_from_file_location("poc_dino_batch_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_images_are_deterministic_and_client_specific(benchmark_module):
    first = benchmark_module._fixed_image(0, 192)
    assert (first == benchmark_module._fixed_image(0, 192)).all()
    assert not (first == benchmark_module._fixed_image(1, 192)).all()


def test_percentile_interpolates(benchmark_module):
    assert benchmark_module._percentile([], 0.5) == 0.0
    assert benchmark_module._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert benchmark_module._percentile([1.0, 3.0], 0.5) == 2.0


def test_equivalence_reports_semantic_and_numeric_drift(benchmark_module):
    expected = {
        "detections": [
            {
                "label": "object",
                "score": 0.9,
                "box": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
            }
        ]
    }
    close = {
        "detections": [
            {
                "label": "object",
                "score": 0.9001,
                "box": {"x1": 1.0001, "y1": 2.0, "x2": 3.0, "y2": 4.0},
            }
        ]
    }
    drifted = {
        "detections": [
            {
                "label": "other",
                "score": 0.5,
                "box": {"x1": 10.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
            }
        ]
    }
    assert not benchmark_module._equivalence_errors(expected, close, rtol=1e-3, atol=1e-3)
    errors = benchmark_module._equivalence_errors(expected, drifted, rtol=1e-3, atol=1e-3)
    assert any("labels" in error for error in errors)
    assert any("box" in error for error in errors)
    assert any("score" in error for error in errors)
