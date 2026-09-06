from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_benchmark():
    path = Path(__file__).parents[1] / "tools" / "sam3" / "benchmark_shared_rpc.py"
    spec = importlib.util.spec_from_file_location("sam3_batch_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_image_is_representative_resolution():
    benchmark = _load_benchmark()
    image = benchmark._fixed_image(2)
    assert image.shape == (720, 1280, 3)
    assert image.dtype == np.uint8


def test_equivalence_reports_score_and_mask_drift():
    benchmark = _load_benchmark()
    expected = {
        "masks": [np.zeros((2, 2), dtype=np.uint8)],
        "scores": [0.9],
        "boxes": [{"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}],
    }
    actual = {
        "masks": [np.ones((2, 2), dtype=np.uint8)],
        "scores": [0.8],
        "boxes": list(expected["boxes"]),
    }
    errors = benchmark._equivalence_errors(expected, actual)
    assert any("scores differ" in error for error in errors)
    assert any("IoU" in error for error in errors)


def test_equivalence_accepts_observed_bfloat16_batch_drift():
    benchmark = _load_benchmark()
    expected_mask = np.zeros((100, 100), dtype=np.uint8)
    expected_mask[10:90, 10:90] = 255
    actual_mask = expected_mask.copy()
    actual_mask[10, 10] = 0
    expected = {
        "masks": [expected_mask],
        "scores": [0.9],
        "boxes": [{"x1": 10.0, "y1": 10.0, "x2": 90.0, "y2": 90.0}],
    }
    actual = {
        "masks": [actual_mask],
        "scores": [0.898],
        "boxes": [{"x1": 10.1, "y1": 10.1, "x2": 90.1, "y2": 90.1}],
    }
    assert benchmark._equivalence_errors(expected, actual) == []
