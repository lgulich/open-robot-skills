"""CPU-only regression tests for embodiment-aware cuRobo TCP conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_MODULE_PATH = Path(__file__).parents[1] / "tools" / "curobo" / "tcp_frames.py"
_SPEC = importlib.util.spec_from_file_location("curobo_tcp_frames", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
tcp_frames = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tcp_frames)


def _wxyz(rotation: Rotation) -> np.ndarray:
    return np.roll(rotation.as_quat(), 1)


def test_unset_environment_preserves_panda_conversion(monkeypatch):
    monkeypatch.delenv("GAP_HAND_TO_FINGERTIP_Z", raising=False)
    monkeypatch.delenv("GAP_TCP_ROTATION_Z", raising=False)
    position = np.array([0.3, -0.2, 0.7])
    quat = _wxyz(Rotation.from_euler("xyz", [0.2, -0.3, 0.4]))

    hand_position, hand_quat = tcp_frames.fingertip_pose_to_hand_pose(position, quat)

    rotation = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
    expected = position - rotation @ np.array([0.0, 0.0, 0.1029])
    np.testing.assert_allclose(hand_position, expected, atol=1e-12)
    np.testing.assert_array_equal(hand_quat, quat)


def test_robotiq_environment_converts_length_and_mount_yaw(monkeypatch):
    monkeypatch.setenv("GAP_HAND_TO_FINGERTIP_Z", "0.157")
    monkeypatch.setenv("GAP_TCP_ROTATION_Z", str(np.pi / 4.0))
    position = np.array([0.4, 0.1, 0.5])
    tcp_rotation = Rotation.from_euler("xyz", [0.1, np.pi, -0.2])
    quat = _wxyz(tcp_rotation)

    hand_position, hand_quat = tcp_frames.fingertip_pose_to_hand_pose(position, quat)

    expected_rotation = tcp_rotation * Rotation.from_euler("z", -np.pi / 4.0)
    expected_position = position - expected_rotation.as_matrix() @ np.array([0.0, 0.0, 0.157])
    np.testing.assert_allclose(hand_position, expected_position, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(np.roll(hand_quat, -1)).as_matrix(),
        expected_rotation.as_matrix(),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("GAP_HAND_TO_FINGERTIP_Z", "nan"),
        ("GAP_HAND_TO_FINGERTIP_Z", "0"),
        ("GAP_TCP_ROTATION_Z", "inf"),
    ],
)
def test_invalid_environment_fails_closed(monkeypatch, name, value):
    monkeypatch.delenv("GAP_HAND_TO_FINGERTIP_Z", raising=False)
    monkeypatch.delenv("GAP_TCP_ROTATION_Z", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        tcp_frames.fingertip_pose_to_hand_pose([0, 0, 0], [1, 0, 0, 0])
