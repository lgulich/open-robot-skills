"""Embodiment-aware fingertip-to-hand frame conversion for cuRobo grasps."""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation

PANDA_HAND_TO_FINGERTIP_Z_M = 0.0584 + 0.0445


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = default if raw in (None, "") else float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def fingertip_pose_to_hand_pose(
    position,
    quat_wxyz,
    *,
    hand_to_fingertip_z: float | None = None,
    tcp_rotation_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a TCP/fingertip target pose to the robot hand-link pose.

    The optional environment variables ``GAP_HAND_TO_FINGERTIP_Z`` and
    ``GAP_TCP_ROTATION_Z`` let an embodiment adapter describe its tool mount
    without changing a graph. With both variables unset, this exactly preserves
    the existing Panda conversion: 0.1029 m along hand Z and no rotation.
    """
    distance = (
        _env_float("GAP_HAND_TO_FINGERTIP_Z", PANDA_HAND_TO_FINGERTIP_Z_M)
        if hand_to_fingertip_z is None
        else float(hand_to_fingertip_z)
    )
    yaw = (
        _env_float("GAP_TCP_ROTATION_Z", 0.0)
        if tcp_rotation_z is None
        else float(tcp_rotation_z)
    )
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"hand_to_fingertip_z must be finite and positive, got {distance!r}")
    if not np.isfinite(yaw):
        raise ValueError(f"tcp_rotation_z must be finite, got {yaw!r}")

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
        raise ValueError("fingertip pose must contain only finite values")

    tcp_rotation = Rotation.from_quat(np.roll(quat, -1))
    if yaw == 0.0:
        hand_rotation = tcp_rotation
        hand_quat = quat.copy()
    else:
        hand_rotation = tcp_rotation * Rotation.from_euler("z", -yaw)
        hand_quat = np.roll(hand_rotation.as_quat(), 1)

    offset_world = hand_rotation.as_matrix() @ np.array([0.0, 0.0, distance])
    return pos - offset_world, hand_quat
