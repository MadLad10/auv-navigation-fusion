"""Deterministic kinematic sensor fixtures, separate from MuJoCo physics."""
from copy import deepcopy
import math

import numpy as np

from .ekf import rot
from .sonar_mapmatch import predicted_range

INITIAL_POSITION = (12.0, 5.5, -1.5)
INITIAL_YAW = 0.4
WALL_MAP = (-5.0, 25.0, -10.0, 10.0)
SONAR_OFFSET = (0.2, 0.0, -0.08)


def trajectory(t):
    position = np.array([12 + 0.18 * t + 0.7 * math.sin(0.08 * t),
                         5.5 + 0.7 * math.sin(0.1 * t), -1.5 - 0.15 * math.sin(0.12 * t)])
    velocity = np.array([0.18 + 0.056 * math.cos(0.08 * t),
                         0.07 * math.cos(0.1 * t), -0.018 * math.cos(0.12 * t)])
    acceleration = np.array([-0.00448 * math.sin(0.08 * t),
                             -0.007 * math.sin(0.1 * t), 0.00216 * math.sin(0.12 * t)])
    yaw = INITIAL_YAW + 0.25 * math.sin(0.06 * t)
    return position, velocity, acceleration, yaw, 0.015 * math.cos(0.06 * t)


def generate_trace(seconds=40.0, seed=0, dropout_at=None):
    """20 Hz IMU; DVL 5 Hz; compass/depth 10 Hz; idealized wall sonar 2 Hz.

    Truth is an explicit evaluation field. It is never part of a sensor payload.
    A fixed accelerometer offset exposes dead-reckoning drift in this fixture.
    """
    if not 0 < seconds <= 60:
        raise ValueError("Fixture duration must be in (0, 60] seconds to keep the route inside its map")
    rng = np.random.default_rng(seed)
    bearings = np.linspace(-math.pi / 3, math.pi / 3, 81)
    latest = {}
    for tick in range(int(round(seconds / 0.05)) + 1):
        t = round(tick * 0.05, 6)
        p, velocity, acceleration, yaw, yaw_rate = trajectory(t)
        rotation = rot(0.0, 0.0, yaw)
        latest["imu"] = {"t": t, "gyro": (np.array([0, 0, yaw_rate]) + rng.normal(0, 0.002, 3)).tolist(),
                         "accel": (rotation.T @ (acceleration + [0, 0, 9.81])
                                   + [0.008, -0.006, 0.002] + rng.normal(0, 0.03, 3)).tolist(),
                         "roll": 0.0, "pitch": 0.0}
        if tick % 2 == 0:
            latest["pressure"] = {"t": t, "depth": float(-p[2] + rng.normal(0, 0.005))}
            latest["compass"] = {"t": t, "yaw": float(yaw + rng.normal(0, math.radians(1.5)))}
        if tick % 4 == 0:
            valid = dropout_at is None or t < dropout_at
            latest["dvl"] = {"t": t, "valid": valid,
                             "vel": ((rotation.T @ velocity) * 1.002 + rng.normal(0, 0.01, 3)).tolist() if valid else None}
        if tick % 10 == 0:
            ranges = []
            for bearing in bearings:
                distance, _ = predicted_range(*p[:2], yaw, bearing, SONAR_OFFSET, WALL_MAP)
                ranges.append(float(distance + rng.normal(0, 0.05)) if distance <= 15 and rng.random() >= 0.05 else -1.0)
            latest["sonar"] = {"t": t, "valid": True, "bearings": bearings.tolist(), "ranges": ranges}
        yield {"time_s": t, "sensors": deepcopy(latest),
               "truth": {"position_m": p.tolist(), "rpy_rad": [0.0, 0.0, yaw],
                         "velocity_body_mps": (rotation.T @ velocity).tolist()}}
