from copy import deepcopy
import math

import numpy as np
import pytest

from auv_navigation_fusion import NavEKF, NavigationFusion, load_config
from auv_navigation_fusion.replay import replay
from auv_navigation_fusion.synthetic import INITIAL_POSITION, INITIAL_YAW, generate_trace


def stationary(t=0.0):
    return {"imu": {"t": t, "gyro": [0, 0, 0], "accel": [0, 0, 9.81], "roll": 0, "pitch": 0},
            "dvl": {"t": t, "valid": True, "vel": [0, 0, 0]},
            "pressure": {"t": t, "depth": 1.5}, "compass": {"t": t, "yaw": 0}}


def test_static_fusion_preserves_pose_and_positive_covariance():
    nav = NavigationFusion(x0=(0, 0, -1.5))
    for k in range(401):
        state = nav.update(stationary(k * 0.05))
    assert np.linalg.norm([state["x"], state["y"], state["z"] + 1.5]) < 1e-6
    assert np.linalg.eigvalsh(nav.filter.P).min() >= 0


def test_dead_reckoning_uncertainty_grows_and_landmarks_reject_outliers():
    filt = NavEKF(load_config(), (0, 0, -1))
    initial = filt.pos_std()
    for k in range(601):
        filt.predict(stationary(k * 0.05)["imu"])
    assert filt.pos_std() > initial * 3
    assert filt.update_landmark((8, 0), 8, 0, 0.3, math.radians(2))
    assert not filt.update_landmark((8, 0), 2, 1.2, 0.3, math.radians(2))


def test_dvl_rotates_body_velocity_into_world_coordinates():
    cfg = load_config(overrides={"initial_std": {"vel": 2.0}})
    filt = NavEKF(cfg, yaw0=math.pi / 2)
    assert filt.update_dvl({"valid": True, "vel": [1, 0, 0]})
    assert abs(filt.state()["vx"]) < 1e-6
    assert abs(filt.state()["vy"] - 1) < 0.01


def test_duplicate_sensor_samples_are_not_fused_twice():
    nav = NavigationFusion(x0=(0, 0, -1.5))
    packet = stationary()
    nav.update(packet)
    covariance = nav.filter.P.copy()
    accepted = nav.filter.stats["dvl"].accepted
    nav.update(packet, time_s=0.05)
    assert np.array_equal(nav.filter.P, covariance)
    assert nav.filter.stats["dvl"].accepted == accepted


@pytest.mark.parametrize("sensor,field,value", [
    ("imu", "accel", [0, 0, float("nan")]), ("dvl", "vel", [0, 0]),
    ("pressure", "depth", float("inf")), ("compass", "yaw", float("nan")),
])
def test_invalid_measurements_fail_before_estimate_changes(sensor, field, value):
    nav = NavigationFusion(x0=(0, 0, -1.5))
    before = nav.filter.s.copy()
    packet = stationary()
    packet[sensor][field] = value
    with pytest.raises(ValueError):
        nav.update(packet)
    assert np.array_equal(before, nav.filter.s)


def test_future_and_backwards_packet_times_are_rejected():
    nav = NavigationFusion()
    with pytest.raises(ValueError, match="future"):
        nav.update(stationary(1), time_s=0)
    nav.update(stationary(1))
    with pytest.raises(ValueError, match="backwards"):
        nav.update(stationary(0))


def test_blocked_depth_reading_does_not_update_estimator():
    nav = NavigationFusion(x0=(0, 0, -1.5))
    nav.update(stationary(), blocked=["pressure"])
    assert nav.filter.stats["pressure"].accepted == 0
    assert nav.filter.stats["dvl"].accepted == 1


def test_wall_correction_requires_explicit_map_and_fgo_flags_are_explicit():
    cfg = load_config(overrides={"sonar_walls": {"enabled": True}})
    with pytest.raises(ValueError, match="wall_map"):
        NavigationFusion(cfg)
    cfg = load_config(overrides={"estimator": "fgo", "learned_velocity": {"enabled": True}})
    with pytest.raises(ValueError, match="require EKF"):
        NavigationFusion(cfg)


def test_singular_landmark_geometry_is_not_fused():
    nav = NavigationFusion()
    assert not nav.update_landmark([0, 0], 0, 0, 0.1, 0.1)


def test_replay_truth_does_not_change_estimates():
    records = list(generate_trace(1, seed=4))
    altered = deepcopy(records)
    for record in altered:
        record["truth"]["position_m"][0] += 100
    a = replay(records, NavigationFusion(x0=INITIAL_POSITION, yaw0=INITIAL_YAW))
    b = replay(altered, NavigationFusion(x0=INITIAL_POSITION, yaw0=INITIAL_YAW))
    assert a["predictions"] == b["predictions"]
    assert b["metrics"]["rmse_xy_m"] > 99


def test_replay_without_truth_returns_estimates_without_accuracy_claim():
    records = [{"time_s": 0, "sensors": stationary()}]
    result = replay(records, NavigationFusion(x0=(0, 0, -1.5)))
    assert result["metrics"]["truth_samples"] == 0
    assert "rmse_xy_m" not in result["metrics"]


def test_trajectory_replay_remains_finite_and_tracks_motion():
    result = replay(generate_trace(8, seed=2), NavigationFusion(x0=INITIAL_POSITION, yaw0=INITIAL_YAW))
    assert result["metrics"]["rmse_xy_m"] < 0.2
    assert result["metrics"]["rmse_depth_m"] < 0.03
