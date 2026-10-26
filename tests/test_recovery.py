from pathlib import Path

import numpy as np
import pytest

from auv_navigation_fusion import NavigationFusion, load_config
from auv_navigation_fusion.learned_velocity import LearnedVelocity, default_path
from auv_navigation_fusion.synthetic import INITIAL_POSITION, INITIAL_YAW, WALL_MAP, generate_trace
from auv_navigation_fusion.replay import replay
from auv_navigation_fusion.sonar_mapmatch import ray_to_rect


def test_known_wall_geometry():
    assert ray_to_rect(0, 0, 0, *WALL_MAP) == (25, 0)
    assert np.isinf(ray_to_rect(30, 0, 0, *WALL_MAP)[0])


def test_mapped_sonar_reduces_dropout_drift_on_fixed_replay():
    records = list(generate_trace(30, seed=3, dropout_at=10))
    baseline = replay(records, NavigationFusion(x0=INITIAL_POSITION, yaw0=INITIAL_YAW))
    walls = NavigationFusion(load_config(overrides={"sonar_walls": {"enabled": True}}),
                             x0=INITIAL_POSITION, yaw0=INITIAL_YAW, wall_map=WALL_MAP)
    corrected = replay(records, walls)
    assert walls.filter.stats["walls"].accepted > 50
    assert corrected["metrics"]["rmse_xy_m"] < baseline["metrics"]["rmse_xy_m"]


def test_bundled_gru_agrees_with_pytorch_reference():
    model = LearnedVelocity(default_path())
    with np.load(Path(__file__).parent / "fixtures/gru_reference.npz") as reference:
        model.buf.extend(reference["normalized_features"])
        assert model.ready()
        assert np.allclose(model.predict(), reference["torch_prediction"], atol=1e-6)
    assert np.all(model.std >= 0.01)


def test_missing_explicit_model_path_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("AUV_DVL_NET", str(tmp_path / "missing.npz"))
    with pytest.raises(FileNotFoundError):
        default_path()


def test_current_is_estimated_before_dvl_loss_and_used_during_outage():
    nav = NavigationFusion(load_config(overrides={"learned_velocity": {"enabled": True, "current_tau_s": 2}}))

    class ConstantThroughWater:
        std = np.array([0.05] * 3)
        def push(self, *args):
            pass
        def ready(self):
            return True
        def predict(self):
            return np.array([0.2, 0, 0])

    nav.learned = ConstantThroughWater()
    for k in range(501):
        t = k * 0.05
        nav.update({"imu": {"t": t, "gyro": [0, 0, 0], "accel": [0, 0, 9.81], "roll": 0, "pitch": 0},
                    "pressure": {"t": t, "depth": 0}, "compass": {"t": t, "yaw": 0},
                    "dvl": {"t": t, "valid": t < 15, "vel": [0.3, 0, 0] if t < 15 else None}}, commands=[0] * 6)
    assert abs(nav.current_est[0] - 0.1) < 0.002
    assert nav.filter.stats["learned_vel"].accepted > 5
    assert abs(nav.state()["vx"] - 0.3) < 0.02


def test_learned_model_rejects_incompatible_sampling_rate():
    nav = NavigationFusion(load_config(overrides={"learned_velocity": {"enabled": True}}))
    packet = {"imu": {"t": 0, "gyro": [0, 0, 0], "accel": [0, 0, 9.81], "roll": 0, "pitch": 0},
              "pressure": {"t": 0, "depth": 0}}
    nav.update(packet, commands=[0] * 6)
    packet["imu"]["t"] = 0.1
    with pytest.raises(ValueError, match="20 Hz"):
        nav.update(packet, commands=[0] * 6)
