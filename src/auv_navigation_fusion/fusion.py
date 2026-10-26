"""Sensor scheduling and optional recovery around the extracted estimators.

Inputs are observations and externally supplied map/calibration/command records.
Truth, mission planning, health decisions and actuator publishing have no role here.
"""
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from .config import load_config
from .ekf import NavEKF, rot
from .learned_velocity import LearnedVelocity, default_path
from .sonar_mapmatch import select_and_linearise


def _number(value, field):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _vector(value, size, field):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Non-numeric vector")
        array = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain {size} finite numbers") from error
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must contain {size} finite numbers")
    return array


class NavigationFusion:
    def __init__(self, config=None, *, x0=(0.0, 0.0, 0.0), yaw0=0.0, gravity=9.81,
                 wall_map=None, sonar_offset=(0.2, 0.0, -0.08), sonar_max_range=15.0):
        self.config = deepcopy(config) if config is not None else load_config()
        cfg = self.config
        estimator = cfg.get("estimator", "ekf")
        self.wall_map = None if wall_map is None else tuple(_vector(wall_map, 4, "wall_map"))
        if self.wall_map is not None:
            x_min, x_max, y_min, y_max = self.wall_map
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("wall_map requires x_min < x_max and y_min < y_max")
        self.sonar_offset = _vector(sonar_offset, 3, "sonar_offset")
        self.sonar_max_range = _number(sonar_max_range, "sonar_max_range")
        if self.sonar_max_range <= 0:
            raise ValueError("sonar_max_range must be positive")
        walls = cfg.get("sonar_walls", {}).get("enabled", False)
        learned = cfg.get("learned_velocity", {}).get("enabled", False)
        if walls and self.wall_map is None:
            raise ValueError("sonar wall updates require an explicit known wall_map")
        if estimator == "fgo" and (walls or learned):
            raise ValueError("The standalone FGO supports physical sensor/landmark updates; wall and learned updates require EKF")
        if estimator == "fgo":
            from .fgo import FactorGraphNav
            cls = FactorGraphNav
        elif estimator == "ekf":
            cls = NavEKF
        else:
            raise ValueError(f"Unknown estimator: {estimator}")
        self.filter = cls(cfg, tuple(_vector(x0, 3, "x0")), _number(yaw0, "yaw0"),
                          _number(gravity, "gravity"))
        self.learned = None
        if learned:
            lv = cfg["learned_velocity"]
            if lv["rate_hz"] <= 0 or lv["current_tau_s"] <= 0 or lv["sample_dt_s"] <= 0:
                raise ValueError("Learned velocity rates and time constants must be positive")
            self.learned = LearnedVelocity(lv.get("weights") or default_path(),
                                           lv["std_inflation"], lv.get("std_floor_mps", 0.0))
        self.current_est = np.zeros(3)
        self._last = {}
        self._time = None
        self._next_learned = 0.0
        self.duplicate_or_older = {name: 0 for name in ("imu", "dvl", "pressure", "compass", "sonar")}

    def _validate(self, sensors, time_s, commands):
        if not isinstance(sensors, dict):
            raise ValueError("sensors must be a mapping")
        observed = [_number(sensors[name]["t"], name + ".t") for name in self.duplicate_or_older if name in sensors]
        if not observed:
            raise ValueError("At least one navigation sensor is required")
        t = _number(max(observed) if time_s is None else time_s, "time_s")
        if self._time is not None and t < self._time:
            raise ValueError("Packet time must not go backwards")
        for name in self.duplicate_or_older:
            if name not in sensors:
                continue
            item = sensors[name]
            stamp = _number(item["t"], name + ".t")
            if stamp > t + 1e-8:
                raise ValueError(f"{name} has a future timestamp")
            if name == "imu":
                _vector(item["accel"], 3, "imu.accel")
                _vector(item["gyro"], 3, "imu.gyro")
                _number(item["roll"], "imu.roll")
                _number(item["pitch"], "imu.pitch")
            elif name == "dvl" and item.get("valid"):
                _vector(item.get("vel"), 3, "dvl.vel")
            elif name == "pressure":
                _number(item["depth"], "pressure.depth")
            elif name == "compass":
                _number(item["yaw"], "compass.yaw")
            elif name == "sonar" and item.get("valid"):
                bearings, ranges = np.asarray(item["bearings"], float), np.asarray(item["ranges"], float)
                if bearings.ndim != 1 or ranges.shape != bearings.shape or not np.isfinite(bearings).all() or not np.isfinite(ranges).all():
                    raise ValueError("Sonar bearings/ranges must have equal finite one-dimensional shapes")
        if self.learned is not None and "imu" in sensors:
            if commands is None or "pressure" not in sensors:
                raise ValueError("Learned velocity requires six command values and a pressure reading")
            _vector(commands, 6, "commands")
            old = self._last.get("imu")
            dt = sensors["imu"]["t"] - old if old is not None else self.config["learned_velocity"]["sample_dt_s"]
            nominal = self.config["learned_velocity"]["sample_dt_s"]
            if dt > 0 and abs(dt - nominal) > nominal * 0.1:
                raise ValueError("Bundled learned model requires regular 20 Hz IMU samples")
        return t

    def _fresh(self, name, sensors):
        item = sensors.get(name)
        if item is None:
            return None
        if item["t"] <= self._last.get(name, -math.inf):
            self.duplicate_or_older[name] += 1
            return None
        self._last[name] = item["t"]
        return item

    def update(self, sensors, *, time_s=None, commands=None, blocked=()):
        """Consume a timestamped sensor snapshot; repeated samples are fused once.

        `commands` are six observed thruster requests in N, never actuator outputs.
        `blocked` names sensors excluded by a caller-owned health policy.
        """
        if commands is None and isinstance(sensors, dict):
            commands = sensors.get("thrusters", {}).get("cmd_n")
        t = self._validate(sensors, time_s, commands)
        blocked = set(blocked)
        unknown = blocked - self.duplicate_or_older.keys()
        if unknown:
            raise ValueError(f"Unknown blocked navigation sensors: {sorted(unknown)}")
        previous_imu = self._last.get("imu")
        imu = self._fresh("imu", sensors)
        if imu is not None and "imu" not in blocked:
            self.filter.predict(imu)
            if self.learned is not None and "pressure" not in blocked:
                lv = self.config["learned_velocity"]
                dt = lv["sample_dt_s"] if previous_imu is None else imu["t"] - previous_imu
                self.learned.push(imu, commands, sensors["pressure"]["depth"], dt)
                if self.learned.ready() and t + 1e-9 >= self._next_learned:
                    self._next_learned = t + 1.0 / lv["rate_hz"]
                    through_water = self.learned.predict()
                    rotation = rot(self.filter.roll, self.filter.pitch, float(self.filter.s[6]))
                    dvl = sensors.get("dvl", {})
                    valid = dvl.get("valid", False) and "dvl" not in blocked and t - dvl["t"] <= lv["dvl_max_age_s"]
                    if valid:
                        if lv.get("current_estimation"):
                            measured_current = rotation @ (np.asarray(dvl["vel"]) - through_water)
                            alpha = min(1.0, 1.0 / (lv["rate_hz"] * lv["current_tau_s"]))
                            self.current_est[:2] += alpha * (measured_current[:2] - self.current_est[:2])
                    else:
                        velocity = through_water + rotation.T @ self.current_est if lv.get("current_estimation") else through_water
                        self.filter.update_velocity_pseudo(velocity, self.learned.std)
        for name, method in (("dvl", self.filter.update_dvl), ("pressure", self.filter.update_pressure),
                             ("compass", self.filter.update_compass)):
            item = self._fresh(name, sensors)
            if item is not None and name not in blocked:
                method(item)
        scan = self._fresh("sonar", sensors)
        mm = self.config.get("sonar_walls", {})
        if scan is not None and "sonar" not in blocked and mm.get("enabled"):
            state = self.filter.s
            cfg = dict(mm, max_range_m=self.sonar_max_range)
            for innovation, jacobian, variance in select_and_linearise(
                    (state[0], state[1], state[6]), scan, self.sonar_offset, self.wall_map, cfg):
                self.filter.update_scalar("walls", innovation, jacobian, variance)
        self._time = t
        return self.state()

    def update_landmark(self, lm_xy, range_m, bearing_rad, range_std, bearing_std):
        point = _vector(lm_xy, 2, "lm_xy")
        values = [_number(value, name) for value, name in zip(
            [range_m, bearing_rad, range_std, bearing_std], ["range", "bearing", "range_std", "bearing_std"])]
        if values[0] < 0 or min(values[2:]) <= 0:
            raise ValueError("Landmark range must be nonnegative and standard deviations positive")
        if np.linalg.norm(point - self.filter.s[:2]) < 1e-6:
            return False
        return self.filter.update_landmark(point, *values)

    def state(self):
        return {name: float(value) for name, value in self.filter.state().items()}

    def summary(self):
        return {"estimator": self.config.get("estimator", "ekf"), **self.filter.summary(),
                "current_estimate_world_mps": self.current_est.tolist(),
                "duplicate_or_older_samples": dict(self.duplicate_or_older)}
