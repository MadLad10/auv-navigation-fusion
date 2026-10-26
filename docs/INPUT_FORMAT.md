# Sensor records and coordinate conventions

`NavigationFusion.update(sensors, time_s=..., commands=..., blocked=...)` accepts observations. It never accepts a simulator object, mission backend or truth argument.

JSONL replay accepts one snapshot per line:

```json
{"time_s":0.0,"sensors":{"imu":{"t":0.0,"gyro":[0,0,0],"accel":[0,0,9.81],"roll":0,"pitch":0},"dvl":{"t":0.0,"valid":true,"vel":[0,0,0]},"pressure":{"t":0.0,"depth":1.5},"compass":{"t":0.0,"yaw":0}},"truth":{"position_m":[0,0,-1.5],"rpy_rad":[0,0,0]}}
```

The optional `truth` field is read by the replay scorer after `update()` returns. It is omitted from estimator inputs. Optional `commands_n` contains six observed commands. Alternatively, simulator telemetry can supply `sensors.thrusters.cmd_n`. Optional `blocked_sensors` is a list such as `["pressure"]`, supplied by an external health policy.

| Record | Required fields |
|---|---|
| `imu` | `t`, `gyro` (3), `accel` (3), `roll`, `pitch` |
| `dvl` | `t`, `valid`; `vel` (3) when valid; `vel: null` is accepted when invalid |
| `pressure` | `t`, `depth` |
| `compass` | `t`, `yaw` |
| `sonar` | `t`, `valid`; equally sized finite `bearings` and `ranges` arrays when valid; negative range indicates no return |

At least one navigation sensor must be present. Sensor vectors/scalars must be finite. A sensor timestamp cannot exceed packet time; packet time cannot move backward. Each sensor reading is fused at most once for its timestamp; repeated/older per-sensor readings are counted and ignored. A held sonar scan is not treated as a fresh ping each IMU tick. Replay does not resample streams or smooth measurements backward in time.

Use meters, seconds, radians, m/s, m/s², rad/s and newtons. Body x is forward, y left, z up. The world z axis is up; pressure-derived depth is positive down. The initial yaw and supplied map use the same world frame. The simulator's generic course frame is used without geographical latitude/longitude.

For known-wall updates, supply `(x_min, x_max, y_min, y_max)`, the sonar mounting offset in body meters and its range limit. These are map/calibration inputs. No private sonar class labels or echo annotations are consumed.

Mapped landmark updates are explicit calls to `update_landmark(lm_xy, range_m, bearing_rad, range_std, bearing_std)`. Observations must be horizontal range and body-origin bearing. A camera/sonar adapter must correct its mounting offset and associate the observation with the known landmark before calling it.

Learned recovery additionally requires regular 20 Hz IMU samples, depth and six command observations in this order: `surge_left`, `surge_right`, `heave_front`, `heave_back`, `sway_front`, `sway_back`. The original model normalizes commands by 40 N and uses 40 consecutive feature vectors. It is specific to that vehicle/training scale.

`state()` returns world `x`, `y`, `z`, `vx`, `vy`, `vz`, yaw, horizontal forward speed `u`, and estimated yaw rate `r`. `summary()` returns uncertainty summaries, accepted/rejected updates, last-update times, duplicate counts and the horizontal current estimate. EKF covariance is available as `nav.filter.P`; FGO pose covariance is its last-keyframe internal covariance, not an equivalent eight-state EKF matrix.
