"""A stationary sensor record; all units are SI and the world frame has z up."""
from auv_navigation_fusion import NavigationFusion

nav = NavigationFusion(x0=(0, 0, -1.5))
for k in range(101):
    t = k * 0.05
    state = nav.update({
        "imu": {"t": t, "gyro": [0, 0, 0], "accel": [0, 0, 9.81], "roll": 0, "pitch": 0},
        "dvl": {"t": t, "valid": True, "vel": [0, 0, 0]},
        "pressure": {"t": t, "depth": 1.5},
        "compass": {"t": t, "yaw": 0},
    })
print(state)
print(nav.summary())
