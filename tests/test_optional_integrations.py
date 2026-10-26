import os
os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
import pytest

from auv_navigation_fusion import NavigationFusion, load_config
from auv_navigation_fusion.replay import replay
from auv_navigation_fusion.synthetic import INITIAL_POSITION, INITIAL_YAW, generate_trace


@pytest.mark.factor_graph
def test_factor_graph_runs_measurement_replay():
    pytest.importorskip("gtsam")
    nav = NavigationFusion(load_config(overrides={"estimator": "fgo"}), x0=INITIAL_POSITION, yaw0=INITIAL_YAW)
    result = replay(generate_trace(8, seed=1), nav)
    assert nav.summary()["keyframes"] >= 30
    assert result["metrics"]["rmse_xy_m"] < 0.2
    assert result["metrics"]["rmse_depth_m"] < 0.04


@pytest.mark.simulator
def test_standalone_simulator_provides_compatible_measurements():
    pytest.importorskip("auv_marine_sim")
    from auv_marine_sim import MarineSim
    from auv_marine_sim.world import load_arena
    from auv_navigation_fusion.sonar_mapmatch import pool_rect
    arena = load_arena("default")
    arena["start"].update(x=16, y=6, z=-2, yaw=1)
    sim = MarineSim(arena, seed=3)
    nav = NavigationFusion(load_config(overrides={"sonar_walls": {"enabled": True}}),
                           x0=(16, 6, -2), yaw0=1, wall_map=pool_rect(arena["pool"]))
    errors = []
    try:
        for _ in range(200):
            sim.step()
            state = nav.update(sim.readings, time_s=sim.t)
            errors.append(np.linalg.norm(np.array([state["x"], state["y"]]) - sim.truth().pos[:2]))
        assert np.sqrt(np.mean(np.square(errors))) < 0.3
        assert nav.filter.stats["walls"].accepted > 0
        assert not np.any(sim.cmd)
    finally:
        sim.close()
