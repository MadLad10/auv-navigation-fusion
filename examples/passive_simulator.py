"""Optional Marine Sim integration: passive drift, no motor commands."""
import os
os.environ.setdefault("MUJOCO_GL", "disable")

from auv_marine_sim import MarineSim
from auv_marine_sim.world import load_arena
from auv_navigation_fusion import NavigationFusion, load_config
from auv_navigation_fusion.sonar_mapmatch import pool_rect

arena = load_arena("default")
arena["start"].update(x=16.0, y=6.0, z=-2.0, yaw=1.0)
sim = MarineSim(arena, seed=0)
nav = NavigationFusion(load_config(overrides={"sonar_walls": {"enabled": True}}),
                       x0=(16, 6, -2), yaw0=1.0, wall_map=pool_rect(arena["pool"]),
                       sonar_offset=sim.cfg["sensors"]["sonar"]["offset"])
try:
    for _ in range(400):
        sim.step()
        state = nav.update(sim.readings, time_s=sim.t)
        sim.referee.record_estimate(sim.t, [state["x"], state["y"], state["z"]], sim.truth().pos)
    print(sim.referee.report()["localisation"])
finally:
    sim.close()
