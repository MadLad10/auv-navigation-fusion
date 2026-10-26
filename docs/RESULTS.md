# Evaluation evidence

The package keeps three kinds of evidence separate:

1. Fresh standalone sensor replay from the deterministic analytic fixture. Ground-truth positions define the kinematic trajectory; the estimator receives noisy sensor records and a known initial pose. This is not a rigid-body or mission-success evaluation.
2. A passive Marine Sim integration check, using its independently installed package, noisy sensor outputs and zero motor commands.
3. Imported historical studies from the integrated mission application. These retain their original seeds, conditions, task outcomes and measurements; they have not been relabeled as fresh standalone runs.

The fixture's position/yaw are analytic functions of time. IMU samples are generated at 20 Hz with sensor noise and a fixed accelerometer offset; DVL is 5 Hz, compass/depth 10 Hz, and idealized rectangular-wall sonar 2 Hz. The dropout condition invalidates DVL at t=10 s. The comparison uses the same generated sensor trace for each estimator, three seeds by default, a caller-supplied initial pose and no trajectory alignment. The CLI writes the exact fixture settings and trace SHA-256 values.

The recorded run contains **18 replays**: 3 seeds × 2 conditions × 3 estimators. [Summary](fresh_replay_summary.json), [individual runs](fresh_replay_runs.json), [plot](media/trajectory.png), [animation](media/dvl_dropout.gif). The full input logs remain under the local `runs/` directory after extraction; their exact hashes are in the summary. Recreate them using:

```bash
auv-nav demo --seconds 40 --seeds 3 --estimators ekf,ekf_walls,fgo --plot --output runs/comparison
```

Metrics are horizontal position RMSE, maximum/final horizontal error, depth RMSE and yaw RMSE. Errors use supplied world coordinates with no rigid or scale alignment. A JSONL trace without truth yields estimates and update diagnostics rather than error metrics.

## Historical evidence

| Study | Runs | Original setting |
|---|---|---|
| [Fusion study](historical/nav_study_fusion/summary.md) | 400 | 20 seeds × 5 conditions × 4 EKF fusion configurations |
| [Current study](historical/nav_study_current/summary.md) | 120 | 20 seeds × 2 dropout conditions × 3 recovery configurations |
| [Estimator study](historical/nav_study_20seeds/summary.md) | 320 | 20 seeds × 4 conditions × EKF/FGO with/without learned recovery |

These 840 rows are imported records across related campaigns, not 840 guaranteed independent trajectories. Some baseline values are reused between campaigns. The original studies use the application's controller, course priors and perception adapter. Mission tool success and ground-truth task success are different columns. The historical FGO+learned configuration predates the unsupported-path restriction in this release.

The bundled GRU's original validation report is in `src/auv_navigation_fusion/models/dvl_net_eval.json`. Its very low synthetic validation errors reflect the original still-water vehicle distribution and should not be transferred to the kinematic fixture or another vehicle. The historical current condition shows that adding learned recovery can underperform wall correction alone.
