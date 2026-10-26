# Component extraction

The original EKF, GTSAM estimator, wall geometry and NumPy GRU implementations come from `src/auv_onboard/auv_onboard` in Agentic AUV. The extraction revision and original file hashes are in [extraction_source.json](extraction_source.json). The application and private research workspace are preserved.

The standalone layer adds packaged configuration/model lookup, the measurement-driven `NavigationFusion` API, timestamp validation and duplicate suppression, explicit sensor exclusions, JSONL replay and scoring, analytic sensor fixtures, CLI examples, tests and packaging.

Core adaptations:

- GTSAM imports its update statistics and angle helper from this package; its keyframe documentation follows configuration.
- The original GTSAM pseudo-velocity method ignored the supplied uncertainty. That method is disabled explicitly, and the wrapper rejects learned/wall correction flags for FGO. The EKF supports those options.
- The GRU resolver uses the bundled model or an explicit `AUV_DVL_NET` path, without searching an unrelated working directory or ROS workspace.
- Defaults are sensor-only EKF. Known-wall and learned updates require explicit opt-in. The map is supplied independently of the observations.
- Current/recovery scheduling was adapted from the application backend. The wrapper checks DVL age before using it as a current witness and verifies the bundled model's sampling interval.
- The original training recipe is retained with creation of its output parent directory added. Training/data collection are not run as part of installing the package.

Historical experiment files were copied byte-for-byte and their hashes recorded. Their source files did not record exact experiment revisions; the extraction revision is not evidence of which revision originally produced each run. Those files remain historical integrated-system observations. Fresh replay results and package tests are recorded separately.

The original application still contains these implementations. Changing its dependencies to this package is a separate integration migration; this extraction does not claim that duplicate application sources have already been retired. The simulator is consumed as an optional installed dependency and is not copied here.
