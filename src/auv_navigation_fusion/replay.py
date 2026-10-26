"""Causal replay with truth used only after estimation to compute errors."""
import json
import math
from pathlib import Path

import numpy as np

from .ekf import wrap


def read_records(path):
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at line {line_number}") from error


def replay(records, navigation, estimates_path=None):
    stream = Path(estimates_path).open("w") if estimates_path is not None else None
    horizontal, vertical, yaw_errors, predictions, positions = [], [], [], [], []
    samples = 0
    try:
        for record in records:
            # The estimator only receives observations, optional command telemetry
            # and the caller's sensor-exclusion flags. Truth is read below.
            state = navigation.update(record["sensors"], time_s=record.get("time_s"),
                                      commands=record.get("commands_n"), blocked=record.get("blocked_sensors", ()))
            t = record.get("time_s", navigation.filter.t)
            prediction = [state["x"], state["y"], state["z"]]
            predictions.append([t, *prediction])
            samples += 1
            if stream is not None:
                stream.write(json.dumps({"time_s": t, "state": state, "diagnostics": navigation.summary()}, allow_nan=False) + "\n")
            truth = record.get("truth")
            if truth is not None:
                p = np.asarray(truth["position_m"], float)
                if p.shape != (3,) or not np.isfinite(p).all():
                    raise ValueError("truth.position_m must contain three finite coordinates")
                positions.append([t, *p.tolist()])
                horizontal.append(math.hypot(state["x"] - p[0], state["y"] - p[1]))
                vertical.append(state["z"] - p[2])
                if "rpy_rad" in truth:
                    yaw_errors.append(wrap(state["yaw"] - truth["rpy_rad"][2]))
    finally:
        if stream is not None:
            stream.close()
    if not samples:
        raise ValueError("Input trace contains no observations")
    metrics = {"samples": samples, "truth_samples": len(horizontal), "alignment": "none; known initial pose supplied by caller"}
    if horizontal:
        metrics.update(rmse_xy_m=float(np.sqrt(np.mean(np.square(horizontal)))),
                       max_xy_m=float(max(horizontal)), final_xy_m=float(horizontal[-1]),
                       rmse_depth_m=float(np.sqrt(np.mean(np.square(vertical)))))
    if yaw_errors:
        metrics["rmse_yaw_deg"] = float(np.degrees(np.sqrt(np.mean(np.square(yaw_errors)))))
    return {"metrics": metrics, "diagnostics": navigation.summary(),
            "predictions": predictions, "truth_positions": positions}
