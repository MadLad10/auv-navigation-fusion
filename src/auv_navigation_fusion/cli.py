"""Inspect configuration, replay sensor logs and compare deterministic fixtures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics

from . import __version__
from .config import load_config
from .fusion import NavigationFusion
from .replay import read_records, replay
from .synthetic import INITIAL_POSITION, INITIAL_YAW, SONAR_OFFSET, WALL_MAP, generate_trace


def positive(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def output_dir(path):
    directory = Path(path).expanduser().resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Output directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def do_replay(args):
    cfg = load_config(args.config)
    if args.estimator is not None:
        cfg["estimator"] = args.estimator
    if args.wall_map is not None:
        cfg["sonar_walls"]["enabled"] = True
    if args.learned_velocity:
        cfg["learned_velocity"]["enabled"] = True
    if args.current_estimation:
        if not cfg["learned_velocity"]["enabled"]:
            raise ValueError("Current estimation requires learned velocity")
        cfg["learned_velocity"]["current_estimation"] = True
    nav = NavigationFusion(cfg, x0=args.initial_position, yaw0=args.initial_yaw,
                           wall_map=args.wall_map, sonar_offset=args.sonar_offset,
                           sonar_max_range=args.sonar_max_range)
    source = Path(args.trace).resolve()
    if not source.is_file():
        raise ValueError(f"Trace not found: {source}")
    out = output_dir(args.output)
    result = replay(read_records(source), nav, out / "estimates.jsonl")
    save(out / "summary.json", {"package_version": __version__, "input_trace": str(source),
                               "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                               "config": cfg, "initial_position_m": args.initial_position,
                               "initial_yaw_rad": args.initial_yaw,
                               **{key: result[key] for key in ("metrics", "diagnostics")}})
    print(json.dumps(result["metrics"]))


def do_demo(args):
    if args.seconds > 60 or args.seeds < 1:
        raise ValueError("Demo requires 1 or more seeds and a duration <= 60 seconds")
    methods = args.estimators.split(",")
    if not set(methods) <= {"ekf", "ekf_walls", "fgo"}:
        raise ValueError("Demo estimators must be ekf,ekf_walls,fgo")
    if "fgo" in methods:
        from .fgo import FactorGraphNav  # noqa: F401
    out = output_dir(args.output)
    rows, showcase = [], {}
    for condition in ("nominal", "dvl_dropout"):
        for seed in range(args.seeds):
            records = list(generate_trace(args.seconds, seed, dropout_at=10.0 if condition == "dvl_dropout" else None))
            trace_path = out / f"{condition}_seed{seed}.jsonl"
            with trace_path.open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            for method in methods:
                cfg = load_config(overrides={"estimator": "fgo" if method == "fgo" else "ekf",
                                              "sonar_walls": {"enabled": method == "ekf_walls"}})
                nav = NavigationFusion(cfg, x0=INITIAL_POSITION, yaw0=INITIAL_YAW,
                                       wall_map=WALL_MAP if method == "ekf_walls" else None)
                result = replay(copy.deepcopy(records), nav)
                rows.append({"condition": condition, "seed": seed, "estimator": method, **result["metrics"]})
                if condition == "dvl_dropout" and seed == 0:
                    showcase[method] = result
    save(out / "runs.json", rows)
    groups = []
    for condition in ("nominal", "dvl_dropout"):
        for method in methods:
            values = [row["rmse_xy_m"] for row in rows if row["condition"] == condition and row["estimator"] == method]
            groups.append({"condition": condition, "estimator": method, "seeds": len(values),
                           "mean_rmse_xy_m": statistics.mean(values), "population_sd_rmse_xy_m": statistics.pstdev(values)})
    save(out / "summary.json", {"package_version": __version__,
                               "kind": "kinematic sensor replay; no rigid-body simulation or mission controller",
                               "duration_s": args.seconds, "dropout_at_s": 10.0,
                               "initial_position_m": INITIAL_POSITION, "initial_yaw_rad": INITIAL_YAW,
                               "wall_map": WALL_MAP, "sonar_offset": SONAR_OFFSET,
                               "alignment": "none", "rows": groups,
                               "trace_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.jsonl"))}})
    if args.plot:
        from .plotting import plot_showcase
        plot_showcase(showcase, out)
    print(json.dumps({"output": str(out), "runs": len(rows), "summary": groups}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="Print packaged navigation defaults")
    replay_parser = commands.add_parser("replay", help="Estimate state from JSONL sensor snapshots")
    replay_parser.add_argument("trace")
    replay_parser.add_argument("--output", required=True)
    replay_parser.add_argument("--config")
    replay_parser.add_argument("--estimator", choices=("ekf", "fgo"))
    replay_parser.add_argument("--initial-position", nargs=3, type=float, default=[0, 0, -1.5])
    replay_parser.add_argument("--initial-yaw", type=float, default=0.0)
    replay_parser.add_argument("--wall-map", nargs=4, type=float)
    replay_parser.add_argument("--sonar-offset", nargs=3, type=float, default=list(SONAR_OFFSET))
    replay_parser.add_argument("--sonar-max-range", type=positive, default=15.0)
    replay_parser.add_argument("--learned-velocity", action="store_true")
    replay_parser.add_argument("--current-estimation", action="store_true")
    demo_parser = commands.add_parser("demo", help="Generate and replay analytic noisy-sensor fixtures")
    demo_parser.add_argument("--seconds", type=positive, default=40.0)
    demo_parser.add_argument("--seeds", type=integer, default=3)
    demo_parser.add_argument("--estimators", default="ekf,ekf_walls")
    demo_parser.add_argument("--output", required=True)
    demo_parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            print(json.dumps(load_config(), indent=2))
        elif args.command == "replay":
            do_replay(args)
        else:
            do_demo(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.error(str(error))
    return 0
