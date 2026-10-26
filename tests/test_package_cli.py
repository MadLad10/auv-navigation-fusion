import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from auv_navigation_fusion import load_config
from auv_navigation_fusion.config import config_path


def cli(*args, cwd):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-m", "auv_navigation_fusion", *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True)


def test_packaged_configuration_and_overrides_are_independent():
    assert Path(config_path()).is_file()
    customized = load_config(overrides={"sonar_walls": {"enabled": True}})
    assert customized["sonar_walls"]["enabled"]
    assert not load_config()["sonar_walls"]["enabled"]


def test_cli_outside_source_checkout(tmp_path):
    result = cli("config", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["estimator"] == "ekf"
    result = cli("demo", "--seconds", "1", "--seeds", "1", "--output", tmp_path / "demo", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(json.loads((tmp_path / "demo/runs.json").read_text())) == 4


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_bad_duration_does_not_create_output(duration, tmp_path):
    target = tmp_path / "never"
    result = cli("demo", "--seconds", duration, "--output", target, cwd=tmp_path)
    assert result.returncode == 2
    assert not target.exists()


def test_existing_output_is_preserved(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep"
    marker.write_text("preserve")
    result = cli("demo", "--seconds", "1", "--output", target, cwd=tmp_path)
    assert result.returncode == 2
    assert marker.read_text() == "preserve"
