"""Installed resources and explicit per-run overrides."""
from copy import deepcopy
from importlib.resources import files

import yaml


def merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def config_path():
    return str(files("auv_navigation_fusion") / "config/navigation.yaml")


def load_config(path=None, overrides=None):
    with open(path or config_path()) as stream:
        cfg = yaml.safe_load(stream)
    return merge(cfg, overrides or {})
