"""Measurement-driven underwater state estimation; no ROS or simulator required."""
from .config import load_config
from .ekf import NavEKF
from .fusion import NavigationFusion

__version__ = "0.1.0"
__all__ = ["NavEKF", "NavigationFusion", "load_config", "FactorGraphNav"]


def __getattr__(name):
    if name == "FactorGraphNav":
        from .fgo import FactorGraphNav
        return FactorGraphNav
    raise AttributeError(name)
