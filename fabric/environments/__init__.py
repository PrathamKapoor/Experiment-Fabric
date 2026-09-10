"""Environment and hardware capture, and the differencing that makes it useful."""

from fabric.environments.capture import capture_environment
from fabric.environments.capture import describe as describe_environment
from fabric.environments.fingerprint import (
    Difference,
    environment_diff,
    hardware_diff,
    likely_causes,
    summarise_differences,
)
from fabric.environments.hardware import capture_hardware
from fabric.environments.hardware import describe as describe_hardware

__all__ = [
    "Difference",
    "capture_environment",
    "capture_hardware",
    "describe_environment",
    "describe_hardware",
    "environment_diff",
    "hardware_diff",
    "likely_causes",
    "summarise_differences",
]
