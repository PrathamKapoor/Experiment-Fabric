"""Environment and hardware capture, and the differencing that makes it useful."""

from fabric.environments.capture import capture_environment, describe as describe_environment
from fabric.environments.fingerprint import (
    Difference,
    environment_diff,
    hardware_diff,
    likely_causes,
    summarise_differences,
)
from fabric.environments.hardware import capture_hardware, describe as describe_hardware

__all__ = [
    "capture_environment",
    "capture_hardware",
    "describe_environment",
    "describe_hardware",
    "Difference",
    "environment_diff",
    "hardware_diff",
    "likely_causes",
    "summarise_differences",
]
