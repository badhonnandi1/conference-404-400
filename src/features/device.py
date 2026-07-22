"""Device selection utilities for feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform

import torch


class DeviceSelectionError(ValueError):
    """Raised when a requested feature extraction device cannot be used."""


@dataclass(frozen=True)
class DeviceInfo:
    """Runtime device information for feature extraction."""

    requested_device: str
    selected_device: str
    mps_built: bool
    mps_available: bool
    architecture: str

    def to_dict(self) -> dict[str, str | bool]:
        """Return a JSON-serializable device information dictionary."""

        return asdict(self)


def select_device(requested_device: str = "auto") -> DeviceInfo:
    """Select MPS when available, otherwise CPU.

    CUDA is intentionally not considered because the target environment is
    macOS Apple Silicon with portable CPU fallback.
    """

    requested = requested_device.lower()
    if requested not in {"auto", "cpu", "mps"}:
        raise DeviceSelectionError(
            f"Invalid device '{requested_device}'. Use 'auto', 'cpu', or 'mps'."
        )

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise DeviceSelectionError(
            "MPS was requested but is not available. Use --device auto or --device cpu."
        )

    selected = "mps" if requested == "auto" and mps_available else requested
    if selected == "auto":
        selected = "cpu"

    return DeviceInfo(
        requested_device=requested,
        selected_device=selected,
        mps_built=mps_built,
        mps_available=mps_available,
        architecture=platform.machine(),
    )


def torch_device(device_info: DeviceInfo) -> torch.device:
    """Convert DeviceInfo to a torch.device."""

    return torch.device(device_info.selected_device)
