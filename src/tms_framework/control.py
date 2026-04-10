from __future__ import annotations

from typing import Iterable


def scale_action(action: Iterable[float], intensity_bounds: tuple[float, float], isi_bounds_ms: tuple[float, float]) -> tuple[float, float]:
    """Map normalized action in [-1, 1] to physical parameter ranges."""
    arr = list(action)
    if len(arr) != 2:
        raise ValueError("action must contain exactly two values: [intensity_control, isi_control].")

    a0 = min(1.0, max(-1.0, float(arr[0])))
    a1 = min(1.0, max(-1.0, float(arr[1])))

    intensity = (a0 + 1.0) / 2.0 * (intensity_bounds[1] - intensity_bounds[0]) + intensity_bounds[0]
    isi_ms = (a1 + 1.0) / 2.0 * (isi_bounds_ms[1] - isi_bounds_ms[0]) + isi_bounds_ms[0]
    return float(intensity), float(isi_ms)
