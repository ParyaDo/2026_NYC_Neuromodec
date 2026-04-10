from __future__ import annotations

from src.tms_framework.control import scale_action
from src.tms_framework.safety import SafetyLayer, SafetyState


def test_scale_action_limits():
    intensity, isi = scale_action([-1.0, -1.0], (80.0, 120.0), (200.0, 2500.0))
    assert intensity == 80.0
    assert isi == 200.0

    intensity, isi = scale_action([1.0, 1.0], (80.0, 120.0), (200.0, 2500.0))
    assert intensity == 120.0
    assert isi == 2500.0


def test_safety_layer_veto():
    layer = SafetyLayer(
        max_intensity_pct_rmt=120.0,
        min_inter_pulse_interval_ms=200.0,
        max_tep_amplitude_uv=200.0,
    )

    intensity, isi, veto = layer.apply(130.0, 100.0, SafetyState(latest_tep_amplitude_uv=250.0))
    assert veto is True
    assert intensity <= 100.0
    assert isi >= 1000.0
