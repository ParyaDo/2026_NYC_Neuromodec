from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyState:
    latest_tep_amplitude_uv: float


class SafetyLayer:
    def __init__(
        self,
        max_intensity_pct_rmt: float,
        min_inter_pulse_interval_ms: float,
        max_tep_amplitude_uv: float,
    ):
        self.max_intensity = max_intensity_pct_rmt
        self.min_isi_ms = min_inter_pulse_interval_ms
        self.max_tep = max_tep_amplitude_uv

    def apply(self, proposed_intensity: float, proposed_isi_ms: float, state: SafetyState) -> tuple[float, float, bool]:
        veto = False

        intensity = min(proposed_intensity, self.max_intensity)
        if intensity != proposed_intensity:
            veto = True

        isi_ms = max(proposed_isi_ms, self.min_isi_ms)
        if isi_ms != proposed_isi_ms:
            veto = True

        if state.latest_tep_amplitude_uv > self.max_tep:
            intensity = min(intensity, 100.0)
            isi_ms = max(isi_ms, 1000.0)
            veto = True

        return intensity, isi_ms, veto
