from dataclasses import dataclass, field


@dataclass
class SafetyConfig:
    max_intensity_pct_rmt: float = 120.0
    min_inter_pulse_interval_ms: float = 200.0
    max_tep_amplitude_uv: float = 200.0


@dataclass
class FrameworkConfig:
    latent_dim: int = 64
    eeg_feature_dim: int = 16
    fmri_node_dim: int = 32
    fmri_num_nodes: int = 90
    action_intensity_bounds: tuple[float, float] = (80.0, 120.0)
    action_isi_bounds_ms: tuple[float, float] = (200.0, 2500.0)
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    lr: float = 3e-4
    device: str = "cpu"
    safety: SafetyConfig = field(default_factory=SafetyConfig)
