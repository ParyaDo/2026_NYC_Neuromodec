from __future__ import annotations

import numpy as np
import torch

from .config import FrameworkConfig
from .losses import info_nce
from .models import EEGEncoder, FMRIEncoder
from .control import scale_action
from .rl import SACAgent, ReplayBuffer
from .safety import SafetyLayer, SafetyState


class TMSFramework:
    """End-to-end scaffold implementing CL + RL + safety for individualized TMS."""

    def __init__(self, cfg: FrameworkConfig):
        self.cfg = cfg
        self.fmri_encoder = FMRIEncoder(cfg.fmri_node_dim, 128, cfg.latent_dim).to(cfg.device)
        self.eeg_encoder = EEGEncoder(cfg.eeg_feature_dim, 128, cfg.latent_dim).to(cfg.device)
        self.opt = torch.optim.Adam(
            list(self.fmri_encoder.parameters()) + list(self.eeg_encoder.parameters()),
            lr=cfg.lr,
        )

        obs_dim = cfg.latent_dim * 2
        self.agent = SACAgent(obs_dim=obs_dim, device=cfg.device, gamma=cfg.gamma, tau=cfg.tau, alpha=cfg.alpha, lr=cfg.lr)
        self.replay = ReplayBuffer()

        self.safety = SafetyLayer(
            max_intensity_pct_rmt=cfg.safety.max_intensity_pct_rmt,
            min_inter_pulse_interval_ms=cfg.safety.min_inter_pulse_interval_ms,
            max_tep_amplitude_uv=cfg.safety.max_tep_amplitude_uv,
        )

    def train_cross_modal_step(self, fmri_x: torch.Tensor, fmri_adj: torch.Tensor, eeg_feat: torch.Tensor) -> float:
        """One gradient step of symmetric cross-modal InfoNCE."""
        fmri_x = fmri_x.to(self.cfg.device)
        fmri_adj = fmri_adj.to(self.cfg.device)
        eeg_feat = eeg_feat.to(self.cfg.device)

        z_fmri = self.fmri_encoder(fmri_x, fmri_adj)
        z_eeg = self.eeg_encoder(eeg_feat)
        loss = info_nce(z_fmri, z_eeg)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def target_from_fmri(self, fmri_x: torch.Tensor, fmri_adj: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            z = self.fmri_encoder(fmri_x.to(self.cfg.device), fmri_adj.to(self.cfg.device))
        return z.squeeze(0).cpu().numpy()

    def state_from_eeg(self, eeg_feat: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            z = self.eeg_encoder(eeg_feat.to(self.cfg.device))
        return z.squeeze(0).cpu().numpy()

    def choose_safe_stimulation(self, z_t: np.ndarray, z_star: np.ndarray, latest_tep_amplitude_uv: float) -> dict[str, object]:
        """Generate safe stimulation command from current state and therapeutic target."""
        obs = np.concatenate([z_t, z_star], axis=0)
        raw_action = self.agent.act(obs, deterministic=False)
        proposed_intensity, proposed_isi_ms = scale_action(
            raw_action,
            self.cfg.action_intensity_bounds,
            self.cfg.action_isi_bounds_ms,
        )
        intensity, isi_ms, veto = self.safety.apply(
            proposed_intensity,
            proposed_isi_ms,
            SafetyState(latest_tep_amplitude_uv=latest_tep_amplitude_uv),
        )
        return {
            "raw_action": raw_action.tolist(),
            "proposed": {"intensity_pct_rmt": proposed_intensity, "isi_ms": proposed_isi_ms},
            "safe": {"intensity_pct_rmt": intensity, "isi_ms": isi_ms},
            "veto": veto,
        }

    def rl_store_transition(self, z_t: np.ndarray, z_star: np.ndarray, action_2d: np.ndarray, z_next: np.ndarray, reward: float, done: bool):
        s = np.concatenate([z_t, z_star], axis=0)
        ns = np.concatenate([z_next, z_star], axis=0)
        self.replay.add(s, action_2d, reward, ns, done)

    def rl_update(self, batch_size: int = 128):
        return self.agent.update(self.replay, batch_size=batch_size)
