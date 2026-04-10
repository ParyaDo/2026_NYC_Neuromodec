from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class FMRIEncoder(nn.Module):
    """Lightweight graph encoder for fMRI connectivity graphs.

    Inputs:
        x: [batch, nodes, node_feat]
        adj: [batch, nodes, nodes]
    """

    def __init__(self, node_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(node_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(adj, x)
        h = F.gelu(self.lin1(h))
        h = F.gelu(self.lin2(h))
        h = h.mean(dim=1)
        return F.normalize(self.proj(h), dim=-1)


class EEGEncoder(nn.Module):
    """EEG feature encoder for handcrafted bandpower/TEP vectors."""

    def __init__(self, feature_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, eeg_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(eeg_features), dim=-1)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Linear(hidden_dim, 2)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5, 2)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tanh-squashed action and corrected log-probability."""
        mu, log_std = self(obs)
        std = log_std.exp()
        noise = torch.randn_like(mu)
        pre_tanh = mu + noise * std
        action = torch.tanh(pre_tanh)

        log_two_pi = math.log(2.0 * math.pi)
        gaussian_log_prob = -0.5 * ((((pre_tanh - mu) / (std + 1e-6)) ** 2) + 2 * log_std + log_two_pi)
        gaussian_log_prob = gaussian_log_prob.sum(-1, keepdim=True)
        squash_correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        log_prob = gaussian_log_prob - squash_correction
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)
