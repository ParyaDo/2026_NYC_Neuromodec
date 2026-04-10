from __future__ import annotations

import numpy as np


class SyntheticPatientEnv:
    """Simple latent-state simulator for pretraining SAC.

    State is the current latent vector z(t); target is z*.
    Action contains normalized intensity and ISI terms in [-1, 1].
    """

    def __init__(self, latent_dim: int, max_steps: int = 200, seed: int = 42):
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.state = None
        self.target = None

    def reset(self):
        self.step_count = 0
        self.target = self.rng.normal(size=(self.latent_dim,)).astype(np.float32)
        self.target /= np.linalg.norm(self.target) + 1e-8
        self.state = self.rng.normal(size=(self.latent_dim,)).astype(np.float32)
        self.state /= np.linalg.norm(self.state) + 1e-8
        return self._obs()

    def _obs(self):
        return np.concatenate([self.state, self.target], axis=0).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)

        intensity_term, isi_term = action
        drive = 0.06 * intensity_term - 0.02 * isi_term
        noise = self.rng.normal(scale=0.01, size=self.latent_dim).astype(np.float32)
        self.state = self.state + drive * (self.target - self.state) + noise
        self.state /= np.linalg.norm(self.state) + 1e-8

        distance = np.linalg.norm(self.state - self.target)
        reward = -distance
        done = distance < 0.15 or self.step_count >= self.max_steps
        info = {"distance": float(distance)}
        return self._obs(), float(reward), bool(done), info
