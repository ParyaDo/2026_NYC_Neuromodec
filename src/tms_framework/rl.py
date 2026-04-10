from __future__ import annotations

from collections import deque
import random

import torch
import torch.nn.functional as F

from .models import Actor, Critic

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - handled in runtime checks
    np = None


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        if np is None:
            raise RuntimeError("numpy is required for replay sampling.")
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r[:, None], ns, d[:, None]

    def __len__(self):
        return len(self.buf)


class SACAgent:
    def __init__(self, obs_dim: int, device: str = "cpu", gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2, lr: float = 3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(obs_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.target_critic = Critic(obs_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, obs, deterministic: bool = False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(x)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(x)
        if np is None:
            return action.squeeze(0).cpu().tolist()
        return action.squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer, batch_size: int = 128) -> dict[str, float]:
        if len(replay) < batch_size:
            return {}

        s, a, r, ns, d = replay.sample(batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            na, nlogp = self.actor.sample(ns)
            tq1, tq2 = self.target_critic(ns, na)
            target_v = torch.min(tq1, tq2) - self.alpha * nlogp
            target_q = r + (1 - d) * self.gamma * target_v

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_a, logp = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, new_a)
        actor_loss = (self.alpha * logp - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }
