"""Minimal runnable demo for CL + RL + safety-aware TMS control.

Usage:
    python scripts_demo.py --cl-steps 30 --episodes 10
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic demo for the TMS framework.")
    parser.add_argument("--cl-steps", type=int, default=20, help="Number of cross-modal warm-up steps.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of RL pretraining episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        print(
            "Missing dependency: "
            f"{exc}. Install requirements first (e.g. `pip install -r requirements.txt`).",
            file=sys.stderr,
        )
        return 1

    from src.tms_framework import FrameworkConfig, TMSFramework
    from src.tms_framework.env import SyntheticPatientEnv

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def fake_batch(batch_size: int, num_nodes: int, node_dim: int, eeg_dim: int):
        x = torch.randn(batch_size, num_nodes, node_dim)
        a = torch.softmax(torch.randn(batch_size, num_nodes, num_nodes), dim=-1)
        eeg = torch.randn(batch_size, eeg_dim)
        return x, a, eeg

    cfg = FrameworkConfig()
    model = TMSFramework(cfg)

    for step in range(args.cl_steps):
        x, adj, eeg = fake_batch(32, cfg.fmri_num_nodes, cfg.fmri_node_dim, cfg.eeg_feature_dim)
        loss = model.train_cross_modal_step(x, adj, eeg)
        if step % max(1, args.cl_steps // 4) == 0:
            print(f"contrastive_step={step:02d}, loss={loss:.4f}")

    env = SyntheticPatientEnv(latent_dim=cfg.latent_dim, seed=args.seed)
    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = model.agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            model.replay.add(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            model.rl_update(batch_size=64)
        print(f"episode={ep:02d}, return={total_reward:.3f}, final_distance={info['distance']:.3f}")

    fmri_x = torch.randn(1, cfg.fmri_num_nodes, cfg.fmri_node_dim)
    fmri_adj = torch.softmax(torch.randn(1, cfg.fmri_num_nodes, cfg.fmri_num_nodes), dim=-1)
    eeg_t = torch.randn(1, cfg.eeg_feature_dim)

    z_star = model.target_from_fmri(fmri_x, fmri_adj)
    z_t = model.state_from_eeg(eeg_t)
    command = model.choose_safe_stimulation(z_t, z_star, latest_tep_amplitude_uv=220.0)
    print("safe_command", command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
