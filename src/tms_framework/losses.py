from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(fmri_latent: torch.Tensor, eeg_latent: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric cross-modal InfoNCE loss.

    Same index across modalities is treated as a positive pair.
    """
    logits = fmri_latent @ eeg_latent.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i + loss_t)
