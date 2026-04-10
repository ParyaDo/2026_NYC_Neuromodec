from __future__ import annotations

from pathlib import Path
import numpy as np


def zscore_cols(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def corr_connectivity(ts: np.ndarray) -> np.ndarray:
    """ts shape: [time, roi] -> adj shape [roi, roi]."""
    ts = zscore_cols(ts)
    adj = np.corrcoef(ts.T)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)


def make_node_features(ts: np.ndarray) -> np.ndarray:
    """Simple ROI stats as node features: mean/std/power."""
    mean = ts.mean(axis=0)
    std = ts.std(axis=0)
    power = (ts ** 2).mean(axis=0)
    x = np.stack([mean, std, power], axis=1)
    return x.astype(np.float32)


def save_graph_npz(out_path: Path, x: np.ndarray, adj: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=x.astype(np.float32), adj=adj.astype(np.float32))


def save_eeg_npy(out_path: Path, feat: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat.astype(np.float32))


def simple_eeg_features(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """
    eeg shape [channels, time]
    Very simple baseline features (placeholder):
      - global mean abs amplitude
      - per-band rough power proxies using FFT bins
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    abs_mean = np.abs(eeg).mean()

    # rough bandpowers from averaged PSD
    fft = np.fft.rfft(eeg, axis=1)
    psd = (np.abs(fft) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(eeg.shape[1], d=1.0 / sfreq)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(psd[m].mean()) if m.any() else 0.0

    theta = band(4, 8)
    alpha = band(8, 13)
    beta = band(13, 30)

    return np.array([abs_mean, theta, alpha, beta], dtype=np.float32)
