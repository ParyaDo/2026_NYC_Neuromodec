from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np

from src.tms_framework.preprocess_utils import (
    corr_connectivity,
    make_node_features,
    save_graph_npz,
    save_eeg_npy,
    simple_eeg_features,
)

"""
Expected inputs:
  --fmri-dir data/raw/hcp/fmri_timeseries   # *.npy [time, roi]
  --eeg-dir  data/raw/hcp/eeg_timeseries    # *.npy [channels, time] (or surrogate)
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fmri-dir", required=True)
    p.add_argument("--eeg-dir", required=True)
    p.add_argument("--out-dir", default="data/processed/hcp")
    p.add_argument("--manifest", default="manifests/hcp_manifest.csv")
    p.add_argument("--sfreq", type=float, default=250.0)
    args = p.parse_args()

    fmri_dir = Path(args.fmri_dir)
    eeg_dir = Path(args.eeg_dir)
    out_dir = Path(args.out_dir)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fmri_file in sorted(fmri_dir.glob("*.npy")):
        sid = fmri_file.stem
        eeg_file = eeg_dir / f"{sid}.npy"
        if not eeg_file.exists():
            continue

        ts = np.load(fmri_file)
        eeg = np.load(eeg_file)

        x = make_node_features(ts)
        adj = corr_connectivity(ts)
        eeg_feat = simple_eeg_features(eeg, sfreq=args.sfreq)

        fmri_out = out_dir / "fmri_graphs" / f"{sid}.npz"
        eeg_out = out_dir / "eeg_features" / f"{sid}.npy"

        save_graph_npz(fmri_out, x, adj)
        save_eeg_npy(eeg_out, eeg_feat)

        rows.append((sid, str(fmri_out), str(eeg_out)))

    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "fmri_path", "eeg_path"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} subjects to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
