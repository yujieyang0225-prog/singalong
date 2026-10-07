#!/usr/bin/env python3
"""Real a cappella recordings (Hackathon/08 Songs) -> training data with YIN pseudo-labels for the melody head
(D10: 'teacher pitch labels'). Chord labels are unknown (all-zero rows), so only the melody head trains on them.

    python3 pseudo_label_real.py --out ../../data/real8            # ~2 min (YIN)

Split: by file. Test = the two files named in TEST (held out for real_voice_check.py --only), rest train.
Melody label = YIN pitch where confident (CMND < 0.15) else -1 (not scored). Because YIN is a teacher, not truth,
expect the learned head to inherit YIN's octave/semitone slips.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "scripts"))
from singalong import frontend as fe  # noqa: E402
from real_voice_check import FILES, SONGS  # noqa: E402

TEST = ["michellelindemann1", "Twinkle_twinkle_little_star_(vocal)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "data" / "real8")
    ap.add_argument("--secs", type=float, default=130.0)
    a = ap.parse_args()
    from frontend_check import yin
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel in FILES:
        path = SONGS / rel
        if not path.exists():
            continue
        y = fe.decode_audio(path)[: int(a.secs * fe.SR)]
        f0, conf = yin(y, fe.SR, fe.WIN, fe.HOP)
        X = fe.log_bands(y)
        v = fe.voiced(y)
        T = min(len(X), len(f0))
        midi = np.round(69 + 12 * np.log2(np.maximum(f0[:T], 1) / 440.0)).astype(np.int16)
        melody = np.where(conf[:T] & v[:T], midi, -1).astype(np.int16)
        rid = Path(rel).stem[:40].replace(" ", "_")
        np.savez_compressed(a.out / f"{rid}.npz", X=X[:T].astype(np.float16), voiced=v[:T].astype(np.uint8),
                            melody=melody, chord_pcs=np.zeros((T, 12), np.uint8), chord_root=np.full(T, -1, np.int8),
                            degree=np.full(T, -1, np.int8), key_tonic=np.full(T, -1, np.int8), octave=0, transpose=0)
        n_lab = int((melody >= 0).sum())
        rows.append(dict(id=rid, status="ok", usable=int(n_lab > 50), split="test" if any(k in rel for k in TEST) else "train",
                         duration_s=round(T / fe.FPS, 1), octave=0, transpose=0, singing=round(n_lab / max(T, 1), 3),
                         fit_in_key=1.0, fit_best=1.0, sung_lo=int(melody[melody >= 0].min()) if n_lab else 0,
                         sung_hi=int(melody[melody >= 0].max()) if n_lab else 0, source=rel))
        print(f"{rid:42s} {T:5d} frames, {n_lab:5d} YIN-labelled ({n_lab / max(T, 1):.0%}), split {rows[-1]['split']}", flush=True)
    with open(a.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
