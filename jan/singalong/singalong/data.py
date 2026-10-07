"""DAMP-S-AG data precomputed by jan/prep/precompute_damp.py -> Talamo DenseDatasets of 2 s crops.

X: [N, T, 48] integer-valued 0..100 = 48 semitone bands (D02), per-frame normalised (40 dB range),
   silence-gated (D02), x100 for the IF encoder (Talamo trap: scale before IF).
Y: [N, T, 13] = 12 chord pitch classes (MIDI 'Chords', rotated by the singer's transposition)
   + 1 mask channel (1 = frame is scored: inside the sung part of the song and after the burn-in).
"""
import csv
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("SINGALONG_DATA", REPO / "data" / "damp100"))
FPS = 62.5
NAMES = "C C# D D# E F F# G G# A A# B".split()


def load_recordings(split, data=DATA):
    """Whole recordings of one split ('train' / 'val' / 'test') as numpy arrays."""
    ref = np.load(data / "reference.npz")
    mel, chords = ref["melody"], ref["chord_pcs"]
    on = np.flatnonzero(mel >= 0)
    sung = np.zeros(len(mel), bool)
    sung[on[0]:on[-1] + 1] = True
    recs = []
    for r in csv.DictReader(open(data / "manifest.csv")):
        if r["split"] != split:
            continue
        d = np.load(data / f"{r['id']}.npz")
        T = min(len(d["X"]), len(mel))
        X = d["X"][:T].astype(np.float32)
        Xn = np.maximum(X - X.max(1, keepdims=True), -13.3) / 13.3 + 1
        Xn *= d["voiced"][:T, None]
        recs.append(dict(id=r["id"], X=np.round(Xn * 100).astype(np.float32),
                         Y=np.roll(chords[:T], int(d["transpose"]), axis=1).astype(np.float32), M=sung[:T]))
    return recs


def crops(recs, crop=125, stride=125, burn=31, start_offset=0):
    """Cut recordings into crops (sung part only). Returns X [N,T,48], Y [N,T,13] (last channel = mask)."""
    X, Y = [], []
    for r in recs:
        idx = np.flatnonzero(r["M"])
        for t0 in range(max(idx[0] - crop // 4, 0) + start_offset, idx[-1] - crop, stride):
            m = r["M"][t0:t0 + crop].astype(np.float32).copy()
            m[:burn] = 0
            X.append(r["X"][t0:t0 + crop])
            Y.append(np.concatenate([r["Y"][t0:t0 + crop], m[:, None]], 1))
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(Y))


def dense_dataset(X, Y):
    import talamo
    return talamo.utils.data.DenseDataset(X, Y, np.arange(len(X)))


def baseline(train_recs, Y_val):
    """Best constant output (most common chord pitch-class set in train), scored on val crops."""
    from .model import scores
    tr = np.concatenate([r["Y"][r["M"]] for r in train_recs])
    sets, counts = np.unique(tr, axis=0, return_counts=True)
    best = torch.tensor(sets[counts.argmax()]).float()
    s = scores(best.expand_as(Y_val[..., :12]).clone(), Y_val)
    return {**s, "chord": " ".join(NAMES[i] for i in np.flatnonzero(sets[counts.argmax()]))}
