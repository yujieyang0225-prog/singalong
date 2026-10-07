"""Precomputed recordings (data/damp100 from jan/prep/precompute_damp.py, data/synth* from singalong.synth)
-> 2 s crops for training.

X: [N, T, 48] integer-valued 0..100 = 48 semitone bands (D02), per-frame normalised (40 dB range),
   silence-gated, x100 for the IF encoder (Talamo trap: scale before IF).
Y: [N, T, 14] = 12 chord pitch classes (multi-hot)
             + 1 mask channel (1 = frame is scored: inside the sung part of the song and after the burn-in)
             + 1 melody class (0..11 = pitch class of the sung note, 12 = no note / not singing)
   Older code that expects [.., 13] still works: the first 13 channels are unchanged.

Label sources, per recording: its own 'melody'/'chord_pcs' arrays if the npz has them (synthetic takes, other
songs), else the dataset's reference.npz rotated by the singer's 'transpose' (DAMP, one song for all singers).

Augmentation (D11) happens here on whole recordings before cropping: see singalong.augment.
No talamo import at module level, so the host app and the torch trainer can use this without the SDK.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch

from . import frontend as fe

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("SINGALONG_DATA", REPO / "data" / "damp100"))
FPS = fe.FPS
NAMES = fe.NAMES
N_CH = 14                       # channels of Y
MEL_CH, MASK_CH = 13, 12        # Y[..., MASK_CH] = scored mask, Y[..., MEL_CH] = melody class
NO_NOTE = 12


def _reference(data):
    p = Path(data) / "reference.npz"
    if not p.exists():
        return None
    ref = np.load(p)
    return {k: ref[k] for k in ref.files}


def load_recordings(split, data=None, datasets=None):
    """Whole recordings of one split ('train' / 'val' / 'test') as a list of dicts with
    X [T,48] (0..100), Y [T,12] chord pcs, MEL [T] melody class 0..12, M [T] = 'sung part' mask, id.

    data: one dataset folder; datasets: several (concatenated). Default: $SINGALONG_DATA or data/damp100.
    """
    folders = [Path(d) for d in (datasets or [data or DATA])]
    recs = []
    for folder in folders:
        ref = _reference(folder)
        for r in csv.DictReader(open(folder / "manifest.csv")):
            if r["split"] != split or r.get("usable", "1") not in ("1", "True"):
                continue
            d = np.load(folder / f"{r['id']}.npz")
            X = d["X"].astype(np.float32)
            if "melody" in d.files:
                mel, chords = d["melody"], d["chord_pcs"]
            else:
                mel, chords = ref["melody"], np.roll(ref["chord_pcs"], int(d["transpose"]), axis=1)
                mel = np.where(mel >= 0, mel + 12 * int(d["octave"]) + int(d["transpose"]), -1)
            T = min(len(X), len(mel))
            X, mel, chords, v = X[:T], mel[:T], chords[:T], d["voiced"][:T].astype(bool)
            on = np.flatnonzero(mel >= 0)
            sung = np.zeros(T, bool)
            if len(on):
                sung[on[0]:on[-1] + 1] = True
            mel_cls = np.where((mel >= 0) & v, mel % 12, NO_NOTE).astype(np.int64)
            recs.append(dict(id=r["id"], X_log=X, voiced=v, X=fe.normalise(X, v), Y=chords.astype(np.float32),
                             MEL=mel_cls, M=sung, folder=str(folder)))
    return recs


def crops(recs, crop=125, stride=125, burn=31, start_offset=0):
    """Cut recordings into crops (sung part only). Returns X [N,T,48], Y [N,T,14]."""
    X, Y = [], []
    for r in recs:
        idx = np.flatnonzero(r["M"])
        if len(idx) < crop:
            continue
        for t0 in range(max(idx[0] - crop // 4, 0) + start_offset, idx[-1] - crop, stride):
            m = r["M"][t0:t0 + crop].astype(np.float32).copy()
            m[:burn] = 0
            X.append(r["X"][t0:t0 + crop])
            Y.append(np.concatenate([r["Y"][t0:t0 + crop], m[:, None], r["MEL"][t0:t0 + crop, None].astype(np.float32)], 1))
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(Y))


def dense_dataset(X, Y):
    """Talamo DenseDataset (imported lazily: only the Talamo trainer needs the SDK)."""
    import talamo
    return talamo.utils.data.DenseDataset(X, Y, np.arange(len(X)))


def baseline(train_recs, Y_val):
    """Best constant output (most common chord pitch-class set in train), scored on val crops."""
    from .metrics import scores
    tr = np.concatenate([r["Y"][r["M"]] for r in train_recs])
    sets, counts = np.unique(tr, axis=0, return_counts=True)
    best = torch.tensor(sets[counts.argmax()]).float()
    s = scores(best.expand(*Y_val.shape[:2], 12).clone(), Y_val)
    return {**s, "chord": " ".join(NAMES[i] for i in np.flatnonzero(sets[counts.argmax()]))}
