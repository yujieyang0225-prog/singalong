"""Scores shared by the Talamo trainer, the torch trainer and the host app (D13)."""
from __future__ import annotations

import numpy as np
import torch

from .data import MASK_CH, MEL_CH, NO_NOTE


def scores(rate, Y, thr=0.5):
    """Chroma F1 and exact-chord match of chord rates [N,T,12] against Y [N,T,>=13] on scored frames."""
    m = (Y[..., MASK_CH] > 0) & (Y[..., :12].sum(-1) > 0)    # all-zero chord row = chord unknown (VocalSet), like the loss
    p, y = (rate[m] >= thr).float(), Y[..., :12][m]
    tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()
    return dict(f1=round(float(2 * tp / (2 * tp + fp + fn + 1e-9)), 3),
                exact=round(float((p == y).all(1).float().mean()), 3))


def pitch_scores(mel_rate, Y):
    """Melody head: accuracy of argmax over the 12 melody output rates on scored, sung frames [N,T,12]."""
    if Y.shape[-1] <= MEL_CH:
        return {}
    m = (Y[..., MASK_CH] > 0) & (Y[..., MEL_CH] < NO_NOTE)
    if m.sum() == 0:
        return dict(pitch_acc=float("nan"))
    pred = mel_rate[m].argmax(-1)
    return dict(pitch_acc=round(float((pred == Y[..., MEL_CH][m].long()).float().mean()), 3))


def note_in_chord(chord_pred_pcs, melody_pc):
    """Fraction of sung frames whose melody pitch class is in the predicted chord (what listeners hear)."""
    ok = melody_pc >= 0
    if ok.sum() == 0:
        return float("nan")
    return float(chord_pred_pcs[np.arange(len(melody_pc))[ok], melody_pc[ok]].mean())


def changes_per_minute(chord_ids, fps):
    ch = np.flatnonzero(np.diff(chord_ids) != 0)
    return 60.0 * len(ch) / (len(chord_ids) / fps)
