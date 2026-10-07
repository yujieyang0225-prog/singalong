#!/usr/bin/env python3
"""Are DAMP Amazing Grace recordings already aligned with the MIDI? Visual check.

Rows: the backing track (reference), then N singers. Each row is the D02 SNN input
(48 semitone bands MIDI 40-87, 16 ms hop) for the first seconds of the song; the MIDI
melody (track 'Pitch') is drawn on top at delay 0, shifted to the singer's octave.

Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 jan/scripts/damp_start_compare.py [--n 6 --secs 40]
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np

from band_layout_check import HOP, SR, features, semitone_bank
from damp_band_histograms import DAMP, LO, N, REPO, decode

FPS = SR / HOP


def melody_notes():
    """(start_s, end_s, pitch) of the 'Pitch' track, using the file's tempo map."""
    mf = mido.MidiFile(DAMP / "amazing_grace.midi")
    tempos = sorted((t, m.tempo) for tr in mf.tracks for t, m in zip(np.cumsum([x.time for x in tr]), tr)
                    if m.type == "set_tempo")
    def sec(tick):
        s, last_t, last_tempo = 0.0, 0, 500000
        for t, tempo in tempos:
            if t >= tick:
                break
            s += mido.tick2second(t - last_t, mf.ticks_per_beat, last_tempo)
            last_t, last_tempo = t, tempo
        return s + mido.tick2second(tick - last_t, mf.ticks_per_beat, last_tempo)
    tr = next(t for t in mf.tracks if t.name.strip() == "Pitch")
    notes, on, t = [], {}, 0
    for m in tr:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            on[m.note] = t
        elif m.type in ("note_off", "note_on") and m.note in on:
            notes.append((sec(on.pop(m.note)), sec(t), m.note))
    return sorted(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--secs", type=float, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "Hackathon" / "Attachments" / "damp-start-compare.png")
    a = ap.parse_args()

    fb = semitone_bank(LO, N)
    notes = melody_notes()
    files = sorted((DAMP / "vocals" / "amazing_grace").glob("*.m4a"))
    random.Random(a.seed).shuffle(files)
    rows = [("backing track (reference)", DAMP / "amazing_grace.m4a")] + [(f.stem, f) for f in files[:a.n]]

    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 1.9 * len(rows)), sharex=True)
    for ax, (name, f) in zip(axes, rows):
        y = decode(f)[: int(a.secs * SR)]
        X = features(y, fb)                      # [frames, 48], 0..1 per frame
        frames = np.lib.stride_tricks.sliding_window_view(y, 1024)[::HOP]
        db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
        if f != DAMP / "amazing_grace.m4a":  # backing track never pauses: no silence gate
            X = X * (db >= np.percentile(db, 10) + 15)[:, None]   # blank silence like the picture
        # singer octave: shift (0..3 oct down) where the melody notes collect most band energy
        def fit(s):
            return sum(X[int(st * FPS):int(en * FPS), p - s - LO].sum()
                       for st, en, p in notes if LO <= p - s < LO + N and st < a.secs)
        shift = max(range(0, 37, 12), key=fit)
        ax.imshow(X.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
                  extent=[0, len(X) / FPS, LO - 0.5, LO + N - 0.5])
        for st, en, p in notes:
            if st < a.secs and LO <= p - shift < LO + N:
                ax.plot([st, min(en, a.secs)], [p - shift] * 2, color="#e8590c", lw=2.2, solid_capstyle="butt")
        ax.set_yticks([40, 52, 64, 76], ["E2", "E3", "E4", "E5"], fontsize=8)
        ax.set_title(f"{name}   (melody drawn −{shift // 12} oct, delay 0)", fontsize=9, loc="left")
        print(f"{name:<26} melody shift -{shift // 12} oct")
    axes[-1].set_xlabel("time (s) from file start · orange = MIDI melody at delay 0 · white = silence")
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
