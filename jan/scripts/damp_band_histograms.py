#!/usr/bin/env python3
"""Which notes do DAMP Amazing Grace singers actually sing? One histogram per recording.

Input = the D02 front-end (16 kHz, 1024 Hann, hop 256 = 16 ms, 48 semitone bands MIDI 40-87,
82 Hz - 1.25 kHz), via features() from band_layout_check.py. For every singing frame (energy
>= 15 dB above the recording's noise floor) the strongest band is counted. The MIDI melody
(track 'Pitch', time per note) is overlaid, shifted to the octave that fits the singer best.
No time alignment is needed for this plot.

Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 jan/scripts/damp_band_histograms.py [--n 10]
"""
import argparse
import random
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np

from band_layout_check import HOP, SR, WIN, features, semitone_bank

REPO = Path(__file__).resolve().parents[2]
DAMP = REPO / "data" / "DAMP-S-AG"
LO, N = 40, 48
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def decode(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


def melody_hist():
    """Time (ticks) per MIDI pitch in the 'Pitch' track."""
    mf = mido.MidiFile(DAMP / "amazing_grace.midi")
    tr = next(t for t in mf.tracks if t.name.strip() == "Pitch")
    h, on, t = np.zeros(128), {}, 0
    for msg in tr:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = t
        elif msg.type in ("note_off", "note_on") and msg.note in on:
            h[msg.note] += t - on.pop(msg.note)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "Hackathon" / "Attachments" / "damp-band-histograms.png")
    a = ap.parse_args()

    fb = semitone_bank(LO, N)
    mel = melody_hist()
    files = sorted((DAMP / "vocals" / "amazing_grace").glob("*.m4a"))
    random.Random(a.seed).shuffle(files)

    fig, axes = plt.subplots(2, (a.n + 1) // 2, figsize=(4 * ((a.n + 1) // 2), 6.5), sharex=True)
    x = np.arange(LO, LO + N)
    done = 0
    for f in files:
        if done == a.n:
            break
        try:
            y = decode(f)
        except subprocess.CalledProcessError:
            print(f"skip {f.stem} (decode error)")
            continue
        frames = np.lib.stride_tricks.sliding_window_view(y, WIN)[::HOP]
        db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
        sing = db >= np.percentile(db, 10) + 15
        if sing.mean() < 0.1:
            print(f"skip {f.stem} (almost silent)")
            continue
        feat = features(y, fb)[sing]
        hist = np.bincount(feat.argmax(1), minlength=N) / len(feat) * 100

        # melody shifted down by whole octaves; pick the shift that best matches the histogram
        best = max(range(0, 37, 12), key=lambda s: np.dot(hist, mel[x + s]))
        m = mel[x + best]
        m = m / m.sum() * 100 if m.sum() else m

        ax = axes.flat[done]
        ax.bar(x, hist, width=0.9, color="#4a7ab5", label="strongest band (sung)")
        ax.step(x, m, where="mid", color="#e07b39", lw=1.5, label="MIDI melody")
        ax.set_title(f"{f.stem}\nsinging {sing.mean():.0%} · melody −{best // 12} oct", fontsize=9)
        ax.set_xticks(range(LO, LO + N, 12), [f"{NAMES[k % 12]}{k // 12 - 1}" for k in range(LO, LO + N, 12)])
        print(f"{f.stem:<22} singing={sing.mean():.0%} melody shift=-{best // 12} oct "
              f"peak band={NAMES[x[hist.argmax()] % 12]}{x[hist.argmax()] // 12 - 1}")
        done += 1
    axes.flat[0].legend(fontsize=8)
    fig.supylabel("% of singing frames")
    fig.supxlabel("band (MIDI 40–87, 82 Hz – 1.25 kHz)")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
