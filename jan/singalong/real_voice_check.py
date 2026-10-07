#!/usr/bin/env python3
"""Does the melody head work on REAL voices? Compare the SNN's pitch class with an independent YIN pitch tracker
on the a cappella recordings in Hackathon/08 Songs (Twinkle, Happy Birthday, Frère Jacques), which the network
never saw (it was trained on synthetic Amazing Grace singers).

    python3 real_voice_check.py --evidence snn:pretrained/synth200_rnn80_int8.npz [--evidence baseline]

Per file: agreement of argmax pitch class with YIN on frames where YIN is confident, for the SNN melody head
and for the harmonic-sum baseline; also 'within ±1 semitone' because YIN itself is off by a semitone on
glides. Honest caveats: YIN is a reference, not ground truth (the vault's front-end check used it the same
way, 2026-10-06); children + clapping (Frère Jacques field recording) is hard for both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "scripts"))                  # jan/scripts for frontend_check.yin
from singalong import frontend as fe  # noqa: E402
import live  # noqa: E402

SONGS = HERE.parents[1] / "Hackathon" / "08 Songs" / "sources"
FILES = [
    "02_Twinkle/audio/Twinkle_twinkle_little_star_(vocal).ogg",
    "02_Twinkle/audio/Twinkle_Twinkle_Little_Star_-_sung_with_full_lyrics.ogg",
    "02_Twinkle/audio/Twinkle_Twinkle_Little_Star_plain.ogg",
    "01_Happy_Birthday/audio/624721__asiilanna__happy-birthday.wav",
    "01_Happy_Birthday/audio/521217__michellelindemann1__happy-birthday-song.wav",
    "01_Happy_Birthday/audio/574342__wkg-518__happy-birthday-to-you.mp3",
    "03_Frere_Jacques/audio/Frere_Jacques_-_canon.ogg",
    "03_Frere_Jacques/audio/575367__trp__children-singing-small-audience-clapping-library-10.flac",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", action="append", default=None)
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--only", action="append", help="substring filter on file names (e.g. held-out test files)")
    a = ap.parse_args()
    specs = a.evidence or ["snn:" + str(HERE / "pretrained" / "synth200_rnn80_int8.npz"), "baseline"]
    from frontend_check import yin
    rows, counts = [], []
    print(f"{'file':58s} {'YIN frames':>10s} " + " ".join(f"{s.split(':')[0]:>14s} ±1st" for s in specs))
    for rel in FILES:
        path = SONGS / rel
        if not path.exists():
            continue
        if a.only and not any(k in rel for k in a.only):
            continue
        y = fe.decode_audio(path)[: int(a.secs * fe.SR)]
        f0, conf = yin(y, fe.SR, fe.WIN, fe.HOP)
        X = fe.normalise(fe.log_bands(y), fe.voiced(y))
        T = min(len(X), len(f0))
        ref_pc = np.round(69 + 12 * np.log2(f0[:T] / 440.0)) % 12
        ok = conf[:T] & (X[:T].max(1) > 0)
        line = f"{rel.split('/')[-1][:58]:58s} {int(ok.sum()):10d} "
        res = {}
        for spec in specs:
            evid = live.make_evidence(spec)
            pred = np.zeros(T, int)
            for t in range(T):
                m, c = evid(X[t], bool(X[t].max() > 0))
                e = m if m is not None else c
                pred[t] = int(np.argmax(e)) if e is not None and np.any(e > 0) else -1
            exact = float(np.mean(pred[ok] == ref_pc[ok]))
            near = float(np.mean(np.minimum((pred[ok] - ref_pc[ok]) % 12, (ref_pc[ok] - pred[ok]) % 12) <= 1))
            res[spec] = (exact, near)
            line += f"{exact:14.2f} {near:4.2f} "
        print(line, flush=True)
        rows.append(res); counts.append(int(ok.sum()))
    if rows:
        good = [r for r, n in zip(rows, counts) if n >= 50]            # files with enough confident frames
        print(f"{'MEAN (files with >= 50 YIN frames)':58s} {len(good):10d} "
              + " ".join(f"{np.mean([r[s][0] for r in good]):14.2f} {np.mean([r[s][1] for r in good]):4.2f}" for s in specs))
        print("chance: 0.08 exact, 0.25 within ±1 semitone")


if __name__ == "__main__":
    main()
