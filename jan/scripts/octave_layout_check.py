#!/usr/bin/env python3
"""Limit the input to ~3 octaves? How to encode octaves into 32 / 48 / 64 lines? How much context does the SNN need?

Part 1 (audio, D02): same test as band_layout_check.py (one learned linear layer -> sung pitch class; 16 real
recordings leave-one-recording-out + synthetic voices by register), for octave-limited and octave-folded layouts.
All layouts are cut from one wide semitone bank (MIDI 36-99) so they share everything except the channel choice.

Part 2 (symbolic, D20): at every melody onset of the 6 harmonised songs, predict the arranger's chord from the
melody heard in the last W beats (duration-weighted scale-degree histogram), leave-one-song-out. Shows how much
*short* melody context a chord recommendation needs.

YIN pitch tracks are cached in .cache/yin/ (gitignored).
Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 -u jan/scripts/octave_layout_check.py
"""
import math
from pathlib import Path

import mido
import numpy as np

import analyze_song_harmony as ash
import band_layout_check as bl
import frontend_check as fc

SR, WIN, HOP = 16000, 1024, 256
LO, HI = 36, 99  # wide bank C2..D#7
CACHE = Path(__file__).resolve().parents[2] / ".cache" / "yin"
WIDE = bl.semitone_bank(LO, HI - LO + 1)


def cols(m_lo, m_hi):
    return list(range(m_lo - LO, m_hi - LO + 1))


def chroma(F, m_lo, m_hi):
    c = cols(m_lo, m_hi)
    return np.stack([F[:, [k for k in c if (k + LO) % 12 == p]].max(1) for p in range(12)], 1)


def registers(F, edges):
    return np.stack([F[:, cols(a, b - 1)].max(1) for a, b in zip(edges[:-1], edges[1:])], 1)


LAYOUTS = {  # name: (lines, function of the wide feature matrix)
    "C. 48 semitone, 4 oct, 82 Hz–1.25 kHz (reference)": lambda F: F[:, cols(40, 87)],
    "O1. 36 semitone, 3 oct low, 82–622 Hz (E2–D#5)": lambda F: F[:, cols(40, 75)],
    "O2. 36 semitone, 3 oct mid, 131–988 Hz (C3–B5)": lambda F: F[:, cols(48, 83)],
    "O3. 32 semitone (1 sector), 131–784 Hz (C3–G5)": lambda F: F[:, cols(48, 79)],
    "O4. 36 semitone (C3–B5) + 12 chroma folded from 82 Hz–2.5 kHz = 48": lambda F: np.hstack([F[:, cols(48, 83)], chroma(F, 40, 99)]),
    "O5. 12 chroma + 4 octave-register levels = 16": lambda F: np.hstack([chroma(F, 40, 87), registers(F, [40, 52, 64, 76, 88])]),
    "O6. 2 × 12 chroma (82–330 Hz, 330 Hz–1.3 kHz) = 24": lambda F: np.hstack([chroma(F, 40, 63), chroma(F, 64, 87)]),
    "I. 12 chroma, 82 Hz–1.25 kHz": lambda F: chroma(F, 40, 87),
}


def yin_cached(f, x):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (f.parent.parent.name + "__" + f.stem + ".npz")
    if p.exists():
        d = np.load(p)
        return d["f0"], d["conf"]
    f0, conf = fc.yin(x, SR, WIN, HOP)
    np.savez(p, f0=f0, conf=conf)
    return f0, conf


def audio_part():
    files = sorted(p for p in fc.SONGS.glob("*/audio/*") if p.suffix in (".wav", ".ogg", ".flac", ".mp3", ".webm"))
    rec = []
    for f in files:
        x = fc.load(f, SR)[: SR * 60]
        f0, conf = yin_cached(f, x)
        if conf.sum() < 200:
            continue
        F = bl.features(x, WIDE)
        n = min(len(F), len(conf))
        pc = (np.round(69 + 12 * np.log2(f0[:n] / 440)).astype(int)) % 12
        rec.append((conf[:n], pc, F[:n]))
    print(f"### 1a. Real recordings ({len(rec)}, leave-one-recording-out, one learned linear layer)")
    print("| Layout | lines | pitch-class accuracy |\n|---|---|---|")
    for name, fn in LAYOUTS.items():
        ok = tot = 0
        for i in range(len(rec)):
            tr = [r for j, r in enumerate(rec) if j != i]
            pred = bl.fit_predict(np.vstack([fn(r[2])[r[0]] for r in tr]), np.concatenate([r[1][r[0]] for r in tr]),
                                  fn(rec[i][2])[rec[i][0]])
            ok += np.sum(pred == rec[i][1][rec[i][0]])
            tot += rec[i][0].sum()
        print(f"| {name} | {fn(rec[0][2]).shape[1]} | {ok / tot:.0%} |", flush=True)

    print("\n### 1b. Synthetic voices by register (trained on set 1, tested on set 2)")
    print("| Layout | F0 73–130 Hz | 131–261 Hz | 262–1047 Hz |\n|---|---|---|---|")
    sets = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        X, y, reg = [], [], []
        for m in rng.integers(38, 85, 900):
            F = bl.features(bl.synth_voice(m, rng), WIDE)[2:-2]
            X.append(F)
            y += [m % 12] * len(F)
            reg += [m] * len(F)
        sets.append((np.vstack(X), np.array(y), np.array(reg)))
    for name, fn in LAYOUTS.items():
        pred = bl.fit_predict(fn(sets[0][0]), sets[0][1], fn(sets[1][0]))
        yte, reg = sets[1][1], sets[1][2]
        acc = [np.mean(pred[(reg >= a) & (reg < b)] == yte[(reg >= a) & (reg < b)]) for a, b in ((38, 48), (48, 60), (60, 85))]
        print(f"| {name} | " + " | ".join(f"{a:.0%}" for a in acc) + " |", flush=True)


DIATONIC = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]


def song_events(rel, mel_idx):
    """(onset beat, melody notes, per-beat simple chord labels, tonic) for one harmonised MIDI file."""
    m = mido.MidiFile(ash.SRC / rel)
    tpb = m.ticks_per_beat
    tracks = [(i, ash.notes_of(t, tpb)) for i, t in enumerate(m.tracks)]
    drums = {i for i, t in enumerate(m.tracks) for x in t if x.type == "note_on" and x.channel == 9}
    harm = [n for i, ns in tracks if i not in drums for n in ns]
    mel = sorted(dict(tracks)[mel_idx])
    tot = np.zeros(12)
    for s, e, p in harm:
        tot[p % 12] += e - s
    sc = [(np.corrcoef(np.roll(ash.KK_MAJ, k), tot)[0, 1], k) for k in range(12)]
    tonic = max(sc)[1]
    ksig = next((x.key for t in m.tracks for x in t if x.type == "key_signature"), None)
    if ksig and not ksig.endswith("m"):
        tonic = ash.NAMES.index({"Db": "C#", "Gb": "F#"}.get(ksig, ksig))
    end = max(n[1] for n in harm)
    lab = []
    for b in range(int(math.ceil(end))):
        c, low = np.zeros(12), (99, None)
        for s, e, p in harm:
            ov = min(e, b + 1) - max(s, b)
            if ov > 0:
                c[p % 12] += ov
                low = min(low, (p, p % 12))
        if c.sum() == 0:
            lab.append(None)
            continue
        r, q = ash.chord_of(c / np.linalg.norm(c), low[1])
        lab.append(ash.roman(r, q, tonic).replace("7", ""))
    return mel, lab, tonic


def symbolic_part():
    data = [song_events(rel, mi) for _, rel, mi, _ in ash.SONGS]
    windows = [0, 0.5, 1, 2, 4, 8]
    print("\n### 2. Chord from the melody of the last W beats (6 songs, leave-one-song-out, onsets with a diatonic chord)")
    print("| Melody context W | correct chord |\n|---|---|")
    for W in windows:
        Xs, ys = [], []
        for mel, lab, tonic in data:
            X, y = [], []
            for s, e, p in mel:
                b = int(s)
                if b >= len(lab) or lab[b] not in DIATONIC:
                    continue
                h = np.zeros(12)
                for s2, e2, p2 in mel:
                    if W == 0:
                        if s2 == s:
                            h[(p2 - tonic) % 12] += 1
                    else:
                        ov = min(e2, s + 1e-6) - max(s2, s - W)
                        if ov > 0 or s2 == s:
                            h[(p2 - tonic) % 12] += max(ov, 0.25)
                X.append(h / (h.sum() + 1e-9))
                y.append(DIATONIC.index(lab[b]))
            Xs.append(np.array(X))
            ys.append(np.array(y))
        ok = n = 0
        for i in range(len(Xs)):
            pred = bl.fit_predict(np.vstack([Xs[j] for j in range(len(Xs)) if j != i]),
                                  np.concatenate([ys[j] for j in range(len(Xs)) if j != i]), Xs[i], lam=1e-2)
            ok += np.sum(pred == ys[i])
            n += len(ys[i])
        label = "current note only" if W == 0 else f"last {W:g} beat{'s' if W != 1 else ''} (≈ {W * 0.5:g} s at 120 bpm)"
        print(f"| {label} | {ok / n:.0%} ({n} onsets) |", flush=True)


if __name__ == "__main__":
    symbolic_part()
    audio_part()
