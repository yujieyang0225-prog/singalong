"""Synthetic singers of Amazing Grace -> training/test data in the data/damp100 format (D10: synthetic bulk).

A 'take' = the lead sheet in a random key, octave (voice type), tempo, with detune, vibrato, slow drift,
short gaps between notes, random vowels, noise. Output per take: X [T,48] log2 bands (float16, not normalised),
voiced [T], melody [T] (MIDI, -1 = rest), chord_pcs [T,12], onsets. manifest.csv has the same columns as the
DAMP prep plus key/bpm, so singalong.data.load_recordings() reads both datasets the same way.

    python3 -m singalong.synth --n 40 --out data/synth40            # ~1 s per take
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from . import amazing_grace as ag
from . import frontend as fe

VOWELS = [(700, 1200, 2600), (400, 2000, 2550), (300, 2300, 3000), (500, 900, 2500), (350, 800, 2400)]
VOICE_OCTAVE = {"bass": 2, "tenor": 3, "alto": 3, "soprano": 4}       # MIDI octave of the tonic (G2..G4)


def synth_note(midi, dur_s, rng, vowel=None, detune_cents=0.0, vib_hz=5.5, vib_cents=20.0, sr=fe.SR):
    """One sung vowel at a MIDI pitch (band_layout_check.synth_voice, made streamable)."""
    F = VOWELS[rng.integers(len(VOWELS))] if vowel is None else vowel
    n = max(int(dur_s * sr), 1)
    t = np.arange(n) / sr
    f0 = fe.hz(midi + detune_cents / 100) * 2 ** (vib_cents / 1200 * np.sin(2 * np.pi * vib_hz * t + rng.uniform(0, 6.3)))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros(n)
    for h in range(1, 80):
        fh = h * f0.mean()
        if fh > 7000:
            break
        env = sum(np.exp(-((fh - Fk) / (90 + 0.06 * Fk)) ** 2) for Fk in F)
        x += (1 / h) * (0.1 + env) * np.sin(h * ph)
    # attack / release 30 ms so note boundaries are not clicks
    a = min(int(0.03 * sr), n // 2)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    env[-a:] *= np.linspace(1, 0, a)
    return x * env


def synth_take(rng, tonic=None, voice=None, bpm=None, snr_db=None, gap=0.08, drift_cents=20.0):
    """Render one take. Returns (audio float32 @16 kHz, timeline dict, meta dict)."""
    tonic = rng.integers(12) if tonic is None else tonic
    voice = voice or rng.choice(list(VOICE_OCTAVE))
    bpm = float(rng.uniform(60, 100)) if bpm is None else bpm
    snr_db = float(rng.uniform(20, 40)) if snr_db is None else snr_db
    lead_s = float(rng.uniform(2.5, 4.0))                               # silence before the pickup (like DAMP's 9 s)
    tl = ag.render_timeline(tonic=tonic, octave=VOICE_OCTAVE[voice], bpm=bpm, fps=fe.FPS, pickup_silence_s=lead_s)
    end_s = tl["notes"][-1][1] + 2.0
    y = np.zeros(int(end_s * fe.SR))
    base_detune = rng.uniform(-30, 30)
    for (s, e, m) in tl["notes"]:
        g = min(gap, (e - s) * 0.3)                                   # breath gap, shorter for short notes
        dur = e - s - g
        drift = drift_cents * (s / end_s) * rng.choice([-1, 1])
        x = synth_note(m, dur, rng, detune_cents=base_detune + drift + rng.uniform(-10, 10),
                       vib_cents=rng.uniform(5, 25), vib_hz=rng.uniform(4.5, 6.5))
        i0 = int(s * fe.SR)
        y[i0:i0 + len(x)] += x * rng.uniform(0.5, 1.0)
    y /= max(np.abs(y).max(), 1e-9)
    noise = rng.normal(0, 1, len(y))
    y = y + noise * (y.std() / (10 ** (snr_db / 20)))
    return y.astype(np.float32), tl, dict(tonic=int(tonic), voice=str(voice), bpm=round(bpm, 1), snr_db=round(snr_db, 1),
                                         detune=round(float(base_detune), 1))


def take_to_arrays(y, tl):
    X = fe.log_bands(y)
    v = fe.voiced(y)
    T = min(len(X), len(tl["melody"]))
    return dict(X=X[:T].astype(np.float16), voiced=v[:T].astype(np.uint8), melody=tl["melody"][:T],
                chord_pcs=tl["chord_pcs"][:T], chord_root=tl["chord_root"][:T], degree=tl["degree"][:T],
                key_tonic=tl["key_tonic"][:T], octave=0, transpose=0)


def make_dataset(out: Path, n=40, seed=0, write_wav=False):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y, tl, meta = synth_take(rng)
        arrs = take_to_arrays(y, tl)
        rid = f"synth{seed:02d}_{i:03d}"
        np.savez_compressed(out / f"{rid}.npz", **arrs)
        if write_wav:
            import wave
            with wave.open(str(out / f"{rid}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(fe.SR)
                w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
        sung = arrs["melody"] >= 0
        rows.append(dict(id=rid, status="ok", usable=1, duration_s=round(len(y) / fe.SR, 1), octave=0, transpose=0,
                         singing=round(float(arrs["voiced"][sung].mean()), 3), fit_in_key=1.0, fit_best=1.0,
                         sung_lo=int(arrs["melody"][sung].min()), sung_hi=int(arrs["melody"][sung].max()), **meta))
    ids = [r["id"] for r in rows]
    random.Random(seed + 1).shuffle(ids)
    split = {r: "test" for r in ids[:max(n // 10, 1)]} | {r: "val" for r in ids[max(n // 10, 1):max(n // 5, 2)]}
    cols = ["id", "status", "usable", "split", "duration_s", "octave", "transpose", "singing", "fit_in_key", "fit_best",
            "sung_lo", "sung_hi", "tonic", "voice", "bpm", "snr_db", "detune"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r["split"] = split.get(r["id"], "train")
            w.writerow(r)
    # a reference.npz for tools that expect one (G major, soprano register); per-take labels override it
    tl = ag.render_timeline()
    np.savez_compressed(out / "reference.npz", melody=tl["melody"], chord_pcs=tl["chord_pcs"], chord_root=tl["chord_root"],
                        degree=tl["degree"], key_tonic=tl["key_tonic"], lo_band=fe.LO, n_bands=fe.NB, sr=fe.SR,
                        hop=fe.HOP, win=fe.WIN, synthetic=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[3] / "data" / "synth40")
    ap.add_argument("--wav", action="store_true", help="also write 16 kHz wav files (for the live demo from file)")
    a = ap.parse_args()
    rows = make_dataset(a.out, a.n, a.seed, a.wav)
    print(f"{len(rows)} takes -> {a.out}  (voices: {sorted(set(r['voice'] for r in rows))})")
