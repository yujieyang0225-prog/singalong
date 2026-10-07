#!/usr/bin/env python3
"""Built-in MFCC (as in Innatera's audio example) vs our semitone bands, for reading the sung pitch (D02).

Same test as jan/scripts/band_layout_check.py (one learned linear layer -> sung pitch class; 16 real recordings
leave-one-recording-out + synthetic voices by register). Feature variants, all from the 16 kHz input:
  * our layout C: 48 semitone bands 82 Hz - 1.25 kHz (reference)
  * example-like: 32 MFCC from 128 mel, 512-pt FFT (32 ms)   [example: 22.05 kHz, n_fft 512, n_mels 128, n_mfcc 32]
  * 32 MFCC from 128 mel / 32 MFCC from 64 mel, 1024-pt FFT (64 ms, our window)
  * 64 / 32 log-mel bands (the mel filterbank without the DCT step), 1024-pt FFT
MFCC follows librosa (Slaney mel, power_to_db with top_db 80, DCT-II ortho), which Talamo's MFCC matches
within ~5 % (Talamo MFCC docs / embedded pipeline tutorial).

Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 -u jan/scripts/mfcc_check.py
"""
import numpy as np
from scipy.fft import dct

import band_layout_check as bl
import frontend_check as fc

SR, HOP = 16000, 256


def mel_bank(n_mels, nfft, sr=SR, fmin=0.0, fmax=None):
    """librosa.filters.mel defaults (Slaney mel scale, Slaney area normalisation)."""
    fmax = fmax or sr / 2

    def hz2mel(f):
        f = np.asarray(f, float)
        lin = f / (200 / 3)
        return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27), lin)

    def mel2hz(m):
        m = np.asarray(m, float)
        return np.where(m >= 15, 1000 * np.exp((np.log(6.4) / 27) * (m - 15)), (200 / 3) * m)

    freqs = np.arange(nfft // 2 + 1) * sr / nfft
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
        fb[i] *= 2.0 / (hi - lo)
    return fb


def power_spec(x, win, nfft, offset=0):
    frames = np.lib.stride_tricks.sliding_window_view(x[offset:], win)[::HOP] * np.hanning(win)
    return np.abs(np.fft.rfft(frames, nfft)) ** 2


def log_mel(S, fb, norm_frame=True):
    e = np.log2(S @ fb.T + 1e-10)
    if norm_frame:  # same per-frame normalisation as our bands
        e = np.maximum(e - e.max(axis=1, keepdims=True), -13.3) / 13.3 + 1
    return e


def mfcc(S, fb, n):
    db = 10 * np.log10(S @ fb.T + 1e-10)
    db = np.maximum(db, db.max() - 80)  # librosa power_to_db(top_db=80)
    return dct(db, type=2, norm="ortho", axis=1)[:, :n]


FEATS = {
    "C. ours: 48 semitone bands 82 Hz–1.25 kHz (64 ms)": lambda x: bl.features(x, bl.LAYOUTS["C. 48 semitone, 82 Hz–1.25 kHz (MIDI 40–87)"]),
    "M1. example-like: 32 MFCC from 128 mel, 512-pt FFT (32 ms)": lambda x: mfcc(power_spec(x, 512, 512, 256), mel_bank(128, 512), 32),
    "M2. 32 MFCC from 128 mel, 1024-pt FFT (64 ms)": lambda x: mfcc(power_spec(x, 1024, 1024), mel_bank(128, 1024), 32),
    "M3. 32 MFCC from 64 mel, 1024-pt FFT (64 ms)": lambda x: mfcc(power_spec(x, 1024, 1024), mel_bank(64, 1024), 32),
    "L1. 64 log-mel bands (no DCT), 1024-pt FFT": lambda x: log_mel(power_spec(x, 1024, 1024), mel_bank(64, 1024)),
    "L2. 32 log-mel bands (no DCT), 1024-pt FFT": lambda x: log_mel(power_spec(x, 1024, 1024), mel_bank(32, 1024)),
}


def main():
    files = sorted(p for p in fc.SONGS.glob("*/audio/*") if p.suffix in (".wav", ".ogg", ".flac", ".mp3", ".webm"))
    rec = []
    for f in files:
        x = fc.load(f, SR)[: SR * 60]
        f0, conf = fc.yin(x, SR, 1024, HOP)
        if conf.sum() < 200:
            continue
        pc = (np.round(69 + 12 * np.log2(f0 / 440)).astype(int)) % 12
        feats = {k: fn(x) for k, fn in FEATS.items()}
        n = min(len(conf), *(len(v) for v in feats.values()))
        rec.append((conf[:n], pc[:n], {k: v[:n] for k, v in feats.items()}))
        print("loaded", f.stem[:40], flush=True)
    print(f"\n### 1. Real recordings ({len(rec)}, leave-one-recording-out, one learned linear layer)")
    print("| Features | channels | pitch-class accuracy |\n|---|---|---|")
    for name in FEATS:
        ok = tot = 0
        for i in range(len(rec)):
            tr = [r for j, r in enumerate(rec) if j != i]
            pred = bl.fit_predict(np.vstack([r[2][name][r[0]] for r in tr]), np.concatenate([r[1][r[0]] for r in tr]),
                                  rec[i][2][name][rec[i][0]])
            ok += np.sum(pred == rec[i][1][rec[i][0]])
            tot += rec[i][0].sum()
        print(f"| {name} | {rec[0][2][name].shape[1]} | {ok / tot:.0%} |", flush=True)

    print("\n### 2. Synthetic voices by register (trained on set 1, tested on set 2)")
    print("| Features | F0 73–130 Hz | 131–261 Hz | 262–1047 Hz |\n|---|---|---|---|")
    sets = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        sets.append([(m, bl.synth_voice(m, rng)) for m in rng.integers(38, 85, 900)])
    for name, fn in FEATS.items():
        def stack(notes):
            X, y, reg = [], [], []
            for m, x in notes:
                v = fn(x)[2:-2]
                X.append(v)
                y += [m % 12] * len(v)
                reg += [m] * len(v)
            return np.vstack(X), np.array(y), np.array(reg)
        Xtr, ytr, _ = stack(sets[0])
        Xte, yte, reg = stack(sets[1])
        pred = bl.fit_predict(Xtr, ytr, Xte)
        acc = [np.mean(pred[(reg >= a) & (reg < b)] == yte[(reg >= a) & (reg < b)]) for a, b in ((38, 48), (48, 60), (60, 85))]
        print(f"| {name} | " + " | ".join(f"{a:.0%}" for a in acc) + " |", flush=True)


if __name__ == "__main__":
    main()
