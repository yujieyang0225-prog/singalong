#!/usr/bin/env python3
"""Which frequency range and band layout should the SNN input use (D02)? Could the filterbank be learned?

Each layout turns the same 1024-pt FFT (16 kHz, 64 ms Hann, 16 ms hop) into N input channels
(log energy, per-frame normalised to a 40 dB range, as in D02). A *trained* linear readout
(multinomial logistic regression, i.e. one learned layer) then predicts the sung pitch class.
Because the readout is learned, layouts with different spacing are compared fairly, and the
"raw FFT bins" layout shows what a fully learned filterbank could at best add.

Tests
  1. Real recordings (08 Songs), leave-one-recording-out, labels = YIN pitch class on confident frames.
  2. Synthetic voices with random vowel, detune ±50 cents and vibrato, F0 from 73 Hz (D2) to 1 kHz:
     train on one random set, test on another, scored by register. This covers low male voices,
     which our recordings barely contain.

Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 jan/scripts/band_layout_check.py
"""
import numpy as np
from scipy.optimize import minimize

import frontend_check as fc

SR, WIN, NFFT, HOP = 16000, 1024, 1024, 256
FREQS = np.arange(NFFT // 2 + 1) * SR / NFFT


def tri_bank(centres_hz, widths_lo, widths_hi):
    fb = np.zeros((len(centres_hz), len(FREQS)))
    for b, (c, lo, hi) in enumerate(zip(centres_hz, widths_lo, widths_hi)):
        w = np.clip(np.minimum((FREQS - lo) / (c - lo), (hi - FREQS) / (hi - c)), 0, None)
        if w.sum() == 0:  # narrower than a bin: interpolate at the centre
            k = c * NFFT / SR
            w[int(k)] += 1 - (k - int(k))
            w[int(k) + 1] += k - int(k)
        fb[b] = w / w.sum()
    return fb


def semitone_bank(lo_midi, n, per_semitone=1):
    m = lo_midi + np.arange(n) / per_semitone
    step = 1 / per_semitone
    return tri_bank(fc.hz(m), fc.hz(m - step), fc.hz(m + step))


def erb_bank(n, f_lo, f_hi):
    erb = lambda f: 21.4 * np.log10(1 + 0.00437 * f)  # noqa: E731  ERB-number scale (Glasberg & Moore 1990)
    inv = lambda e: (10 ** (e / 21.4) - 1) / 0.00437  # noqa: E731
    e = np.linspace(erb(f_lo), erb(f_hi), n + 2)
    f = inv(e)
    return tri_bank(f[1:-1], f[:-2], f[2:])


def bins_bank(f_lo, f_hi):
    idx = np.where((FREQS >= f_lo) & (FREQS <= f_hi))[0]
    fb = np.zeros((len(idx), len(FREQS)))
    fb[np.arange(len(idx)), idx] = 1
    return fb


LAYOUTS = {  # name: (filterbank, channels used on the chip)
    "A. plan: 64 semitone, 131 Hz–5.0 kHz (MIDI 48–111)": semitone_bank(48, 64),
    "B. 64 semitone, 65 Hz–2.5 kHz (MIDI 36–99)": semitone_bank(36, 64),
    "C. 48 semitone, 82 Hz–1.25 kHz (MIDI 40–87)": semitone_bank(40, 48),
    "D. 32 semitone, 98–587 Hz (MIDI 43–74, ~F0 only)": semitone_bank(43, 32),
    "E. 96 quarter-tone, 82 Hz–1.25 kHz (MIDI 40–87.5)": semitone_bank(40, 96, 2),
    "F. 64 ERB-spaced (cochlea-like), 80 Hz–4 kHz": erb_bank(64, 80, 4000),
    "G. 32 ERB-spaced, 80 Hz–2 kHz": erb_bank(32, 80, 2000),
    "H. raw FFT bins 80 Hz–2 kHz (123 ch, 'learned filterbank' upper bound)": bins_bank(80, 2000),
}
CHROMA_FROM = "C. 48 semitone, 82 Hz–1.25 kHz (MIDI 40–87)"


def features(x, fb):
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP] * np.hanning(WIN)
    spec = np.abs(np.fft.rfft(frames, NFFT)) ** 2
    e = np.log2(spec @ fb.T + 1e-10)
    return np.maximum(e - e.max(axis=1, keepdims=True), -13.3) / 13.3 + 1  # 0..1


def chroma(f48):
    return np.stack([f48[:, (k - 40) % 12::12].max(axis=1) for k in range(40, 52)], axis=1)


def fit_predict(Xtr, ytr, Xte, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A, B = np.hstack([A, np.ones((len(A), 1))]), np.hstack([B, np.ones((len(B), 1))])
    Y = np.eye(12)[ytr]
    d = A.shape[1]

    def f(w):
        W = w.reshape(d, 12)
        Z = A @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        loss = -np.mean(np.sum(Y * np.log(P + 1e-12), 1)) + lam * np.sum(W[:-1] ** 2)
        g = A.T @ (P - Y) / len(A)
        g[:-1] += 2 * lam * W[:-1]
        return loss, g.ravel()

    w = minimize(f, np.zeros(d * 12), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x
    return (B @ w.reshape(d, 12)).argmax(1)


def synth_voice(midi, rng, dur=0.5):
    vowels = [(700, 1200, 2600), (400, 2000, 2550), (300, 2300, 3000), (500, 900, 2500), (350, 800, 2400)]  # a e i o u (approx.)
    F = vowels[rng.integers(len(vowels))]
    t = np.arange(int(dur * SR)) / SR
    f0 = fc.hz(midi + rng.uniform(-0.5, 0.5)) * 2 ** (rng.uniform(0, 0.3) / 12 * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = np.zeros_like(t)
    for h in range(1, 80):
        fh = h * f0.mean()
        if fh > 7000:
            break
        env = sum(np.exp(-((fh - Fk) / (90 + 0.06 * Fk)) ** 2) for Fk in F)
        x += (1 / h) * (0.1 + env) * np.sin(h * ph)
    x /= np.abs(x).max()
    return x + rng.normal(0, 10 ** (-rng.uniform(10, 30) / 20) * x.std(), len(x))


def main():
    # ---------- 1. real recordings
    files = sorted(p for p in fc.SONGS.glob("*/audio/*") if p.suffix in (".wav", ".ogg", ".flac", ".mp3", ".webm"))
    rec = []
    for f in files:
        x = fc.load(f, SR)[: SR * 60]
        f0, conf = fc.yin(x, SR, WIN, HOP)
        if conf.sum() < 200:
            continue
        pc = (np.round(69 + 12 * np.log2(f0 / 440)).astype(int)) % 12
        feats = {k: features(x, fb) for k, fb in LAYOUTS.items()}
        n = min(len(conf), *(len(v) for v in feats.values()))
        rec.append((f.stem, conf[:n], pc[:n], {k: v[:n] for k, v in feats.items()}))
    names = list(LAYOUTS) + ["I. 12 chroma (layout C folded to pitch classes)"]
    print(f"### 1. Real recordings ({len(rec)} recordings, leave-one-recording-out, one learned linear layer)")
    print("| Layout | channels | pitch-class accuracy |")
    print("|---|---|---|")
    for name in names:
        ok = tot = 0
        for i in range(len(rec)):
            def X(r):
                v = r[3][CHROMA_FROM] if name.startswith("I.") else r[3][name]
                return chroma(v) if name.startswith("I.") else v
            tr = [r for j, r in enumerate(rec) if j != i]
            Xtr = np.vstack([X(r)[r[1]] for r in tr])
            ytr = np.concatenate([r[2][r[1]] for r in tr])
            te = rec[i]
            pred = fit_predict(Xtr, ytr, X(te)[te[1]])
            ok += np.sum(pred == te[2][te[1]])
            tot += te[1].sum()
        ch = 12 if name.startswith("I.") else LAYOUTS[name].shape[0]
        print(f"| {name} | {ch} | {ok / tot:.0%} |")

    # ---------- 2. synthetic voices by register
    print("\n### 2. Synthetic voices (random vowel, ±50 cent detune, vibrato, 10–30 dB SNR), trained on set 1, tested on set 2")
    print("| Layout | F0 73–130 Hz (low male) | 131–261 Hz | 262–1047 Hz |")
    print("|---|---|---|---|")
    sets = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        notes = [(m, synth_voice(m, rng)) for m in rng.integers(38, 85, 900)]
        sets.append(notes)
    for name in names:
        def X(x):
            v = features(x, LAYOUTS[CHROMA_FROM] if name.startswith("I.") else LAYOUTS[name])
            return chroma(v) if name.startswith("I.") else v
        Xtr, ytr = [], []
        for m, x in sets[0]:
            v = X(x)[2:-2]
            Xtr.append(v)
            ytr += [m % 12] * len(v)
        Xte, yte, reg = [], [], []
        for m, x in sets[1]:
            v = X(x)[2:-2]
            Xte.append(v)
            yte += [m % 12] * len(v)
            reg += [m] * len(v)
        pred = fit_predict(np.vstack(Xtr), np.array(ytr), np.vstack(Xte))
        yte, reg = np.array(yte), np.array(reg)
        acc = [np.mean(pred[(reg >= a) & (reg < b)] == yte[(reg >= a) & (reg < b)]) for a, b in ((38, 48), (48, 60), (60, 85))]
        print(f"| {name} | " + " | ".join(f"{a:.0%}" for a in acc) + " |")


if __name__ == "__main__":
    main()
