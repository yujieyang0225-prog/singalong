#!/usr/bin/env python3
"""Does the melody note survive the proposed front-end (D02)?

Front-end: Hann window -> HW-sized FFT (<=1024) -> 64 semitone bands (MIDI 48-111 = 131 Hz-4.98 kHz) -> log.
Readout: a fixed harmonic-sum over the bands (no learning) -> pitch class. This is a LOWER BOUND
for what a trained network could read from the same bands.

Tests
  1. Synthetic vowel-like tones with known F0 (MIDI 40-84), vibrato + noise.
  2. Real recordings from 08 Songs vs a YIN reference pitch (confident frames only).
Configs compare 16 kHz (our input) with the old 8 kHz plan and a shorter window.

Run from repo root:  PYTHONPATH=.pydeps python3 jan/scripts/frontend_check.py
"""
import subprocess
from pathlib import Path

import numpy as np

SONGS = Path(__file__).resolve().parents[2] / "Hackathon" / "08 Songs" / "sources"
CONFIGS = {  # name: (sample rate, window samples, FFT size, hop samples)
    "16k win1024 (plan)": (16000, 1024, 1024, 256),
    "16k win512 (low-latency)": (16000, 512, 1024, 256),
    "8k win512→1024 (old plan)": (8000, 512, 1024, 128),
    "8k win1024 (128 ms)": (8000, 1024, 1024, 128),
}
MIDI_LO, N_BANDS = 48, 64
HARM = [(1, 1.0), (2, 0.8), (3, 0.6), (4, 0.5), (5, 0.4), (6, 0.3)]


def hz(m):
    return 440.0 * 2 ** ((np.asarray(m) - 69) / 12)


def filterbank(sr, nfft):
    freqs = np.arange(nfft // 2 + 1) * sr / nfft
    fb = np.zeros((N_BANDS, len(freqs)))
    for b in range(N_BANDS):
        m = MIDI_LO + b
        lo, c, hi = hz(m - 1), hz(m), hz(m + 1)
        if hi >= sr / 2:  # band above Nyquist (8 kHz configs): leave empty
            continue
        w = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
        if w.sum() == 0:  # band narrower than a bin: linear interpolation at the centre
            k = c * nfft / sr
            w[int(k)] += 1 - (k - int(k))
            w[int(k) + 1] += k - int(k)
        fb[b] = w / w.sum()
    fb[fb.sum(axis=1) == 0, 0] = 1e-6  # keep empty bands finite
    return fb


def bands(x, sr, win, nfft, hop):
    fb, w = filterbank(sr, nfft), np.hanning(win)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop] * w
    spec = np.abs(np.fft.rfft(frames, nfft)) ** 2
    e = np.log2(spec @ fb.T + 1e-10)
    return np.maximum(e - e.max(axis=1, keepdims=True), -13.3)  # 40 dB range


def pitch_from_bands(e):
    """Harmonic sum over semitone bands; candidates MIDI 36-84 (65 Hz - 1 kHz)."""
    cands = np.arange(36, 85)
    score = np.zeros((len(e), len(cands)))
    for h, wgt in HARM:
        off = int(round(12 * np.log2(h)))
        idx = cands + off - MIDI_LO
        ok = (idx >= 0) & (idx < N_BANDS)
        score[:, ok] += wgt * (e[:, idx[ok]] + 13.3)
    return cands[score.argmax(axis=1)]


def yin(x, sr, win, hop, fmin=70, fmax=1000, thr=0.15):
    taus = np.arange(int(sr / fmax), int(sr / fmin))
    frames = np.lib.stride_tricks.sliding_window_view(x, win + taus[-1] + 1)[::hop]
    f0, conf = [], []
    for fr in frames:
        a = fr[:win]
        d = np.array([np.sum((a - fr[t:t + win]) ** 2) for t in range(taus[-1] + 1)])
        cmnd = d[1:] * np.arange(1, len(d)) / np.maximum(np.cumsum(d[1:]), 1e-12)
        cm = cmnd[taus[0] - 1:]
        below = np.where(cm < thr)[0]
        i = below[0] if len(below) else cm.argmin()
        while i + 1 < len(cm) and cm[i + 1] < cm[i]:
            i += 1
        f0.append(sr / (taus[0] + i)), conf.append(cm[i] < thr)
    return np.array(f0), np.array(conf)


def synth(midi, sr, dur=0.6, rng=np.random.default_rng(0)):
    t = np.arange(int(dur * sr)) / sr
    f0 = hz(midi) * 2 ** (0.2 / 12 * np.sin(2 * np.pi * 5.5 * t))  # ±20 cent vibrato
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros_like(t)
    for h in range(1, 60):
        fh = h * hz(midi)
        if fh > min(4000, sr / 2 - 200):
            break
        env = sum(np.exp(-((fh - F) / B) ** 2) for F, B in ((700, 130), (1200, 150), (2600, 250)))  # /a/ formants
        x += (1 / h) * (0.15 + env) * np.sin(h * ph)
    x /= np.abs(x).max()
    return x + rng.normal(0, 0.1 * x.std(), len(x))  # 20 dB SNR


def load(path, sr):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def main():
    print("### 1. Synthetic voice-like tones (pitch-class accuracy of the fixed harmonic-sum readout)")
    print("| Config | latency (window) | F0 < C4 (MIDI 40-59) | F0 ≥ C4 (MIDI 60-84) |")
    print("|---|---|---|---|")
    for name, (sr, win, nfft, hop) in CONFIGS.items():
        acc = {}
        for m in range(40, 85):
            p = pitch_from_bands(bands(synth(m, sr), sr, win, nfft, hop))
            acc[m] = np.mean(p % 12 == m % 12)
        lo = np.mean([acc[m] for m in range(40, 60)])
        hi = np.mean([acc[m] for m in range(60, 85)])
        print(f"| {name} | {1000 * win / sr:.0f} ms | {lo:.0%} | {hi:.0%} |")

    print("\n### 2. Real recordings vs YIN reference (frames where YIN is confident)")
    files = sorted(p for p in SONGS.glob("*/audio/*") if p.suffix in (".wav", ".ogg", ".flac", ".mp3", ".webm"))
    print("| Recording | voiced frames | " + " | ".join(CONFIGS) + " |")
    print("|---|---|" + "---|" * len(CONFIGS))
    tot = {k: [0, 0] for k in CONFIGS}
    for f in files:
        x16 = load(f, 16000)[: 16000 * 60]
        f0, conf = yin(x16, 16000, 1024, 256)
        ref = np.round(69 + 12 * np.log2(f0 / 440))
        row = []
        for name, (sr, win, nfft, hop) in CONFIGS.items():
            x = x16 if sr == 16000 else load(f, sr)[: sr * 60]
            p = pitch_from_bands(bands(x, sr, win, nfft, hop))
            # align frames by time (both hops = 16 ms; account for window-centre offset)
            shift = (win - 1024 * sr // 16000) // 2 // hop
            n = min(len(p) - max(shift, 0), len(ref))
            pp, rr, cc = p[max(shift, 0):max(shift, 0) + n], ref[:n], conf[:n]
            ok = int(np.sum((pp[cc] % 12) == (rr[cc] % 12)))
            row.append(f"{ok / max(1, cc.sum()):.0%}")
            tot[name][0] += ok
            tot[name][1] += int(cc.sum())
        print(f"| {f.parent.parent.name[:2]} {f.stem[:38]} | {int(conf.sum())} | " + " | ".join(row) + " |")
    print("| **all** | " + str(tot[next(iter(tot))][1]) + " | " + " | ".join(f"**{a / max(1, n):.0%}**" for a, n in tot.values()) + " |")


if __name__ == "__main__":
    main()
