#!/usr/bin/env python3
"""Deterministic harmonic tones with quiet lead-in; generated audio stays in ignored data/."""
import argparse
from pathlib import Path
import wave
import numpy as np

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--seconds', type=float, default=190)
    a = ap.parse_args()
    sr = 16000
    t = np.arange(int(sr * a.seconds)) / sr
    pitches = np.array([60, 64, 67, 69, 67, 64, 62, 60])
    midi = pitches[(np.maximum(0, t - 1) // 2).astype(int) % len(pitches)]
    phase = 2 * np.pi * np.cumsum(440 * 2 ** ((midi - 69) / 12)) / sr
    local = np.maximum(0, t - 1) % 2
    envelope = np.clip(local / .1, 0, 1) * np.clip((1.8 - local) / .1, 0, 1) * (t >= 1)
    y = envelope * (.15 * np.sin(phase) + .07 * np.sin(2 * phase) + .04 * np.sin(3 * phase))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(a.output), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((y * 32767).astype('<i2').tobytes())

if __name__ == '__main__':
    main()
