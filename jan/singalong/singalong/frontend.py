"""Audio front-end (D02), the one definition shared by data prep, training, the live host app and the C kernel.

    16 kHz · 1024-point Hann window (64 ms) · hop 256 (16 ms, 62.5 frames/s)
    48 semitone bands, MIDI 40..87 (82 Hz .. 1.25 kHz), triangular on the FFT bins, each row summing to 1
    log2 band energy -> per-frame normalisation (40 dB = 13.3 octaves range) -> voicing gate -> x100 -> int

Numerically identical to jan/prep/precompute_damp.py (which took it from jan/scripts/band_layout_check.py),
so a model trained on data/damp100 sees the same numbers from this module. kernels_c/bands/ is the C port;
tests/test_frontend.py checks the two agree.

Offline:  X = log_bands(y);  Xn = normalise(X, voiced(y))          -> [T, 48] ints 0..100
Online:   fe = StreamingFrontEnd();  for frame in fe.feed(pcm_chunk): ...  (frame = 48 ints + level in dB)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR, WIN, HOP, NFFT = 16000, 1024, 256, 1024
LO, NB = 40, 48                     # first band MIDI 40 (E2, 82 Hz), 48 bands -> MIDI 87 (D#6, 1.245 kHz)
RANGE_OCT = 13.3                    # 40 dB dynamic range per frame, in octaves (log2 units)
GATE_DB = 15.0                      # a frame is 'singing' if its level is >= noise floor + 15 dB
FPS = SR / HOP
FREQS = np.arange(NFFT // 2 + 1) * SR / NFFT
NAMES = "C C# D D# E F F# G G# A A# B".split()


def hz(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def tri_bank(centres_hz, lo_hz, hi_hz):
    """Triangular filters on the FFT bins, each normalised to sum 1 (band_layout_check.tri_bank)."""
    fb = np.zeros((len(centres_hz), len(FREQS)))
    for b, (c, lo, hi) in enumerate(zip(centres_hz, lo_hz, hi_hz)):
        w = np.clip(np.minimum((FREQS - lo) / (c - lo), (hi - FREQS) / (hi - c)), 0, None)
        if w.sum() == 0:            # band narrower than a bin: interpolate between the two nearest bins
            k = c * NFFT / SR
            w[int(k)] += 1 - (k - int(k))
            w[int(k) + 1] += k - int(k)
        fb[b] = w / w.sum()
    return fb


def semitone_bank(lo_midi=LO, n=NB):
    m = lo_midi + np.arange(n)
    return tri_bank(hz(m), hz(m - 1), hz(m + 1))


FB = semitone_bank()                # [48, 513]
HANN = np.hanning(WIN)              # symmetric Hann, as numpy (and precompute_damp.py) use it


def decode_audio(path, sr=SR):
    """Any audio file -> mono float32 at 16 kHz via ffmpeg."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).copy()


def frames_of(y):
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    return np.lib.stride_tricks.sliding_window_view(y, WIN)[::HOP]


def log_bands(y):
    """float audio -> [T, 48] log2 band energy (not normalised; this is what data/damp100/<id>.npz X holds)."""
    frames = frames_of(y)
    spec = np.abs(np.fft.rfft(frames * HANN, NFFT)) ** 2
    return np.log2(spec @ FB.T + 1e-10).astype(np.float32)


def level_db(y):
    """Frame level in dB (mean square of the raw frame), [T]."""
    frames = frames_of(y)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def voiced(y, gate_db=GATE_DB):
    """Offline voicing gate: level >= 10th percentile + 15 dB (precompute_damp.py)."""
    db = level_db(y)
    return db >= np.percentile(db, 10) + gate_db


def normalise(X, voiced_mask=None, scale=100):
    """log2 bands -> per-frame normalised 0..1 (40 dB range), gated, x100 -> integer-valued float32 [T, 48]."""
    Xn = np.maximum(X - X.max(axis=1, keepdims=True), -RANGE_OCT) / RANGE_OCT + 1
    if voiced_mask is not None:
        Xn = Xn * voiced_mask[:, None]
    return np.round(Xn * scale).astype(np.float32)


def bands_to_chroma(X):
    """[T, 48] -> [T, 12] pitch classes by max over octaves (band k is MIDI LO + k, pitch class (LO + k) % 12)."""
    out = np.zeros((len(X), 12), X.dtype)
    for k in range(NB):
        pc = (LO + k) % 12
        out[:, pc] = np.maximum(out[:, pc], X[:, k])
    return out


class OnlineGate:
    """Streaming replacement for the offline 10th-percentile noise floor.

    The floor follows the level down quickly and up slowly (so a long sung phrase does not raise it),
    and a frame is 'voiced' when level >= floor + gate_db. Starts from the first frame's level.
    """

    def __init__(self, gate_db=GATE_DB, up_db_per_s=1.0, fps=FPS):
        self.gate_db, self.up = gate_db, up_db_per_s / fps
        self.floor = None

    def __call__(self, db):
        if self.floor is None:
            self.floor = db
        elif db < self.floor:
            self.floor = db
        else:
            self.floor = min(self.floor + self.up, db)
        return db >= self.floor + self.gate_db


class StreamingFrontEnd:
    """Feed int16 or float PCM chunks of any length; get one 48-band frame per 256 new samples.

    Each yielded frame is (bands[48] as int-valued float32 0..100, level_db, is_voiced). The first frame is
    produced once 1024 samples have arrived, so the latency is 64 ms + one hop, like the chip's FFT window.
    """

    def __init__(self, gate: OnlineGate | None = None):
        self.buf = np.zeros(0, np.float32)
        self.gate = gate or OnlineGate()

    def feed(self, pcm):
        pcm = np.asarray(pcm)
        if pcm.dtype.kind in "iu":
            pcm = pcm.astype(np.float32) / 32768.0
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        while len(self.buf) >= WIN:
            frame = self.buf[:WIN]
            spec = np.abs(np.fft.rfft(frame * HANN, NFFT)) ** 2
            X = np.log2(spec @ FB.T + 1e-10)
            db = float(10 * np.log10(np.mean(frame ** 2) + 1e-12))
            v = self.gate(db)
            yield normalise(X[None], np.array([v]))[0], db, v
            self.buf = self.buf[HOP:]


def analyse_file(path):
    """Convenience: file -> (X_int [T,48], voiced [T], level_db [T])."""
    y = decode_audio(path)
    v = voiced(y)
    return normalise(log_bands(y), v), v, level_db(y)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        X, v, db = analyse_file(Path(p))
        print(f"{p}: {len(X)} frames, {v.mean():.0%} voiced, level {db.min():.0f}..{db.max():.0f} dB, "
              f"strongest band mean MIDI {LO + X.argmax(1)[v].mean():.1f}")
