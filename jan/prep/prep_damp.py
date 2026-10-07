#!/usr/bin/env python3
"""Prepare a subset of DAMP-S-AG (Amazing Grace) for SNN training.

For each selected performance:
  1. decode the .m4a to 16 kHz mono and save it as FLAC,
  2. track the sung pitch (pYIN, 10 ms frames),
  3. align the singer to the reference melody (MIDI track 'Pitch'):
     global lag + transposition search, then banded DTW so the labels follow
     the singer's *own* timing (rubato, rushing, dragging), not the backing track,
  4. write frame labels in singer time (10 ms): melody pitch in the singer's octave,
     pitch class, note onsets, chord pitch classes, score position and local tempo.

Tempo handling: `score_time[t]` maps every singer frame to a position in the score
and `tempo_ratio[t]` = d score_time / d t (1.0 = backing-track tempo). Training-time
time-stretch augmentation only has to resample audio and label arrays together.

Run inside the SDK container (needs pretty_midi; ffmpeg binary or PyAV for m4a):
    pip install pretty_midi av
    python3 jan/prep/prep_damp.py --n 100
Output (gitignored): data/damp_prep/{audio/*.flac, labels/*.npz, manifest.csv, reference.npz}
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SR = 16000          # audio sample rate (matches the 16 kHz front-end check)
FPS = 100           # label frame rate (10 ms)
HOP = SR // FPS
DTW_DS = 5          # DTW runs on 50 ms blocks (5 label frames)
MAX_LAG_S = 2.0     # global lag search range
BAND_S = 1.5        # DTW may deviate this far from the global lag
MIN_DURATION_S = 150.0

REPO = Path(__file__).resolve().parents[2]
DAMP = REPO / "data" / "DAMP-S-AG"


# ---------------------------------------------------------------- reference --
def load_reference(midi_path: Path) -> dict:
    import pretty_midi

    m = pretty_midi.PrettyMIDI(str(midi_path))
    tracks = {i.name.strip(): i for i in m.instruments}
    mel = sorted((n.start, n.end, n.pitch) for n in tracks["Pitch"].notes)
    chords = [(n.start, n.end, n.pitch) for n in tracks["Chords"].notes] if "Chords" in tracks else []
    T = int(np.ceil(m.get_end_time() * FPS)) + 1
    pitch = np.full(T, -1, np.int16)
    for s, e, p in mel:  # later onsets overwrite earlier ones on overlap
        a = int(round(s * FPS))
        pitch[a:max(int(round(e * FPS)), a + 1)] = p
    chord = np.zeros((T, 12), np.uint8)
    for s, e, p in chords:
        chord[int(round(s * FPS)):int(round(e * FPS)), p % 12] = 1
    return dict(notes=np.array(mel, np.float64), pitch=pitch, chord=chord,
                beats=np.asarray(m.get_beats(), np.float64))


# ------------------------------------------------------------------- audio --
def decode(path: Path) -> np.ndarray:
    if shutil.which("ffmpeg"):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                              "-f", "f32le", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(out, np.float32).copy()
    import av  # pip install av (bundles ffmpeg libs)

    chunks = []
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for fr in c.decode(audio=0):
            chunks += [o.to_ndarray().reshape(-1) for o in rs.resample(fr)]
        chunks += [o.to_ndarray().reshape(-1) for o in rs.resample(None)]
    return np.concatenate(chunks).astype(np.float32)


def track_pitch(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pYIN f0 in MIDI (nan = unvoiced) and voicing probability, 10 ms frames."""
    import librosa

    # 30 s chunks: pYIN on a whole 3 min file needs several GB of RAM
    chunk = 30 * SR
    T = len(y) // HOP + 1
    f0, vp = np.full(T, np.nan), np.zeros(T)
    for a in range(0, len(y), chunk):
        c0, _, c1 = librosa.pyin(y[a:a + chunk], fmin=65.0, fmax=1100.0, sr=SR, frame_length=1024,
                                 hop_length=HOP, center=True)
        i = a // HOP
        n = min(len(c0), T - i)
        f0[i:i + n], vp[i:i + n] = c0[:n], c1[:n]
    midi = np.where(np.isfinite(f0), 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    return midi.astype(np.float32), np.nan_to_num(vp).astype(np.float32)


# --------------------------------------------------------------- alignment --
def block(x: np.ndarray, n: int) -> np.ndarray:
    T = len(x) // n * n
    return x[:T].reshape(T // n, n, *x.shape[1:]).mean(1)


def vocal_features(f0: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """13-dim per frame: soft pitch-class distribution * voicing, + silence."""
    T = len(f0)
    feat = np.zeros((T, 13), np.float32)
    ok = np.isfinite(f0)
    d = (np.arange(12)[None, :] - f0[ok, None] + 6) % 12 - 6  # circular semitone distance
    g = np.exp(-0.5 * (d / 0.5) ** 2)
    feat[ok, :12] = g / g.sum(1, keepdims=True) * vp[ok, None]
    feat[:, 12] = 1 - np.where(ok, vp, 0)
    return feat


def reference_features(pitch: np.ndarray, k: int) -> np.ndarray:
    feat = np.zeros((len(pitch), 13), np.float32)
    on = pitch >= 0
    feat[np.where(on)[0], (pitch[on] + k) % 12] = 1
    feat[~on, 12] = 1
    return feat


def global_lag(R: np.ndarray, V: np.ndarray, max_lag: int) -> tuple[int, int, np.ndarray]:
    """Best (lag, transpose) by pitch-class correlation on 50 ms blocks. j = i + lag."""
    N, M = len(R), len(V)
    scores = np.full((12, 2 * max_lag + 1), -np.inf)
    for k in range(12):
        Rk = np.roll(R[:, :12], k, axis=1)
        for li, lag in enumerate(range(-max_lag, max_lag + 1)):
            i0, i1 = max(0, -lag), min(N, M - lag)
            if i1 - i0 > N // 2:
                scores[k, li] = np.sum(Rk[i0:i1] * V[i0 + lag:i1 + lag, :12]) / (i1 - i0)
    k, li = np.unravel_index(np.argmax(scores), scores.shape)
    return li - max_lag, int(k), scores


def banded_dtw(R: np.ndarray, V: np.ndarray, lag: int, B: int) -> np.ndarray:
    """DTW of reference blocks R[i] against vocal blocks V[j], j = i + lag + b, |b| <= B.

    Steps (di, dj) in {(1,1),(1,2),(2,1),(1,3),(3,1)}: local tempo between 1/3x and 3x, and no
    horizontal/vertical steps, so every row depends only on earlier rows (row-vectorised).
    Free start in the first row, free end in the last row. Returns path [(i, j), ...].
    """
    N, M, W = len(R), len(V), 2 * B + 1
    b = np.arange(W) - B
    J = np.arange(N)[:, None] + lag + b[None, :]
    valid = (J >= 0) & (J < M)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    C = np.ones((N, W), np.float64)  # outside the recording: max cost, not forbidden
    C[valid] = 1 - np.einsum("ik,ik->i", np.repeat(Rn, W, 0)[valid.ravel()], Vn[J[valid]])
    steps = [(1, 1, 0.0), (1, 2, 0.05), (2, 1, 0.05), (1, 3, 0.15), (3, 1, 0.15)]
    D = np.full((N, W), np.inf)
    P = np.full((N, W), -1, np.int8)
    D[0] = C[0]
    for i in range(1, N):
        for s, (di, dj, pen) in enumerate(steps):
            if i - di < 0:
                continue
            shift = di - dj          # predecessor band index = b + shift
            src = D[i - di]
            prev = np.full(W, np.inf)
            if shift >= 0:
                prev[:W - shift] = src[shift:]
            else:
                prev[-shift:] = src[:W + shift]
            cand = prev + C[i] * (di + dj) / 2 + pen
            better = cand < D[i]
            D[i, better] = cand[better]
            P[i, better] = s
    path, i, w = [], N - 1, int(np.argmin(D[N - 1]))
    while True:
        path.append((i, i + lag + w - B))
        s = P[i, w]
        if s < 0:
            break
        di, dj, _ = steps[s]
        w, i = w + di - dj, i - di
    return np.array(path[::-1], np.float64)


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    pad = np.pad(x, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(pad, np.ones(win) / win, mode="valid")


# ------------------------------------------------------------------ worker --
def process(args: tuple) -> dict:
    path, out_dir, ref = args
    pid = path.stem
    row = dict(id=pid, status="ok")
    try:
        y = decode(path)
    except Exception as e:  # truncated archive member etc.
        return {**row, "status": f"decode_error: {str(e)[:80]}"}
    dur = len(y) / SR
    row["duration_s"] = round(dur, 2)
    if dur < MIN_DURATION_S:
        return {**row, "status": "too_short"}

    f0, vp = track_pitch(y)
    T = len(f0)

    # --- alignment on 50 ms blocks, restricted to the melody region of the score
    on = np.where(ref["pitch"] >= 0)[0]
    r0 = max(0, on[0] - 2 * FPS)
    r1 = min(len(ref["pitch"]), on[-1] + 2 * FPS)
    r0 -= r0 % DTW_DS
    V = block(vocal_features(f0, vp), DTW_DS)
    R0 = block(reference_features(ref["pitch"][r0:r1], 0), DTW_DS)
    off = r0 // DTW_DS  # block index of the score region start
    lag_blocks, k, scores = global_lag(R0, V[off:], int(MAX_LAG_S * FPS / DTW_DS))
    # only accept a transposition if it is clearly better than singing in the track's key
    if k != 0 and scores[k].max() < 1.2 * scores[0].max():
        k = 0
        lag_blocks = int(np.argmax(scores[0])) - int(MAX_LAG_S * FPS / DTW_DS)
    R = block(reference_features(ref["pitch"][r0:r1], k), DTW_DS)
    path = banded_dtw(R, V[off:], lag_blocks, int(BAND_S * FPS / DTW_DS))

    # --- singer time -> score time, 10 ms grid (piecewise linear, smoothed, monotone)
    tr_p = (path[:, 0] * DTW_DS + r0) / FPS
    tv_p = (path[:, 1] + off) * DTW_DS / FPS
    tv = np.arange(T) / FPS
    score_time = np.interp(tv, tv_p, tr_p)
    score_time[tv < tv_p[0]] = tr_p[0] + tv[tv < tv_p[0]] - tv_p[0]      # slope-1 extrapolation
    score_time[tv > tv_p[-1]] = tr_p[-1] + tv[tv > tv_p[-1]] - tv_p[-1]
    score_time = smooth(score_time, FPS // 2 + 1)
    tempo_ratio = smooth(np.gradient(score_time) * FPS, 2 * FPS + 1)

    # --- labels in singer time
    idx = np.round(score_time * FPS).astype(int)
    inside = (idx >= 0) & (idx < len(ref["pitch"]))
    pitch_ref = np.full(T, -1, np.int16)
    pitch_ref[inside] = ref["pitch"][idx[inside]]
    chord = np.zeros((T, 12), np.uint8)
    chord[inside] = np.roll(ref["chord"][idx[inside]], k, axis=1)

    lab = pitch_ref >= 0
    voiced = np.isfinite(f0) & (vp > 0.5)
    both = lab & voiced
    if both.sum() < 5 * FPS:
        return {**row, "status": "silent", "voiced_frac": round(float(both.sum() / max(lab.sum(), 1)), 3)}
    octave = int(np.round(np.median(f0[both] - (pitch_ref[both] + k)) / 12))
    pitch = np.where(lab, pitch_ref + k + 12 * octave, -1).astype(np.int16)
    pc = np.where(lab, pitch % 12, 12).astype(np.int8)

    onset = np.zeros(T, np.uint8)
    tr_mono = score_time + np.arange(T) * 1e-9  # guard against flat segments for interp
    for s, _, _ in ref["notes"]:
        j = int(round(np.interp(s, tr_mono, tv) * FPS))
        if 0 <= j < T and tr_mono[0] <= s <= tr_mono[-1]:
            onset[j] = 1
    beats = ref["beats"][(ref["beats"] >= score_time[0]) & (ref["beats"] <= score_time[-1])]
    beat_times = np.interp(beats, tr_mono, tv).astype(np.float32)

    err = (f0[both] - pitch[both] + 6) % 12 - 6  # octave-folded error in semitones
    tr_lab = tempo_ratio[lab]
    row.update(
        lag_s=round(lag_blocks * DTW_DS / FPS, 3), transpose=k, octave_shift=octave,
        voiced_frac=round(float(both.sum() / lab.sum()), 3),
        pitch_acc=round(float(np.mean(np.abs(err) < 1.0)), 3),
        median_abs_cents=round(float(np.median(np.abs(err)) * 100), 1),
        tempo_p05=round(float(np.percentile(tr_lab, 5)), 3),
        tempo_p95=round(float(np.percentile(tr_lab, 95)), 3),
        drift_s=round(float(np.max(np.abs(score_time[lab] - tv[lab] - lag_blocks * DTW_DS / FPS))), 3),
    )

    import soundfile as sf

    sf.write(out_dir / "audio" / f"{pid}.flac", np.clip(y, -1, 1), SR, subtype="PCM_16")
    np.savez_compressed(
        out_dir / "labels" / f"{pid}.npz",
        fps=FPS, sr=SR,
        pitch=pitch, pc=pc, onset=onset, chord=chord,
        f0=f0, voiced_prob=vp.astype(np.float16),
        score_time=score_time.astype(np.float32), tempo_ratio=tempo_ratio.astype(np.float32),
        beat_times=beat_times, meta=json.dumps(row),
    )
    return row


# -------------------------------------------------------------------- main --
def read_tsv(path: Path) -> dict:
    with open(path, newline="") as f:
        return {r["performance_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of accepted performances")
    ap.add_argument("--src", type=Path, default=DAMP / "vocals" / "amazing_grace")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "damp_prep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-voiced", type=float, default=0.5, help="min fraction of melody frames sung")
    ap.add_argument("--min-acc", type=float, default=0.5, help="min fraction of frames within 1 semitone")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    a = ap.parse_args()

    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(parents=True, exist_ok=True)
    ref = load_reference(DAMP / "amazing_grace.midi")
    np.savez_compressed(a.out / "reference.npz", **{k: v for k, v in ref.items()}, fps=FPS)
    meta = read_tsv(DAMP / "amazing_grace.tsv")

    files = sorted(a.src.glob("*.m4a"))
    random.Random(a.seed).shuffle(files)
    print(f"{len(files)} candidate recordings, target {a.n}, {a.workers} workers", flush=True)

    rows, accepted, t0, pos = [], 0, time.time(), 0
    with Pool(a.workers) as pool:
        while accepted < a.n and pos < len(files):
            batch = files[pos:pos + max(a.workers, a.n - accepted)]
            pos += len(batch)
            for r in pool.imap_unordered(process, [(f, a.out, ref) for f in batch]):
                good = (r["status"] == "ok" and r["voiced_frac"] >= a.min_voiced
                        and r["pitch_acc"] >= a.min_acc and accepted < a.n)
                if r["status"] == "ok" and not good:
                    r["status"] = "rejected"
                    for p in (a.out / "audio" / f"{r['id']}.flac", a.out / "labels" / f"{r['id']}.npz"):
                        p.unlink(missing_ok=True)
                accepted += good
                m = meta.get(r["id"], {})
                r.update(gender=m.get("gender"), headphones=m.get("headphones"), device_os=m.get("device_os"))
                rows.append(r)
                print(f"[{accepted:3d}/{a.n}] {r['id']:<22} {r['status']:<9} "
                      f"acc={r.get('pitch_acc', '-')} voiced={r.get('voiced_frac', '-')} "
                      f"tempo={r.get('tempo_p05', '-')}..{r.get('tempo_p95', '-')} "
                      f"oct={r.get('octave_shift', '-')} ({time.time() - t0:.0f}s)", flush=True)

    ok = sorted(r["id"] for r in rows if r["status"] == "ok")
    random.Random(a.seed + 1).shuffle(ok)
    n_val, n_test = round(a.val_frac * len(ok)), round(a.test_frac * len(ok))
    split = {pid: "test" for pid in ok[:n_test]}
    split |= {pid: "val" for pid in ok[n_test:n_test + n_val]}
    for r in rows:
        r["split"] = split.get(r["id"], "train" if r["status"] == "ok" else "")
    cols = ["id", "status", "split", "duration_s", "lag_s", "transpose", "octave_shift", "voiced_frac",
            "pitch_acc", "median_abs_cents", "tempo_p05", "tempo_p95", "drift_s",
            "gender", "headphones", "device_os"]
    with open(a.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["status"] != "ok", r["id"])))
    print(f"done: {accepted} accepted, {len(rows) - accepted} skipped, {time.time() - t0:.0f}s -> {a.out}")
    if accepted < a.n:
        sys.exit(f"only {accepted} recordings passed the filters")


if __name__ == "__main__":
    main()
