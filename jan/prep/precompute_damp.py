#!/usr/bin/env python3
"""Precompute SNN input X and labels for N DAMP-S-AG (Amazing Grace) recordings.

The recordings are aligned with the backing-track MIDI at delay 0
(Hackathon/07 Experiments/2026-10-07 DAMP alignment check.md), so the labels are the same for
every singer except an octave / transposition shift. Therefore:

  reference.npz  (once, 16 ms grid, frame t centred at (t*256+512)/16000 s)
      melody      int16 [T]      MIDI note of the 'Pitch' track, -1 = no note (MIDI octave)
      chord_pcs   uint8 [T,12]   pitch classes sounding in the 'Chords' track
      chord_root  int8  [T]      root pitch class, -1 = no chord
      chord_qual  int8  [T]      index into QUALITIES, -1 = no chord / unknown
      key_tonic   int8  [T]      tonic pitch class of the 'Key' track (major keys), -1 = none
      degree      int8  [T]      (chord_root - key_tonic) % 12 in semitones, -1 = none
  <id>.npz       (per recording)
      X           float16 [T,48] log2 band energy, bands MIDI 40-87 (82 Hz - 1.25 kHz), NOT normalised
      voiced      uint8   [T]    1 = singing (frame energy >= noise floor + 15 dB)
      octave      int            add 12*octave to reference melody to get the sung note
      transpose   int            semitones the singer is away from the track key (0 = in key)
  manifest.csv   one row per processed recording, with quality numbers and a train/val/test split

Front-end = D02 (16 kHz, 1024 Hann, hop 256, 48 semitone bands), from band_layout_check.py.
Per-frame normalisation (40 dB range) is NOT applied here, so augmentation can work on log energy.

Run from repo root (host, needs ffmpeg):
    PYTHONPATH=.pydeps:jan/scripts python3 jan/prep/precompute_damp.py --n 100
"""
import argparse
import csv
import random
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import mido
import numpy as np

from band_layout_check import HOP, SR, WIN, semitone_bank

REPO = Path(__file__).resolve().parents[2]
DAMP = REPO / "data" / "DAMP-S-AG"
LO, NB = 40, 48
FB = semitone_bank(LO, NB)
QUALITIES = {  # intervals above the root
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "sus4": (0, 5, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
}
QNAMES = list(QUALITIES)
MAJOR = (0, 2, 4, 5, 7, 9, 11)


# ---------------------------------------------------------------- reference --
def midi_track_notes(mf, name):
    """[(start_s, end_s, pitch)] of one track, using the file's tempo map."""
    tempo_ev = sorted((t, m.tempo) for tr in mf.tracks
                      for t, m in zip(np.cumsum([x.time for x in tr]), tr) if m.type == "set_tempo")

    def sec(tick):
        s, lt, lv = 0.0, 0, 500000
        for t, v in tempo_ev:
            if t >= tick:
                break
            s += mido.tick2second(t - lt, mf.ticks_per_beat, lv)
            lt, lv = t, v
        return s + mido.tick2second(tick - lt, mf.ticks_per_beat, lv)

    tr = next(t for t in mf.tracks if t.name.strip() == name)
    out, on, t = [], {}, 0
    for m in tr:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            on[m.note] = t
        elif m.type in ("note_off", "note_on") and m.note in on:
            out.append((sec(on.pop(m.note)), sec(t), m.note))
    return sorted(out)


def name_chord(pcs):
    """Pitch-class set -> (root, quality index); exact template match, else (-1, -1)."""
    for r in sorted(pcs):
        for qi, iv in enumerate(QUALITIES.values()):
            if {(r + i) % 12 for i in iv} == pcs:
                return r, qi
    return -1, -1


def build_reference(n_frames):
    mf = mido.MidiFile(DAMP / "amazing_grace.midi")
    tc = (np.arange(n_frames) * HOP + WIN / 2) / SR          # frame centre times
    ref = dict(melody=np.full(n_frames, -1, np.int16), chord_pcs=np.zeros((n_frames, 12), np.uint8),
               chord_root=np.full(n_frames, -1, np.int8), chord_qual=np.full(n_frames, -1, np.int8),
               key_tonic=np.full(n_frames, -1, np.int8))
    for s, e, p in midi_track_notes(mf, "Pitch"):
        ref["melody"][(tc >= s) & (tc < e)] = p
    for s, e, p in midi_track_notes(mf, "Chords"):
        ref["chord_pcs"][(tc >= s) & (tc < e), p % 12] = 1
    # 'Key' = a short marker chord (the 7 scale notes) at the start of each key section;
    # the key holds until the next marker
    markers = {}
    for s, _, p in midi_track_notes(mf, "Key"):
        markers.setdefault(round(s, 2), set()).add(p % 12)
    for s, pcs in sorted(markers.items()):
        tonic = [r for r in range(12) if {(r + i) % 12 for i in MAJOR} == pcs]
        if tonic:
            ref["key_tonic"][tc >= s] = tonic[0]
    cache, unknown = {}, set()
    for t in range(n_frames):
        pcs = frozenset(np.flatnonzero(ref["chord_pcs"][t]))
        if pcs:
            if pcs not in cache:
                cache[pcs] = name_chord(set(pcs))
                if cache[pcs][0] < 0:
                    unknown.add(tuple(sorted(pcs)))
            ref["chord_root"][t], ref["chord_qual"][t] = cache[pcs]
    ok = (ref["chord_root"] >= 0) & (ref["key_tonic"] >= 0)
    ref["degree"] = np.where(ok, (ref["chord_root"] - ref["key_tonic"]) % 12, -1).astype(np.int8)
    return ref, unknown


# ------------------------------------------------------------- recordings --
def decode(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32)


def process(args):
    path, out_dir, melody = args
    row = dict(id=path.stem)
    try:
        y = decode(path)
    except subprocess.CalledProcessError:
        return {**row, "status": "decode_error"}
    if len(y) < 150 * SR:
        return {**row, "status": "too_short", "duration_s": round(len(y) / SR, 1)}
    frames = np.lib.stride_tricks.sliding_window_view(y, WIN)[::HOP]
    spec = np.abs(np.fft.rfft(frames * np.hanning(WIN), WIN)) ** 2
    X = np.log2(spec @ FB.T + 1e-10).astype(np.float32)           # [T, 48] log2 energy
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    voiced = db >= np.percentile(db, 10) + 15
    T = min(len(X), len(melody))
    X, voiced, mel = X[:T], voiced[:T], melody[:T]

    # fit of the melody: mean normalised energy (0..1) at the target band, singing frames only
    Xn = np.maximum(X - X.max(1, keepdims=True), -13.3) / 13.3 + 1
    lab = (mel >= 0) & voiced

    def fit(shift):
        b = mel[lab] + shift - LO
        ok = (b >= 0) & (b < NB)
        return Xn[np.flatnonzero(lab)[ok], b[ok]].sum() / max(lab.sum(), 1)

    shifts = {s: fit(s) for s in range(-48, 13)}                    # 4 octaves down .. 1 up
    best = max(shifts, key=shifts.get)
    octv = int(round(best / 12)) if abs(best - 12 * round(best / 12)) <= 1 else int(np.floor(best / 12))
    transpose = best - 12 * octv
    if transpose > 6:
        transpose, octv = transpose - 12, octv + 1
    in_key = shifts[12 * octv]
    singing = float(voiced[mel >= 0].mean())
    row.update(usable=int(singing >= 0.5 and shifts[best] >= 0.5))
    row.update(status="ok", duration_s=round(len(y) / SR, 1), octave=octv, transpose=transpose,
               singing=round(singing, 3),
               fit_best=round(float(shifts[best]), 3), fit_in_key=round(float(in_key), 3),
               sung_lo=int(mel[mel >= 0].min() + 12 * octv + transpose),
               sung_hi=int(mel[mel >= 0].max() + 12 * octv + transpose))
    np.savez_compressed(out_dir / f"{path.stem}.npz", X=X.astype(np.float16), voiced=voiced.astype(np.uint8),
                        octave=octv, transpose=transpose)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "damp100")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    n_frames = (int(191.5 * SR) - WIN) // HOP + 1
    ref, unknown = build_reference(n_frames)
    np.savez_compressed(a.out / "reference.npz", **ref, qualities=np.array(QNAMES), lo_band=LO,
                        n_bands=NB, sr=SR, hop=HOP, win=WIN)
    print(f"reference: {n_frames} frames, keys {sorted(set(ref['key_tonic'][ref['key_tonic'] >= 0]))}, "
          f"chords named {np.mean(ref['chord_root'][ref['chord_pcs'].any(1)] >= 0):.0%}, unknown: {unknown or '-'}")

    files = sorted((DAMP / "vocals" / "amazing_grace").glob("*.m4a"))
    random.Random(a.seed).shuffle(files)
    files = files[:a.n]
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, [(f, a.out, ref["melody"]) for f in files]), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  ({time.time() - t0:.0f} s)", flush=True)

    ok = sorted(r["id"] for r in rows if r["status"] == "ok" and r["usable"])
    random.Random(a.seed + 1).shuffle(ok)
    split = {pid: "test" for pid in ok[:len(ok) // 10]} | {pid: "val" for pid in ok[len(ok) // 10:len(ok) // 5]}
    cols = ["id", "status", "usable", "split", "duration_s", "octave", "transpose", "singing", "fit_in_key", "fit_best",
            "sung_lo", "sung_hi"]
    with open(a.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["id"]):
            r["split"] = split.get(r["id"], "train") if r["id"] in ok else ""
            w.writerow(r)
    print(f"done: {len(ok)} usable, {len(rows) - len(ok)} failed or unusable, {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
