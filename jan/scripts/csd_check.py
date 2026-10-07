#!/usr/bin/env python3
"""Is the Children's Song Dataset (CSD) good SNN training data? (D10)

1. Inventory: recordings, duration, sung range (from the MIDI), English titles that overlap our 7 songs.
2. Label check: does the CSD MIDI note agree with what is sung (YIN pitch class) on a sample of recordings?
3. Transfer (the key test): train one learned linear layer (same as band_layout_check.py, D02 input:
   48 semitone bands 82 Hz - 1.25 kHz) on (a) synthetic voices, (b) CSD, (c) both, and test on
   - our 16 real song recordings (YIN labels, never seen in training),
   - synthetic voices by register (to show what a female-only dataset misses).

Data: data/CSD (download from Zenodo 10.5281/zenodo.4916302, CC BY-NC-SA 4.0; not in git).
Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 -u jan/scripts/csd_check.py
"""
import re
from pathlib import Path

import mido
import numpy as np

import band_layout_check as bl
import frontend_check as fc
import octave_layout_check as ol

ROOT = Path(__file__).resolve().parents[2]
CSD = ROOT / "data"
SR, HOP, WIN = 16000, 256, 1024
BANK = bl.LAYOUTS["C. 48 semitone, 4 oct, 82 Hz–1.25 kHz (MIDI 40–87)"] if "C. 48 semitone, 4 oct, 82 Hz–1.25 kHz (MIDI 40–87)" in bl.LAYOUTS \
    else bl.LAYOUTS["C. 48 semitone, 82 Hz–1.25 kHz (MIDI 40–87)"]
OURS = ["happy", "twinkle", "jacques", "jakob", "hänschen", "haenschen", "vogel", "mond", "wien"]


def find_pairs():
    wavs = sorted(CSD.rglob("*.wav"))
    mids = {m.stem: m for m in CSD.rglob("*.mid")}
    return [(w, mids[w.stem]) for w in wavs if w.stem in mids]


def midi_notes(path):
    """[(start_s, end_s, pitch)] from a monophonic MIDI file."""
    t, on, out = 0.0, {}, []
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = t
        elif msg.type in ("note_off", "note_on") and msg.note in on:
            out.append((on.pop(msg.note), t, msg.note))
    return sorted(out)


def frame_labels(notes, n):
    """Pitch class per 16 ms frame (window centre), -1 where no note sounds."""
    centre = (np.arange(n) * HOP + WIN / 2) / SR
    lab = np.full(n, -1)
    for s, e, p in notes:
        lab[(centre >= s) & (centre < e)] = p % 12
    return lab


def main():
    pairs = find_pairs()
    print(f"### 1. Inventory\n{len(pairs)} recordings with MIDI found under {CSD.relative_to(ROOT)}/")
    langs = {}
    for w, _ in pairs:
        key = "english" if re.search(r"en", w.stem.lower()) or "english" in str(w).lower() else "korean" if ("kr" in w.stem.lower() or "korean" in str(w).lower()) else "?"
        langs[key] = langs.get(key, 0) + 1
    print("by language (from names):", langs)
    texts = list(CSD.rglob("*.txt")) + list(CSD.rglob("*.csv")) + list(CSD.rglob("*.json"))
    hits = sorted({t.name for t in texts for o in OURS if o in t.read_text(errors="ignore").lower()})
    print("annotation/metadata files mentioning our songs:", hits[:20] or "none")

    data, lo, hi, dur = [], 128, 0, 0.0
    for w, m in pairs:
        x = fc.load(w, SR)
        notes = midi_notes(m)
        if not notes:
            continue
        lo, hi = min(lo, min(p for _, _, p in notes)), max(hi, max(p for _, _, p in notes))
        F = bl.features(x, BANK)
        lab = frame_labels(notes, len(F))
        dur += len(x) / SR
        data.append((w, x, F, lab))
    print(f"total {dur / 3600:.2f} h; sung range MIDI {lo}–{hi} ({fc.hz(lo):.0f}–{fc.hz(hi):.0f} Hz); "
          f"labelled frames {sum(int((d[3] >= 0).sum()) for d in data)}", flush=True)

    print("\n### 2. Label check (CSD MIDI vs YIN pitch class, 30 recordings, frames where both are voiced)")
    agree = tot = 0
    for w, x, F, lab in data[:: max(1, len(data) // 30)]:
        f0, conf = fc.yin(x, SR, WIN, HOP)
        n = min(len(f0), len(lab))
        ok = conf[:n] & (lab[:n] >= 0)
        pc = np.round(69 + 12 * np.log2(f0[:n] / 440)).astype(int) % 12
        agree += int(np.sum(pc[ok] == lab[:n][ok]))
        tot += int(ok.sum())
    print(f"MIDI label = sung pitch class on {agree / max(tot, 1):.0%} of {tot} frames", flush=True)

    # ---------- 3. transfer
    Xc = np.vstack([d[2][d[3] >= 0][::3] for d in data])  # every 3rd labelled frame
    yc = np.concatenate([d[3][d[3] >= 0][::3] for d in data])
    rng = np.random.default_rng(1)
    Xs, ys, Xt, yt, rt = [], [], [], [], []
    for m in rng.integers(38, 85, 900):
        v = bl.features(bl.synth_voice(m, rng), BANK)[2:-2]
        Xs.append(v)
        ys += [m % 12] * len(v)
    rng2 = np.random.default_rng(2)
    for m in rng2.integers(38, 85, 900):
        v = bl.features(bl.synth_voice(m, rng2), BANK)[2:-2]
        Xt.append(v)
        yt += [m % 12] * len(v)
        rt += [m] * len(v)
    Xs, ys, Xt, yt, rt = np.vstack(Xs), np.array(ys), np.vstack(Xt), np.array(yt), np.array(rt)
    ours = []
    for f in sorted(p for p in fc.SONGS.glob("*/audio/*") if p.suffix in (".wav", ".ogg", ".flac", ".mp3", ".webm")):
        x = fc.load(f, SR)[: SR * 60]
        f0, conf = ol.yin_cached(f, x)
        if conf.sum() < 200:
            continue
        F = bl.features(x, BANK)
        n = min(len(F), len(conf))
        ours.append((F[:n][conf[:n]], (np.round(69 + 12 * np.log2(f0[:n][conf[:n]] / 440)).astype(int)) % 12))
    Xo, yo = np.vstack([o[0] for o in ours]), np.concatenate([o[1] for o in ours])
    print(f"\n### 3. Transfer: train on X, test on our {len(ours)} song recordings (never seen) and on synthetic registers")
    print("| Trained on | frames | our recordings | synth 73–130 Hz | 131–261 Hz | 262–1047 Hz |\n|---|---|---|---|---|---|")
    for name, (X, y) in {"synthetic voices only": (Xs, ys), "CSD only": (Xc, yc),
                         "synthetic + CSD": (np.vstack([Xs, Xc]), np.concatenate([ys, yc]))}.items():
        po = bl.fit_predict(X, y, Xo)
        pt = bl.fit_predict(X, y, Xt)
        reg = [np.mean(pt[(rt >= a) & (rt < b)] == yt[(rt >= a) & (rt < b)]) for a, b in ((38, 48), (48, 60), (60, 85))]
        print(f"| {name} | {len(y)} | {np.mean(po == yo):.0%} | " + " | ".join(f"{r:.0%}" for r in reg) + " |", flush=True)


if __name__ == "__main__":
    main()
