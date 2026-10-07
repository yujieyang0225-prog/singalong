#!/usr/bin/env python3
"""Harmony facts from the challenge songs' harmonised MIDI files.

For each song: estimate one chord per beat from all non-drum voices (triad/7th template matching
on duration-weighted chroma, bass-weighted), express it relative to the key (Roman numerals),
and measure how predictable the chord is from the melody.

Run from repo root:  PYTHONPATH=.pydeps python3 jan/scripts/analyze_song_harmony.py
Output: markdown on stdout (pasted into the experiment note).
"""
import math
from collections import Counter, defaultdict
from pathlib import Path

import mido
import numpy as np

SRC = Path(__file__).resolve().parents[2] / "Hackathon" / "08 Songs" / "sources"
# file, melody track index, beats per bar (Gerloff files have no time-signature event)
SONGS = [
    ("Happy Birthday", "01_Happy_Birthday/midi/Happy-Birthday-To-You-4.mid", 2, 3),
    ("Twinkle (Gerloff)", "02_Twinkle/midi/Ah!_vous_dirai-je,_Maman.mid", 1, 4),
    ("Frère Jacques (Gerloff)", "03_Frere_Jacques/midi/Frère_Jacques_(pg).mid", 1, 4),
    ("Hänschen klein (Gerloff)", "05_Haenschen_klein/midi/Hänschen_klein.mid", 3, 2),
    ("Kommt ein Vogel (Gerloff)", "06_Kommt_ein_Vogel_geflogen/midi/Kommt_ein_Vogel_geflogen.mid", 1, 3),
    ("Der Mond (Gerloff)", "07_Der_Mond_ist_aufgegangen/midi/Der_Mond_ist_aufgegangen.mid", 1, 4),
]
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
QUAL = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "dim": (0, 3, 6)}
MAJ_ROMAN = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
# Krumhansl-Kessler key profiles
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def notes_of(track, tpb):
    """(onset_beats, offset_beats, pitch) for one track."""
    t, on, out = 0, {}, []
    for msg in track:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on.setdefault(msg.note, []).append(t)
        elif msg.type in ("note_off", "note_on") and on.get(msg.note):
            s = on[msg.note].pop(0)
            out.append((s / tpb, t / tpb, msg.note))
    return out


def chord_of(chroma, bass_pc):
    best, bs = None, -1
    for root in range(12):
        for q, ivs in QUAL.items():
            tmpl = np.zeros(12)
            tmpl[[(root + i) % 12 for i in ivs]] = 1
            score = chroma @ tmpl / math.sqrt(len(ivs)) + (0.15 if bass_pc == root else 0) - (0.02 if q in ("7", "dim") else 0)
            if score > bs:
                best, bs = (root, q), score
    return best


def roman(root, q, tonic):
    r = MAJ_ROMAN[(root - tonic) % 12]
    return (r.lower() if q in ("m", "dim") else r) + ("°" if q == "dim" else "7" if q == "7" else "")


def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values())


def cond_entropy(pairs):
    by = defaultdict(Counter)
    for x, y in pairs:
        by[x][y] += 1
    n = len(pairs)
    return sum(sum(c.values()) / n * entropy(c) for c in by.values())


def analyse(name, rel, mel_idx, bpb):
    m = mido.MidiFile(SRC / rel)
    tpb = m.ticks_per_beat
    tracks = [(i, notes_of(t, tpb)) for i, t in enumerate(m.tracks)]
    drums = {i for i, t in enumerate(m.tracks) for x in t if x.type == "note_on" and x.channel == 9}
    harm = [n for i, ns in tracks if i not in drums for n in ns]
    mel = sorted(dict(tracks)[mel_idx])
    end = max(n[1] for n in harm)
    # key: Krumhansl-Kessler on duration-weighted chroma of everything
    tot = np.zeros(12)
    for s, e, p in harm:
        tot[p % 12] += e - s
    scores = [(np.corrcoef(np.roll(KK_MAJ, k), tot)[0, 1], k, "major") for k in range(12)]
    scores += [(np.corrcoef(np.roll(KK_MIN, k), tot)[0, 1], k, "minor") for k in range(12)]
    _, tonic, mode = max(scores)
    ksig = next((x.key for t in m.tracks for x in t if x.type == "key_signature"), None)
    if ksig and not ksig.endswith("m"):  # trust the notated (major) key signature over the estimate
        tonic, mode = NAMES.index({"Db": "C#", "Gb": "F#"}.get(ksig, ksig)), "major"
    # one chord per beat
    labels = []
    for b in range(int(math.ceil(end))):
        c, low = np.zeros(12), (99, None)
        for s, e, p in harm:
            ov = min(e, b + 1) - max(s, b)
            if ov > 0:
                c[p % 12] += ov
                low = min(low, (p, p % 12))
        labels.append(None if c.sum() == 0 else chord_of(c / np.linalg.norm(c), low[1]))
    rom = [roman(r, q, tonic) if r is not None else "N" for r, q in (l if l else (None, None) for l in labels)]
    simple = [r.replace("7", "") for r in rom]  # fold 7ths into triads (V7 -> V)
    onsets = {int(s) for s, _, _ in mel}
    # segments (merge repeated beats)
    segs = []
    for r in rom:
        if segs and segs[-1][0] == r:
            segs[-1][1] += 1
        else:
            segs.append([r, 1])
    # melody facts: note-in-chord, predictability of chord from melody degree
    changes = [i for i in range(1, len(simple)) if simple[i] != simple[i - 1] and simple[i] != "N"]
    chg_on_onset = sum(i in onsets for i in changes) / max(1, len(changes))
    in_chord, pairs1, pairs2, prev = 0, [], [], "N"
    for s, e, p in mel:
        b = int(s)
        if b >= len(labels) or labels[b] is None:
            continue
        root, q = labels[b]
        pcs = {(root + i) % 12 for i in QUAL[q]}
        in_chord += p % 12 in pcs
        deg = (p - tonic) % 12
        pairs1.append((deg, simple[b]))
        pairs2.append(((deg, prev), simple[b]))
        prev = simple[b]
    vocab = Counter(r for r in simple if r != "N")
    nb = sum(vocab.values())
    return dict(name=name, key=f"{NAMES[tonic]} {mode}", ksig=next((x.key for t in m.tracks for x in t if x.type == "key_signature"), "—"),
                beats=len(rom), vocab=vocab, chg_on_onset=chg_on_onset, pairs1=pairs1, pairs2=pairs2, nb=nb, segs=segs, bpb=bpb, mel_n=len(pairs1),
                in_chord=in_chord / max(1, len(pairs1)), H=entropy(Counter(y for _, y in pairs1)),
                H1=cond_entropy(pairs1), H2=cond_entropy(pairs2),
                mel_rng=(min(p for _, _, p in mel), max(p for _, _, p in mel)))


def loso(res):
    """Leave-one-song-out: train chord predictors on 5 songs, test on the 6th (at melody-note onsets)."""
    print("\n| Held-out song | always I | P(chord \\| melody note) | P(chord \\| melody note, previous chord) | chord changes on a melody onset |")
    print("|---|---|---|---|---|")
    tot = Counter()
    for i, r in enumerate(res):
        train1, train2 = defaultdict(Counter), defaultdict(Counter)
        for j, o in enumerate(res):
            if j != i:
                for x, y in o["pairs1"]:
                    train1[x][y] += 1
                for x, y in o["pairs2"]:
                    train2[x][y] += 1
        n = len(r["pairs1"])
        a0 = sum(y == "I" for _, y in r["pairs1"])
        a1 = sum(train1[x].most_common(1)[0][0] == y if train1[x] else y == "I" for x, y in r["pairs1"])
        a2 = 0
        for (x, y), (x1, _) in zip(r["pairs2"], r["pairs1"]):
            pred = train2[x].most_common(1)[0][0] if train2[x] else (train1[x1].most_common(1)[0][0] if train1[x1] else "I")
            a2 += pred == y
        tot.update(n=n, a0=a0, a1=a1, a2=a2)
        print(f"| {r['name']} | {a0 / n:.0%} | {a1 / n:.0%} | {a2 / n:.0%} | {r['chg_on_onset']:.0%} |")
    n = tot["n"]
    print(f"| **all ({n} notes)** | **{tot['a0'] / n:.0%}** | **{tot['a1'] / n:.0%}** | **{tot['a2'] / n:.0%}** | |")


def main():
    res = [analyse(*s) for s in SONGS]
    print("| Song | Key (est. / signature) | Chords used (share of beats) | Distinct | Mean chord length | Melody notes in chord |")
    print("|---|---|---|---|---|---|")
    allv = Counter()
    for r in res:
        allv += r["vocab"]
        top = ", ".join(f"{k} {v / r['nb']:.0%}" for k, v in r["vocab"].most_common(6))
        lens = [n for c, n in r["segs"] if c != "N"]
        print(f"| {r['name']} | {r['key']} / {r['ksig']} | {top} | {len(r['vocab'])} | {np.mean(lens):.1f} beats ({np.mean(lens) / r['bpb']:.2f} bars) | {r['in_chord']:.0%} |")
    tot = sum(allv.values())
    print("\n**All 6 songs pooled:** " + ", ".join(f"{k} {v / tot:.1%}" for k, v in allv.most_common(12)))
    cov = sum(v for k, v in allv.items() if k in ("I", "IV", "V", "V7")) / tot
    print(f"\nI + IV + V(7) cover **{cov:.0%}** of all beats.")
    loso(res)
    for r in []:
        print(f"| {r['name']} | {r['H']:.2f} bit | {r['H1']:.2f} bit | {r['H2']:.2f} bit | {r['mel_n']} |")


if __name__ == "__main__":
    main()
