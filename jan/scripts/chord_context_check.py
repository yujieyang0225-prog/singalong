#!/usr/bin/env python3
"""How much does chord history help choose the next chord? (input for D20)

Uses the chord-per-beat labels of jan/scripts/analyze_song_harmony.py (6 harmonised songs, relative to the key).
At every melody-note onset, predict the arranger's chord from growing contexts, trained on 5 songs and
tested on the 6th (leave-one-song-out). Unseen contexts back off to the next shorter one.

Run from repo root:  PYTHONPATH=.pydeps:jan/scripts python3 jan/scripts/chord_context_check.py
"""
from collections import Counter, defaultdict

import analyze_song_harmony as ash

CTX = {
    "always I": lambda d: (),
    "melody note": lambda d: (d["deg"],),
    "+ previous chord": lambda d: (d["deg"], d["p1"]),
    "+ 2 previous chords": lambda d: (d["deg"], d["p1"], d["p2"]),
    "+ previous chord + downbeat?": lambda d: (d["deg"], d["p1"], d["down"]),
    "+ 2 previous chords + downbeat?": lambda d: (d["deg"], d["p1"], d["p2"], d["down"]),
    "previous chord only (no melody)": lambda d: (d["p1"],),
}


def events(r, song):
    """One record per melody onset: scale degree, previous distinct chords, downbeat flag, target chord."""
    out = []
    for (deg, simple), ((_, prev), _) in zip(r["pairs1"], r["pairs2"]):
        out.append(dict(deg=deg, y=simple, p1=prev))
    # rebuild the chord-change history (distinct chords) to get the chord before the previous one
    hist, p2s = ["N", "N"], []
    for e in out:
        if e["p1"] != hist[-1]:
            hist.append(e["p1"])
        p2s.append(hist[-2])
    for e, p2 in zip(out, p2s):
        e["p2"] = p2
    onsets = sorted(s for s, _, _ in r["mel_onsets"])
    for e, s in zip(out, onsets):
        e["down"] = int(s) % r["bpb"] == 0
    return out


def main():
    res = []
    for s in ash.SONGS:
        r = ash.analyse(*s)
        import mido  # melody onsets (beats) for the downbeat flag
        m = mido.MidiFile(ash.SRC / s[1])
        r["mel_onsets"] = sorted(ash.notes_of(m.tracks[s[2]], m.ticks_per_beat))
        res.append((s[0], events(r, s[0])))
    print("| Context used to pick the chord | correct (leave-one-song-out, 651 onsets) |")
    print("|---|---|")
    for name, f in CTX.items():
        ok = n = 0
        for i, (_, test) in enumerate(res):
            tables = [defaultdict(Counter) for _ in range(5)]
            for j, (_, train) in enumerate(res):
                if j == i:
                    continue
                for e in train:
                    c = f(e)
                    for k in range(len(c) + 1):  # all prefixes for back-off
                        tables[k][c[:k]][e["y"]] += 1
            for e in test:
                c = f(e)
                pred = "I"
                for k in range(len(c), -1, -1):
                    if tables[k][c[:k]]:
                        pred = tables[k][c[:k]].most_common(1)[0][0]
                        break
                ok += pred == e["y"]
                n += 1
        print(f"| {name} | {ok / n:.0%} |")


if __name__ == "__main__":
    main()
