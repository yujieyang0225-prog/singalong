"""Amazing Grace lead sheet in scale degrees (any key), plus helpers to read the DAMP reference if present.

Melody: common hymnal version ('New Britain'), 3/4, 16 bars + 1-beat pickup. Degrees are semitones above the
tonic; octave 0 = the tonic's octave. Chords: one or two per bar, as (beat, root degree, quality).

    > [!question] The DAMP backing track (Eb, two key changes) may use different passing chords. When
    > data/damp100/reference.npz exists, prefer it (see reference_from_npz).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

QUALITIES = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "dim": (0, 3, 6)}
MAJOR = (0, 2, 4, 5, 7, 9, 11)

# (degree in semitones above tonic, octave offset, duration in beats). Pickup first. 3/4 time.
MELODY = [
    (7, -1, 1),                                   # A-        D4 in G
    (0, 0, 2), (4, 0, .5), (0, 0, .5),            # ma-zing   G  B G
    (4, 0, 2), (2, 0, 1),                         # grace how B  A
    (0, 0, 2), (9, -1, 1),                        # sweet the G  E
    (7, -1, 2), (7, -1, 1),                       # sound that D  D
    (0, 0, 2), (4, 0, .5), (0, 0, .5),            # saved a   G  B G
    (4, 0, 2), (2, 0, 1),                         # wretch like
    (7, 0, 3),                                    # me        D5
    (7, 0, 2), (4, 0, 1),                         # (me) I    D5 B
    (7, 0, 2), (4, 0, .5), (0, 0, .5),            # once was  D5 B G
    (4, 0, 2), (2, 0, 1),                         # lost but  B A
    (0, 0, 2), (9, -1, 1),                        # now am    G E
    (7, -1, 2), (7, -1, 1),                       # found was D D
    (0, 0, 2), (4, 0, .5), (0, 0, .5),            # blind but G B G
    (4, 0, 2), (2, 0, 1),                         # now I     B A
    (0, 0, 3),                                    # see       G
    (0, 0, 3),                                    # (see)
]
# (bar index 0-based, root degree, quality); the chord holds until the next entry. Bar 0 = the pickup bar.
CHORDS = [
    (0, 7, "maj"),      # pickup on V (optional; many play nothing)
    (1, 0, "maj"), (3, 5, "maj"), (4, 0, "maj"), (7, 7, "maj"),
    (9, 0, "maj"), (10, 0, "7"), (11, 5, "maj"), (12, 0, "maj"), (14, 7, "maj"), (15, 0, "maj"),
]
# Scale-degree chords the selector may choose in 'key' mode (D07 F: I ii iii IV V vi + V7)
DEGREE_CHORDS = [(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min"), (7, "7")]
DEGREE_NAMES = ["I", "ii", "iii", "IV", "V", "vi", "V7"]
NAMES = "C C# D D# E F F# G G# A A# B".split()
BEATS_PER_BAR = 3
PICKUP_BEATS = 1


def chord_pcs(root, quality, tonic=0):
    """Pitch-class set (as 12-vector) of a chord given as degree root + quality in a key."""
    v = np.zeros(12, np.uint8)
    for i in QUALITIES[quality]:
        v[(tonic + root + i) % 12] = 1
    return v


def chord_name(root_pc, quality):
    return NAMES[root_pc % 12] + {"maj": "", "min": "m", "7": "7", "dim": "dim"}[quality]


def total_beats():
    return sum(d for _, _, d in MELODY)


def render_timeline(tonic=7, octave=4, bpm=80, fps=62.5, pickup_silence_s=1.0):
    """Lead sheet -> frame-wise arrays like reference.npz: melody (MIDI, -1 = rest), chord_pcs [T,12],
    chord_root, degree, key_tonic, plus note_onsets (frame indices) for the renderer tests.

    tonic: pitch class of the key (7 = G). octave: MIDI octave of the tonic (4 -> G4 = 67 is degree 0, octave 0).
    """
    spb = 60.0 / bpm
    t = pickup_silence_s
    notes = []                                            # (start_s, end_s, midi)
    for deg, octv, beats in MELODY:
        midi = 12 * (octave + 1) + tonic + deg + 12 * octv
        notes.append((t, t + beats * spb, midi))
        t += beats * spb
    end_s = t + 1.0
    T = int(np.ceil(end_s * fps))
    tc = (np.arange(T) + 0.5) / fps
    melody = np.full(T, -1, np.int16)
    onsets = []
    for s, e, m in notes:
        melody[(tc >= s) & (tc < e)] = m
        onsets.append(int(s * fps))
    # chords: bar b starts at pickup_silence + (b*3 - 1) beats (bar 0 is the one-beat pickup)
    def bar_start(b):
        return pickup_silence_s + max(b * BEATS_PER_BAR - PICKUP_BEATS, 0) * spb

    pcs = np.zeros((T, 12), np.uint8)
    root = np.full(T, -1, np.int8)
    degree = np.full(T, -1, np.int8)
    for i, (bar, deg, q) in enumerate(CHORDS):
        s = bar_start(bar)
        e = bar_start(CHORDS[i + 1][0]) if i + 1 < len(CHORDS) else notes[-1][1]
        sel = (tc >= s) & (tc < e)
        pcs[sel] = chord_pcs(deg, q, tonic)
        root[sel] = (tonic + deg) % 12
        degree[sel] = deg
    key = np.full(T, tonic, np.int8)
    return dict(melody=melody, chord_pcs=pcs, chord_root=root, degree=degree, key_tonic=key,
                onsets=np.array(onsets), notes=notes, fps=fps)


def degree_sequence():
    """Chord progression as a list of (root degree, quality) per bar (for the HMM 'song' prior)."""
    n_bars = int(np.ceil((total_beats() + PICKUP_BEATS) / BEATS_PER_BAR)) + 1
    out, cur = [], None
    for b in range(n_bars):
        for bar, deg, q in CHORDS:
            if bar == b:
                cur = (deg, q)
        out.append(cur)
    return out


def reference_from_npz(path=None):
    """Load data/damp100/reference.npz if it exists (frame-wise melody/chords/key of the DAMP track) else None."""
    path = Path(path) if path else Path(__file__).resolve().parents[3] / "data" / "damp100" / "reference.npz"
    if not path.exists():
        return None
    r = np.load(path)
    return {k: r[k] for k in r.files}


def transition_counts(chord_root, degree_or_key=None, n_states=12, hold_weight=True):
    """Count chord-to-chord transitions on a frame-wise chord sequence (ignores -1)."""
    seq = chord_root[chord_root >= 0]
    C = np.zeros((n_states, n_states))
    for a, b in zip(seq[:-1], seq[1:]):
        if hold_weight or a != b:
            C[a, b] += 1
    return C


if __name__ == "__main__":
    tl = render_timeline()
    sung = tl["melody"] >= 0
    print(f"{len(tl['melody'])} frames, {sung.mean():.0%} sung, range MIDI {tl['melody'][sung].min()}..{tl['melody'][sung].max()}")
    print("bars:", " ".join(chord_name(7 + d, q) for d, q in degree_sequence()))
