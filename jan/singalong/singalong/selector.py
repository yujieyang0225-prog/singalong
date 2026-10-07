"""Chord selector (D20 B): online HMM forward filter over chords, fed by per-frame evidence.

Evidence per frame: a 12-vector in 0..1 per pitch class. Two kinds, same emission model:
  - melody evidence  (SNN melody head rates, or the pitch-tracker baseline): "this pitch class is being sung"
  - chord evidence   (SNN chord head rates): "these pitch classes fit now"

Emission log-likelihood of chord c:  beta * ( weighted mean of evidence on chord tones - mean on non-chord tones )
with root/fifth/third weights, so a sung root supports the chord more than a sung third. Chord evidence uses
flat weights. Both are combined additively when both are available.

Two state spaces:
  KeyChordHMM  : 12 keys x 7 degree chords (I ii iii IV V vi V7) = 84 states. Transitions: stay (sticky), change
                 within the key by a degree-transition matrix (counted from the song or a generic prior), rare key
                 change to a neighbour on the circle of fifths. Works for any song.
  SongHMM      : left-to-right over the Amazing Grace progression (one state per bar-chord), self-loops with the
                 expected dwell time from the tempo, small skip/back probabilities, in all 12 keys (12 x n_bars
                 states). Tracks *where in the song* the singer is: the "know the song, follow it" part.

Decoding: argmax posterior with hysteresis (switch when the new chord beats the current by `margin` for
`hold_frames` frames, D08). The hysteresis is mild because the transition prior already smooths.
"""
from __future__ import annotations

import numpy as np

from . import amazing_grace as ag

TONE_W = {0: 1.0, 7: 0.8, 4: 0.6, 3: 0.6, 10: 0.4}            # weight of a chord tone by interval above the root


def chord_template(root, quality):
    """12-vector of chord-tone weights (0 for non-chord tones)."""
    t = np.zeros(12)
    for iv in ag.QUALITIES[quality]:
        t[(root + iv) % 12] = TONE_W.get(iv, 0.5)
    return t


class Emission:
    """Per-frame log-likelihood. The betas are small on purpose: frames within one sung note are not independent
    observations, so ~20 frames of one note should add up to a few nats, not tens (beta ≈ 1 / note length)."""

    def __init__(self, roots, qualities, beta_mel=0.25, beta_chord=0.2):
        self.T = np.stack([chord_template(r, q) for r, q in zip(roots, qualities)])          # [S,12]
        self.in_chord = (self.T > 0).astype(float)
        self.beta_mel, self.beta_chord = beta_mel, beta_chord
        self.n_in = self.in_chord.sum(1)
        self.n_out = 12 - self.n_in

    def loglik(self, melody=None, chord=None):
        ll = np.zeros(len(self.T))
        if melody is not None and np.any(melody > 0):
            e = np.asarray(melody, float)
            e = e / max(e.sum(), 1e-6)                                   # distribution over pitch classes
            # weight of the sung tone in the chord (root 1, fifth .8, third .6, seventh .4) minus mass outside the chord
            ll += self.beta_mel * (self.T @ e - (1 - self.in_chord) @ e)
        if chord is not None and np.any(chord > 0):
            e = np.clip(np.asarray(chord, float), 0, 1)
            ll += self.beta_chord * ((self.in_chord @ e) / self.n_in - ((1 - self.in_chord) @ e) / self.n_out) * 3
        return ll


class _HMM:
    def __init__(self, logA, emission, names, hold_frames=3, margin=0.1, min_dwell=8):
        self.logA, self.em, self.names = logA, emission, names
        self.hold_frames, self.margin, self.min_dwell = hold_frames, margin, min_dwell
        self.reset()

    def reset(self):
        self.logp = np.full(len(self.names), -np.log(len(self.names)))
        self.current, self.candidate, self.cand_count, self.dwell = None, None, 0, 0

    def step(self, melody=None, chord=None, onset=True):
        """Forward filter one frame. Returns (chord index to play, posterior)."""
        # predict: logsumexp over previous states
        m = self.logp.max()
        self.logp = m + np.log(np.exp(self.logp - m) @ np.exp(self.logA) + 1e-300)
        self.logp += self.em.loglik(melody, chord)
        self.logp -= self.logp.max()
        post = np.exp(self.logp)
        post /= post.sum()
        best = int(post.argmax())
        self.dwell += 1
        if self.current is None:
            self.current, self.dwell = best, 0
        elif best != self.current and post[best] > post[self.current] + self.margin:
            self.cand_count = self.cand_count + 1 if best == self.candidate else 1
            self.candidate = best
            if self.cand_count >= self.hold_frames and self.dwell >= self.min_dwell and onset:
                self.current, self.dwell, self.cand_count = best, 0, 0
        else:
            self.cand_count = 0
        return self.current, post

    def pcs(self, state):
        return self.em.in_chord[state]


def _log_normalise(A):
    A = A / A.sum(1, keepdims=True)
    return np.log(A + 1e-12)


class KeyChordHMM(_HMM):
    def __init__(self, p_stay=0.985, p_key_change=1e-6, degree_transitions=None, **kw):
        degs = ag.DEGREE_CHORDS
        roots, quals, names = [], [], []
        for key in range(12):
            for d, q in degs:
                roots.append((key + d) % 12); quals.append(q)
                names.append(f"{ag.chord_name((key + d) % 12, q)} ({ag.DEGREE_NAMES[degs.index((d, q))]} in {ag.NAMES[key]})")
        nD, S = len(degs), 12 * len(degs)
        D = degree_transitions if degree_transitions is not None else generic_degree_transitions()
        A = np.zeros((S, S))
        for key in range(12):
            blk = slice(key * nD, (key + 1) * nD)
            within = (1 - p_stay) * (1 - 12 * p_key_change) * D
            A[blk, blk] = within + np.eye(nD) * p_stay
            for other in ((key + 7) % 12, (key + 5) % 12):          # neighbours on the circle of fifths
                ob = slice(other * nD, (other + 1) * nD)
                A[blk, ob] += p_key_change * 6 / nD
        super().__init__(_log_normalise(A), Emission(roots, quals), names, **kw)
        self.nD = nD

    def key_of(self, state):
        return state // self.nD


def generic_degree_transitions():
    """Rough functional-harmony prior over DEGREE_CHORDS = I ii iii IV V vi V7 (rows: from, cols: to; no stay)."""
    I, ii, iii, IV, V, vi, V7 = range(7)
    D = np.ones((7, 7)) * 0.3
    for a, b, w in [(I, IV, 3), (I, V, 3), (I, vi, 1.5), (I, ii, 1), (ii, V, 3), (ii, V7, 2), (IV, I, 3), (IV, V, 2.5),
                    (IV, ii, 1), (V, I, 4), (V7, I, 4), (V, vi, 1), (vi, IV, 2), (vi, ii, 2), (vi, V, 1), (iii, vi, 2),
                    (iii, IV, 1.5), (V, V7, 1.5)]:
        D[a, b] += w
    np.fill_diagonal(D, 0)
    return D


def degree_transitions_from_sequence(seq):
    """Count degree-to-degree changes from a per-bar (root degree, quality) sequence (e.g. ag.degree_sequence())."""
    idx = {dq: i for i, dq in enumerate(ag.DEGREE_CHORDS)}
    D = np.ones((len(idx), len(idx))) * 0.1
    prev = None
    for dq in seq:
        if dq is None:
            continue
        i = idx.get(dq, idx.get((dq[0], "maj")))
        if prev is not None and i != prev and i is not None:
            D[prev, i] += 1
        prev = i
    np.fill_diagonal(D, 0)
    return D


class SongHMM(_HMM):
    """Follow the Amazing Grace progression in any key. One state per (key, chord event); chord events are the
    bars where the chord changes (consecutive identical bars merge). Dwell ~ expected frames per event."""

    def __init__(self, bpm=80, fps=62.5, p_skip=0.02, p_back=0.01, p_key_change=1e-7, events=None, start_mass=0.7, **kw):
        ev = events or self.events_from_lead_sheet()
        self.events = ev                                              # [(degree root, quality, beats)]
        nE = len(ev)
        roots, quals, names = [], [], []
        for key in range(12):
            for d, q, _ in ev:
                roots.append((key + d) % 12); quals.append(q)
                names.append(f"{ag.chord_name((key + d) % 12, q)} in {ag.NAMES[key]}")
        A = np.zeros((12 * nE, 12 * nE))
        for key in range(12):
            for i, (_, _, beats) in enumerate(ev):
                dwell = max(beats * 60 / bpm * fps, 2)
                p_adv = 1 / dwell
                row = key * nE + i
                A[row, row] = 1 - p_adv
                nxt = key * nE + (i + 1) % nE
                A[row, nxt] += p_adv * (1 - p_skip - p_back)
                A[row, key * nE + (i + 2) % nE] += p_adv * p_skip
                A[row, key * nE + (i - 1) % nE] += p_adv * p_back
            for other in range(12):
                if other != key:
                    A[key * nE:(key + 1) * nE, other * nE:(other + 1) * nE] += p_key_change / nE
        super().__init__(_log_normalise(A), Emission(roots, quals), names, **kw)
        self.nE = nE
        self.start_mass = start_mass
        self.reset()

    def reset(self):
        super().reset()
        if getattr(self, "nE", None):
            p = np.full(12 * self.nE, (1 - self.start_mass) / (12 * self.nE))
            for key in range(12):
                p[key * self.nE] += self.start_mass / 24
                p[key * self.nE + 1] += self.start_mass / 24
            self.logp = np.log(p)

    @staticmethod
    def events_from_lead_sheet():
        seq = ag.degree_sequence()
        out = []
        for b, dq in enumerate(seq):
            if dq is None:
                continue
            beats = ag.PICKUP_BEATS if b == 0 else ag.BEATS_PER_BAR
            if out and out[-1][:2] == dq:
                out[-1] = (dq[0], dq[1], out[-1][2] + beats)
            else:
                out.append((dq[0], dq[1], beats))
        return out

    def position(self, state):
        """(key, event index) of a state: where in the song we are."""
        return state // self.nE, state % self.nE


class ScoreFollowerHMM(_HMM):
    """Follow the *melody* of Amazing Grace (the lead sheet) in any key: one state per (key, melody note).

    Emission: the sung pitch class against the expected note of that position (plus chord evidence against the
    chord written over that note). Transitions: stay (dwell = note length at the given tempo), advance, skip one
    note, a small 'restart' (singer begins again) and a tiny uniform jump (lost -> re-found). The played chord is
    the lead-sheet chord at the current position, so the accompaniment follows the song, not the note.
    This is the "know the song and follow where they are" part of the challenge (D01 option E).
    """

    def __init__(self, bpm=80, fps=62.5, p_skip=0.03, p_restart=0.002, p_lost=1e-5, p_key_change=1e-8,
                 start_mass=0.8, anticipate_s=0.15, **kw):
        tl = ag.render_timeline(tonic=0, octave=4, bpm=bpm, fps=fps, pickup_silence_s=0.0)
        notes = tl["notes"]                                          # (start_s, end_s, midi) in C
        # chord written over the note, sampled at the note's middle (its start sits exactly on a bar line)
        starts = np.array([int((s + e) / 2 * fps) for s, e, _ in notes])
        roots_c = tl["chord_root"][np.clip(starts, 0, len(tl["chord_root"]) - 1)]
        pcs_c = tl["chord_pcs"][np.clip(starts, 0, len(tl["chord_pcs"]) - 1)]
        nN = len(notes)
        self.note_pc = np.array([[(m + key) % 12 for _, _, m in notes] for key in range(12)]).reshape(-1)
        chord_T, in_chord, names = [], [], []
        for key in range(12):
            for i in range(nN):
                pcs = np.roll(pcs_c[i], key)
                root = (roots_c[i] + key) % 12 if roots_c[i] >= 0 else int(np.flatnonzero(pcs)[0])
                q = next((q for q, iv in ag.QUALITIES.items() if {(root + j) % 12 for j in iv} == set(np.flatnonzero(pcs))), "maj")
                chord_T.append(chord_template(root, q))
                in_chord.append(pcs.astype(float))
                names.append(f"{ag.chord_name(root, q)} in {ag.NAMES[key]} @note{i + 1}")
        em = Emission([0] * (12 * nN), ["maj"] * (12 * nN), **{k: v for k, v in kw.items() if k.startswith("beta")})
        em.T = np.stack(chord_T)
        em.in_chord = np.stack(in_chord)
        em.n_in, em.n_out = em.in_chord.sum(1), 12 - em.in_chord.sum(1)
        self.em_note = em
        A = np.zeros((12 * nN, 12 * nN))
        for key in range(12):
            for i, (s, e, _) in enumerate(notes):
                dwell = max((e - s) * fps, 2)
                p_adv = 1 / dwell
                row = key * nN + i
                A[row, row] = 1 - p_adv
                A[row, key * nN + (i + 1) % nN] += p_adv * (1 - p_skip - p_restart)
                A[row, key * nN + (i + 2) % nN] += p_adv * p_skip
                A[row, key * nN] += p_adv * p_restart
            A[key * nN:(key + 1) * nN, :] += p_lost / (12 * nN)
            for other in range(12):
                if other != key:
                    A[key * nN:(key + 1) * nN, other * nN:(other + 1) * nN] += p_key_change / nN
        hmm_kw = {k: v for k, v in kw.items() if not k.startswith("beta")}
        hmm_kw.setdefault("min_dwell", 4)
        hmm_kw.setdefault("margin", 0.05)
        super().__init__(_log_normalise(A), em, names, **hmm_kw)
        self.nE, self.start_mass = nN, start_mass
        self.expected = np.array([max((e - s) * fps, 2) for s, e, _ in notes])     # frames per note at this tempo
        self.anticipate = anticipate_s * fps
        self.in_note = 0
        self.reset()

    def reset(self):
        super().reset()
        if getattr(self, "nE", None):
            p = np.full(12 * self.nE, (1 - self.start_mass) / (12 * self.nE))
            for key in range(12):
                p[key * self.nE] += self.start_mass / 12
            self.logp = np.log(p)

    def step(self, melody=None, chord=None, onset=True):
        # melody evidence against the expected *note*, chord evidence against the written chord
        m = self.logp.max()
        self.logp = m + np.log(np.exp(self.logp - m) @ np.exp(self.logA) + 1e-300)
        if melody is not None and np.any(melody > 0):
            e = np.asarray(melody, float)
            e = e / max(e.sum(), 1e-6)
            self.logp += self.em.beta_mel * 1.5 * (e[self.note_pc] - (1 - e[self.note_pc]))
        if chord is not None and np.any(chord > 0):
            self.logp += self.em.loglik(None, chord)
        self.logp -= self.logp.max()
        post = np.exp(self.logp)
        post /= post.sum()
        best = int(post.argmax())
        self.dwell += 1
        # the chord changes only when the *chord at the position* changes, so compare chords, not states
        if self.current is None:
            self.current, self.dwell, self.in_note = best, 0, 0
        elif best != self.current:
            same_chord = np.array_equal(self.em.in_chord[best], self.em.in_chord[self.current])
            if same_chord or (post[best] > post[self.current] + self.margin and self.dwell >= self.min_dwell and onset):
                self.current, self.dwell, self.in_note = best, (self.dwell if same_chord else 0), 0
        else:
            self.in_note += 1
        # anticipation: near the expected end of the current note, already answer with the next note's chord
        # if that chord differs (the singer will be there by the time the violin sounds)
        key, i = self.position(self.current)
        if self.anticipate and i + 1 < self.nE and self.in_note >= self.expected[i] - self.anticipate:
            nxt = key * self.nE + i + 1
            if not np.array_equal(self.em.in_chord[nxt], self.em.in_chord[self.current]) and post[nxt] > 0.05:
                return nxt, post
        return self.current, post

    def position(self, state):
        return state // self.nE, state % self.nE
