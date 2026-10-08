"""Direct singing-to-note tracking: the sung note, in any song. No score, no chords.

Input per frame is what the existing loop already has: the 12 pitch-class rates (chip SNN or
laptop baseline), the 48 front-end bands and the front-end's voiced flag. Output is one MIDI
note or silence.

  pitch class  strongest of the 12 rates, averaged over `smooth` frames
  switching    another class must lead by `margin` for `switch` frames, and the current note
               must have lasted `min_note` frames
  silence      no voiced frame for `release_s` releases the note
  octave       NOT from the SNN (it has 12 outputs, no octave): the lowest strong band of that
               pitch class among the 48 bands, computed here on the laptop

It offers the calls live_chip.py makes on a selector (step, pcs, names, em.T), so the loop
does not need to know which selector runs.
"""
from collections import deque
from types import SimpleNamespace

import numpy as np

from singalong import frontend as fe, renderer as rd

SILENCE = 12


class PitchTracker:
    def __init__(self, fps=fe.FPS, smooth=12, switch=6, min_note=9, margin=0.25, release_s=0.4):
        self.fps, self.switch, self.min_note, self.margin = fps, switch, min_note, margin
        self.release = max(1, int(round(release_s * fps)))
        self.names = fe.NAMES + ['silence']
        self.em = SimpleNamespace(T=np.eye(13, 12))            # row = state, so argmax is its pitch class
        self.rates = deque(maxlen=smooth)
        self.bands = deque(maxlen=smooth)
        self.pc = self.note = None
        self.voiced, self.quiet, self.dwell, self.lead, self.octave_votes = False, 0, 0, 0, 0
        self.candidate = None

    def observe(self, bands, voiced):
        """The front-end's bands and voiced flag for the frame that step() is about to get."""
        self.voiced = bool(voiced)
        self.bands.append(np.asarray(bands, float))

    def _octave_note(self, pc):
        """Lowest band of this pitch class that is at least 60 % as strong as its strongest octave."""
        x = np.mean(self.bands, axis=0) if self.bands else np.zeros(fe.NB)
        candidates = [k for k in range(fe.NB) if (fe.LO + k) % 12 == pc]
        top = max(x[k] for k in candidates)
        if top <= 0:
            return None
        return fe.LO + next(k for k in candidates if x[k] >= 0.6 * top)

    def step(self, melody=None, chord=None, onset=True):
        self.rates.append(np.zeros(12) if melody is None else np.asarray(melody, float))
        avg = np.mean(self.rates, axis=0)
        self.quiet = 0 if self.voiced else self.quiet + 1
        self.dwell += 1
        if self.quiet >= self.release:                         # nobody is singing: let go
            self.pc = self.note = self.candidate = None
            self.lead = 0
        elif avg.max() > 0:
            best = int(avg.argmax())
            current = 0.0 if self.pc is None else avg[self.pc]
            if best != self.pc and avg[best] > current * (1 + self.margin):
                self.lead = self.lead + 1 if best == self.candidate else 1
                self.candidate = best
                if self.lead >= self.switch and (self.pc is None or self.dwell >= self.min_note):
                    note = self._octave_note(best)
                    if note is not None:
                        self.pc, self.note, self.dwell, self.lead, self.octave_votes = best, note, 0, 0, 0
            else:
                self.lead = 0
            if self.pc is not None and self.lead == 0:         # same class: follow a lasting octave change
                note = self._octave_note(self.pc)
                self.octave_votes = self.octave_votes + 1 if note not in (None, self.note) else 0
                if self.octave_votes >= 2 * self.switch:
                    self.note, self.octave_votes = note, 0
        post = np.append(avg, 0.0)
        post = post / post.sum() if post.sum() > 0 else np.eye(13)[SILENCE]
        return (SILENCE if self.pc is None else self.pc), post

    def pcs(self, state):
        return None if state == SILENCE else self.em.T[state]

    def renderer(self, sink):
        return NoteRenderer(sink, self)


class NoteRenderer(rd.Renderer):
    """One voice: plays the tracker's note. Dynamics (CC11 from the singer's level) as in the Renderer."""

    def __init__(self, sink, tracker):
        super().__init__(sink, n_voices=1, fps=tracker.fps)
        self.tracker = tracker

    def step(self, pcs, root_pc=None, confidence=0.7, level_db=None, cents=0.0):
        if level_db is not None:
            cc = self.dyn(level_db)
            if abs(cc - self.last_cc) >= 2:
                self.sink.control(self.ch, 11, cc)
                self.last_cc = cc
        want = [] if pcs is None or self.tracker.note is None else [int(self.tracker.note)]
        if want == self.notes:
            return
        for n in want:
            self.sink.note_on(self.ch, n, int(40 + 60 * float(np.clip(confidence, 0, 1))))
        for n in self.notes:
            self.sink.note_off(self.ch, n)
        self.notes = want
