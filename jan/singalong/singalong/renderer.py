"""Violin renderer (D14-D18): chord pitch classes -> 2-3 sustained violin notes with voice leading -> MIDI.

  - register: violin G3 (55) .. E6 (88); voices kept in [55, 81], below or around the singer (D18)
  - voicing: root in the lowest voice when possible, then nearest chord tones; common tones are held (D16)
  - timing: change notes only when the selector changes chord (it already waits for onsets) and at least
    `min_hold_s` after the last change; note-off before note-on; legato by overlapping 30 ms (D16)
  - dynamics: CC11 follows the singer's level: attack 0.15 s, release 1.2 s, 2:1, -8 dB (D15), velocity from
    confidence (D08)
  - tuning: optional pitch bend following the singer's offset in cents (D17), vibrato left to the sound (D19)

The renderer is synth-agnostic: it yields mido messages. Sinks: MidiSink (any MIDI port, e.g. FluidSynth,
a DAW, a virtual port), FluidSynthSink (pyfluidsynth in-process), PrintSink.
"""
from __future__ import annotations

import time

import numpy as np

from . import amazing_grace as ag

LOW, HIGH = 55, 81


def voice_chord(pcs, prev_notes=None, n_voices=3, root_pc=None, low=LOW, high=HIGH):
    """Pick concrete MIDI notes for a chord. pcs: 12-vector (>0 = chord tone) or list of pitch classes."""
    tones = [i for i in range(12) if np.asarray(pcs)[i] > 0] if len(pcs) == 12 else list(pcs)
    if not tones:
        return list(prev_notes or [])
    candidates = [n for n in range(low, high + 1) if n % 12 in tones]
    if prev_notes:
        # nearest-note voice leading: each previous voice moves to the closest chord tone (common tones stay)
        notes = []
        for p in prev_notes:
            best = min(candidates, key=lambda n: (abs(n - p), n))
            if best not in notes:
                notes.append(best)
        # make sure the chord is complete-ish: add missing tones nearest to the centre of the previous voicing
        centre = int(np.mean(prev_notes))
        for t in tones:
            if t not in {n % 12 for n in notes} and len(notes) < n_voices:
                extra = min((n for n in candidates if n % 12 == t), key=lambda n: abs(n - centre))
                notes.append(extra)
        notes = sorted(notes)[:n_voices]
    else:
        root = root_pc if root_pc is not None else tones[0]
        bass = min((n for n in candidates if n % 12 == root), default=candidates[0])
        notes = [bass]
        for t in tones:
            if t != root and len(notes) < n_voices:
                notes.append(min((n for n in candidates if n % 12 == t and n > bass), default=bass + 12))
        notes = sorted(notes)
    return notes


class Dynamics:
    """Singer level (dB) -> CC11 value: smoothed (attack 0.15 s, release 1.2 s), 2:1 compressed, -8 dB (D15)."""

    def __init__(self, fps=62.5, attack_s=0.15, release_s=1.2, ratio=2.0, offset_db=-8.0, floor_db=-50.0, top_db=-10.0):
        self.a_up, self.a_dn = 1 - np.exp(-1 / (attack_s * fps)), 1 - np.exp(-1 / (release_s * fps))
        self.ratio, self.offset, self.floor, self.top = ratio, offset_db, floor_db, top_db
        self.level = floor_db

    def __call__(self, db):
        a = self.a_up if db > self.level else self.a_dn
        self.level += a * (db - self.level)
        comp = self.top + (self.level - self.top) / self.ratio + self.offset
        return int(np.clip((comp - self.floor) / (self.top - self.floor), 0, 1) * 127)


class Renderer:
    def __init__(self, sink, channel=0, n_voices=3, min_hold_s=0.3, fps=62.5, legato_s=0.03, follow_tuning=False):
        self.sink, self.ch, self.n_voices = sink, channel, n_voices
        self.min_hold, self.fps, self.legato = int(min_hold_s * fps), fps, legato_s
        self.notes, self.since_change, self.last_cc = [], 10 ** 6, -1
        self.dyn = Dynamics(fps)
        self.follow_tuning = follow_tuning
        self.pcs_playing = None

    def step(self, pcs, root_pc=None, confidence=0.7, level_db=None, cents=0.0):
        """Call once per frame with the chord to play (12-vector or None for silence)."""
        self.since_change += 1
        if level_db is not None:
            cc = self.dyn(level_db)
            if abs(cc - self.last_cc) >= 2:
                self.sink.control(self.ch, 11, cc)
                self.last_cc = cc
        if self.follow_tuning:
            self.sink.pitch_bend(self.ch, int(np.clip(cents / 200 * 8192, -8192, 8191)))      # bend range ±2 st
        want = None if pcs is None else tuple(int(i) for i in np.flatnonzero(np.asarray(pcs) > 0))
        if want == self.pcs_playing:
            return
        if self.since_change < self.min_hold and want is not None and self.pcs_playing is not None:
            return
        new = voice_chord(list(want), self.notes or None, self.n_voices, root_pc) if want else []
        vel = int(40 + 60 * float(np.clip(confidence, 0, 1)))
        for n in new:
            if n not in self.notes:
                self.sink.note_on(self.ch, n, vel)
        if self.legato and self.notes:
            self.sink.sleep(self.legato)
        for n in self.notes:
            if n not in new:
                self.sink.note_off(self.ch, n)
        self.notes, self.pcs_playing, self.since_change = new, want, 0

    def all_off(self):
        for n in self.notes:
            self.sink.note_off(self.ch, n)
        self.notes, self.pcs_playing = [], None


class PrintSink:
    def __init__(self, quiet=False):
        self.events, self.quiet = [], quiet

    def _log(self, *a):
        self.events.append(a)
        if not self.quiet:
            print("   MIDI", *a)

    def note_on(self, ch, n, vel):
        self._log("on", n, vel)

    def note_off(self, ch, n):
        self._log("off", n)

    def control(self, ch, cc, v):
        self._log(f"cc{cc}", v)

    def pitch_bend(self, ch, v):
        self._log("bend", v)

    def sleep(self, s):
        pass


class MidiSink:
    """Any MIDI output port via mido (python-rtmidi). Name a port or a substring; `virtual=True` creates one."""

    def __init__(self, port=None, virtual=False, program=40):
        import mido
        if virtual:
            self.out = mido.open_output(port or "singalong", virtual=True)
        else:
            names = mido.get_output_names()
            match = next((n for n in names if port is None or port.lower() in n.lower()), None)
            if match is None:
                raise RuntimeError(f"no MIDI output matching {port!r}; available: {names}")
            self.out = mido.open_output(match)
        self.mido = mido
        self.out.send(mido.Message("program_change", program=program))                        # 40 = GM violin

    def note_on(self, ch, n, vel):
        self.out.send(self.mido.Message("note_on", channel=ch, note=n, velocity=vel))

    def note_off(self, ch, n):
        self.out.send(self.mido.Message("note_off", channel=ch, note=n))

    def control(self, ch, cc, v):
        self.out.send(self.mido.Message("control_change", channel=ch, control=cc, value=v))

    def pitch_bend(self, ch, v):
        self.out.send(self.mido.Message("pitchwheel", channel=ch, pitch=v))

    def sleep(self, s):
        time.sleep(s)


class FluidSynthSink:
    """In-process FluidSynth (pip install pyfluidsynth; needs libfluidsynth and a .sf2 with a violin, GM 40)."""

    def __init__(self, sf2, program=40, driver=None, gain=0.8):
        import fluidsynth
        self.fs = fluidsynth.Synth(gain=gain)
        self.fs.start(driver=driver)
        sfid = self.fs.sfload(str(sf2))
        self.fs.program_select(0, sfid, 0, program)

    def note_on(self, ch, n, vel):
        self.fs.noteon(ch, n, vel)

    def note_off(self, ch, n):
        self.fs.noteoff(ch, n)

    def control(self, ch, cc, v):
        self.fs.cc(ch, cc, v)

    def pitch_bend(self, ch, v):
        self.fs.pitch_bend(ch, v)

    def sleep(self, s):
        time.sleep(s)


class WavSink:
    """Offline sink: records the violin's MIDI events with timestamps and renders them to a WAV file.

    For listening to the demo without FluidSynth / a MIDI synth (e.g. in the SDK container). The tone is a
    simple additive string sound (harmonics 1..7, light vibrato, 80 ms attack, 150 ms release), shaped by
    velocity, CC11 (dynamics, D15) and pitch bend (±2 semitones, D17). Optionally mixes in the singer.
    The caller sets the clock with set_time(seconds) once per frame (live.py does).
    """

    def __init__(self, path, sr=16000):
        self.path, self.sr, self.t = str(path), sr, 0.0
        self.on, self.notes = {}, []               # note -> (t0, vel); finished (t0, t1, note, vel)
        self.cc11, self.bend = [(0.0, 100)], [(0.0, 0)]

    def set_time(self, t):
        self.t = float(t)

    def note_on(self, ch, n, vel):
        if n in self.on:
            self.note_off(ch, n)
        self.on[n] = (self.t, vel)

    def note_off(self, ch, n):
        if n in self.on:
            t0, vel = self.on.pop(n)
            self.notes.append((t0, self.t, n, vel))

    def control(self, ch, cc, v):
        if cc == 11:
            self.cc11.append((self.t, v))

    def pitch_bend(self, ch, v):
        self.bend.append((self.t, v))

    def sleep(self, s):
        pass

    def close(self, voice=None, stereo=False, violin_db=-4.0, release_s=0.15):
        """Render and write the file. voice: mono float array at self.sr (the singer).

        Mono: singer + violin mixed. stereo=True: singer left, violin right. With a voice, levels follow the
        typical singing loudness, and the violin is gated to the singing (silent before the first note).
        """
        import wave
        for n in list(self.on):
            self.note_off(0, n)
        sr = self.sr
        dur = max([t1 for _, t1, _, _ in self.notes] + [self.t])
        if voice is not None:
            dur = max(dur, len(voice) / sr)
        N = int((dur + release_s) * sr) + 1
        t = np.arange(N) / sr
        ct, cv = np.array(self.cc11, dtype=float).T
        expr = np.interp(t, ct, cv) / 127.0
        bt, bv = np.array(self.bend, dtype=float).T
        bend_st = np.interp(t, bt, bv) / 8192.0 * 2.0
        out = np.zeros(N)
        for t0, t1, n, vel in self.notes:
            a, b = int(t0 * sr), min(int((t1 + release_s) * sr), N)
            if b <= a:
                continue
            tt = t[a:b]
            f = 440.0 * 2 ** ((n - 69 + bend_st[a:b]) / 12) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * tt))
            ph = 2 * np.pi * np.cumsum(f) / sr
            tone = sum(np.sin(h * ph) / h ** 1.2 for h in range(1, 8))
            env = np.clip((tt - t0) / 0.08, 0, 1) * np.clip((t1 + release_s - tt) / release_s, 0, 1)
            out[a:b] += tone * env * (vel / 127.0)
        out *= expr
        vio = out
        v = np.zeros(N)
        if voice is not None and len(voice):
            v[:min(len(voice), N)] = np.asarray(voice, dtype=float)[:N]
            # loudness from the typical singing level, not from the loudest click of the phone recording
            hop = sr // 50                                                  # 20 ms
            nb = N // hop
            env = np.sqrt(np.mean(v[:nb * hop].reshape(nb, hop) ** 2, axis=1) + 1e-12)
            singing = env > np.percentile(env, 95) * 10 ** (-30 / 20)    # within 30 dB of the loud parts
            v_rms = float(np.sqrt(np.mean(env[singing] ** 2))) if singing.any() else float(env.max())
            v = v / (v_rms + 1e-9) * 0.12
            # violin only while someone sings: gate opens in ~50 ms, holds through breaths (closes after ~1.5 s)
            gate, g = np.zeros(nb), 0.0
            for i, on in enumerate(singing):
                g = min(1.0, g + 0.4) if on else max(0.0, g - 1 / 75)
                gate[i] = g
            vio = vio * np.interp(np.arange(N), np.arange(nb) * hop + hop // 2, gate)
            mask = np.zeros(N, bool)
            mask[:nb * hop] = np.repeat(singing, hop)                          # N is not a multiple of hop
            on_rms = np.sqrt(np.mean(vio[mask] ** 2)) if mask.any() else 0.0
            vio = vio / (on_rms + 1e-9) * 0.12 * 10 ** (violin_db / 20)          # violin ~4 dB under the singer
        else:
            vio = vio / (np.abs(vio).max() + 1e-9) * 0.5
        chans = [v, vio] if stereo else [v + vio]
        y = np.stack(chans, 1)
        y = np.tanh(y * 1.2) / np.tanh(1.2)                                # soft limiter: peaks never clip
        with wave.open(self.path, "wb") as w:
            w.setnchannels(len(chans))
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
        return self.path


def chord_label(pcs, root_pc=None):
    tones = [i for i in range(12) if pcs[i] > 0]
    if root_pc is None:
        for r in tones:
            for q, iv in ag.QUALITIES.items():
                if {(r + i) % 12 for i in iv} == set(tones):
                    return ag.chord_name(r, q)
        return "+".join(ag.NAMES[t] for t in tones)
    for q, iv in ag.QUALITIES.items():
        if {(root_pc + i) % 12 for i in iv} == set(tones):
            return ag.chord_name(root_pc, q)
    return ag.NAMES[root_pc] + "?"
