#!/usr/bin/env python3
"""Live violin-like synth sink: no GarageBand, no FluidSynth, nothing to install.

Takes the same calls as the existing sinks (note_on, note_off, control, pitch_bend, sleep) and
plays them through sounddevice. The tone is the additive string sound of renderer.WavSink
(harmonics 1..7, light vibrato, 80 ms attack, 150 ms release, CC11 expression). Audio is made
in sounddevice's own callback thread, so the callers never wait for it.

Stand-alone check with a known melody (no board, no microphone, no model):

    python yujie/chip/synth_sink.py
"""
import threading
import time

import numpy as np

ATTACK_S, RELEASE_S, VOICE_GAIN = 0.08, 0.15, 0.15


class LiveSynthSink:
    def __init__(self, device=None, gain=1.0):
        import sounddevice as sd
        self.lock = threading.Lock()
        self.voices = {}                         # note -> dict(phase, env, gate, vel)
        self.expr = self.expr_target = 100 / 127
        self.bend, self.gain, self.clock = 0.0, gain, 0
        self.peak, self.frames, self.status_count = 0.0, 0, 0
        self.stream = sd.OutputStream(device=device, channels=1, dtype='float32', callback=self._callback)
        self.sr = float(self.stream.samplerate)
        self.label = f"Python synth -> {sd.query_devices(self.stream.device, 'output')['name']}"
        self.stream.start()

    def _callback(self, outdata, frames, _time, status):
        if status:
            self.status_count += 1
        with self.lock:
            voices = list(self.voices.items())
        steps = np.arange(1, frames + 1)
        vibrato = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * (self.clock + steps) / self.sr)
        out = np.zeros(frames)
        for note, v in voices:
            f = 440.0 * 2 ** ((note - 69 + self.bend) / 12) * vibrato
            phase = v['phase'] + 2 * np.pi * np.cumsum(f) / self.sr
            slope = 1 / (ATTACK_S * self.sr) if v['gate'] else -1 / (RELEASE_S * self.sr)
            env = np.clip(v['env'] + slope * steps, 0, 1)
            top = int(min(7, (self.sr / 2) // f.max()))
            out += sum(np.sin(h * phase) / h ** 1.2 for h in range(1, top + 1)) * env * (v['vel'] / 127)
            v['phase'], v['env'] = float(phase[-1] % (2 * np.pi)), float(env[-1])
        with self.lock:
            for note, v in voices:
                if not v['gate'] and v['env'] <= 0 and self.voices.get(note) is v:
                    del self.voices[note]
        expr = np.linspace(self.expr, self.expr_target, frames + 1)[1:]     # no zipper noise
        self.expr = float(expr[-1])
        out = np.tanh(out * expr * VOICE_GAIN * self.gain)
        self.clock += frames
        self.frames += frames
        self.peak = max(self.peak, float(np.max(np.abs(out))) if frames else 0.0)
        outdata[:, 0] = out

    def note_on(self, ch, n, vel):
        with self.lock:
            voice = self.voices.setdefault(int(n), dict(phase=0.0, env=0.0, gate=True, vel=vel))
            voice.update(gate=True, vel=vel)

    def note_off(self, ch, n):
        with self.lock:
            if int(n) in self.voices:
                self.voices[int(n)]['gate'] = False

    def all_notes_off(self):
        with self.lock:
            for v in self.voices.values():
                v['gate'] = False

    def control(self, ch, cc, v):
        if cc == 11:
            self.expr_target = v / 127

    def pitch_bend(self, ch, v):
        self.bend = v / 8192 * 2

    def sleep(self, s):
        pass                                     # the release tail already overlaps the next note

    def close(self):
        with self.lock:
            for v in self.voices.values():
                v['gate'] = False
        time.sleep(RELEASE_S + 0.1)
        self.stream.stop()
        self.stream.close()
        return f'{self.label} (peak {self.peak:.2f}, {self.frames / self.sr:.1f} s played)'


if __name__ == '__main__':
    sink = LiveSynthSink()
    print(f'Playing the first phrase of Amazing Grace: {sink.label}', flush=True)
    sink.control(0, 11, 110)
    try:
        for note, beats in ((62, 1), (67, 2), (71, .5), (67, .5), (71, 2), (69, 1), (67, 2), (64, 1), (62, 2)):
            sink.note_on(0, note, 90)
            time.sleep(beats * 0.45)
            sink.note_off(0, note)
        for note in (55, 59, 62):                # closing chord: three voices at once
            sink.note_on(0, note, 80)
        time.sleep(1.5)
    finally:
        print(sink.close())
        print(f'output stream warnings: {sink.status_count}')
