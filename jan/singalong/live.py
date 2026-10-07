#!/usr/bin/env python3
"""Laptop side of the demo: audio or board spikes in -> chord selector -> violin out, live.

Sources (--source):
  wav:<file>          any audio file (ffmpeg); --realtime plays it at 1x, else as fast as possible
  npz:<file>          a precomputed recording (data/<set>/<id>.npz) -> bands straight in
  mic[:<device>]      laptop microphone via sounddevice (pip install sounddevice), 16 kHz
  board:<serial port> the Pulsar EVK streaming evidence as JSON lines (Pulsar/IF_DNR/deploy_live.py);
                      the audio front-end and the SNN then run on the chip, this script only selects + renders

Evidence (--evidence), for audio sources:
  baseline            harmonic-sum pitch-class salience from the bands (no SNN; D01 option D)
  snn:<weights.npz>   exported int8 network, integer simulation on the laptop (singalong.snn_numpy)
  torch:<checkpoint>  train_torch.py checkpoint in PyTorch

Selector (--selector): key (any song: 12 keys x 7 degree chords) | song (follows the Amazing Grace progression)
Sink (--sink): print | midi[:<port substring>] | virtual[:<name>] | fluidsynth:<file.sf2>

Examples
  python3 live.py --source wav:../../data/synth40/synth00_000.wav --evidence baseline --selector song --sink print
  python3 live.py --source mic --evidence snn:../../data/runs/<run>/weights_int8.npz --selector song --sink fluidsynth:violin.sf2
  python3 live.py --source board:/dev/ttyUSB1 --selector song --sink midi:fluid
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from singalong import amazing_grace as ag  # noqa: E402
from singalong import evidence as ev  # noqa: E402
from singalong import frontend as fe  # noqa: E402
from singalong import renderer as rd  # noqa: E402
from singalong import selector as sel  # noqa: E402


def make_sink(spec):
    kind, _, arg = spec.partition(":")
    if kind == "print":
        return rd.PrintSink(quiet=True)
    if kind == "midi":
        return rd.MidiSink(arg or None)
    if kind == "virtual":
        return rd.MidiSink(arg or "singalong", virtual=True)
    if kind == "fluidsynth":
        return rd.FluidSynthSink(arg)
    raise SystemExit(f"unknown sink {spec}")


def make_evidence(spec):
    kind, _, arg = spec.partition(":")
    if kind == "baseline":
        return ev.BaselineEvidence()
    if kind == "snn":
        return ev.NumpySnnEvidence(arg)
    if kind == "torch":
        return ev.TorchSnnEvidence(arg)
    raise SystemExit(f"unknown evidence {spec}")


def make_selector(kind, bpm):
    if kind == "song":
        return sel.SongHMM(bpm=bpm)
    if kind == "score":
        return sel.ScoreFollowerHMM(bpm=bpm)
    return sel.KeyChordHMM(degree_transitions=sel.degree_transitions_from_sequence(ag.degree_sequence()))


def audio_frames(source, realtime):
    """Yield (bands[48], level_db, voiced) frames from an audio source."""
    kind, _, arg = source.partition(":")
    if kind == "wav":
        y = fe.decode_audio(arg)
        X, v, db = fe.normalise(fe.log_bands(y), fe.voiced(y)), fe.voiced(y), fe.level_db(y)
        t0 = time.time()
        for i in range(len(X)):
            if realtime:
                time.sleep(max(0, t0 + i / fe.FPS - time.time()))
            yield X[i], float(db[i]), bool(v[i])
    elif kind == "npz":
        d = np.load(arg)
        X = fe.normalise(d["X"].astype(np.float32), d["voiced"].astype(bool))
        t0 = time.time()
        for i in range(len(X)):
            if realtime:
                time.sleep(max(0, t0 + i / fe.FPS - time.time()))
            yield X[i], float(d["X"][i].max() * 3.01 - 100), bool(d["voiced"][i])      # level ≈ from log2 energy
    elif kind == "mic":
        import queue
        import sounddevice as sd
        q = queue.Queue()
        dev = int(arg) if arg.isdigit() else (arg or None)
        with sd.InputStream(samplerate=fe.SR, channels=1, dtype="float32", blocksize=fe.HOP, device=dev,
                            callback=lambda indata, frames, t, status: q.put(indata[:, 0].copy())):
            front = fe.StreamingFrontEnd()
            while True:
                for bands, db, v in front.feed(q.get()):
                    yield bands, db, v
    else:
        raise SystemExit(f"unknown audio source {source}")


def board_frames(port, baud=115200):
    import serial
    parser = ev.BoardEvidence()
    with serial.Serial(port, baud, timeout=1) as s:
        while True:
            line = s.readline().decode("utf-8", "ignore")
            out = parser.parse(line)
            if out is not None:
                yield out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--evidence", default="baseline")
    ap.add_argument("--selector", choices=["key", "song", "score"], default="score")
    ap.add_argument("--sink", default="print")
    ap.add_argument("--bpm", type=float, default=80)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--use", choices=["melody", "chord", "both"], default="both", help="which SNN head feeds the HMM")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sink = make_sink(a.sink)
    hmm = make_selector(a.selector, a.bpm)
    render = rd.Renderer(sink, fps=fe.FPS)
    is_board = a.source.startswith("board:")
    evid = None if is_board else make_evidence(a.evidence)
    last_label, n = None, 0
    try:
        if is_board:
            frames = ((m, c, None, True) for m, c in board_frames(a.source.partition(":")[2]))
        else:
            def gen():
                for bands, db, v in audio_frames(a.source, a.realtime):
                    m, c = evid(bands, v)
                    yield m, c, db, v
            frames = gen()
        for mel, chord, db, voiced in frames:
            n += 1
            if a.use == "melody":
                chord = None
            elif a.use == "chord":
                mel = None
            state, post = hmm.step(mel, chord, onset=True)
            pcs = hmm.pcs(state)
            root = hmm.em.T[state].argmax()
            render.step(pcs, root_pc=int(root), confidence=float(post[state]), level_db=db)
            label = hmm.names[state]
            if label != last_label and not a.quiet:
                extra = f"  (song position {hmm.position(state)[1] + 1}/{hmm.nE})" if a.selector in ("song", "score") else ""
                print(f"{n / fe.FPS:7.2f} s  {label:28s} p={post[state]:.2f}  notes {render.notes}{extra}", flush=True)
                last_label = label
    except KeyboardInterrupt:
        pass
    finally:
        render.all_off()
        if isinstance(sink, rd.PrintSink):
            print(f"{n} frames, {sum(1 for e in sink.events if e[0] == 'on')} note-ons")


if __name__ == "__main__":
    main()
