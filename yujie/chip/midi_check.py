#!/usr/bin/env python3
"""Does MIDI from this Mac make GarageBand sound? No board, no microphone, no model.

Sends a short G major phrase through the existing MidiSink to the named port. GarageBand must be
running with a software-instrument track selected. Sending succeeds even if nothing sounds, so
only a person listening can confirm the result.

    python yujie/chip/midi_check.py
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'jan/singalong'))
from singalong import renderer as rd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--port', default='GarageBand')
a = ap.parse_args()
sink = rd.MidiSink(a.port)
print(f'Sending to MIDI port: {sink.out.name}', flush=True)
sink.control(0, 11, 110)                      # the live renderer drives expression from the singer's level
try:
    for note in (55, 59, 62, 67):             # G3 B3 D4 G4, one after another
        print(f'  note {note}', flush=True)
        sink.note_on(0, note, 90)
        time.sleep(0.6)
        sink.note_off(0, note)
    print('  chord G3 B3 D4', flush=True)
    for note in (55, 59, 62):
        sink.note_on(0, note, 90)
    time.sleep(2)
finally:
    for note in (55, 59, 62, 67):
        sink.note_off(0, note)
    sink.out.close()
print('Sent. If you heard nothing: is a software-instrument track selected in GarageBand, and is the volume up?')
