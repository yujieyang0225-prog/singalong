"""Software-only tests of what the live page is given to draw. No board, no audio, no browser.

Covers the note timeline, the run-length encoded chip output and the decoder state that
live_view.View collects, and that everything /state had before is still there.
"""
import json
import queue
import threading
import time
from types import SimpleNamespace
import unittest

import live_view

# One /state response of the page before the timeline was added (laptop-only run of the tone file).
BEFORE = json.loads('''{"phase": "running", "error": null, "elapsed": 8.2, "mode": "baseline",
 "source": "wav:data/live_chip/tones.wav", "sink": "print", "port": "PrintSink", "audio_seconds": 8.0,
 "mic": {"db": -18.3, "voiced": true},
 "evidence": [0.0, 0.008, 0.015, 0.082, 0.0, 0.001, 0.0, 0.0, 0.452, 1.0, 0.556, 0.0],
 "follower": {"label": "A", "confidence": 0.47, "melody": null, "tracking": true, "tracked": 69, "chord": null,
              "key": null, "note_index": null},
 "midi": {"notes": [69], "cc11": 88, "note_ons": 4, "events": [
   {"seconds": 7.41, "kind": "off", "note": 67, "velocity": null}, {"seconds": 7.41, "kind": "on", "note": 69, "velocity": 60}]},
 "snn": {"windows": 0, "roundtrip": null, "recent_roundtrips": [], "exact": null, "age": null, "boot_seconds": null},
 "latency": {"last": 0.10326149998581968, "median": 0.10300279149669223, "playout_delay": 0.1, "late": 0, "skipped": 0,
             "connection_failures": 0, "delivered": 492}}''')


class Clock:
    """Stands in for View.now(), so a test decides what time it is."""

    def __init__(self, view):
        self.t = 0.0
        view.now = lambda: self.t


def tracker(**attributes):
    """An object with the attributes of pitch_tracker.PitchTracker that the page reads."""
    values = dict(names=live_view.NAMES + ['silence'], note=None, candidate=None, lead=0, switch=6, dwell=0,
                  min_note=9, quiet=0, release=25, fps=62.5)
    values.update(attributes)
    return SimpleNamespace(**values)


def frame(view, top=None, selector=None):
    """One 16 ms frame whose strongest pitch class is `top` (None: all twelve values are zero)."""
    melody = [0.0] * 12
    if top is not None:
        melody[top] = 1.0
    view.follow(selector or tracker(), melody, 12 if top is None else top, [1.0] * 13)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.view = live_view.View()
        self.clock = Clock(self.view)

    def send(self, t, kind, note, velocity=None):
        self.clock.t = t
        self.view.midi(kind, note, velocity)

    def test_1_1_on_and_off_give_segments(self):
        self.send(1.0, 'on', 60, 52)
        self.send(3.0, 'on', 64, 55)                          # the renderer sends the new note first,
        self.send(3.0, 'off', 60)                             # then ends the old one
        self.assertEqual(list(self.view.segments), [dict(start=1.0, end=3.0, note=60, velocity=52),
                                                    dict(start=3.0, end=None, note=64, velocity=55)])

    def test_1_1_note_off_without_note_on_is_ignored(self):
        self.send(1.0, 'off', 60)
        self.assertEqual(len(self.view.segments), 0)
        self.send(2.0, 'on', 64, 50)
        self.send(3.0, 'off', 60)                             # a different note: 64 stays open
        self.assertEqual(list(self.view.segments), [dict(start=2.0, end=None, note=64, velocity=50)])

    def test_1_1_same_pitch_twice_gives_two_segments(self):
        for t, kind in ((1.0, 'on'), (2.0, 'off'), (2.5, 'on'), (4.0, 'off')):
            self.send(t, kind, 60, 50 if kind == 'on' else None)
        self.assertEqual([(s['start'], s['end']) for s in self.view.segments], [(1.0, 2.0), (2.5, 4.0)])

    def test_1_1_same_note_sent_again_without_note_off_leaves_no_open_bar_behind(self):
        self.send(1.0, 'on', 60, 50)
        self.send(2.0, 'on', 60, 50)
        self.send(3.0, 'off', 60)
        self.assertEqual([(s['start'], s['end']) for s in self.view.segments], [(1.0, 2.0), (2.0, 3.0)])

    def test_1_2_histories_stay_within_their_limits(self):
        longest_segments = longest_runs = 0
        for i in range(10000):
            self.clock.t = i * 0.016
            self.view.midi('on' if i % 2 == 0 else 'off', 48 + (i // 2) % 24, 50)
            frame(self.view, i % 12)
            longest_segments = max(longest_segments, len(self.view.segments))
            longest_runs = max(longest_runs, len(self.view.runs))
        self.assertEqual(longest_segments, 120)
        self.assertEqual(longest_runs, 400)
        self.assertEqual(len(self.view.events), 8)

    def test_1_3_snapshot_holds_only_the_last_15_seconds(self):
        self.send(0.0, 'on', 60, 50); self.send(1.0, 'off', 60)       # ended 19 s ago: not shown
        self.send(4.0, 'on', 62, 50); self.send(6.0, 'off', 62)       # ended inside the window: shown
        self.send(18.0, 'on', 64, 50)                                 # still sounding: shown
        for t, top in ((2.0, 0), (5.0, 2), (5.5, 2), (19.0, 4)):      # runs: [2, 2] old, [5, 5.5] shown, [19, 19] shown
            self.clock.t = t
            frame(self.view, top)
        self.clock.t = 20.0
        s = self.view.snapshot()
        self.assertEqual(s['timeline']['now'], 20.0)
        self.assertEqual(s['timeline']['window'], 15)
        # Times are seconds from the left edge of the window, which is at 20 - 15 = 5 s.
        self.assertEqual(s['timeline']['notes'], [[-1.0, 1.0, 62, 50], [13.0, None, 64, 50]])
        self.assertEqual(s['raw'], [[0.0, 0.5, 2], [14.0, 14.0, 4]])
        self.assertEqual(len(self.view.segments), 3)                  # nothing was removed from the history itself

    def test_1_4_raw_output_is_run_length_encoded(self):
        for i in range(100):
            self.clock.t = i * 0.016
            frame(self.view, 7)
        self.assertEqual(len(self.view.runs), 1)
        start, end, top = self.view.runs[0]
        self.assertEqual((start, top), (0.0, 7))
        self.assertAlmostEqual(end, 99 * 0.016)
        for i, top in enumerate((0, 4, 0, 4, 0)):
            self.clock.t = 2.0 + i * 0.016
            frame(self.view, top)
        self.assertEqual([run[2] for run in self.view.runs], [7, 0, 4, 0, 4, 0])
        for i in range(10):                                           # all twelve values zero: one run of "none"
            self.clock.t = 3.0 + i * 0.016
            frame(self.view, None)
        self.assertEqual([run[2] for run in self.view.runs], [7, 0, 4, 0, 4, 0, None])
        self.assertAlmostEqual(self.view.runs[-1][1] - self.view.runs[-1][0], 9 * 0.016)

    def test_1_4_strongest_class_is_the_one_the_bars_label(self):
        melody = [0.0, 0.2, 0.9, 0.9, 0.1] + [0.0] * 7               # a tie: the first of the two, as on the page
        self.view.follow(tracker(), melody, 2, [1.0] * 13)
        self.assertEqual(self.view.runs[-1][2], 2)
        self.view.follow(tracker(), None, 12, [1.0] * 13)             # no values at all counts as "none"
        self.assertIsNone(self.view.runs[-1][2])

    def test_1_5_decoder_states_in_the_order_of_the_table(self):
        cases = [
            (dict(note=60, quiet=10, release=25, lead=3, candidate=1, dwell=50), dict(state='releasing', progress=0.4)),
            (dict(note=60, quiet=0, lead=3, switch=6, candidate=1, dwell=50),
             dict(state='candidate', progress=0.5, candidate='C#')),
            (dict(note=None, lead=6, switch=6, candidate=11), dict(state='candidate', progress=1.0, candidate='B')),
            (dict(note=60, dwell=75), dict(state='note', held=1.2)),
            (dict(note=None, quiet=40, dwell=75), dict(state='silence')),
        ]
        for attributes, expected in cases:
            selector = tracker(**attributes)
            before = dict(vars(selector))
            frame(self.view, 0, selector)
            decoder = self.view.snapshot()['decoder']
            self.assertEqual(decoder, dict(dict(progress=None, candidate=None, held=None), **expected), attributes)
            self.assertEqual(vars(selector), before)                  # the tracker is read, never written

    def test_1_5_decoder_state_is_there_before_the_first_frame(self):
        view, live_view.VIEW = live_view.VIEW, self.view              # tap_selector writes to the module's view
        try:
            live_view.tap_selector(tracker(step=lambda melody=None, chord=None, onset=True: (12, [1.0] * 13)))
        finally:
            live_view.VIEW = view
        self.assertEqual(self.view.snapshot()['decoder']['state'], 'silence')
        self.assertEqual(len(self.view.runs), 0)                      # nothing was heard yet

    def test_1_5_other_selectors_have_no_decoder_state(self):
        self.assertIsNone(self.view.snapshot()['decoder'])            # before the first frame
        chords = SimpleNamespace(names=['C in C @note1'] * 13)
        frame(self.view, 0, chords)
        self.assertIsNone(self.view.snapshot()['decoder'])


def running_view(windows=0):
    """A View in the state of a run: a stand-in loop and its metrics, a tracker, some notes."""
    view = live_view.View()
    view.args = SimpleNamespace(mode='baseline', source='wav:data/live_chip/tones.wav', sink='print', playout_delay=0.1)
    view.run = SimpleNamespace(total=500, epoch=None, stop=threading.Event(), errors=queue.Queue(), metrics=dict(
        windows=[dict(host_roundtrip_seconds=0.3812102079973556, submitted_seconds=2.0502083750034217 + i / 2,
                      exact_reference=None) for i in range(windows)],
        output_latency_seconds=[0.10326149998581968] * 400, late_outputs=0, skipped_updates=0, slow_round_trips=3,
        connection_failures=0, delivered_frames=492))
    view.phase, view.port, view.cc11 = 'running', 'PrintSink', 88
    view.mic_db, view.voiced = -18.3, True
    return view


def same_types(test, old, new, path='/state'):
    """Every key of `old` is in `new` with a value of the same type. A null in `old` only needs its key."""
    if isinstance(old, dict):
        test.assertIsInstance(new, dict, path)
        for key, value in old.items():
            test.assertIn(key, new, path)
            same_types(test, value, new[key], f'{path}.{key}')
    elif isinstance(old, list):
        test.assertIsInstance(new, list, path)
        for value in new if old else []:
            same_types(test, old[0], value, path + '[]')
    elif old is not None:
        test.assertIs(type(new), type(old), path)


class StateTests(unittest.TestCase):
    def test_1_6_every_earlier_key_is_still_there_with_its_type(self):
        view = running_view()
        clock = Clock(view)
        selector = tracker(note=69, dwell=30)
        for t, kind, note, velocity in ((5.41, 'on', 67, 56), (7.41, 'on', 69, 60), (7.41, 'off', 67, None)):
            clock.t = t
            view.midi(kind, note, velocity)
        view.follow(selector, BEFORE['evidence'], 9, [0.47] * 13)
        clock.t = 8.2
        state = json.loads(view.state())
        same_types(self, BEFORE, state)
        self.assertEqual({key: state[key] for key in ('phase', 'mode', 'port', 'audio_seconds', 'elapsed')},
                         dict(phase='running', mode='baseline', port='PrintSink', audio_seconds=8.0, elapsed=8.2))
        self.assertEqual(state['midi']['events'][:2], BEFORE['midi']['events'][:2])
        self.assertEqual(state['follower'], BEFORE['follower'])
        self.assertEqual(state['latency']['slow'], 3)                 # new: the loop's count of slow round trips
        self.assertEqual(sorted(set(state) - set(BEFORE)), ['decoder', 'model', 'raw', 'timeline'])

    def test_model_is_named_only_when_the_chip_runs_it(self):
        view = running_view()
        self.assertIsNone(view.snapshot()['model'])                   # laptop only: no network is running
        view.args = SimpleNamespace(mode='chip', model='A_vsP_chip_shift3')
        self.assertEqual(view.snapshot()['model'], 'A_vsP_chip_shift3')
        view.args = SimpleNamespace(mode='chip', model=None)          # --model left out: the loop's default
        self.assertEqual(view.snapshot()['model'], live_view.live_chip.DEFAULT_MODEL)

    def test_1_7_snapshot_is_quick_and_small_with_full_histories(self):
        """The largest /state there can be: both histories full and all of it inside the 15 s window, in a
        chip run that has lasted ten minutes (60 round trips in the list, 8 events)."""
        view = running_view(windows=80)
        view.args.mode = 'chip'
        clock = Clock(view)
        selector = tracker(note=60, lead=3, candidate=10, dwell=40)
        for i in range(400):                                          # a new strongest class every second frame
            for half in range(2):
                clock.t = 587.2 + (2 * i + half) * 0.016
                frame(view, 10 + i % 2, selector)
        for i in range(120):                                          # 120 notes of 0.1 s
            clock.t = 588.03 + i * 0.1
            view.midi('on', 100 + i % 8, 100)
            view.midi('off', 100 + (i - 1) % 8)
        clock.t = 600.0
        state = json.loads(view.state())
        self.assertEqual((len(state['timeline']['notes']), len(state['raw'])), (120, 400))
        self.assertEqual((len(state['midi']['events']), len(state['snn']['recent_roundtrips'])), (8, 60))
        size, calls = len(view.state().encode()), 1000
        started = time.perf_counter()
        for _ in range(calls):
            view.snapshot()
        average_ms = (time.perf_counter() - started) * 1000 / calls
        print(f'\n  /state with full histories: {size} bytes; snapshot() {average_ms:.3f} ms on average', end=' ')
        self.assertLess(size, 12000)
        self.assertLess(average_ms, 5.0)


class PageTests(unittest.TestCase):
    """What can be checked in the page's source without a browser."""

    def test_2_2_the_page_loads_nothing_from_outside(self):
        for forbidden in ('http://', 'https://', '@import', 'url('):
            self.assertNotIn(forbidden, live_view.PAGE)

    def test_2_3_removed_and_kept_parts(self):
        for gone in ('id="keys"', 'id="events"', 'id="midinotes"', 'id="clock"', 'class="card tile"',
                     'MIDI notes sounding', 'Last MIDI messages', 'Notes on the keyboard', 'prefers-color-scheme'):
            self.assertNotIn(gone, live_view.PAGE)
        for kept in ('id="mode"', 'id="status"', 'id="banner"', 'id="note"', 'id="bars"', 'id="barlabels"',
                     'id="micdb"', 'id="micbar"', 'Voice detected', "'Silence'", '−70 dB', '0 dB',   # the microphone monitor stays
                     'id="timeline"', 'id="rawstrip"', 'id="decoder"', 'Loudness (expression)', 'voice to note',
                     'CHIP · Pulsar C1', 'LAPTOP ONLY', '104 spiking neurons', 'octave chosen on the laptop'):
            self.assertIn(kept, live_view.PAGE)
        for text in ('Running on the chip', 'Starting the board', 'No chip result for', 'Stopped with an error',
                     "'Stopped'", 'Not connected'):
            self.assertIn(text, live_view.PAGE)

    def test_2_5_words_the_demo_must_not_claim(self):
        for word in ('velocity', 'pitch bend', 'pitchbend', 'legato'):
            self.assertNotIn(word, live_view.PAGE.lower())

    def test_colours_are_the_ones_of_the_deck(self):
        for token in ('--page:#f3f1ec', '--card:#fcfcfb', '--line:#e8e5de', '--ink:#16202e', '--text:#4a5363',
                      '--muted:#5f6673', '--hears:#2563c9', '--hears-soft:#e6eefb', '--decides:#177e6b', '--plays:#c2531b',
                      '--accent:#a8441a', '--rule:#000000'):
            self.assertIn(token, live_view.PAGE)


if __name__ == '__main__':
    unittest.main()
