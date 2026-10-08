"""Software-only tests of what happens when the board answers late. No board.

A stand-in worker answers chosen windows late. Three rules can stop a run on slowness, and each
must keep working: the count of slow round trips in a row (--slow-limit), the limit on how many
frames one window may deliver (windowing.py), and the per-window timeout.
"""
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'data', 'live_chip')
WAV = os.path.join(OUT, 'tones.wav')
DRIVER = '''
import argparse, sys
from pathlib import Path
sys.path.insert(0, {here!r})
import live_chip
a = argparse.Namespace(mode='chip', source='wav:' + {wav!r}, period=.5, playout_delay=1.25, duration={duration}, sink='print',
                       selector='pitch', bpm=80, verify_reference=False, container='none', metrics=Path({metrics!r}),
                       window_timeout=3.0, start_attempts=1,
                       worker_command=[sys.executable, {fake!r}, 'delays', {delays!r}])
{extra}
live_chip.LiveRun(a).run()
'''
TONIGHT = '5:0.55,6:1.6,7:0.62'        # like the run of 2026-10-08: a stall with a slightly slow window on each side


class SlowRoundTripTests(unittest.TestCase):
    def run_loop(self, name, delays, slow_limit=None, duration=11):
        metrics = os.path.join(OUT, f'test_slow_{name}.json')
        if os.path.exists(metrics):
            os.remove(metrics)
        extra = '' if slow_limit is None else f'a.slow_limit = {slow_limit}'
        code = DRIVER.format(here=HERE, wav=WAV, duration=duration, metrics=metrics, delays=delays, extra=extra,
                             fake=os.path.join(HERE, 'fake_worker.py'))
        with open(metrics + '.log', 'w') as log:
            exit_code = subprocess.run([sys.executable, '-c', code], stdout=log, stderr=log, timeout=90).returncode
        with open(metrics) as f:
            return exit_code, json.load(f)

    def test_1_default_still_stops_at_three_in_a_row(self):
        exit_code, m = self.run_loop('default', TONIGHT)
        self.assertNotEqual(exit_code, 0)
        self.assertIn('3 consecutive round trips exceeded update period', m['error'])
        self.assertEqual(m['longest_slow_streak'], 3)
        self.assertEqual(m['shutdown']['notes_off'], 'ok')

    def test_2_limit_six_rides_through_the_same_pattern(self):
        exit_code, m = self.run_loop('limit6', TONIGHT, slow_limit=6)
        self.assertEqual(exit_code, 0)
        self.assertTrue(m['completed'])                       # every frame reached the selector once, in order
        self.assertEqual(m['longest_slow_streak'], 3)
        self.assertGreaterEqual(m['slow_round_trips'], 3)
        self.assertGreaterEqual(m['skipped_updates'], 1)      # it caught up by skipping windows
        self.assertGreater(m['late_outputs'], 0)              # the stall made frames late ...
        self.assertLess(m['output_lateness_seconds_summary']['median'], 0.02)   # ... but nothing stayed late
        self.assertEqual(m['duplicate_frames'], 0)

    def test_3_lasting_slowness_still_stops_at_the_limit(self):
        exit_code, m = self.run_loop('lasting', '4+:0.6', slow_limit=6, duration=14)
        self.assertNotEqual(exit_code, 0)
        self.assertIn('6 consecutive round trips exceeded update period', m['error'])
        self.assertEqual(m['longest_slow_streak'], 6)

    def test_4_window_timeout_is_unchanged(self):
        exit_code, m = self.run_loop('timeout', '5:3.5', slow_limit=6)
        self.assertNotEqual(exit_code, 0)
        self.assertIn('Board timeout', m['error'])

    def test_5_one_stall_beyond_the_frame_limit_still_stops(self):
        exit_code, m = self.run_loop('gap', '5:2.4', slow_limit=6)
        self.assertNotEqual(exit_code, 0)
        self.assertIn('Output gap exceeds valid post-burn-in window', m['error'])


if __name__ == '__main__':
    unittest.main()
