"""Software-only tests of how a run ends: notes released, metrics kept. No board."""
import json
import os
import signal
import subprocess
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'data', 'live_chip')
WAV = os.path.join(OUT, 'tones.wav')
DRIVER = '''
import argparse, json, signal, sys, time
from pathlib import Path
sys.path.insert(0, {here!r})
import live_chip
from singalong import renderer as rd

class Stamped(rd.PrintSink):                     # the existing print sink, with a time on every event
    def _log(self, *event):
        super()._log(*event)
        stamped.append([time.time(), *event])
        Path({events!r}).write_text(json.dumps(stamped))

stamped = []
live_chip.host.make_sink = lambda spec: Stamped(quiet=True)
def stop(*_):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, stop)
a = argparse.Namespace(mode={mode!r}, source='wav:' + {wav!r}, period=.5, playout_delay=1., duration={duration}, sink='print',
                       selector='score', bpm=80, verify_reference=False, container='none', metrics=Path({metrics!r}),
                       window_timeout=1.0, start_attempts=1, release={release},
                       worker_command={worker})
live_chip.LiveRun(a).run()
'''
OLD_FIELDS = ['mode', 'source', 'period_seconds', 'playout_delay_seconds', 'windows', 'capture_status', 'skipped_updates',
              'startup_frames_omitted', 'duplicate_frames', 'captured_frames', 'delivered_frames', 'late_outputs',
              'queue_peak', 'connection_failures', 'recoveries', 'feature_nonzero_frames', 'pcm_peak',
              'pcm_clipped_samples', 'completed', 'note_ons', 'wall_seconds', 'host_cpu_seconds', 'host_peak_rss_bytes',
              'frontend_seconds_summary', 'output_latency_seconds_summary', 'output_lateness_seconds_summary']


class ShutdownTests(unittest.TestCase):
    def start(self, name, mode='baseline', duration=30, worker=None, release=None):
        self.metrics = os.path.join(OUT, f'test_shutdown_{name}.json')
        self.events = self.metrics + '.events'
        for path in (self.metrics, self.events, self.metrics + '.partial'):
            if os.path.exists(path):
                os.remove(path)
        fake = None if worker is None else [sys.executable, os.path.join(HERE, 'fake_worker.py'), worker]
        code = DRIVER.format(here=HERE, events=self.events, mode=mode, wav=WAV, duration=duration, metrics=self.metrics,
                             worker=repr(fake), release=release)
        self.log = open(self.metrics + '.log', 'w')
        self.addCleanup(self.log.close)
        self.started = time.perf_counter()
        return subprocess.Popen([sys.executable, '-c', code], stdout=self.log, stderr=self.log)

    def unreleased(self):
        sounding = {}
        for _, kind, *rest in json.load(open(self.events)):
            if kind in ('on', 'off'):
                sounding[rest[0]] = sounding.get(rest[0], 0) + (1 if kind == 'on' else -1)
        return {n: c for n, c in sounding.items() if c}

    def read_metrics(self):
        with open(self.metrics) as f:
            return json.load(f)

    def test_1_worker_failure_releases_every_note_within_a_second(self):
        process = self.start('worker_fail', mode='chip', worker='hang')
        process.wait(timeout=40)
        m = self.read_metrics()
        self.assertIn('Board timeout', m['error'])
        self.assertGreater(m['note_ons'], 0)
        self.assertEqual(self.unreleased(), {})
        self.assertLessEqual(m['shutdown']['notes_off_seconds_after_stop'], 1.0)
        self.assertEqual(m['shutdown']['metrics'], 'ok')

    def test_2_terminate_signal_releases_notes_and_keeps_metrics(self):
        process = self.start('terminate')
        time.sleep(5)
        sent = time.perf_counter()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        self.assertLess(time.perf_counter() - sent, 5)
        m = self.read_metrics()
        self.assertFalse(m['completed'])
        self.assertGreater(m['delivered_frames'], 100)
        self.assertEqual(self.unreleased(), {})

    def test_3_second_signal_during_shutdown_does_not_lose_them(self):
        process = self.start('twice', mode='chip', worker='stubborn')      # its shutdown takes about 15 s
        try:
            time.sleep(5)
            process.send_signal(signal.SIGTERM)
            time.sleep(2)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=40)
            m = self.read_metrics()
            self.assertEqual(self.unreleased(), {})
            self.assertEqual(m['shutdown']['notes_off'], 'ok')
            self.assertEqual(m['shutdown']['metrics'], 'ok')
            self.assertIn('KeyboardInterrupt', json.dumps(m['shutdown']))  # the second signal was caught mid-shutdown
        finally:                                                           # the abandoned test helper only
            subprocess.run(['pkill', '-KILL', '-f', 'fake_worker.py stubborn'])

    def test_4_abrupt_kill_leaves_a_recent_checkpoint(self):
        process = self.start('killed')
        time.sleep(14)
        process.send_signal(signal.SIGKILL)                                # a baseline test run, never a board worker
        process.wait()
        killed = time.time()
        self.assertFalse(os.path.exists(self.metrics))
        partial = self.metrics + '.partial'
        self.assertLessEqual(killed - os.path.getmtime(partial), 10.5)
        with open(partial) as f:
            m = json.load(f)
        self.assertTrue(m['partial'])
        self.assertGreater(m['delivered_frames'], 100)

    def test_5_normal_run_keeps_every_old_field_and_cleans_up(self):
        process = self.start('normal', duration=8)
        self.assertEqual(process.wait(timeout=40), 0)
        m = self.read_metrics()
        self.assertEqual([f for f in OLD_FIELDS if f not in m], [])
        self.assertTrue(m['completed'])
        self.assertEqual(self.unreleased(), {})
        self.assertEqual(set(m['shutdown']) - {'notes_off_seconds_after_stop'},
                         {'notes_off', 'metrics_before_worker_close', 'threads', 'sink', 'worker', 'metrics'})
        self.assertTrue(all(v == 'ok' for k, v in m['shutdown'].items() if k != 'notes_off_seconds_after_stop'))
        self.assertFalse(os.path.exists(self.metrics + '.partial'))

    def test_6_chord_release_is_off_by_default_and_silences_when_asked(self):
        def notes_during_trailing_silence(release):
            # tones.wav starts with 1 s of silence: without release the follower's chord sounds through it
            process = self.start(f'release_{release}', duration=1.0, release=release)
            process.wait(timeout=40)
            return self.read_metrics()['note_ons']
        self.assertGreater(notes_during_trailing_silence(None), 0)
        self.assertEqual(notes_during_trailing_silence(0.1), 0)


if __name__ == '__main__':
    unittest.main()
