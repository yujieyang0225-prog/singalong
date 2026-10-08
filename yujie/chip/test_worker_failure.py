"""Software-only tests: the live loop notices a dead or hung worker and always exits. No board."""
import os
import subprocess
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = '''
import argparse, sys
from pathlib import Path
sys.path.insert(0, {here!r})
import live_chip
a = argparse.Namespace(mode='chip', source='wav:' + {wav!r}, period=.5, playout_delay=1., duration=12, sink='print',
                       selector='score', bpm=80, verify_reference=False, container='none', metrics=Path({metrics!r}),
                       window_timeout=1.0, start_attempts=1,
                       worker_command=[sys.executable, {fake!r}, {mode!r}])
live_chip.LiveRun(a).run()
'''
WAV = os.path.join(HERE, '..', '..', 'data', 'live_chip', 'tones.wav')


class WorkerFailureTests(unittest.TestCase):
    def run_loop(self, mode):
        """Run the real loop against a fake worker. Returns (seconds, exit code, stderr, fake worker pid)."""
        metrics = os.path.join(HERE, '..', '..', 'data', 'live_chip', f'test_worker_{mode}.json')
        code = DRIVER.format(here=HERE, wav=WAV, metrics=metrics, fake=os.path.join(HERE, 'fake_worker.py'), mode=mode)
        log = metrics + '.log'                      # a file, not a pipe: a surviving fake worker would hold a pipe open
        started = time.perf_counter()
        with open(log, 'w') as out:
            code = subprocess.run([sys.executable, '-c', code], stdout=out, stderr=out, timeout=60).returncode
        return time.perf_counter() - started, code, open(log).read()

    def test_hung_worker_is_detected_and_terminated(self):
        seconds, code, err = self.run_loop('hang')
        self.assertNotEqual(code, 0)
        self.assertIn('Board timeout', err)
        self.assertIn('sending it a terminate signal', err)
        self.assertNotIn('still running', err)
        self.assertLess(seconds, 25)

    def test_worker_that_exits_is_detected_at_once(self):
        seconds, code, err = self.run_loop('exit')
        self.assertNotEqual(code, 0)
        self.assertIn('Board worker exited', err)
        self.assertLess(seconds, 12)                # ~3 s of audio before the third window, then stop

    def test_stubborn_worker_is_reported_not_killed_and_loop_still_exits(self):
        seconds, code, err = self.run_loop('stubborn')
        try:
            self.assertNotEqual(code, 0)
            self.assertIn('Board timeout', err)
            self.assertIn('still running and holds the', err)
            self.assertLess(seconds, 30)
        finally:                                     # the abandoned test helper, never a board worker
            subprocess.run(['pkill', '-KILL', '-f', 'fake_worker.py stubborn'])


if __name__ == '__main__':
    unittest.main()
