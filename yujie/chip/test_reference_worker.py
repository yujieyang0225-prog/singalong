"""Software-only tests of the simulated chip: reference_worker.py alone, and the live loop with --mode reference.
No board, no Docker, no SDK. The loop tests need data/live_chip/tones.wav (make_test_audio.py makes it).

Run from yujie/chip with the host Python:  python test_reference_worker.py
"""
import glob
import hashlib
import json
import os
import pathlib
import queue
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import chipref

WORKER = os.path.join(HERE, 'reference_worker.py')
WAV = os.path.join(ROOT, 'data', 'live_chip', 'tones.wav')
MODELS = ('A_vsP_chip_shift3', 'A_vocalset_P')                  # the validated model and the older default


def npz_of(model):
    return os.path.join(ROOT, 'handoff', 'runs', model, 'weights_int8.npz')


def real_windows(per_take=9):
    """Windows [125, 48] of real band features: from each of the 24 DAMP takes the `per_take` windows with the most
    voiced frames, none overlapping (216 in all). The network's output is sparse even for singing."""
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, 'handoff', 'testdata', 'damp250', '*.npz')))
             if not f.endswith('reference.npz')]
    windows = []
    for path in files:
        take = np.load(path)
        features, voiced = take['X'].astype(np.float32), take['voiced'].astype(float)
        starts = np.arange(0, len(features) - 125, 125)
        best = np.argsort([-voiced[s:s + 125].mean() for s in starts], kind='stable')[:per_take]
        windows += [features[starts[i]:starts[i] + 125] for i in sorted(best)]
    return windows


def random_windows(count=40):
    """Integer features 0..100 like the board worker's start-up input; they make the network fire about 50 times a window."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 101, (125, 48)).astype(np.float32) for _ in range(count)]


class Worker:
    """reference_worker.py as a subprocess, spoken to the way live_chip.ChipClient speaks to board_worker.py."""

    def __init__(self, model=MODELS[0], latency=0.0, npz=None):
        self.process = subprocess.Popen(
            [sys.executable, WORKER, '--npz', npz or npz_of(model), '--sim-latency', str(latency)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.messages = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.hello = self.receive()
        self.ready = self.receive()

    def _read(self):
        for line in self.process.stdout:
            self.messages.put(json.loads(line))
        self.messages.put(None)                                  # end of output

    def receive(self, timeout=20):
        return self.messages.get(timeout=timeout)

    def ask(self, index, window):
        self.process.stdin.write(json.dumps(dict(id=index, features=window.tolist())) + '\n')
        self.process.stdin.flush()
        return self.receive()

    def finish(self, timeout=5):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait(timeout=timeout)
        return self.process.returncode

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe and not pipe.closed:
                pipe.close()


class WorkerProtocolTests(unittest.TestCase):
    def test_start_up_messages(self):
        with Worker() as w:
            self.assertEqual(w.hello['type'], 'hello')
            self.assertEqual(w.hello['pid'], w.process.pid)
            self.assertEqual(w.ready['type'], 'ready')
            self.assertEqual(w.ready['mode'], 'reference-simulation')
            self.assertIs(w.ready['simulated'], True)
            self.assertEqual(w.ready['application_boots'], 0)
            self.assertEqual(w.ready['weights_md5'], hashlib.md5(pathlib.Path(npz_of(MODELS[0])).read_bytes()).hexdigest())

    def test_spikes_are_the_reference_exactly_on_256_windows(self):
        real, random = real_windows(), random_windows()
        self.assertEqual((len(real), len(random)), (216, 40))
        for model in MODELS:
            weights = np.load(npz_of(model), allow_pickle=False)
            spikes_in = dict(real=0, random=0)
            with Worker(model) as w:
                for i, window in enumerate(real + random):
                    result = w.ask(i, window)
                    self.assertEqual(result['type'], 'result')
                    self.assertEqual(result['id'], i)
                    spikes = np.asarray(result['spikes'])
                    self.assertEqual(spikes.shape, (125, 24))
                    self.assertTrue(np.isin(spikes, [0, 1]).all())
                    expected = chipref.run(weights, window, **chipref.CHIP)
                    self.assertTrue(np.array_equal(spikes, expected), f'{model}: window {i} differs from the reference')
                    spikes_in['real' if i < len(real) else 'random'] += int(spikes.sum())
                self.assertEqual(w.finish(), 0)
            # a comparison of two silent rasters would prove nothing
            self.assertGreater(spikes_in['real'], 0, f'{model}: no spike at all in the real windows')
            self.assertGreater(spikes_in['random'], 1000, f'{model}: the random windows produced almost no spikes')

    def test_nothing_is_claimed_that_was_not_done(self):
        with Worker() as w:
            result = w.ask(0, random_windows(1)[0])
            self.assertIsNone(result['exact_reference'])         # no comparison was made: the answer is the reference
            self.assertIs(result['simulated'], True)
            for key in ('inference_seconds', 'worker_cpu_seconds', 'worker_peak_rss_bytes'):
                self.assertIn(key, result)

    def test_every_window_starts_from_zero_state(self):
        active = random_windows(1)[0]
        with Worker() as w:
            first = np.asarray(w.ask(0, active)['spikes'])
            silence = np.asarray(w.ask(1, np.zeros((125, 48), np.float32))['spikes'])
            again = np.asarray(w.ask(2, active)['spikes'])
        self.assertGreater(first.sum(), 0)
        self.assertEqual(int(silence.sum()), 0)                  # nothing left over from the active window
        self.assertTrue(np.array_equal(first, again))

    def test_simulated_chip_time_is_waited_out(self):
        window = random_windows(1)[0]
        with Worker(latency=0.25) as w:
            started = time.perf_counter()
            result = w.ask(0, window)
            seconds = time.perf_counter() - started
        self.assertGreaterEqual(result['inference_seconds'], 0.25)
        self.assertGreaterEqual(seconds, 0.25)
        self.assertLess(seconds, 0.6)

    def test_without_simulated_chip_time_it_answers_in_milliseconds(self):
        windows = random_windows(20)
        with Worker(latency=0) as w:
            w.ask(-1, windows[0])                                # the first call loads the arrays
            times = []
            for i, window in enumerate(windows):
                started = time.perf_counter()
                w.ask(i, window)
                times.append(time.perf_counter() - started)
        self.assertLess(float(np.median(times)), 0.05)

    def test_wrong_shape_is_an_error_and_the_worker_stops(self):
        with Worker() as w:
            message = w.ask(0, np.zeros((10, 48), np.float32))
            self.assertEqual(message['type'], 'error')
            self.assertIn('Expected features of shape (125, 48)', message['error'])
            w.process.wait(timeout=5)
            self.assertNotEqual(w.process.returncode, 0)

    def test_missing_weights_is_an_error(self):
        process = subprocess.run([sys.executable, WORKER, '--npz', '/no/such/weights.npz'], input='', capture_output=True,
                                 text=True, timeout=20)
        messages = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(messages[0]['type'], 'hello')
        self.assertEqual(messages[-1]['type'], 'error')
        self.assertNotEqual(process.returncode, 0)

    def test_end_of_input_close_request_and_sigterm_all_stop_it_quietly(self):
        with Worker() as w:                                       # end of input
            self.assertEqual(w.finish(), 0)
        with Worker() as w:                                       # an explicit close request
            w.process.stdin.write(json.dumps(dict(type='close')) + '\n')
            w.process.stdin.flush()
            w.process.wait(timeout=5)
            self.assertEqual(w.process.returncode, 0)
        with Worker() as w:                                       # SIGTERM while waiting for a request
            w.process.send_signal(signal.SIGTERM)
            w.process.wait(timeout=5)
            self.assertEqual(w.process.returncode, 0)
            self.assertEqual(w.process.stderr.read(), '')         # no traceback for a requested stop
        with Worker(latency=5) as w:                              # SIGTERM in the middle of a window
            w.process.stdin.write(json.dumps(dict(id=0, features=random_windows(1)[0].tolist())) + '\n')
            w.process.stdin.flush()
            time.sleep(0.5)
            started = time.perf_counter()
            w.process.send_signal(signal.SIGTERM)
            w.process.wait(timeout=5)
            self.assertLess(time.perf_counter() - started, 2)
            self.assertEqual(w.process.returncode, 0)


DRIVER = '''
import argparse, sys
from pathlib import Path
sys.path.insert(0, {here!r})
import live_chip
a = argparse.Namespace(mode='reference', source='wav:' + {wav!r}, period=.5, playout_delay=1.25, duration={duration},
                       sink='print', selector='pitch', bpm=80, verify_reference=False, container='none',
                       metrics=Path({metrics!r}), window_timeout={window_timeout}, start_attempts=1,
                       model={model!r}, sim_latency={latency})
live_chip.LiveRun(a).run()
'''
NOTE_LINE = re.compile(r'^\s*[\d.]+s audio \| reference \| .+ \| notes=\[.*\]$')


class Loop:
    """The real live loop in a subprocess, in --mode reference, on tones.wav. A docker that fails loudly is first on PATH."""

    def __init__(self, tmp, name, duration=12, latency=0.38, model=MODELS[0], window_timeout=3.0):
        self.metrics_path = os.path.join(tmp, name + '.json')
        self.log_path = os.path.join(tmp, name + '.log')
        self.marker = os.path.join(tmp, name + '.docker-was-called')
        shim_dir = os.path.join(tmp, name + '-bin')
        os.makedirs(shim_dir)
        shim = os.path.join(shim_dir, 'docker')
        with open(shim, 'w') as f:
            f.write(f'#!/bin/sh\necho "$@" >> {self.marker}\nexit 99\n')
        os.chmod(shim, os.stat(shim).st_mode | stat.S_IXUSR)
        code = DRIVER.format(here=HERE, wav=WAV, metrics=self.metrics_path, duration=duration, model=model,
                             latency=latency, window_timeout=window_timeout)
        self.log = open(self.log_path, 'w')                  # a file, not a pipe: a surviving worker would hold a pipe open
        self.started = time.perf_counter()
        self.process = subprocess.Popen([sys.executable, '-c', code], stdout=self.log, stderr=self.log,
                                        env=dict(os.environ, PATH=shim_dir + os.pathsep + os.environ['PATH']))

    def wait(self, timeout=90):
        self.process.wait(timeout=timeout)
        self.seconds = time.perf_counter() - self.started
        self.log.close()
        with open(self.log_path) as f:
            self.text = f.read()
        self.code = self.process.returncode
        self.docker_called = os.path.exists(self.marker)
        self.metrics = None
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path) as f:
                self.metrics = json.load(f)
        self.notes = [line for line in self.text.splitlines() if NOTE_LINE.match(line)]
        return self


def worker_pids(marker):
    out = subprocess.run(['pgrep', '-f', marker], capture_output=True, text=True).stdout.split()
    return [pid for pid in out if int(pid) != os.getpid()]


@unittest.skipUnless(os.path.exists(WAV), 'data/live_chip/tones.wav is missing: python make_test_audio.py data/live_chip/tones.wav --seconds 190')
class ReferenceModeLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_runs_without_docker_and_says_it_is_simulated(self):
        run = Loop(self.tmp.name, 'default').wait()
        self.assertEqual(run.code, 0, run.text[-1500:])
        self.assertFalse(run.docker_called, 'the loop called docker')
        self.assertIn('ACTIVE INFERENCE MODE: REFERENCE', run.text)
        self.assertIn('Starting the simulated chip', run.text)
        self.assertNotIn('physical board worker', run.text)
        m = run.metrics
        self.assertIs(m['completed'], True)
        self.assertEqual(m['mode'], 'reference-simulation')
        self.assertIs(m['simulated'], True)
        self.assertEqual(m['application_boots'], 0)
        self.assertEqual(len(m['weights_md5']), 32)
        self.assertGreaterEqual(len(m['windows']), 15)
        self.assertTrue(all(w['exact_reference'] is None for w in m['windows']), 'a window claims it was compared')
        round_trips = [w['host_roundtrip_seconds'] for w in m['windows']]
        self.assertTrue(0.38 <= sorted(round_trips)[len(round_trips) // 2] <= 0.45, round_trips)
        self.assertEqual((m['slow_round_trips'], m['late_outputs'], m['connection_failures']), (0, 0, 0))
        self.assertGreaterEqual(len(run.notes), 3, run.text[-800:])

    def test_notes_do_not_depend_on_the_simulated_time_and_repeat_exactly(self):
        runs = [Loop(self.tmp.name, name, latency=latency)
                for name, latency in (('fast1', 0), ('fast2', 0), ('slow', 0.38))]        # at the same time
        for run in runs:
            run.wait()
            self.assertEqual(run.code, 0, run.text[-1500:])
        fast1, fast2, slow = runs
        self.assertGreaterEqual(len(fast1.notes), 3)
        self.assertEqual(fast1.notes, fast2.notes)
        self.assertEqual(fast1.notes, slow.notes)
        trips = sorted(w['host_roundtrip_seconds'] for w in fast1.metrics['windows'])
        self.assertLess(trips[len(trips) // 2], 0.05)

    def test_a_worker_that_dies_stops_the_loop_with_an_error(self):
        run = Loop(self.tmp.name, 'killed', duration=30, latency=0.3777)
        time.sleep(7)
        subprocess.run(['pkill', '-TERM', '-f', 'reference_worker.py.*0.3777'])
        run.wait(timeout=30)
        self.assertNotEqual(run.code, 0)
        self.assertIn('Board worker exited', run.text)
        self.assertLess(run.seconds, 20)
        self.assertEqual(worker_pids('reference_worker.py.*0.3777'), [])

    def test_a_worker_that_is_too_slow_is_a_timeout_and_leaves_nothing_behind(self):
        run = Loop(self.tmp.name, 'slow-worker', duration=30, latency=5.0123, window_timeout=1.0).wait(timeout=60)
        self.assertNotEqual(run.code, 0)
        self.assertIn('Board timeout', run.text)
        self.assertLess(run.seconds, 30)
        time.sleep(1)
        self.assertEqual(worker_pids('reference_worker.py.*5.0123'), [])

    def test_an_unknown_model_is_a_clear_error(self):
        run = Loop(self.tmp.name, 'no-model', model='no_such_model').wait()
        self.assertNotEqual(run.code, 0)
        self.assertIn('No weights for model no_such_model', run.text)
        self.assertFalse(run.docker_called)


class CommandLineTests(unittest.TestCase):
    def usage_error(self, *args):
        process = subprocess.run([sys.executable, os.path.join(HERE, 'live_chip.py'), '--source', 'wav:none', *args],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, 2, process.stderr[-500:])
        return process.stderr

    def test_options_that_make_no_sense_without_a_board_are_refused(self):
        self.assertIn('do not use --banner', self.usage_error('--mode', 'reference', '--banner'))
        self.assertIn('do not use --banner', self.usage_error('--mode', 'reference', '--verify-reference'))
        self.assertIn('sim-latency >= 0', self.usage_error('--mode', 'reference', '--sim-latency', '-1'))

    def test_reference_keeps_the_chip_window_rules_and_chip_mode_is_unchanged(self):
        self.assertIn('Reference mode: use period 0.5..1 s', self.usage_error('--mode', 'reference', '--period', '2'))
        self.assertIn('Chip mode: use period 0.5..1 s', self.usage_error('--mode', 'chip', '--period', '2'))
        self.assertIn('Reference mode: use period 0.5..1 s and playout-delay >= period + 0.4 s',
                      self.usage_error('--mode', 'reference', '--playout-delay', '0.5'))


if __name__ == '__main__':
    unittest.main()
