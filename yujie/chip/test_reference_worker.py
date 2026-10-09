"""Software-only tests of the simulated chip (reference_worker.py). No board, no Docker, no SDK.

Run from yujie/chip with the host Python:  python test_reference_worker.py
"""
import glob
import hashlib
import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import chipref

WORKER = os.path.join(HERE, 'reference_worker.py')
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


if __name__ == '__main__':
    unittest.main()
