"""Software-only timing, readout, frontend, and selector contract checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import argparse
import contextlib
import io
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'jan/singalong'))
from singalong import frontend as fe
import live
from windowing import newest_rates, window_end


class WindowTests(unittest.TestCase):
    def test_fractional_hops_and_no_duplicates(self):
        rng = np.random.default_rng(1)
        last = None
        seen = []
        for k in range(400):
            end = window_end(k)
            self.assertLess(end - (125 + k * 31.25), 1)
            spikes = rng.integers(0, 2, (125, 24))
            indices, rates = newest_rates(spikes, end, last)
            # Independent direct eight-frame average, including overlap context.
            expected = np.array([spikes[i - end + 125 - 7:i - end + 126].mean(0) for i in indices])
            np.testing.assert_array_equal(rates, expected)
            seen.extend(indices)
            last = int(indices[-1])
        np.testing.assert_array_equal(np.diff(seen), 1)

    def test_gap_and_duplicate_fail_closed(self):
        s = np.zeros((125, 24))
        with self.assertRaises(RuntimeError):
            newest_rates(s, 300, 125)
        with self.assertRaises(RuntimeError):
            newest_rates(s, 125, 124)

    def test_streaming_features_match_frame_definition(self):
        rng = np.random.default_rng(2)
        y = rng.normal(0, .03, 16000).astype(np.float32)
        # Gate held open to compare feature definition, not offline vs online noise estimators.
        front = fe.StreamingFrontEnd(gate=lambda db: True)
        rows = []
        for start in range(0, len(y), 333):
            rows.extend(front.feed(y[start:start + 333]))
        expected = fe.normalise(fe.log_bands(y))
        np.testing.assert_array_equal(np.stack([r[0] for r in rows]), expected)
        np.testing.assert_allclose([r[1] for r in rows], fe.level_db(y), atol=1e-5)

    def test_score_follower_preserves_frame_steps(self):
        # Rechunking must not change HMM decisions: each new frame is stepped once.
        rng = np.random.default_rng(3)
        evidence = rng.random((200, 12))
        a, b = live.make_selector('score', 80), live.make_selector('score', 80)
        expected = [a.step(m, None, onset=True)[0] for m in evidence]
        actual = []
        for chunk in np.array_split(evidence, 7):
            actual.extend(b.step(m, None, onset=True)[0] for m in chunk)
        self.assertEqual(actual, expected)

    def test_threaded_scheduler_with_fake_board(self):
        # Explicit software test: no Docker, no hardware, no persisted chip claims.
        import live_chip
        class FakeBoard:
            ready = dict(mode='software-test')
            def __init__(self, *_):
                pass
            def infer(self, x, index):
                self_shape = x.shape
                if self_shape != (125, 48):
                    raise ValueError(self_shape)
                return dict(spikes=np.zeros((125, 24), np.int8), inference_seconds=0.,
                            host_roundtrip_seconds=0., exact_reference=None)
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(mode='chip', source='wav:synthetic', period=.5, playout_delay=1.,
                                      container='unused', verify_reference=False, duration=4., sink='print',
                                      selector='score', bpm=80, metrics=Path(directory) / 'metrics.json')
            run = live_chip.LiveRun(args)
            with patch.object(live_chip, 'ChipClient', FakeBoard), \
                 patch.object(fe, 'decode_audio', return_value=np.zeros(64000, np.float32)), \
                 contextlib.redirect_stdout(io.StringIO()):
                run.run()
            self.assertTrue(run.metrics['completed'])
            self.assertEqual(len(run.metrics['windows']), 4)
            self.assertEqual(run.metrics['delivered_frames'], 126)
            self.assertEqual(run.metrics['duplicate_frames'], 0)
            self.assertEqual(run.metrics['skipped_updates'], 0)


if __name__ == '__main__':
    unittest.main()
