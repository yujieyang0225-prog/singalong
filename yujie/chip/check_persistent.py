#!/usr/bin/env python3
"""Milestone A: one boot, independent deterministic calls, exact reset-reference check."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from persistent_board import PersistentBoard
import chipref


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(__file__).with_name("persistent_check.json"))
    ap.add_argument("--silence-first", action="store_true", help="Diagnostic only: prepend silence before the first active input")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    rng = np.random.default_rng(42)
    x = rng.integers(0, 101, (125, 48)).astype(np.float32)
    z = np.zeros_like(x)
    tests = [("random_A", x), ("silence_after_A", z), ("repeat_A", x),
             ("full_scale", np.full_like(x, 100)), ("silence_after_full", z), ("repeat_A_again", x)]
    if a.silence_first:
        tests.insert(0, ("silence_first", z))
    report = dict(tested_on="physical Pulsar C1", talamo="4.7.1", application_boots=0, calls=[])
    try:
        with PersistentBoard(a.run, a.npz) as board:
            report.update(application_boots=1, boot_seconds=board.boot_seconds)
            print(f"RAM boot: {board.boot_seconds:.3f}s", flush=True)
            for label, features in tests:
                expected = chipref.run(board.weights, features, **chipref.CHIP)
                spikes, seconds = board.infer(features)
                row = dict(input=label, seconds=seconds, shape=list(spikes.shape), spikes=int(spikes.sum()),
                           reference_spikes=int(expected.sum()), exact=bool(np.array_equal(spikes, expected)))
                report['calls'].append(row)
                print(json.dumps(row), flush=True)
                if not row['exact']:
                    raise RuntimeError("STOP: persistent inference differs from zero-state chip reference")
                time.sleep(0.15)
            report['passed'] = True
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == '__main__':
    main()
