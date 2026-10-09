#!/usr/bin/env python3
"""A simulated chip for the live loop: board_worker.py's protocol, answered in software on this Mac.

No board, no SDK, no container, no board lock. Every request is one window of band features [125, 48]; the
answer is the integer reference of the network under the chip's rules (chipref.CHIP), which reproduced the
physical chip on every window that was compared (see BOARDLESS.md, section 1). Each window starts from zero
state, as on the board.

What this is not: a measurement of the chip. The compute takes a few milliseconds here, so the time the chip
would need is only waited out (--sim-latency), and every message says "simulated". Start it through
live_chip.py / live_view.py with --mode reference; run alone it reads JSON lines on stdin:

    {"id": 0, "features": [[...48 numbers...] x 125]}   ->   {"type": "result", "id": 0, "spikes": [[0/1 x 24] x 125], ...}
    {"type": "close"} or end of input                   ->   the worker exits

stdout is the protocol; diagnostics go to stderr.
"""
import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

import chipref

FRAMES, BANDS = 125, 48
BOARD_MEDIAN_SECONDS = 0.38             # round trip of the physical chip, median, from HARDWARE_BENCHMARK.md; not measured here


def emit(message):
    print(json.dumps(message), flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', type=Path, default=root / 'handoff/runs/A_vocalset_P/weights_int8.npz',
                    help='int8 weights of the network (the file that would go to the chip)')
    ap.add_argument('--sim-latency', type=float, default=BOARD_MEDIAN_SECONDS,
                    help=f'seconds a window takes before its answer is returned; default {BOARD_MEDIAN_SECONDS}, '
                         'the physical chip\'s median. 0 answers as fast as the reference computes')
    a = ap.parse_args()

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    emit(dict(type='hello', pid=os.getpid()))      # lets the host signal this process if it hangs
    try:
        weights = np.load(a.npz, allow_pickle=False)
        digest = hashlib.md5(a.npz.read_bytes()).hexdigest()
        emit(dict(type='ready', mode='reference-simulation', simulated=True, boot_seconds=0.0,
                  weights=str(a.npz), weights_md5=digest, sim_latency_seconds=a.sim_latency,
                  application_boots=0))
        for line in sys.stdin:
            started = time.perf_counter()
            request = json.loads(line)
            if request.get('type') == 'close':
                break
            x = np.asarray(request['features'], dtype=np.float32)
            if x.shape != (FRAMES, BANDS):
                raise ValueError(f'Expected features of shape ({FRAMES}, {BANDS}), got {x.shape}')
            spikes = chipref.run(weights, x, **chipref.CHIP)
            wait = a.sim_latency - (time.perf_counter() - started)
            if wait > 0:
                time.sleep(wait)
            emit(dict(type='result', id=request['id'], spikes=spikes.tolist(),
                      inference_seconds=time.perf_counter() - started,
                      exact_reference=None,        # nothing was compared: the answer is the reference itself
                      simulated=True,
                      worker_cpu_seconds=time.process_time(),
                      worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      * (1 if sys.platform == 'darwin' else 1024)))
    except KeyboardInterrupt:
        pass                                       # SIGTERM from the host: leave quietly, there is nothing to release
    except Exception as error:
        emit(dict(type='error', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
