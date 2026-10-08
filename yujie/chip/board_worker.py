#!/usr/bin/env python3
"""JSON-lines transport for the Mac host; runs only in singalong_sdk.

stdout is the protocol, stderr is SDK diagnostics. EOF releases the board.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import signal
import resource
import sys
import time

import numpy as np
from persistent_board import PersistentBoard
import chipref


def emit(message):
    print(json.dumps(message), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument('--run', type=Path, default=root / 'data/runs/A_vocalset_P_talamo')
    ap.add_argument('--npz', type=Path, default=root / 'handoff/runs/A_vocalset_P/weights_int8.npz')
    ap.add_argument('--verify-reference', action='store_true')
    ap.add_argument('--comm-debug', action='store_true')
    a = ap.parse_args()
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        with redirect_stdout(sys.stderr):
            board = PersistentBoard(a.run, a.npz)
            board.pipe.embedded_config.enable_comm_debug_logs = a.comm_debug
            board.__enter__()
        try:
            # Silence as the first sample returned no complete output on this
            # setup; active-first standalone tests passed twice. Qualify startup
            # with the same deterministic active input, then prove the reset by
            # running silence. Neither result is delivered to the music app.
            prime = np.random.default_rng(42).integers(0, 101, (125, 48)).astype(np.float32)
            with redirect_stdout(sys.stderr):
                active, prime_seconds = board.infer(prime)
            if not np.array_equal(active, chipref.run(board.weights, prime, **chipref.CHIP)):
                raise RuntimeError('Startup active input differs from reference')
            with redirect_stdout(sys.stderr):
                spikes, silence_seconds = board.infer(np.zeros((125, 48)))
            if spikes.any():
                raise RuntimeError('Startup silence produced spikes')
            emit(dict(type='ready', mode='physical-chip', boot_seconds=board.boot_seconds,
                      prime_seconds=prime_seconds, startup_silence_seconds=silence_seconds,
                      transport_setup_seconds=board.transport_setup_seconds,
                      startup_active_exact=True, startup_silence_exact=True, application_boots=1))
            for line in sys.stdin:
                request = json.loads(line)
                if request.get('type') == 'close':
                    break
                x = np.asarray(request['features'], dtype=np.float32)
                with redirect_stdout(sys.stderr):
                    spikes, seconds = board.infer(x)
                same = None
                if a.verify_reference:
                    same = bool(np.array_equal(spikes, chipref.run(board.weights, x, **chipref.CHIP)))
                    if not same:
                        raise RuntimeError('STOP: physical output differs from zero-state reference')
                emit(dict(type='result', id=request['id'], spikes=spikes.tolist(),
                          inference_seconds=seconds, exact_reference=same,
                          worker_cpu_seconds=time.process_time(),
                          worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          **board.transport_stats()))
        finally:
            with redirect_stdout(sys.stderr):
                board.__exit__(None, None, None)
    except Exception as error:
        emit(dict(type='error', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
