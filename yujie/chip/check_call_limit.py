#!/usr/bin/env python3
"""How many consecutive windows does one board connection survive? Board needed, RAM boot only.

The 185 s stability run stopped after 337 windows with the SDK's wait_events() returning None.
This repeats one deterministic window back to back, with no audio pacing, until a call fails
or --calls is reached, and records the count. If the count repeats, the limit belongs to the
connection (for example a fixed number of transport events) and not to audio timing.

    docker exec -w /workspace/jan/singalong singalong_sdk python3 -u /workspace/yujie/chip/check_call_limit.py \\
        /workspace/data/runs/A_vocalset_P_talamo --npz /workspace/handoff/runs/A_vocalset_P/weights_int8.npz
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from persistent_board import PersistentBoard
import chipref

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", type=Path)
ap.add_argument("--npz", type=Path, required=True)
ap.add_argument("--calls", type=int, default=500)
ap.add_argument("--output", type=Path, default=Path(__file__).with_name("call_limit.json"))
a = ap.parse_args()
x = np.random.default_rng(7).integers(0, 101, size=(125, 48)).astype(np.float32)
result = dict(tested_on="physical Pulsar C1", requested_calls=a.calls, completed_calls=0, exact_calls=0, error=None)
seconds = []
started = time.time()
with PersistentBoard(a.run, a.npz) as board:
    reference = chipref.run(board.weights, x, **chipref.CHIP)
    result["boot_seconds"] = round(board.boot_seconds, 3)
    for i in range(a.calls):
        try:
            spikes, took = board.infer(x)
        except BaseException as error:               # record exactly where and how it stopped
            result["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            break
        result["completed_calls"] += 1
        result["exact_calls"] += int(np.array_equal(spikes, reference))
        seconds.append(took)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} calls, all exact: {result['exact_calls'] == i + 1}, last {took:.3f} s", flush=True)
result.update(wall_seconds=round(time.time() - started, 1),
              call_seconds=dict(median=round(float(np.median(seconds)), 3), max=round(float(np.max(seconds)), 3)) if seconds else None)
a.output.write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result, indent=1))
