#!/usr/bin/env python3
"""Board check of the display-banner application: same output as the reference, and how fast?

Board needed, RAM boot only. Everything goes through the one-boot adapter, one window per call,
which is the path the live loop uses.

  default      --calls N deterministic and random windows (quick)
  --testset    the handoff test set, same crops and scoring as board_check.py, on one boot.
               Exit code 3 means the board booted but the start-up handshake failed; start again.
               Exit code 4 means a window got no answer within --window-limit; the result file says which.

    docker exec -w /workspace/jan/singalong singalong_sdk python3 -u /workspace/yujie/chip/check_banner.py \\
        /workspace/data/runs/A_vsP_chip_shift3_talamo --npz /workspace/handoff/runs/A_vsP_chip_shift3/weights_int8.npz --testset
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

import build_banner
import chipref
from persistent_board import PersistentBoard

START_FAILED, STALLED = 3, 4


class BannerBoard(PersistentBoard):
    def __init__(self, run, npz):
        super().__init__(run, npz)
        self.app = Path(run).resolve() / build_banner.APP
        self.pipe, _, self.weights, _ = build_banner.load(run, npz)
        self.pipe.to(self.soc)


def quick(a):
    rng = np.random.default_rng(42)
    inputs = [rng.integers(0, 101, (125, 48)).astype(np.float32), np.zeros((125, 48), np.float32),
              np.full((125, 48), 100, np.float32)]
    result = dict(tested_on="physical Pulsar C1", app=build_banner.APP, calls=0, exact=0, error=None)
    seconds = []
    with BannerBoard(a.run, a.npz) as board:
        result["boot_seconds"] = round(board.boot_seconds, 3)
        for i in range(a.calls):
            x = inputs[i % 3] if i < 6 else rng.integers(0, 101, (125, 48)).astype(np.float32)
            try:
                spikes, took = board.infer(x)
            except BaseException as error:
                result["error"] = f"{type(error).__name__}: {str(error)[:300]}"
                break
            result["calls"] += 1
            result["exact"] += int(np.array_equal(spikes, chipref.run(board.weights, x, **chipref.CHIP)))
            seconds.append(took)
    if seconds:
        later = seconds[1:] or seconds
        result["call_seconds"] = dict(first=round(seconds[0], 3), median=round(float(np.median(later)), 3),
                                      max=round(float(np.max(later)), 3))
    Path(__file__).with_name("banner_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def testset(a):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "simon"))
    import eval_pitch
    from singalong import data, metrics
    from singalong.torchsnn import spike_rate

    sets = {"vs_songs": eval_pitch.crops_of(eval_pitch.vocalset_unseen(a.vocalset, ["song"]), 512, 125),
            "vs_exer": eval_pitch.crops_of(eval_pitch.vocalset_unseen(a.vocalset, ["val", "test"]), 512, 125),
            "damp250": eval_pitch.crops_of(data.load_recordings("val", data=a.damp), 512, 125)}
    result = dict(tested_on="physical Pulsar C1", app=build_banner.APP, application_boots=1,
                  path="one boot, one window per call (the live loop's path)", windows_per_set=a.crops, sets={})
    saved, seconds, started = {}, [], time.time()
    out = Path(a.run) / "banner_testset_check.json"
    alarm = dict(fired=False)

    def on_alarm(*_):
        alarm["fired"] = True
        raise TimeoutError("no answer from the board within the per-window limit")

    def save(complete):
        result.update(complete=complete, comm_debug_logs=a.debug_logs, total_windows=len(seconds),
                      total_identical=sum(v["windows_identical_to_reference"] for v in result["sets"].values()),
                      wall_seconds=round(time.time() - started, 1))
        if len(seconds) > 1:
            later = seconds[1:]
            result["call_seconds"] = dict(first=round(seconds[0], 3), median=round(float(np.median(later)), 3),
                                          p95=round(float(np.percentile(later, 95)), 3), max=round(float(np.max(later)), 3))
        out.write_text(json.dumps(result, indent=1) + "\n")

    signal.signal(signal.SIGALRM, on_alarm)
    board = BannerBoard(a.run, a.npz)
    board.pipe.embedded_config.enable_comm_debug_logs = a.debug_logs      # as board_worker.py --comm-debug
    with board:
        result["boot_seconds"] = round(board.boot_seconds, 3)
        for name, (X, Y) in sets.items():
            X, Y = np.asarray(X)[:a.crops], Y[:a.crops]
            chip = chipref.run(board.weights, X, **chipref.CHIP)
            trained = chipref.run(board.weights, X, **chipref.TRAINED)
            spikes = np.zeros(chip.shape, np.int8)
            for i in range(len(X)):
                signal.alarm(a.window_limit)
                try:
                    with redirect_stdout(sys.stderr):                   # SDK log lines, as in the worker
                        spikes[i], took = board.infer(X[i])
                except BaseException as error:
                    signal.alarm(0)
                    handshake = "waiting for sync" in str(error) or "no complete sample" in str(error)
                    if not seconds and handshake and not alarm["fired"]:  # nothing has come back yet on this boot
                        print(f"START FAILED: {str(error)[:160]}", flush=True)
                        sys.exit(START_FAILED)
                    result["stopped"] = dict(set=name, window_in_set=i, windows_completed=len(seconds),
                                             exact_so_far_in_set=int(sum(np.array_equal(spikes[j], chip[j]) for j in range(i))),
                                             no_answer_within_limit=alarm["fired"],
                                             error=f"{type(error).__name__}: {str(error)[:200]}")
                    save(False)
                    print("STOPPED", json.dumps(result["stopped"]), flush=True)
                    signal.alarm(30)                                    # closing the session must not hang either
                    sys.exit(STALLED)
                signal.alarm(0)
                seconds.append(took)
                if len(seconds) % 50 == 0:
                    print(f"{len(seconds)} windows, last {took:.3f} s", flush=True)
            score = lambda s: metrics.pitch_scores(spike_rate(torch.tensor(s, dtype=torch.float32))[..., 12:24], Y)["pitch_acc"]  # noqa: E731
            identical = [bool(np.array_equal(spikes[i], chip[i])) for i in range(len(X))]
            result["sets"][name] = dict(
                windows=len(X), windows_identical_to_reference=int(sum(identical)),
                spikes_board=int(spikes.sum()), spikes_reference=int(chip.sum()),
                chord_outputs_silent_on_board=bool(spikes[..., :12].sum() == 0),
                pitch_accuracy=dict(board=score(spikes), reference_chip_rules=score(chip), reference_as_trained=score(trained)))
            saved[name + "_board"], saved[name + "_reference"] = spikes, chip
            save(False)                                                 # keep each finished set
            print(name, json.dumps(result["sets"][name]), flush=True)
    save(True)
    np.savez_compressed(Path(a.run) / "banner_testset_spikes.npz", **saved)
    print(json.dumps(result, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--testset", action="store_true", help="the handoff test set on one boot")
    ap.add_argument("--crops", type=int, default=512, help="--testset: windows per set")
    ap.add_argument("--window-limit", type=int, default=20, help="--testset: seconds to wait for one window")
    ap.add_argument("--debug-logs", action="store_true", help="--testset: SDK communication logging on, as in the live demo")
    ap.add_argument("--vocalset", default="../../handoff/testdata/vocalset")
    ap.add_argument("--damp", default="../../handoff/testdata/damp250")
    a = ap.parse_args()
    testset(a) if a.testset else quick(a)
