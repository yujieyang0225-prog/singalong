#!/usr/bin/env python3
"""Run the imported network on the board and compare it with the reference: timing and agreement.

Needs the board (RAM boot through evk-bridge; nothing is flashed) and the application built by build_chip.py.
Windows are the same fixed crops as simon/eval_pitch.py (125 frames, 31-frame burn-in in the labels).

    python3 ../../yujie/chip/board_check.py ../../data/runs/A_vocalset_P_talamo --npz ../../handoff/runs/A_vocalset_P/weights_int8.npz --timing
    python3 ../../yujie/chip/board_check.py ../../data/runs/A_vocalset_P_talamo --npz ../../handoff/runs/A_vocalset_P/weights_int8.npz --crops 150

Run in the SDK container from jan/singalong.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "jan" / "singalong"), str(ROOT / "simon"), str(Path(__file__).parent)]
import talamo  # noqa: E402
import build_chip  # noqa: E402
import chipref  # noqa: E402
import eval_pitch  # noqa: E402
from singalong import data, metrics  # noqa: E402
from singalong.torchsnn import spike_rate  # noqa: E402

FRAMES = 125


def f1(p, q):
    p, q = p.astype(bool), q.astype(bool)
    return float(2 * (p & q).sum() / max(1, p.sum() + q.sum()))


class Board:
    def __init__(self, run, npz):
        self.app = Path(run).resolve() / f"app_hostfed_{FRAMES}"
        self.pipe, _, self.w, _ = build_chip.load(run, npz)
        self.pipe.to(talamo.device.c1.Soc(simulate=False, clock_frequency=build_chip.CLOCK_MHZ))

    def clear(self):
        """RAM boot the application with the plain boot utility. A Talamo boot over a still-running Talamo
        application has failed with 'waiting for sync' on this setup; after this it works."""
        util = Path(__import__("os").environ["SPINE_DIR"]) / "tools/c1util.py"
        subprocess.run([sys.executable, str(util), "-S", str(self.app / "bin/c1/release/app.c1boot")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)

    def run(self, X, attempts=3):
        """X [N, 125, 48] -> (spikes [N, 125, 24] int8, seconds for the whole call including the boot)."""
        ds = data.dense_dataset(torch.tensor(np.asarray(X, dtype=np.float32)), torch.zeros(len(X), FRAMES, 14))
        for attempt in range(attempts):
            started = time.time()
            try:
                out = np.asarray(self.pipe(ds, embedded_application_path=str(self.app)).data_as_numpy())
            except IndexError:                      # Talamo got nothing back: the boot did not sync
                out = np.zeros((0,))
            seconds = time.time() - started
            if len(out) == len(X):
                return out[:, :FRAMES].astype(np.int8), seconds
            print(f"  board returned {len(out)} of {len(X)} windows (attempt {attempt + 1}); clearing the board and retrying", flush=True)
            self.clear()
        raise SystemExit("FAILED: the board did not return all windows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--vocalset", default="../../handoff/testdata/vocalset")
    ap.add_argument("--damp", default="../../handoff/testdata/damp250")
    ap.add_argument("--crops", type=int, default=150, help="windows per test set (the first N of eval_pitch's fixed crops)")
    ap.add_argument("--timing", action="store_true", help="only measure the time per window")
    a = ap.parse_args()
    board = Board(a.run, a.npz)
    out_dir = Path(__file__).parent

    if a.timing:
        X, _ = eval_pitch.crops_of(data.load_recordings("val", data=a.damp), 512, FRAMES)
        X = np.asarray(X)
        rows = []
        for n in (1, 6, 21, 41):
            spikes, seconds = board.run(X[:n])
            same = bool(np.array_equal(spikes, chipref.run(board.w, X[:n], **chipref.CHIP)))
            rows.append(dict(windows=n, seconds=round(seconds, 2), identical_to_reference=same))
            print(rows[-1], flush=True)
        slope, intercept = np.polyfit([r["windows"] for r in rows], [r["seconds"] for r in rows], 1)
        result = dict(calls=rows, seconds_per_window=round(float(slope), 3), fixed_seconds_per_call=round(float(intercept), 2),
                      window_frames=FRAMES, window_audio_seconds=FRAMES * 0.016,
                      note="one pipe() call boots the board, then sends the windows one after another; "
                           "seconds_per_window is the slope over the number of windows")
        (out_dir / "board_timing.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
        return

    sets = {"vs_songs": eval_pitch.crops_of(eval_pitch.vocalset_unseen(a.vocalset, ["song"]), 512, FRAMES),
            "vs_exer": eval_pitch.crops_of(eval_pitch.vocalset_unseen(a.vocalset, ["val", "test"]), 512, FRAMES),
            "damp250": eval_pitch.crops_of(data.load_recordings("val", data=a.damp), 512, FRAMES)}
    result, saved = dict(windows_per_set=a.crops, sets={}), {}
    for name, (X, Y) in sets.items():
        X, Y = np.asarray(X)[:a.crops], Y[:a.crops]
        spikes, seconds = board.run(X)
        chip = chipref.run(board.w, X, **chipref.CHIP)
        trained = chipref.run(board.w, X, **chipref.TRAINED)
        score = lambda s: metrics.pitch_scores(spike_rate(torch.tensor(s, dtype=torch.float32))[..., 12:24], Y)["pitch_acc"]  # noqa: E731
        identical = [bool(np.array_equal(spikes[i], chip[i])) for i in range(len(X))]
        difference = np.abs(spikes.astype(int) - chip.astype(int)).sum(axis=(1, 2))
        result["sets"][name] = dict(
            windows=len(X), seconds=round(seconds, 1), windows_identical_to_reference=int(sum(identical)),
            spikes_board=int(spikes.sum()), spikes_reference=int(chip.sum()), spike_position_f1=round(f1(spikes, chip), 4),
            differing_entries_per_window=dict(mean=round(float(difference.mean()), 3), max=int(difference.max())),
            chord_outputs_silent_on_board=bool(spikes[..., :12].sum() == 0),
            pitch_accuracy=dict(board=score(spikes), reference_chip_rules=score(chip), reference_as_trained=score(trained)))
        saved[name + "_board"], saved[name + "_reference"] = spikes, chip
        print(name, json.dumps(result["sets"][name]), flush=True)
    np.savez_compressed(Path(a.run) / "board_check_spikes.npz", **saved)
    (out_dir / "board_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
