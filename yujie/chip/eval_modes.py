#!/usr/bin/env python3
"""Pitch accuracy of one exported model under the rules it was trained with and under the chip's rules.

Same crops and scoring as simon/eval_pitch.py (melody accuracy on scored, sung frames; rate over 8 frames).
The TRAINED row must reproduce that script's numbers; the CHIP rows are what to expect on the chip from
the same weights. Run in the SDK container from jan/singalong:

    python3 ../../yujie/chip/eval_modes.py ../../handoff/runs/A_vocalset_P --vocalset ../../handoff/testdata/vocalset --damp ../../handoff/testdata/damp250
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "jan" / "singalong"), str(ROOT / "simon"), str(Path(__file__).parent)]
import chipref  # noqa: E402
import eval_pitch  # noqa: E402
from singalong import data, metrics  # noqa: E402
from singalong.torchsnn import spike_rate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--vocalset", default="../../handoff/testdata/vocalset")
ap.add_argument("--damp", default="../../handoff/testdata/damp250")
ap.add_argument("--crops", type=int, default=512)
a = ap.parse_args()
npz = np.load(a.run / "weights_int8.npz", allow_pickle=False)
sets = {"vs_songs": eval_pitch.crops_of(eval_pitch.vocalset_unseen(a.vocalset, ["song"]), a.crops, 125),
        "vs_exer": eval_pitch.crops_of(eval_pitch.vocalset_unseen(a.vocalset, ["val", "test"]), a.crops, 125),
        "damp250": eval_pitch.crops_of(data.load_recordings("val", data=a.damp), a.crops, 125)}


def score(spikes, Y):
    rate = spike_rate(torch.tensor(spikes, dtype=torch.float32))
    return metrics.pitch_scores(rate[..., 12:24], Y)["pitch_acc"]


modes = {"TRAINED (torchsnn rules)": chipref.TRAINED,
         "recurrent as on chip, no layer delay": dict(recurrent="post", layer_delay=0),
         "CHIP, scored as it arrives (one frame late)": chipref.CHIP,
         "CHIP, labels shifted by the one-frame delay": chipref.CHIP,
         "snn_numpy.py default (encoder reset 'subtract')": dict(chipref.TRAINED, encoder_reset="subtract")}
result = {}
print(f"{'rules':48s}" + "".join(f"{k:>10s}" for k in sets) + "   hidden rate  output spikes/frame")
for name, rules in modes.items():
    row, rates = {}, []
    for key, (X, Y) in sets.items():
        spikes, hidden = chipref.run(npz, np.asarray(X), return_hidden=True, **rules)
        if name.endswith("delay"):
            spikes = np.concatenate([spikes[:, 1:], spikes[:, -1:]], axis=1)       # undo the one-frame latency
        row[key] = score(spikes, Y)
        rates.append((float(hidden.mean()), float(spikes[..., 12:24].sum(-1).mean())))
    result[name] = row
    print(f"{name:48s}" + "".join(f"{v:10.3f}" for v in row.values())
          + f"   {np.mean([r[0] for r in rates]):.3f}        {np.mean([r[1] for r in rates]):.3f}")
(Path(__file__).parent / "chip_expectation.json").write_text(json.dumps(dict(run=a.run.name, crops={k: len(v[0]) for k, v in sets.items()},
                                                                         pitch_accuracy=result), indent=1) + "\n")
