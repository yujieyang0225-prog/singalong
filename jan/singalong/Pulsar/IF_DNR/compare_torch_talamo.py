#!/usr/bin/env python3
"""Three-way spike comparison on the same input: Talamo software neuron vs singalong.snn_numpy vs (optionally)
the C1 SoC simulator. Goal: explain the 73.7 % PyTorch-vs-simulator agreement seen in run 2 and pin down the
exact neuron semantics (encoder reset, layer delay) so train_torch.py models what the chip does.

    jan/singalong/run.sh python3 Pulsar/IF_DNR/compare_torch_talamo.py data/runs/<run> [--sim] [--secs 5]

Needs data/runs/<run>/weights_int8.npz (export_npz.py) next to the Talamo run. Prints, per variant of the numpy
reference (enc_reset x layer_delay), the agreement with Talamo and with the simulator, input/hidden/output
separately where probes allow. The variant that agrees best is the one to put into torchsnn.py.

UNTESTED in this session (no SDK here).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import talamo  # noqa: E402
from singalong import data  # noqa: E402
from singalong.snn_numpy import C1Net  # noqa: E402
from Pulsar.IF_DNR.sim_check import CLOCK_MHZ, load_quantized  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--sim", action="store_true", help="also run the C1 SoC simulator")
    a = ap.parse_args()
    pipe, snn_step = load_quantized(a.run)
    recs = data.load_recordings("val", data=a.data)
    X = recs[0]["X"][600:600 + int(a.secs * 62.5)]
    ds = data.dense_dataset(torch.tensor(X[None]), torch.zeros(1, len(X), 14))
    s_tal = pipe(ds, verbose=0).data_as_numpy()[0]                    # [T+1, O]
    s_tal = s_tal[-len(X):]
    results = {}
    for enc in ("subtract", "zero"):
        for delay in (0, 1):
            net = C1Net(a.run / "weights_int8.npz", enc_reset=enc, layer_delay=delay)
            s_np = net.run(X)
            results[(enc, delay)] = (s_tal == s_np).mean()
            print(f"numpy enc_reset={enc:8s} layer_delay={delay}: agreement with Talamo {results[(enc, delay)]:.3f} "
                  f"(spikes {int(s_np.sum())} vs {int(s_tal.sum())})")
    if a.sim:
        soc = talamo.device.c1.Soc(simulate=True, clock_frequency=CLOCK_MHZ * 1_000_000)
        snn_step.to(soc)
        s_sim = pipe(ds, verbose=0).data_as_numpy()[0][-len(X):]
        print(f"simulator vs Talamo software: {(s_sim == s_tal).mean():.3f}")
        for (enc, delay) in results:
            s_np = C1Net(a.run / "weights_int8.npz", enc_reset=enc, layer_delay=delay).run(X)
            print(f"simulator vs numpy enc_reset={enc:8s} layer_delay={delay}: {(s_sim == s_np).mean():.3f}")


if __name__ == "__main__":
    main()
