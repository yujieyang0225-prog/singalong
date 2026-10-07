#!/usr/bin/env python3
"""Export a Talamo run (quantized int8 parameters) to weights_int8.npz for singalong.snn_numpy / live.py.

    jan/singalong/run.sh python3 Pulsar/IF_DNR/export_npz.py data/runs/<run>

Writes data/runs/<run>/weights_int8.npz with: enc_threshold [48], W_in [H,48], W_rec [H,H], b_h [H], thr_h,
W_out [O,H], b_o [O], thr_o. It also dumps *every* parameter under its Talamo name into
data/runs/<run>/talamo_params_dump.npz, so if the name mapping below does not match your SDK version you
can read that file and fix MAP (print the names with --list).

UNTESTED in this session (no SDK here): the parameter names come from model.py's query patterns
('*snn_step.*synapses.weight', 'recurrent_synapses.weight', 'threshold', 'bias') and param_ranges()
('weight', 'bias_current', 'firing_threshold'). Check the shapes it prints.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from singalong import model  # noqa: E402

# substrings that identify each exported array in the Talamo parameter names (layer index 0 = hidden, 1 = output)
# Fully qualified (checked against the real Talamo 4.7.1 names with --list, 2026-10-07). The earlier short
# patterns ["1", "threshold"] / ["1", "bias"] also matched the encoder ("c1_if_encoder_step.threshold/.bias",
# the "1" of "c1"), which comes first in the list: thr_o / b_o were exported as the encoder's values.
MAP = {
    "enc_threshold": ["encoder", "threshold"],
    "W_in": ["snn_step.layers.0.synapses.weight"],
    "W_rec": ["snn_step.layers.0.recurrent_synapses.weight"],
    "b_h": ["snn_step.layers.0.neurons.bias_current"],
    "thr_h": ["snn_step.layers.0.neurons.firing_threshold"],
    "W_out": ["snn_step.layers.1.synapses.weight"],
    "b_o": ["snn_step.layers.1.neurons.bias_current"],
    "thr_o": ["snn_step.layers.1.neurons.firing_threshold"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--list", action="store_true", help="only print parameter names and shapes")
    a = ap.parse_args()
    m = json.load(open(a.run / "metrics.json"))["args"]
    pipe, _ = model.build_pipeline(m["hidden"], recurrent=not m.get("feedforward", False),
                                   enc_threshold=m.get("enc_threshold", 120.0), out_neurons=m.get("out", 12))
    pipe.quantize(quantizers=model.make_quantizers(pipe))
    pipe.trainable_params.load(str(a.run / "trainable_params_quantized.json"))
    params = {}
    for r in pipe.query_torch_params("*", references=True):
        t = r.get_value().detach().cpu().numpy()
        params[r.name] = t
        print(f"{r.name:60s} {t.shape} {t.dtype} integer={bool(np.all(t == np.round(t)))}")
    np.savez(a.run / "talamo_params_dump.npz", **params)
    if a.list:
        return
    out = {}
    for key, subs in MAP.items():
        cands = [n for n in params if all(s in n for s in subs) and ("recurrent" in n) == ("recurrent" in "".join(subs))]
        if key == "enc_threshold":
            cands = [n for n in params if "encoder" in n and "threshold" in n]
        if not cands:
            print(f"!! no parameter for {key} (patterns {subs}); fix MAP using the names above")
            continue
        if len(cands) > 1:
            raise SystemExit(f"several parameters match {key}: {cands}; make MAP unambiguous")
        out[key] = np.round(params[cands[0]]).astype(np.int64)
    # C1 IF encoder: weighted = floor(x * 2**weight) * weight_sign + bias (talamo/encoders/c1/if_encoder.py)
    for key, name in (("enc_weight", "c1_if_encoder_step.weight"), ("enc_weight_sign", "c1_if_encoder_step.weight_sign"),
                      ("enc_bias", "c1_if_encoder_step.bias")):
        if name in params:
            out[key] = np.round(params[name]).astype(np.int64).ravel()
    # shape sanity: W_in should be [H, 48]; Talamo may store [48, H]
    if "W_in" in out and out["W_in"].shape[1] != 48 and out["W_in"].shape[0] == 48:
        for k in ("W_in", "W_rec", "W_out"):
            if k in out:
                out[k] = out[k].T
        print("transposed weight matrices to [out, in]")
    for k in ("thr_h", "thr_o"):
        if k in out:
            out[k] = np.atleast_1d(out[k]).ravel()[:1]
    out["cfg"] = np.array(str(dict(n_in=48, hidden=m["hidden"], out=m.get("out", 12), recurrent=not m.get("feedforward", False))))
    np.savez(a.run / "weights_int8.npz", **out)
    print(f"-> {a.run / 'weights_int8.npz'}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items() if k != "cfg"))


if __name__ == "__main__":
    main()
