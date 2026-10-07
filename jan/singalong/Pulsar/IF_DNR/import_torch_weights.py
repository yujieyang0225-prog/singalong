#!/usr/bin/env python3
"""Load a train_torch.py export (weights_int8.npz) into the Talamo pipeline, so a GPU-trained network goes
through Innatera's quantize -> simulator -> embedded-app path.

    jan/singalong/run.sh python3 Pulsar/IF_DNR/import_torch_weights.py data/runs/<torch run>/weights_int8.npz \
        --out data/runs/<torch run>_talamo [--check data/damp100]

Steps: build the pipeline with the same sizes, write the integers into the Talamo parameters (name mapping as in
export_npz.MAP), save trainable_params_quantized.json + metrics.json in the layout train.py produces, so
sim_check.py / deploy_live.py work unchanged. With --check, run both implementations (Talamo software and
singalong.snn_numpy) on 1 s of a validation recording and report the spike agreement, like sim_check does
for PyTorch vs the simulator.

UNTESTED in this session (no SDK here). If a parameter name does not match, run export_npz.py --list on any
Talamo run to see the real names and adjust export_npz.MAP.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from singalong import data, model  # noqa: E402
from Pulsar.IF_DNR.export_npz import MAP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, help="dataset folder for a software-vs-numpy spike comparison")
    a = ap.parse_args()
    w = np.load(a.npz, allow_pickle=False)
    cfg = ast.literal_eval(str(w["cfg"]))
    pipe, snn_step = model.build_pipeline(cfg["hidden"], recurrent=cfg["recurrent"], out_neurons=cfg["out"],
                                          enc_threshold=float(np.median(w["enc_threshold"])))
    pipe.quantize(quantizers=model.make_quantizers(pipe))          # put the steps in quantized state first
    refs = {r.name: r for r in pipe.query_torch_params("*", references=True)}
    for key, subs in MAP.items():
        if key not in w.files:
            continue
        cands = [n for n in refs if all(s in n for s in subs) and ("recurrent" in n) == ("recurrent" in "".join(subs))]
        if key == "enc_threshold":
            cands = [n for n in refs if "encoder" in n and "threshold" in n]
        if not cands:
            print(f"!! no Talamo parameter for {key}; names: {list(refs)}")
            continue
        ref = refs[cands[0]]
        cur = ref.get_value().detach()
        val = torch.tensor(w[key], dtype=cur.dtype)
        if val.shape != cur.shape:
            if val.T.shape == cur.shape:
                val = val.T
            elif val.numel() == 1:
                val = val.reshape(cur.shape) if cur.numel() == 1 else torch.full_like(cur, float(val))
            else:
                print(f"!! shape mismatch for {key}: npz {tuple(val.shape)} vs talamo {tuple(cur.shape)}")
                continue
        with torch.no_grad():
            ref.get_value().copy_(val)
        print(f"{key:14s} -> {cands[0]} {tuple(cur.shape)}")
    a.out.mkdir(parents=True, exist_ok=True)
    pipe.trainable_params.save(str(a.out / "trainable_params_quantized.json"))
    pipe.trainable_params.save(str(a.out / "trainable_params.json"))
    json.dump(dict(args=dict(hidden=cfg["hidden"], feedforward=not cfg["recurrent"], out=cfg["out"],
                             enc_threshold=float(np.median(w["enc_threshold"])), name=a.out.name, source=str(a.npz)),
                   imported_from="train_torch.py"), open(a.out / "metrics.json", "w"), indent=1)
    print(f"-> {a.out}")

    if a.check:
        from singalong.snn_numpy import C1Net
        recs = data.load_recordings("val", data=a.check)
        X = recs[0]["X"][600:1200]                                   # ~10 s
        tal = model.predict(pipe, torch.tensor(X[None]), n_label_ch=14)          # rates [1,T,O]
        s_tal = pipe(data.dense_dataset(torch.tensor(X[None]), torch.zeros(1, len(X), 14))).data_as_numpy()[0, 1:]
        s_np = C1Net(a.npz).run(X)
        agree = (s_tal == s_np).mean()
        print(f"Talamo software vs numpy reference: spike agreement {agree:.3f} over {s_np.shape}; "
              f"spikes {int(s_tal.sum())} vs {int(s_np.sum())}")
        for delay in (0, 1):
            s_np2 = C1Net(a.npz, layer_delay=delay).run(X)
            for enc in ("subtract", "zero"):
                s3 = C1Net(a.npz, layer_delay=delay, enc_reset=enc).run(X)
                print(f"  layer_delay={delay} enc_reset={enc}: agreement {(s_tal == s3).mean():.3f}")


if __name__ == "__main__":
    main()
