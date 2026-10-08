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

Fixed 2026-10-08 (yujie/chip/TRAINING_FEEDBACK.md has the measurements). Before, the imported pipeline ran random weights:
  - the values were copied into the live tensors only; pipe(...) and the saved file use the stored parameter
    set, so they are now written back with step.save_params() inside the quantized context;
  - quantize() leaves the IF encoder multiplying its input by 2**7 with a threshold scaled to match; the trained
    thresholds are in unscaled units, so the multiplier is reset to 2**0 (sign +1, bias 0);
  - W_rec is square, so the shape test never transposed it; torch stores [to, from], Talamo [from, to].
--check now compares pipe(...) with yujie/chip/chipref.py under the chip's rules and scores spike positions.

If a parameter name does not match, run export_npz.py --list on any Talamo run to see the real names and adjust
export_npz.MAP.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import talamo  # noqa: E402
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
        if key == "W_rec":
            val = val.T                                        # square: must be transposed explicitly
        elif val.shape != cur.shape:
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
    # The encoder computes floor(x * 2**weight) * sign + bias; the trained thresholds assume 2**0, +1, 0.
    for name, value in (("weight", 0.0), ("weight_sign", 1.0), ("bias", 0.0)):
        ref = next(r for n, r in refs.items() if "encoder" in n and n.endswith("." + name))
        with torch.no_grad():
            ref.get_value().fill_(value)
    # Store the tensors in the pipeline's parameter set: pipe(...) and save() read that, not the live tensors.
    with pipe.trainable_params.quantized_context():
        for step in pipe.filter_steps(talamo.pipeline.elements.PipelineStep):
            step.save_params(pipe.trainable_params)
    a.out.mkdir(parents=True, exist_ok=True)
    pipe.trainable_params.save(str(a.out / "trainable_params_quantized.json"))
    pipe.trainable_params.save(str(a.out / "trainable_params.json"))
    json.dump(dict(args=dict(hidden=cfg["hidden"], feedforward=not cfg["recurrent"], out=cfg["out"],
                             enc_threshold=float(np.median(w["enc_threshold"])), name=a.out.name, source=str(a.npz)),
                   imported_from="train_torch.py"), open(a.out / "metrics.json", "w"), indent=1)
    print(f"-> {a.out}")

    if a.check:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "yujie" / "chip"))
        import chipref
        recs = sorted(data.load_recordings("val", data=a.check), key=lambda r: -len(r["X"]))
        X = recs[0]["X"]
        X = X[600:1200] if len(X) >= 1200 else X[:600]              # ~10 s, or the start of a short recording
        ds = data.dense_dataset(torch.tensor(X[None]), torch.zeros(1, len(X), 14))
        s_tal = pipe(ds).data_as_numpy()[0][:len(X)]                  # the Snn appends one frame; chip rules include the delay
        steps = list(pipe.filter_steps(talamo.pipeline.elements.PipelineStep))[0](ds).data_as_numpy()[0][:len(X)]
        reloaded, _ = model.build_pipeline(cfg["hidden"], recurrent=cfg["recurrent"], out_neurons=cfg["out"],
                                           enc_threshold=float(np.median(w["enc_threshold"])))
        reloaded.trainable_params.load(str(a.out / "trainable_params_quantized.json"))
        s_file = reloaded(ds).data_as_numpy()[0][:len(X)]
        s_chip = chipref.run(w, X, **chipref.CHIP)
        s_trained = chipref.run(w, X, **chipref.TRAINED)

        def f1(p, q):
            p, q = p.astype(bool), q.astype(bool)
            return 2 * (p & q).sum() / max(1, p.sum() + q.sum())
        print(f"check on {recs[0]['id']}, {len(X)} frames x {s_tal.shape[1]} outputs (spike-position F1; 1.000 = identical)")
        print(f"  pipe(...)               {int(s_tal.sum()):5d} spikes")
        print(f"  steps called directly   {int(steps.sum()):5d} spikes, F1 {f1(s_tal, steps):.3f}")
        print(f"  saved file, reloaded    {int(s_file.sum()):5d} spikes, F1 {f1(s_tal, s_file):.3f}")
        print(f"  reference, chip rules   {int(s_chip.sum()):5d} spikes, F1 {f1(s_tal, s_chip):.3f}")
        print(f"  reference, as trained   {int(s_trained.sum()):5d} spikes, F1 {f1(s_tal, s_trained):.3f}  (differs by design, see TRAINING_FEEDBACK.md)")
        if not (np.array_equal(s_tal, steps) and np.array_equal(s_tal, s_file) and np.array_equal(s_tal, s_chip)):
            sys.exit("IMPORT CHECK FAILED: the pipeline does not reproduce the reference under the chip's rules")
        print("IMPORT CHECK PASSED")


if __name__ == "__main__":
    main()
