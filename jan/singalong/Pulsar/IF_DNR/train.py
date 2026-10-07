#!/usr/bin/env python3
"""Train the sing-along SNN the Innatera way: Talamo Pipeline -> fit -> evaluate -> quantize -> evaluate.

  IFEncoder(48) -> Snn(GRSnn2LayersRNN 48 -> 64 recurrent -> 12)  (wristband example model, C1 neurons)
  target: MIDI chord pitch classes per 16 ms frame (12 chroma, multi-hot), no augmentation.

Run (from anywhere):
    jan/singalong/run.sh python3 Pulsar/IF_DNR/train.py --epochs 5
Output: data/runs/<name>/ trainable_params.json (float), trainable_params_quantized.json (int8, chip),
        log.csv, metrics.json  -> visualise with jan/train/visualise_run.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from singalong import data, model  # noqa: E402


def evaluate(pipe, Xv, Yv):
    return model.scores(model.predict(pipe, Xv), Yv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=int, default=24, help="12 = chord head only (runs 0-2), 24 = chord + melody heads")
    ap.add_argument("--loss", choices=["bce", "dual"], default="dual")
    ap.add_argument("--data", action="append", help="dataset folder(s); default $SINGALONG_DATA or data/damp100")
    ap.add_argument("--aug", action="store_true", help="fresh random augmentation (D11) of every recording each epoch")
    ap.add_argument("--shift", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="'cuda' is untested with Talamo's Pipeline.fit; see README")
    ap.add_argument("--feedforward", action="store_true", help="no recurrent synapses (D04 variant A)")
    ap.add_argument("--lr", type=float, default=1e-3, help="chip units; the torch trainer needed ~0.3, try --lr 0.1")
    ap.add_argument("--enc-threshold", type=float, default=120.0)
    ap.add_argument("--train-crops", type=int, default=0, help="limit number of train crops (0 = all)")
    ap.add_argument("--val-crops", type=int, default=256)
    ap.add_argument("--name", default=time.strftime("%Y-%m-%d_%H%M") + "_rnn64")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    out = data.REPO / "data" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)

    train_recs = data.load_recordings("train", datasets=a.data)
    val_recs = data.load_recordings("val", datasets=a.data)
    X, Y = data.crops(train_recs)
    if a.train_crops:
        keep = rng.choice(len(X), a.train_crops, replace=False)
        X, Y = X[keep], Y[keep]
    Xv, Yv = data.crops(val_recs)
    keep = np.random.default_rng(1).choice(len(Xv), min(a.val_crops, len(Xv)), replace=False)
    Xv, Yv = Xv[keep], Yv[keep]
    base = data.baseline(train_recs, Yv)
    print(f"train {len(train_recs)} singers -> {len(X)} crops of 2 s · val {len(Xv)} crops · "
          f"baseline (always {base['chord']}): F1 {base['f1']} exact {base['exact']}", flush=True)

    pipe, snn_step = model.build_pipeline(a.hidden, recurrent=not a.feedforward, enc_threshold=a.enc_threshold,
                                          out_neurons=a.out)
    if a.device != "cpu":                                            # untested: Talamo may keep its own device
        try:
            pipe.to(torch.device(a.device))
        except Exception as e:  # noqa: BLE001
            print(f"could not move the pipeline to {a.device}: {e}; training on cpu")
    loss_fn = model.DualRateLoss() if a.loss == "dual" and a.out >= 24 else model.RateBCE()
    log = open(out / "log.csv", "w")
    log.write("step,loss,val_f1,val_exact,out_rate,in_rate,sec\n")
    t0, best = time.time(), None
    ev = evaluate(pipe, Xv, Yv)
    print(f"epoch  0  val F1 {ev['f1']:.3f} exact {ev['exact']:.3f}", flush=True)
    for ep in range(1, a.epochs + 1):
        if a.aug:
            from singalong import augment
            recs = [augment.random_augment(r, rng, dict(augment.DEFAULT, shift=a.shift)) for r in train_recs]
            X, Y = data.crops(recs, start_offset=int(rng.integers(125)))
            if a.train_crops:                                       # honour --train-crops also with --aug
                keep = rng.choice(len(X), min(a.train_crops, len(X)), replace=False)
                X, Y = X[keep], Y[keep]
        model.fit(pipe, snn_step, X, Y, epochs=1, lr=a.lr, loss_function=loss_fn)
        ev = evaluate(pipe, Xv, Yv)
        el = time.time() - t0
        print(f"epoch {ep:2d}  val F1 {ev['f1']:.3f} exact {ev['exact']:.3f}  ({el:.0f} s)", flush=True)
        log.write(f"{ep},nan,{ev['f1']},{ev['exact']},nan,nan,{el:.0f}\n"); log.flush()
        score = ev["f1"] + ev.get("pitch_acc", 0.0)
        if best is None or score > best["score"]:
            best = {**ev, "step": ep, "score": score}
            pipe.trainable_params.save(str(out / "trainable_params.json"))

    # --- quantize the best float model (int8 weights/bias/threshold per layer, as in the wristband notebook)
    pipe.trainable_params.load(str(out / "trainable_params.json"))
    float_ev = evaluate(pipe, Xv, Yv)
    pipe.quantize(quantizers=model.make_quantizers(pipe))
    q_ev = evaluate(pipe, Xv, Yv)
    pipe.trainable_params.save(str(out / "trainable_params_quantized.json"))
    ranges = model.param_ranges(pipe)
    print(f"float  : {float_ev}\nquant. : {q_ev}")
    for k, v in ranges.items():
        print(f"  {k:<55} {v}")
    json.dump(dict(args=vars(a), baseline=base, best=best, float=float_ev, quantized=q_ev, param_ranges=ranges,
                   train_ids=[r["id"] for r in train_recs], val_ids=[r["id"] for r in val_recs]),
              open(out / "metrics.json", "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
