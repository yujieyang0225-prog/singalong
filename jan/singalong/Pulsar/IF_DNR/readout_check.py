#!/usr/bin/env python3
"""How good is a trained run under different readouts? (threshold sweep, top-3 pitch classes, per-class stats)

Uses the same 256 validation crops as train.py. Compares float and quantized models with the
constant baseline chord (stored in metrics.json).

Run:  jan/singalong/run.sh python3 Pulsar/IF_DNR/readout_check.py data/runs/<run>
Writes data/runs/<run>/readout_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from singalong import data, model  # noqa: E402


def topk_sets(rate, k=3):
    idx = rate.topk(k, dim=-1).indices
    return torch.zeros_like(rate).scatter_(-1, idx, 1.0)


def set_scores(pred, Y):
    m = Y[..., 12] > 0
    p, y = pred[m], Y[..., :12][m]
    tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()
    return dict(f1=round(float(2 * tp / (2 * tp + fp + fn + 1e-9)), 3),
                exact=round(float((p == y).all(1).float().mean()), 3),
                # does the predicted set contain all target notes? (chord found, extra notes allowed)
                covers=round(float(((p >= y).all(1)).float().mean()), 3))


def main():
    run = Path(sys.argv[1])
    run = run if run.is_absolute() else data.REPO / run
    meta = json.load(open(run / "metrics.json"))
    a = meta["args"]
    Xv, Yv = data.crops(data.load_recordings("val"))
    keep = np.random.default_rng(1).choice(len(Xv), min(a["val_crops"], len(Xv)), replace=False)
    Xv, Yv = Xv[keep], Yv[keep]

    base_pcs = [data.NAMES.index(n) for n in meta["baseline"]["chord"].split()]
    base = torch.zeros(12); base[base_pcs] = 1
    res = {"baseline_chord": meta["baseline"]["chord"],
           "baseline": set_scores(base.expand_as(Yv[..., :12]).clone(), Yv)}

    for kind in ("float", "quantized"):
        pipe, _ = model.build_pipeline(a["hidden"], recurrent=not a.get("feedforward", False),
                                       enc_threshold=a.get("enc_threshold", 120.0))
        if kind == "quantized":
            pipe.quantize(quantizers=model.make_quantizers(pipe))
            pipe.trainable_params.load(str(run / "trainable_params_quantized.json"))
        else:
            pipe.trainable_params.load(str(run / "trainable_params.json"))
        rate = model.predict(pipe, Xv)
        r = {f"thr_{t}": set_scores((rate >= t).float(), Yv) for t in (0.1, 0.2, 0.3, 0.4, 0.5)}
        r["top3"] = set_scores(topk_sets(rate, 3), Yv)
        m = Yv[..., 12] > 0
        r["mean_rate_per_class"] = [round(float(v), 3) for v in rate[m].mean(0)]
        r["silent_frames_share"] = round(float((rate[m].max(-1).values == 0).float().mean()), 3)
        r["flicker_mean_abs_change"] = round(float((rate[:, 1:] - rate[:, :-1]).abs().mean()), 4)
        res[kind] = r
    res["target_share_per_class"] = [round(float(v), 3) for v in Yv[..., :12][Yv[..., 12] > 0].mean(0)]

    print(f"{'readout':<10} {'F1':>6} {'exact':>6} {'covers':>7}")
    print(f"{'baseline':<10} {res['baseline']['f1']:>6} {res['baseline']['exact']:>6} {res['baseline']['covers']:>7}"
          f"   (always {res['baseline_chord']})")
    for kind in ("float", "quantized"):
        for k in ("thr_0.1", "thr_0.2", "thr_0.3", "thr_0.4", "thr_0.5", "top3"):
            s = res[kind][k]
            print(f"{kind[:5]} {k:<8} {s['f1']:>6} {s['exact']:>6} {s['covers']:>7}")
    print("class      " + " ".join(f"{n:>5}" for n in data.NAMES))
    print("target %   " + " ".join(f"{v:>5.2f}" for v in res["target_share_per_class"]))
    print("rate float " + " ".join(f"{v:>5.2f}" for v in res["float"]["mean_rate_per_class"]))
    print(f"frames with no output spike at all: {res['float']['silent_frames_share']:.0%} · "
          f"flicker (mean |Δrate| per frame): {res['float']['flicker_mean_abs_change']}")
    json.dump(res, open(run / "readout_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
