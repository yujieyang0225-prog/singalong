#!/usr/bin/env python3
"""Train the sing-along C1 network in plain PyTorch (CPU or GPU, no Innatera SDK needed).

    python3 train_torch.py --data ../../data/damp100 --data ../../data/synth200 --hidden 64 --epochs 20 --name rnn64
    python3 train_torch.py --data ../../data/synth40 --epochs 2 --device cpu --name smoke      # quick check

Per epoch: fresh random augmentation of every training recording (D11), fresh 2 s crops (random offset),
BPTT over the crop, AdamW. The last --qat-epochs epochs round weights/thresholds with a straight-through
estimator. Outputs in data/runs/<name>/: checkpoint.pt, weights_int8.npz (for snn_numpy and Talamo import),
log.csv, metrics.json. Picture: plot_chain.py --evidence snn:<weights_int8.npz>; sound: live.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from singalong import augment, data, metrics                     # noqa: E402
from singalong.torchsnn import DualLoss, SingalongSNN, spike_rate  # noqa: E402


def evaluate(net, Xv, Yv, device, batch=128):
    net.eval()
    outs, hid = [], []
    with torch.no_grad():
        for i in range(0, len(Xv), batch):
            s_o, s_h, _ = net(Xv[i:i + batch].to(device), return_hidden=True)
            outs.append(spike_rate(s_o).cpu())
            hid.append(s_h.mean((0, 1)).cpu())
    rate = torch.cat(outs)
    hr = torch.stack(hid).mean(0)
    ev = metrics.scores(rate[..., :12], Yv)
    ev.update({f"{k}_thr0.2": v for k, v in metrics.scores(rate[..., :12], Yv, thr=0.2).items()})
    ev.update(metrics.pitch_scores(rate[..., 12:24], Yv) if rate.shape[-1] >= 24 else {})
    ev.update(hidden_rate=round(float(hr.mean()), 3), dead_hidden=int((hr < 1e-3).sum()),
              out_rate=round(float(rate.mean()), 3))
    net.train()
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="dataset folder(s) with manifest.csv")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=int, default=24, help="12 = chord head only, 24 = chord + melody heads")
    ap.add_argument("--feedforward", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--qat-epochs", type=int, default=4, help="last N epochs with integer rounding (STE)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.3, help="in chip units (int8 weights), so ~100x a float-net lr")
    ap.add_argument("--lr-enc", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.0,
                    help="AdamW weight decay of the hidden/output layers. Keep 0: with fixed thresholds the decay shrinks "
                         "the int8 weights and so the firing rates (AdamW's default 0.01 made every run decline or collapse)")
    ap.add_argument("--crop", type=int, default=125, help="frames per crop (125 = 2 s)")
    ap.add_argument("--stride", type=int, default=0, help="crop stride in frames (0 = crop/2, overlapping)")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--compact", action="store_true", help="recordings in compact dtypes (~1/3 of the RAM, same values)")
    ap.add_argument("--chunk", type=int, default=0,
                    help="augment + crop this many recordings at a time (0 = all at once); bounds the RAM per epoch")
    ap.add_argument("--shift", type=int, default=6, help="max transposition in semitones for augmentation")
    ap.add_argument("--w-pitch", type=float, default=1.0)
    ap.add_argument("--w-chord", type=float, default=1.0)
    ap.add_argument("--pitch-only", action="store_true",
                    help="train the melody head only; the 12 chord outputs stay silent (weights 0, bias -128), so the "
                         "file keeps the 24-output layout and the selector ignores the chord evidence")
    ap.add_argument("--val-crops", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--name", default=time.strftime("%Y-%m-%d_%H%M") + "_torch")
    ap.add_argument("--runs", type=Path, default=data.REPO / "data" / "runs")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = a.runs / a.name
    out.mkdir(parents=True, exist_ok=True)

    train_recs = data.load_recordings("train", datasets=a.data, compact=a.compact)
    val_recs = data.load_recordings("val", datasets=a.data, compact=a.compact)
    Xv, Yv = data.crops(val_recs, crop=a.crop)
    keep = np.random.default_rng(1).choice(len(Xv), min(a.val_crops, len(Xv)), replace=False)
    Xv, Yv = Xv[keep], Yv[keep]
    base = data.baseline(train_recs, Yv)
    print(f"device {device} · train {len(train_recs)} recordings · val {len(Xv)} crops · " +
          (f"baseline (always {base['chord']}): F1 {base['f1']} exact {base['exact']}" if base else "no chord labels"),
          flush=True)

    net = SingalongSNN(48, a.hidden, a.out, recurrent=not a.feedforward).to(device)
    if a.pitch_only:
        if a.out < 24:
            sys.exit("--pitch-only needs --out 24 (outputs 12..23 are the melody head)")
        a.w_chord = 0.0

    def silence_chords():
        if a.pitch_only:
            with torch.no_grad():
                net.o.W[:12] = 0
                net.o.b[:12] = -128
    silence_chords()
    print(f"network 48 -> {a.hidden}{' rec' if not a.feedforward else ''} -> {a.out}: {net.fit_note}")
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": a.lr_enc, "weight_decay": 0},
                             {"params": list(net.h.parameters()) + list(net.o.parameters()), "lr": a.lr,
                              "weight_decay": a.wd}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = DualLoss(w_pitch=a.w_pitch if a.out >= 24 else 0, w_chord=a.w_chord)
    start_ep, best = 1, None
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_ep = ck["epoch"] + 1
        best = ck.get("best")
    aug_cfg = dict(augment.DEFAULT, shift=a.shift)

    logf = open(out / "log.csv", "a")
    w = csv.writer(logf)
    if start_ep == 1:
        w.writerow(["epoch", "loss", "loss_chord", "loss_pitch", "val_f1", "val_exact", "val_f1_0.2", "pitch_acc",
                    "hidden_rate", "dead", "qat", "sec"])
    ev = evaluate(net, Xv, Yv, device)
    print(f"epoch  0  {ev}", flush=True)
    t0 = time.time()
    for ep in range(start_ep, a.epochs + 1):
        qat = ep > a.epochs - a.qat_epochs
        net.set_qat(qat)
        tot, parts_sum, n = 0.0, {}, 0
        order = rng.permutation(len(train_recs)) if a.chunk else np.arange(len(train_recs))
        size = a.chunk or len(train_recs)
        for c in range(0, len(order), size):                          # chunk 0: one chunk = the old behaviour
            part = [train_recs[j] for j in order[c:c + size]]
            recs = part if a.no_aug else [augment.random_augment(r, rng, aug_cfg) for r in part]
            X, Y = data.crops(recs, crop=a.crop, stride=a.stride or a.crop // 2, start_offset=int(rng.integers(a.crop)))
            del part, recs
            perm = torch.randperm(len(X))
            for i in range(0, len(X), a.batch):
                idx = perm[i:i + a.batch]
                xb, yb = X[idx].to(device), Y[idx].to(device)
                s_o, s_h, _ = net(xb, return_hidden=True)
                loss, parts = loss_fn(s_o, yb, s_h)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                silence_chords()
                tot += float(loss.detach()); n += 1
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0) + v
            del X, Y
        sched.step()
        ev = evaluate(net, Xv, Yv, device)
        el = time.time() - t0
        print(f"epoch {ep:2d}{' q' if qat else '  '} loss {tot / n:.3f} ({', '.join(f'{k} {v / n:.3f}' for k, v in parts_sum.items())})"
              f"  val F1 {ev['f1']:.3f} exact {ev['exact']:.3f} F1@0.2 {ev['f1_thr0.2']:.3f}"
              f" pitch {ev.get('pitch_acc', float('nan')):.3f}  hid {ev['hidden_rate']:.2f} dead {ev['dead_hidden']}  ({el:.0f} s)",
              flush=True)
        w.writerow([ep, round(tot / n, 4), round(parts_sum.get("chord", 0) / n, 4), round(parts_sum.get("pitch", 0) / n, 4),
                    ev["f1"], ev["exact"], ev["f1_thr0.2"], ev.get("pitch_acc"), ev["hidden_rate"], ev["dead_hidden"],
                    int(qat), round(el)])
        logf.flush()
        score = (0 if a.pitch_only else ev["f1"]) + ev.get("pitch_acc", 0)
        if qat and (best is None or score > best["score"]):          # keep the best *integer* model
            best = {**ev, "epoch": ep, "score": score}
            net.export_npz(out / "weights_int8.npz")
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), epoch=ep, best=best, cfg=net.cfg,
                        args={k: str(v) for k, v in vars(a).items()}),
                   out / "checkpoint.pt")
    if best is None:                                                  # no QAT epochs: export the final rounded net
        net.set_qat(True)
        best = {**evaluate(net, Xv, Yv, device), "epoch": a.epochs}
        net.export_npz(out / "weights_int8.npz")
    json.dump(dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, device=device, baseline=base, best=best, cfg=net.cfg, fit=net.fit_note,
                   train_ids=[r["id"] for r in train_recs], val_ids=[r["id"] for r in val_recs]),
              open(out / "metrics.json", "w"), indent=1)
    print(f"best int8 model (epoch {best['epoch']}): {best}\n-> {out}")


if __name__ == "__main__":
    main()
