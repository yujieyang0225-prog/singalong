#!/usr/bin/env python3
"""One yardstick for all runs (D3): pitch accuracy of each run's exported int8 model on the SAME validation sets.

train_torch.py scores every run on its own val split, so A (VocalSet), B (DAMP) and C (both) are not comparable.
This loads data/runs/<run>/weights_int8.npz (the file that goes to the chip, not checkpoint.pt) and scores it on
three sets, with the same fixed crops for every run:
  vs_songs  VocalSet *songs* (Caro mio ben, Dona nobis pacem, Row row; not the spoken takes) by the 7 singers no
            model trained on (val + test singers): real songs, annotated notes = what was actually sung
  vs_exer   VocalSet exercises of the same 7 unseen singers (val + test splits)
  damp250   DAMP val: real amateur karaoke, but the label is the score (one transposition per take), not what was
            sung, and every take is the same song, so a DAMP-trained model can score here by knowing the song
Decision rule (D3): rank by vs_songs (real songs with real pitch ground truth), vs_exer as the tie-breaker within
0.02; damp250 is a sanity check (a model far below the others there is suspect).

Run inside the Innatera image from jan/singalong (needs torch), e.g. on the cluster:
  cd ~/MNH26SVR && srun --time=00:20:00 --cpus-per-task=4 singularity exec --bind $HOME:$HOME \
      --pwd $HOME/MNH26SVR/jan/singalong innatera_singularity.sif \
      python3 ../../simon/eval_pitch.py ../../data/runs/A_vocalset_P ../../data/runs/B_damp_P ../../data/runs/C_both_P
Prints a table and writes eval_pitch.json into each run folder.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jan" / "singalong"))
from singalong import data, metrics                                    # noqa: E402
from singalong.torchsnn import SingalongSNN, spike_rate                 # noqa: E402


def load_int8(run):
    z = np.load(Path(run) / "weights_int8.npz")
    cfg = eval(str(z["cfg"]))                                           # dict literal written by export_arrays
    net = SingalongSNN(**cfg)
    t = lambda k: torch.tensor(z[k], dtype=torch.float32)
    with torch.no_grad():
        net.enc.theta.copy_(t("enc_threshold"))
        net.h.W.copy_(t("W_in")); net.h.b.copy_(t("b_h")); net.h.thr.copy_(t("thr_h").reshape(-1)[:1])
        if net.h.W_rec is not None:
            net.h.W_rec.copy_(t("W_rec"))
        net.o.W.copy_(t("W_out")); net.o.b.copy_(t("b_o")); net.o.thr.copy_(t("thr_o").reshape(-1)[:1])
    net.set_qat(True)
    return net.eval()


def vocalset_unseen(folder, splits):
    """Recordings of the 7 VocalSet singers that are in no training split (val + test singers)."""
    import csv
    rows = list(csv.DictReader(open(Path(folder) / "manifest.csv")))
    unseen = {r["singer"] for r in rows if r["split"] in ("val", "test")}
    singer = {r["id"]: r["singer"] for r in rows}
    recs = [r for sp in splits for r in data.load_recordings(sp, data=folder)]
    return [r for r in recs if singer[r["id"]] in unseen and "_spoken" not in r["id"]]


def crops_of(recs, n, crop):
    X, Y = data.crops(recs, crop=crop)
    keep = np.random.default_rng(1).choice(len(X), min(n, len(X)), replace=False)
    return X[np.sort(keep)], Y[np.sort(keep)]


@torch.no_grad()
def pitch_acc(net, X, Y, device, batch=128):
    rates = [spike_rate(net(X[i:i + batch].to(device))).cpu() for i in range(0, len(X), batch)]
    rate = torch.cat(rates)
    out = metrics.pitch_scores(rate[..., 12:24], Y)
    out["chord_out_rate"] = round(float(rate[..., :12].mean()), 4)      # ~0 for --pitch-only models
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--vocalset", default="../../data/vocalset")
    ap.add_argument("--damp", default="../../data/damp250")
    ap.add_argument("--crops", type=int, default=512)
    ap.add_argument("--crop", type=int, default=125)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sets = {"vs_songs": crops_of(vocalset_unseen(a.vocalset, ["song"]), a.crops, a.crop),
            "vs_exer": crops_of(vocalset_unseen(a.vocalset, ["val", "test"]), a.crops, a.crop),
            "damp250": crops_of(data.load_recordings("val", data=a.damp), a.crops, a.crop)}
    print(f"crops: " + ", ".join(f"{k} {len(v[0])}" for k, v in sets.items()) + "   (ranked by vs_songs)")
    print(f"{'run':32s}" + "".join(f"{k:>12s}" for k in sets) + f"{'chord out':>11s}")
    for run in a.runs:
        if not (run / "weights_int8.npz").exists():
            print(f"{run.name:32s} (no weights_int8.npz)")
            continue
        net = load_int8(run).to(a.device)
        res = {k: pitch_acc(net, X, Y, a.device) for k, (X, Y) in sets.items()}
        print(f"{run.name:32s}" + "".join(f"{r['pitch_acc']:12.3f}" for r in res.values())
              + f"{max(r['chord_out_rate'] for r in res.values()):11.4f}")
        json.dump(dict(sets=res), open(run / "eval_pitch.json", "w"), indent=1)


if __name__ == "__main__":
    main()
