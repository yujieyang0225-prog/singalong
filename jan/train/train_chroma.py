#!/usr/bin/env python3
"""Simplest baseline: audio bands -> SNN -> 12 chroma outputs that should match the MIDI chords.

No augmentation, no song knowledge at run time, no pitch-recognition stage. Live/causal:
one 16 ms frame in, one SNN timestep, one output step.

  X (48 bands, D02) -> per-frame normalise (40 dB), silence gate, x100 -> c1.IFEncoder (48)
    -> Dense 48->64 (c1.digital_neuron, trainable bias = linear leak, D05)
    -> Dense 64->12 (c1.digital_neuron)                      [D04 variant A, feedforward]
  output = spike rate of each chroma neuron over the last RATE_WIN frames (D08 readout)
  target = MIDI 'Chords' pitch classes at that frame (rotated by the singer's transposition)
  loss   = BCE(rate, target) on scored frames (D09, simplified)
Training on random 2 s crops, the first 0.5 s unscored (burn-in), full BPTT (D06). Talamo (D12).

Run inside the SDK container from the repo root:
    python3 jan/train/train_chroma.py --steps 20          # smoke test
    python3 jan/train/train_chroma.py --steps 1500        # real run
Output: data/runs/<name>/{model.pt, metrics.json, log.csv, val_example.png}
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import talamo
from talamo.snn.neuron.c1 import digital_neuron

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "damp100"
NAMES = "C C# D D# E F F# G G# A A# B".split()


# -------------------------------------------------------------------- data --
def load_split(split, ref):
    rows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] == split]
    mel = ref["melody"]
    on = np.flatnonzero(mel >= 0)
    scored = np.zeros(len(mel), bool)
    scored[on[0]:on[-1] + 1] = True                     # sung part of the song only
    out = []
    for r in rows:
        d = np.load(DATA / f"{r['id']}.npz")
        X = d["X"].astype(np.float32)
        T = min(len(X), len(mel))
        Xn = np.maximum(X[:T] - X[:T].max(1, keepdims=True), -13.3) / 13.3 + 1   # 0..1, 40 dB range
        Xn *= d["voiced"][:T, None]                                            # silence gate (D02)
        Y = np.roll(ref["chord_pcs"][:T], int(d["transpose"]), axis=1).astype(np.float32)
        out.append(dict(id=r["id"], X=np.round(Xn * 100).astype(np.float32), Y=Y, M=scored[:T]))
    return out


def sample_batch(recs, bs, crop, rng):
    X, Y, M = [], [], []
    for _ in range(bs):
        r = recs[rng.integers(len(recs))]
        valid = np.flatnonzero(r["M"])
        t0 = rng.integers(max(valid[0] - crop // 4, 0), valid[-1] - crop)
        X.append(r["X"][t0:t0 + crop]); Y.append(r["Y"][t0:t0 + crop]); M.append(r["M"][t0:t0 + crop])
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)), torch.tensor(np.stack(M))


# ------------------------------------------------------------------- model --
class ChromaSNN(torch.nn.Module):
    def __init__(self, hidden=64, threshold=1.0, rate_win=8):
        super().__init__()
        sg = talamo.snn.gradient.fast_sigmoid_surrogate_gradient_descent(10000)
        syn = talamo.snn.synapse.c1.digital_synapse()
        self.enc = talamo.encoders.c1.IFEncoder(
            num_encoder_channels=48, threshold_initializer=talamo.initializers.RandomUniform(1, 10),
            encoder_timestep=1e-6)
        self.net = talamo.snn.containers.Sequential(
            talamo.snn.layer.Dense(48, hidden, syn, digital_neuron(firing_threshold=threshold, spike_gradient=sg,
                                                                   train_bias=True)),
            talamo.snn.layer.Dense(hidden, 12, syn, digital_neuron(firing_threshold=threshold, spike_gradient=sg,
                                                                   train_bias=True), probe=True))
        self.net.set_timestep(1e-6)
        self.rate_win = rate_win

    def forward(self, x):                      # x [B, T, 48] integer-valued
        s_in = self.enc(x)                     # [B, T, 48] spikes
        self.net.reset()
        out = torch.stack([self.net(s_in[:, t]) for t in range(x.shape[1])], 1)   # [B, T, 12] spikes
        # causal spike rate over the last rate_win frames = what the RISC-V counts (D08)
        pad = F.pad(out.transpose(1, 2), (self.rate_win - 1, 0))
        rate = F.avg_pool1d(pad, self.rate_win, stride=1).transpose(1, 2)
        return rate.clamp(1e-4, 1 - 1e-4), out, s_in


# ----------------------------------------------------------------- metrics --
def scores(rate, Y, M, thr=0.5):
    p = (rate[M] >= thr).float()
    y = Y[M]
    tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()
    f1 = float(2 * tp / (2 * tp + fp + fn + 1e-9))
    exact = float((p == y).all(1).float().mean())
    return dict(f1=round(f1, 3), exact=round(exact, 3))


def baseline(train, val):
    """Best constant output: the most common chord pitch-class set in train, scored on val."""
    sets, counts = np.unique(np.concatenate([r["Y"][r["M"]] for r in train]), axis=0, return_counts=True)
    c = torch.tensor(sets[counts.argmax()])
    Y = torch.tensor(np.concatenate([r["Y"][r["M"]] for r in val]))
    s = scores(c.expand_as(Y), Y, torch.ones(len(Y), dtype=torch.bool))
    return {**s, "chord": " ".join(NAMES[i] for i in np.flatnonzero(sets[counts.argmax()]))}


@torch.no_grad()
def evaluate(model, val, crop, burn):
    model.eval()
    rng = np.random.default_rng(123)
    X, Y, M = sample_batch(val, 64, crop, rng)
    rate, out, s_in = model(X)
    M[:, :burn] = False
    model.train()
    return {**scores(rate, Y, M), "out_rate": round(float(out.mean()), 3), "in_rate": round(float(s_in.mean()), 3)}


@torch.no_grad()
def plot_example(model, rec, path, secs=40, start_s=9.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    t0, n = int(start_s * 62.5), int(secs * 62.5)
    x = torch.tensor(rec["X"][None, t0:t0 + n])
    rate = model(x)[0][0].numpy()
    fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    ext = [start_s, start_s + n / 62.5]
    ax[0].imshow(rec["X"][t0:t0 + n].T, origin="lower", aspect="auto", cmap="Blues", extent=ext + [40, 88])
    ax[0].set_title(f"input: 48 bands ({rec['id']})", loc="left", fontsize=9)
    ax[1].imshow(rec["Y"][t0:t0 + n].T, origin="lower", aspect="auto", cmap="Greys", extent=ext + [-.5, 11.5])
    ax[1].set_title("target: MIDI chord pitch classes", loc="left", fontsize=9)
    ax[2].imshow(rate.T, origin="lower", aspect="auto", cmap="Oranges", vmin=0, vmax=1, extent=ext + [-.5, 11.5])
    ax[2].set_title("SNN output: spike rate per chroma neuron (causal, live)", loc="left", fontsize=9)
    for a in ax[1:]:
        a.set_yticks(range(12), NAMES, fontsize=7)
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    model.train()


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--crop", type=int, default=125, help="frames (16 ms) per crop: 125 = 2 s")
    ap.add_argument("--burn", type=int, default=31, help="unscored frames at crop start: 31 = 0.5 s")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--name", default=time.strftime("%Y-%m-%d_%H%M") + "_chroma_ff64")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(6)
    rng = np.random.default_rng(0)
    out_dir = REPO / "data" / "runs" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = dict(np.load(DATA / "reference.npz"))
    train, val = load_split("train", ref), load_split("val", ref)
    base = baseline(train, val)
    print(f"train {len(train)} / val {len(val)} recordings · baseline (always '{base['chord']}'): "
          f"F1 {base['f1']}, exact {base['exact']}", flush=True)

    model = ChromaSNN(a.hidden)
    opt = torch.optim.AdamW([{"params": model.net.parameters(), "lr": a.lr},
                             {"params": model.enc.parameters(), "lr": a.lr / 10}])
    log = open(out_dir / "log.csv", "w")
    log.write("step,loss,val_f1,val_exact,out_rate,in_rate,sec\n")
    t_start, best = time.time(), None
    for step in range(1, a.steps + 1):
        X, Y, M = sample_batch(train, a.bs, a.crop, rng)
        M[:, :a.burn] = False
        rate, _, _ = model(X)
        loss = F.binary_cross_entropy(rate[M], Y[M])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.eval_every == 0 or step == a.steps or step == 1:
            ev = evaluate(model, val, a.crop, a.burn)
            el = time.time() - t_start
            print(f"step {step:5d}  loss {loss.item():.3f}  val F1 {ev['f1']:.3f}  exact {ev['exact']:.3f}  "
                  f"spikes in/out {ev['in_rate']:.2f}/{ev['out_rate']:.2f}  ({el:.0f} s, "
                  f"{el / step:.2f} s/step)", flush=True)
            log.write(f"{step},{loss.item():.4f},{ev['f1']},{ev['exact']},{ev['out_rate']},{ev['in_rate']},{el:.0f}\n")
            log.flush()
            if best is None or ev["f1"] > best["f1"]:
                best = {**ev, "step": step}
                torch.save(model.state_dict(), out_dir / "model.pt")
    model.load_state_dict(torch.load(out_dir / "model.pt"))
    plot_example(model, val[0], out_dir / "val_example.png")
    json.dump(dict(args=vars(a), baseline=base, best=best, train_ids=[r["id"] for r in train],
                   val_ids=[r["id"] for r in val]), open(out_dir / "metrics.json", "w"), indent=1)
    print(f"best val: {best} · baseline: {base} -> {out_dir}")


if __name__ == "__main__":
    main()
