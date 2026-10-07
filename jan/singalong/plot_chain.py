#!/usr/bin/env python3
"""One figure of the demo chain on one recording: input bands, SNN melody evidence, chosen chord vs reference.

    python3 plot_chain.py --npz ../../data/synth200/synth01_003.npz --evidence snn:<weights_int8.npz> --out chain.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from singalong import amazing_grace as ag  # noqa: E402
from singalong import frontend as fe  # noqa: E402
from singalong import renderer as rd  # noqa: E402
import live  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--evidence", default="baseline")
    ap.add_argument("--selector", default="score")
    ap.add_argument("--out", default="chain.png")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--secs", type=float, default=30.0)
    a = ap.parse_args()
    d = np.load(a.npz)
    v = d["voiced"].astype(bool)
    X = fe.normalise(d["X"].astype(np.float32), v)
    mel = d["melody"]
    ref = d["chord_pcs"]
    evid, hmm = live.make_evidence(a.evidence), live.make_selector(a.selector, 80)
    T = len(X)
    E = np.zeros((T, 12)); P = np.zeros((T, 12)); notes = np.full((T, 3), np.nan)
    render = rd.Renderer(rd.PrintSink(quiet=True), min_hold_s=0.3)
    for t in range(T):
        m, c = evid(X[t], bool(v[t]))
        E[t] = m if m is not None else (c if c is not None else 0)
        s, post = hmm.step(m, c)
        P[t] = hmm.pcs(s)
        render.step(P[t], root_pc=int(hmm.em.T[s].argmax()), confidence=float(post[s]))
        notes[t, :len(render.notes)] = render.notes[:3]
    t0, t1 = int(a.start * fe.FPS), int(min(T, (a.start + a.secs) * fe.FPS))
    tt = np.arange(t0, t1) / fe.FPS
    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 2, 2]))
    ax[0].imshow(X[t0:t1].T, aspect="auto", origin="lower", extent=[tt[0], tt[-1], fe.LO - .5, fe.LO + fe.NB - .5], cmap="magma")
    ax[0].plot(tt, np.where(mel[t0:t1] >= 0, mel[t0:t1], np.nan), "c.", ms=2, label="sung note (label)")
    ax[0].set_ylabel("band (MIDI)"); ax[0].legend(loc="upper right"); ax[0].set_title("input: 48 semitone bands, 16 ms frames")
    ax[1].imshow(E[t0:t1].T, aspect="auto", origin="lower", extent=[tt[0], tt[-1], -.5, 11.5], cmap="viridis")
    ax[1].plot(tt, np.where(mel[t0:t1] >= 0, mel[t0:t1] % 12, np.nan), "w.", ms=2)
    ax[1].set_yticks(range(12)); ax[1].set_yticklabels(ag.NAMES); ax[1].set_ylabel("pitch class")
    ax[1].set_title(f"evidence ({a.evidence.split(':')[0]}): melody pitch-class rate; white = sung pitch class")
    ax[2].imshow(np.stack([ref[t0:t1].T, P[t0:t1].T, np.zeros_like(P[t0:t1].T)], -1).astype(float), aspect="auto",
                 origin="lower", extent=[tt[0], tt[-1], -.5, 11.5], interpolation="nearest")
    ax[2].set_yticks(range(12)); ax[2].set_yticklabels(ag.NAMES)
    ax[2].set_title("chord pitch classes: red = reference, green = played, yellow = both")
    for k in range(3):
        ax[3].step(tt, notes[t0:t1, k], where="post", lw=2)
    ax[3].plot(tt, np.where(mel[t0:t1] >= 0, mel[t0:t1], np.nan), "k.", ms=2, label="singer")
    ax[3].set_ylabel("MIDI note"); ax[3].set_xlabel("time (s)"); ax[3].legend(loc="upper right")
    ax[3].set_title("violin voices (voice leading, G3–A5) under the singer")
    plt.tight_layout()
    plt.savefig(a.out, dpi=110)
    sung = (mel >= 0) & v
    print(f"{a.out}: exact chord {np.mean((P[sung] == ref[sung]).all(1)):.2f} on {sung.sum()} sung frames")


if __name__ == "__main__":
    main()
