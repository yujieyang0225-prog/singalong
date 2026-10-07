#!/usr/bin/env python3
"""Score the whole chain offline on a dataset split: bands -> evidence -> HMM selector -> chord per frame,
against the reference chords (D13: note-in-chord, exact chord, chord changes per minute, selector key accuracy).

    python3 eval_chain.py --data ../../data/synth40 --evidence baseline --selector song
    python3 eval_chain.py --data ../../data/damp100 --evidence snn:../../data/runs/<run>/weights_int8.npz --selector key

Prints one line per recording and the mean. Use it to compare: baseline vs SNN evidence, key vs song selector,
melody head vs chord head (--use).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from singalong import data, metrics  # noqa: E402
from singalong import frontend as fe  # noqa: E402
import live  # noqa: E402


def run_recording(rec, evid_spec, selector, bpm, use):
    evid = live.make_evidence(evid_spec)
    hmm = live.make_selector(selector, bpm)
    X, v = rec["X"], rec["voiced"]
    pred = np.zeros((len(X), 12))
    states = np.zeros(len(X), int)
    for t in range(len(X)):
        m, c = evid(X[t], bool(v[t]))
        if use == "melody":
            c = None
        elif use == "chord":
            m = None
        s, _ = hmm.step(m, c)
        states[t] = s
        pred[t] = hmm.pcs(s)
    return pred, states, hmm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--evidence", default="baseline")
    ap.add_argument("--selector", choices=["key", "song", "score"], default="score")
    ap.add_argument("--use", choices=["melody", "chord", "both"], default="both")
    ap.add_argument("--bpm", type=float, default=80)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    recs = data.load_recordings(a.split, datasets=a.data)
    if a.max:
        recs = recs[:a.max]
    rows = []
    for r in recs:
        pred, states, hmm = run_recording(r, a.evidence, a.selector, a.bpm, a.use)
        sung = r["M"] & (r["MEL"] < 12)
        mel_pc = np.where(sung, r["MEL"], -1)
        ref = r["Y"]
        exact = float((pred[sung] == ref[sung]).all(1).mean()) if sung.any() else float("nan")
        nic = metrics.note_in_chord(pred, mel_pc)
        ref_nic = metrics.note_in_chord(ref, mel_pc)
        pred_ids = np.unique(pred[r["M"]], axis=0, return_inverse=True)[1].ravel()      # chord changes, not states
        cpm = metrics.changes_per_minute(pred_ids, fe.FPS) if r["M"].any() else float("nan")
        ref_ids = np.unique(ref[r["M"]], axis=0, return_inverse=True)[1].ravel()
        ref_cpm = metrics.changes_per_minute(ref_ids, fe.FPS)
        # chroma F1 of the played chord vs reference on sung frames
        p, y = pred[sung], ref[sung]
        tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        rows.append(dict(id=r["id"], f1=f1, exact=exact, nic=nic, ref_nic=ref_nic, cpm=cpm, ref_cpm=ref_cpm))
        print(f"{r['id']:16s} F1 {f1:.2f} exact {exact:.2f} note-in-chord {nic:.2f} (ref {ref_nic:.2f}) "
              f"changes/min {cpm:5.1f} (ref {ref_cpm:5.1f})", flush=True)
    if rows:
        m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0] if k != "id"}
        print(f"MEAN over {len(rows)}: F1 {m['f1']:.3f} exact {m['exact']:.3f} note-in-chord {m['nic']:.3f} "
              f"(ref {m['ref_nic']:.3f}) changes/min {m['cpm']:.1f} (ref {m['ref_cpm']:.1f})")


if __name__ == "__main__":
    main()
