#!/usr/bin/env python3
"""Check datasets in the damp100 format through the trainer's own loader (singalong.data).

Per dataset and split: recordings, hours, 2 s training crops, RAM the loader needs, and a label sanity number
(share of labelled, voiced frames whose strongest pitch class in the bands is within 1 semitone of the label;
harmonics make 'exact' low by nature). VocalSet reference: ~0.84 within 1 semitone on train/val.

Run in the SDK container (needs torch):  jan/singalong/run.sh python3 /workspace/simon/dataprep/check_datasets.py data/damp3k data/vocalset
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "jan" / "singalong"))
from singalong import data, frontend as fe  # noqa: E402

CROP = 125                       # train_torch.py default (2 s); stride 0 -> crop/2


def n_crops(rec, crop=CROP, stride=CROP // 2):
    idx = np.flatnonzero(rec["M"])
    if len(idx) < crop:
        return 0
    return len(range(max(idx[0] - crop // 4, 0), idx[-1] - crop, stride))


def sanity(recs, rng, k=200):
    within = exact = n = 0
    for i in rng.permutation(len(recs))[:k]:
        r = recs[i]
        lab = (r["MEL"] < data.NO_NOTE) & r["voiced"] & r["M"]
        if not lab.any():
            continue
        pc = fe.bands_to_chroma(r["X_log"][lab]).argmax(1)
        d = np.abs(pc - r["MEL"][lab]) % 12
        d = np.minimum(d, 12 - d)
        within, exact, n = within + (d <= 1).sum(), exact + (d == 0).sum(), n + lab.sum()
    return within / max(n, 1), exact / max(n, 1)


def main(paths):
    rng, warn, total_ram = np.random.default_rng(0), [], 0.0
    for p in paths:
        folder = Path(p) if Path(p).is_absolute() else REPO / p
        if not (folder / "manifest.csv").exists():
            warn.append(f"{p}: no manifest.csv, skipped")
            continue
        print(f"\n{folder}")
        print(f"  {'split':6s} {'recs':>6s} {'hours':>6s} {'crops':>7s} {'RAM GB':>7s} {'±1 st':>6s} {'exact':>6s}")
        for split in ("train", "val", "test", "song"):
            recs = data.load_recordings(split, data=folder)
            if not recs:
                if split in ("train", "val"):
                    warn.append(f"{p}: split '{split}' is empty")
                continue
            hours = sum(len(r["X"]) for r in recs) / fe.FPS / 3600
            ram = sum(r["X"].nbytes + r["X_log"].nbytes + r["Y"].nbytes for r in recs) / 1e9
            if split in ("train", "val"):
                total_ram += ram
            within, exact = sanity(recs, rng)
            print(f"  {split:6s} {len(recs):6d} {hours:6.2f} {sum(map(n_crops, recs)):7d} {ram:7.2f} {within:6.2f} {exact:6.2f}")
            if within < 0.6:
                warn.append(f"{p}/{split}: label sanity {within:.2f} < 0.6, labels may be misaligned")
        try:
            some = data.load_recordings("val", data=folder)[:3] or data.load_recordings("train", data=folder)[:3]
            Xc, Yc = data.crops(some)
            print(f"  trainer crops OK: X {tuple(Xc.shape)}, Y {tuple(Yc.shape)}")
        except Exception as e:                                      # noqa: BLE001
            warn.append(f"{p}: data.crops failed: {e!r}")
    print(f"\nRAM for train+val of all datasets together (loader only, estimate): {total_ram:.1f} GB")
    if total_ram > 15:
        warn.append(f"train+val need ~{total_ram:.0f} GB in RAM; the cluster account has 24 GB")
    for w in warn:
        print("WARNING:", w)
    return 1 if any(k in w for w in warn for k in ("empty", "sanity", "crops failed")) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["data/damp3k", "data/vocalset"]))
