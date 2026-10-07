#!/usr/bin/env python3
"""Check datasets in the damp100 format through the trainer's own loader (singalong.data).

Per dataset and split: recordings, hours, 2 s training crops, RAM the loader needs, and a label sanity number
(share of labelled, voiced frames whose strongest pitch class in the bands is within 1 semitone of the label;
harmonics make 'exact' low by nature). VocalSet reference: ~0.84 within 1 semitone on train/val, DAMP ~0.61.

Run in the SDK container (needs torch):  jan/singalong/run.sh python3 /workspace/simon/dataprep/check_datasets.py data/damp250 data/vocalset
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "jan" / "singalong"))
from singalong import data, frontend as fe  # noqa: E402

CROP = 125                       # train_torch.py default (2 s); stride 0 -> crop/2
LIMIT_GB = 12                    # RAM per job on the cluster (sbatch --mem in train_slurm.sh)
# train_torch.py peak RSS, measured on damp_smoke (2 CPU epochs): ~0.5 GB python+torch, + 7x the loader RAM of
# the train split (float32 recordings, a full augmented copy per epoch, all overlapping crops stacked twice),
# + 1x for val; ~1 GB more on the GPU for the CUDA context.
BASE_GB, TRAIN_X, CUDA_GB = 0.5, 7.0, 1.0
# label sanity: random labels give 3/12 = 0.25; VocalSet (studio, annotated) ~0.84, DAMP (amateur karaoke) ~0.61
SANE = 0.45


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
    rng, warn, ram_by = np.random.default_rng(0), [], {}
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
                ram_by.setdefault(p, {})[split] = ram
            within, exact = sanity(recs, rng)
            print(f"  {split:6s} {len(recs):6d} {hours:6.2f} {sum(map(n_crops, recs)):7d} {ram:7.2f} {within:6.2f} {exact:6.2f}")
            if within < SANE:
                warn.append(f"{p}/{split}: label sanity {within:.2f} < {SANE}, labels may be misaligned")
        try:
            some = data.load_recordings("val", data=folder)[:3] or data.load_recordings("train", data=folder)[:3]
            Xc, Yc = data.crops(some)
            print(f"  trainer crops OK: X {tuple(Xc.shape)}, Y {tuple(Yc.shape)}")
        except Exception as e:                                      # noqa: BLE001
            warn.append(f"{p}: data.crops failed: {e!r}")
    runs = {p: [p] for p in ram_by} | ({" + ".join(ram_by): list(ram_by)} if len(ram_by) > 1 else {})
    print(f"\nPeak RAM of train_torch.py on the GPU (estimate; limit {LIMIT_GB} GB per job):")
    for name, ps in runs.items():
        gb = BASE_GB + CUDA_GB + sum(TRAIN_X * ram_by[q].get("train", 0) + ram_by[q].get("val", 0) for q in ps)
        print(f"  {name:40s} {gb:5.1f} GB")
        if gb > LIMIT_GB - 1:
            warn.append(f"{name}: training needs ~{gb:.0f} GB, more than the {LIMIT_GB} GB per job; use a smaller N")
    for w in warn:
        print("WARNING:", w)
    return 1 if any(k in w for w in warn for k in ("empty", "sanity", "crops failed")) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["data/damp250", "data/vocalset"]))
