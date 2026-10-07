"""Convert paired VocalSet recordings to the `damp100` format of MNH26SVR/jan/singalong, so that
`train_torch.py --data <out>` trains the C1 SNN on real, cleanly labelled singing.

Per recording <id>.npz:  X float16 [T,48] log2 bands (same front end as the board, via singalong.frontend),
                         voiced uint8 [T], melody int16 [T] = annotated MIDI note, -1 = rest/transition/margin,
                         chord_pcs uint8 [T,12] = all zeros (VocalSet has no chords: 'unknown', ignored by the baseline)
manifest.csv: id, status, usable, split, singer, music_type, technique, vowel, duration_s
Splits: exercises by singer (official VocalSet test singers; female9/male11 = val); songs -> 'song'.

Usage (on the Mac, where the VocalSet audio is): python3 simon/dataprep/make_damp_format.py --out ~/Downloads/vocalset_damp
"""
import argparse, csv, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

HOME = os.path.expanduser("~")
ap = argparse.ArgumentParser()
ap.add_argument("--singalong", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "jan", "singalong"))
ap.add_argument("--paired", default=f"{HOME}/Downloads/VocalSet_paired")
ap.add_argument("--ann", default=f"{HOME}/Downloads/Annotated VocalSet/extended 1/all files.csv")
ap.add_argument("--out", default=f"{HOME}/Downloads/vocalset_damp")
ap.add_argument("--margin-ms", type=float, default=30)
ap.add_argument("--workers", type=int, default=os.cpu_count())
a = ap.parse_args()
sys.path.insert(0, a.singalong)
from singalong import frontend as fe   # noqa: E402

TEST = {"female2", "female8", "male3", "male5", "male10"}
VAL = {"female9", "male11"}
ann = pd.read_csv(a.ann, skipinitialspace=True, index_col=False).dropna(subset=["File Name"])
SEGS = {k: g[g.Type == "Sound"][["Start time", "End time", "Ground Truth MIDI code"]].dropna().to_numpy()
        for k, g in ann.groupby("File Name")}

def split_of(r):
    if r["music_type"].startswith("MusicalExcerp"):
        return "song"
    return "test" if r["singer"] in TEST else "val" if r["singer"] in VAL else "train"

def work(r):
    sr, x = wavfile.read(r["wav"])
    x = x.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(1)
    y = resample_poly(x, 160, 441).astype(np.float32) if sr == 44100 else resample_poly(x, fe.SR, sr).astype(np.float32)
    X = fe.log_bands(y)
    v = fe.voiced(y)
    T = len(X)
    tc = (np.arange(T) * fe.HOP + fe.WIN / 2) / fe.SR
    mel = np.full(T, -1, np.int16)
    m = a.margin_ms / 1000
    for s, e, p in SEGS.get(r["name"], []):
        mel[(tc >= s + m) & (tc < e - m)] = int(p)
    np.savez_compressed(f"{a.out}/{r['name']}.npz", X=X.astype(np.float16), voiced=v.astype(np.uint8),
                        melody=mel, chord_pcs=np.zeros((T, 12), np.uint8))
    return dict(id=r["name"], status="ok", usable=int((mel >= 0).sum() >= 40), split=split_of(r), singer=r["singer"],
                music_type=r["music_type"], technique=r["technique"], vowel=r["vowel"], duration_s=round(len(y) / fe.SR, 1))

if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    rows = pd.read_csv(f"{a.paired}/manifest.csv").to_dict("records")
    with Pool(a.workers) as p:
        res = p.map(work, rows, chunksize=8)
    cols = ["id", "status", "usable", "split", "singer", "music_type", "technique", "vowel", "duration_s"]
    with open(f"{a.out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(res)
    print(pd.DataFrame(res).groupby("split").agg(files=("id", "size"), usable=("usable", "sum"), hours=("duration_s", lambda s: s.sum() / 3600)))
