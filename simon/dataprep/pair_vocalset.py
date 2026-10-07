"""Pair VocalSet wavs with their note annotations (Annotated VocalSet, 'extended 1').

Writes a symlink tree (no 1.4 GB copy) plus manifest.csv and unmatched.csv.
Usage: python pair_vocalset.py [WAV_ROOT] [ANN_ROOT] [OUT_DIR]
"""
import csv, glob, os, sys
import pandas as pd

HOME = os.path.expanduser("~/Downloads")
WAV_ROOT = sys.argv[1] if len(sys.argv) > 1 else f"{HOME}/VocalSet11/FULL"
ANN_ROOT = sys.argv[2] if len(sys.argv) > 2 else f"{HOME}/Annotated VocalSet/extended 1"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{HOME}/VocalSet_paired"

ALIAS = {"_sow_": "_slow_"}  # m8 slow_forte scales wavs carry a typo

def key(path):
    n = os.path.basename(path)[:-4]
    for a, b in ALIAS.items():
        n = n.replace(a, b)
    return n

wavs = {}
for p in glob.glob(f"{WAV_ROOT}/**/*.wav", recursive=True):
    wavs.setdefault(key(p), []).append(p)

ann = pd.read_csv(f"{ANN_ROOT}/all files.csv", skipinitialspace=True, index_col=False)
ann = ann.dropna(subset=["File Name"])
meta = ann.drop_duplicates("File Name").set_index("File Name")
hdr_csv = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{ANN_ROOT}/with file header/*/*.csv")}

os.makedirs(OUT, exist_ok=True)
rows, missing = [], []
for name, m in meta.iterrows():
    cands = wavs.get(name, [])
    if len(cands) != 1:
        missing.append((name, len(cands)))
        continue
    w = cands[0]
    singer = w.split("/FULL/")[1].split("/")[0]
    mt, tech = m["Music Type"], m["Technique"]
    d = f"{OUT}/{singer}/{mt}/{tech}"
    os.makedirs(d, exist_ok=True)
    for src, ext in ((w, "wav"), (hdr_csv.get(name), "csv")):
        if src and not os.path.lexists(f"{d}/{name}.{ext}"):
            os.symlink(src, f"{d}/{name}.{ext}")
    rows.append(dict(name=name, singer=singer, gender=m["Gender"], music_type=mt,
                     technique=tech, vowel=m["Vowel"], wav=w, csv=hdr_csv.get(name, ""),
                     n_notes=int(((ann["File Name"] == name) & (ann["Type"] == "Sound")).sum())))
pd.DataFrame(rows).to_csv(f"{OUT}/manifest.csv", index=False)
used = {r["wav"] for r in rows}
with open(f"{OUT}/unmatched.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["kind", "path_or_name", "detail"])
    for n, c in missing: wr.writerow(["csv_without_unique_wav", n, c])
    for ps in wavs.values():
        for p in ps:
            if p not in used: wr.writerow(["wav_without_csv", p, ""])
print(f"paired {len(rows)} / {len(meta)} annotated files; csv-side misses {len(missing)}; "
      f"unannotated wavs {sum(len(v) for v in wavs.values()) - len(used)}")
