#!/usr/bin/env python3
"""Stream DAMP-S-AG's amazing_grace.tar.gz (18.7 GB) from stdin and keep only the first N headphone takes.

    curl -sSL https://zenodo.org/api/records/3596940/files/amazing_grace.tar.gz/content \
        | python3 stream_extract.py --tsv data/DAMP-S-AG/amazing_grace.tsv --out data/DAMP-S-AG/vocals/amazing_grace --n 3000

Only the part of the archive up to the N-th kept take is downloaded: the script exits, curl stops on the broken pipe.
Headphone takes (tsv column headphones == 1, 11,014 of 17,582) are clean vocals without backing-track bleed.
Files land as <out>/<performance_id>.m4a (the path jan/prep/precompute_damp.py globs). Re-running skips files
that are already there; a file is written to a temp name first, so an interrupted run never leaves a half file.
"""
import argparse
import csv
import os
import sys
import tarfile
import time

ap = argparse.ArgumentParser()
ap.add_argument("--tsv", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--all", action="store_true", help="keep non-headphone takes too")
a = ap.parse_args()

keep = {r["performance_id"] for r in csv.DictReader(open(a.tsv), delimiter="\t")
        if a.all or r["headphones"].strip() in ("1", "1.0")}
os.makedirs(a.out, exist_ok=True)
have = {f[:-4] for f in os.listdir(a.out) if f.endswith(".m4a")} & keep
print(f"tsv: {len(keep)} takes eligible; already here: {len(have)}; target: {a.n}", flush=True)
if len(have) >= a.n:
    sys.exit(0)

t0, seen, kept, unknown = time.time(), 0, 0, []
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as tf:
    for m in tf:
        if not (m.isfile() and m.name.endswith(".m4a")):
            continue
        seen += 1
        pid = os.path.basename(m.name)[:-4]
        if pid not in keep:
            if len(unknown) < 5:
                unknown.append(m.name)
            if seen == 500 and not have and not kept:
                sys.exit(f"500 files streamed, none matches a tsv performance_id; archive names look like {unknown}")
            continue
        if pid in have:
            continue
        dst = os.path.join(a.out, pid + ".m4a")
        with tf.extractfile(m) as src, open(dst + ".part", "wb") as f:
            f.write(src.read())
        os.replace(dst + ".part", dst)
        kept += 1
        if kept % 100 == 0:
            el = time.time() - t0
            print(f"  kept {len(have) + kept}/{a.n}  (streamed {seen} files, {el:.0f} s, ~{m.offset_data / el / 1e6:.0f} MB/s)",
                  flush=True)
        if len(have) + kept >= a.n:
            break
print(f"done: {len(have) + kept} takes in {a.out} ({kept} new, {seen} streamed, {time.time() - t0:.0f} s)", flush=True)
