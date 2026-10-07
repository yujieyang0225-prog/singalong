#!/usr/bin/env python3
"""Re-split a precompute_damp.py dataset by singer (account_id) instead of by performance.

DAMP-S-AG has 17,582 performances from 14,400 accounts; 1,944 accounts sang more than once (up to 33 times).
precompute_damp.py splits by performance, so one singer can sit in train and test. This assigns each usable
take the split of its account (stable hash, 80/10/10), keeps the original as manifest.performance_split.csv,
and checks that no account is in two splits.

    python3 split_by_singer.py data/damp3k --tsv data/DAMP-S-AG/amazing_grace.tsv
"""
import argparse
import collections
import csv
import hashlib
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--tsv", default="data/DAMP-S-AG/amazing_grace.tsv")
ap.add_argument("--val", type=float, default=0.1)
ap.add_argument("--test", type=float, default=0.1)
ap.add_argument("--salt", default="damp-split-v1")
a = ap.parse_args()

account = {r["performance_id"]: r["account_id"] for r in csv.DictReader(open(a.tsv), delimiter="\t")}
man = os.path.join(a.dataset, "manifest.csv")
orig = os.path.join(a.dataset, "manifest.performance_split.csv")
if not os.path.exists(orig):
    shutil.copy(man, orig)
with open(orig) as f:
    rd = csv.DictReader(f)
    fields, rows = rd.fieldnames, list(rd)
if "account_id" not in fields:
    fields = fields + ["account_id"]


def split_of(acc):
    h = int(hashlib.sha1(f"{a.salt}:{acc}".encode()).hexdigest()[:8], 16) / 2 ** 32
    return "test" if h < a.test else "val" if h < a.test + a.val else "train"


old = {r["id"]: r["split"] for r in rows}
missing = 0
for r in rows:
    acc = account.get(r["id"])
    if acc is None:
        missing += 1
        acc = "perf:" + r["id"]
    r["account_id"] = acc
    usable = r.get("status") == "ok" and r.get("usable") in ("1", "True")
    r["split"] = split_of(acc) if usable else ""

splits_of = collections.defaultdict(set)
for r in rows:
    if r["split"]:
        splits_of[r["account_id"]].add(r["split"])
assert all(len(s) == 1 for s in splits_of.values()), "an account landed in two splits"

# how many takes the old per-performance split put in val/test although the same singer is in train
old_train_acc = {r["account_id"] for r in rows if old[r["id"]] == "train"}
leaked = sum(1 for r in rows if old[r["id"]] in ("val", "test") and r["account_id"] in old_train_acc)

with open(man, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"{man}: split by account_id ({missing} ids not in the tsv, treated as their own singer)")
for s in ("train", "val", "test"):
    takes = [r for r in rows if r["split"] == s]
    print(f"  {s:5s}  {len(takes):5d} takes  {len({r['account_id'] for r in takes}):5d} singers")
multi = sum(1 for acc, n in collections.Counter(r["account_id"] for r in rows if r["split"]).items() if n > 1)
print(f"  singers with >1 take: {multi};  takes the old split had in val/test with the same singer in train: {leaked}")
