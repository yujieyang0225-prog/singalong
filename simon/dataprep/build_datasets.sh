#!/usr/bin/env bash
# Build the training datasets for jan/singalong on the Ubuntu machine (Docker + this repo, no data needed).
#
#   simon/dataprep/build_datasets.sh all                   # download prep split vocalset check
#   simon/dataprep/build_datasets.sh upload                # rsync repo + datasets to the cluster (venue network)
#   N=60 OUT=data/damp_smoke simon/dataprep/build_datasets.sh all     # smoke test first (~1 min)
#
# Stages (each one skips work that is already done; FORCE=1 redoes prep):
#   download  metadata + MIDI from Zenodo 3596940, then stream the 18.7 GB archive and keep N headphone takes
#   prep      jan/prep/precompute_damp.py in the SDK container: ffmpeg -> 16 kHz, 48 bands, labels from the MIDI
#   split     re-split by singer (account_id), so no singer is in train and val/test
#   vocalset  VocalSet comes from the Mac (already converted); prints the rsync command if it is missing
#   check     the trainer's loader reads both datasets; counts, RAM, label sanity
#   upload    rsync to $CLUSTER; then train there with SLURM + Singularity (train_slurm.sh), see the commands printed at the end
# Gain changes and 2 s crops are not baked in: the trainer does both, fresh every epoch (singalong.augment, data.crops).
#
# N=250 because a cluster job gets 12 GB RAM and train_torch.py needs ~30 MB per DAMP take at its peak
# (check prints the estimate per run). Download more (N=3000) only once the trainer streams its crops.
#
# Env: N=250  OUT=data/damp250  WORKERS=$(nproc)  CLUSTER=<user>@<cluster-host>  STARTER=<Innatera starter repo>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
N="${N:-250}"
OUT="${OUT:-data/damp$N}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
CLUSTER="${CLUSTER:-<user>@<cluster-host>}"
DAMP="$REPO/data/DAMP-S-AG"
ZEN="${ZEN:-https://zenodo.org/api/records/3596940/files}"     # override only for testing
RUN="$REPO/jan/singalong/run.sh"

# run.sh mounts $STARTER read-only; Docker would create a missing path as root, so fall back to an empty folder
STARTER="${STARTER:-$HOME/Documents/Hackathon repos/grp-gold-neurotum}"
if [ ! -d "$STARTER" ]; then STARTER="$REPO/data/.no_starter"; mkdir -p "$STARTER"; fi
export STARTER

say() { printf '\n== %s\n' "$*"; }

stage_download() {
  say "download -> $DAMP (N=$N headphone takes)"
  mkdir -p "$DAMP/vocals/amazing_grace"
  for f in amazing_grace.tsv amazing_grace.midi LICENSE.txt README.txt; do
    [ -s "$DAMP/$f" ] || curl -fsSL -o "$DAMP/$f" "$ZEN/$f/content"
  done
  echo "Licence: Smule Research Data License (see $DAMP/LICENSE.txt): academic, non-commercial, no redistribution."
  echo "Keep data/ out of git and do not share the audio or the npz files outside the team."
  # python exits after N takes; curl then fails on the closed pipe, which is expected (hence '|| true')
  { curl -sSL "$ZEN/amazing_grace.tar.gz/content" || true; } \
    | python3 "$HERE/stream_extract.py" --tsv "$DAMP/amazing_grace.tsv" --out "$DAMP/vocals/amazing_grace" --n "$N"
  local have; have=$(find "$DAMP/vocals/amazing_grace" -name '*.m4a' | wc -l)
  [ "$have" -ge "$N" ] || { echo "only $have of $N takes: download interrupted? Re-run 'download' to continue."; exit 1; }
}

stage_prep() {
  say "prep -> $OUT (precompute_damp.py, $WORKERS workers, in the SDK container)"
  if [ -s "$REPO/$OUT/manifest.csv" ] && [ -z "${FORCE:-}" ]; then echo "exists, skipped (FORCE=1 to redo)"; return; fi
  "$RUN" env PYTHONPATH=/workspace/jan/scripts python3 /workspace/jan/prep/precompute_damp.py \
    --n "$N" --workers "$WORKERS" --out "/workspace/$OUT"
}

stage_split() {
  say "split $OUT by singer"
  python3 "$HERE/split_by_singer.py" "$REPO/$OUT" --tsv "$DAMP/amazing_grace.tsv"
}

stage_vocalset() {
  say "vocalset -> data/vocalset"
  if [ -s "$REPO/data/vocalset/manifest.csv" ]; then
    echo "present: $(($(wc -l < "$REPO/data/vocalset/manifest.csv") - 1)) recordings"
  else
    echo "missing. On the Mac (VocalSet already converted by make_damp_format.py), run:"
    echo "  rsync -a ~/Downloads/vocalset_damp/ $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}'):$REPO/data/vocalset/"
    echo "then re-run: simon/dataprep/build_datasets.sh vocalset check"
  fi
}

stage_check() {
  say "check"
  "$RUN" python3 /workspace/simon/dataprep/check_datasets.py "$OUT" data/vocalset
}

stage_upload() {
  say "upload -> $CLUSTER:MNH26SVR"
  rsync -a --info=progress2 --exclude .git --exclude .venv --exclude __pycache__ --exclude '/data/' \
    "$REPO/" "$CLUSTER:MNH26SVR/"
  # data/runs/logs must exist before sbatch, or SLURM cannot open the job's output file and the job fails silently
  ssh "$CLUSTER" 'mkdir -p MNH26SVR/data/runs/logs && chmod 700 MNH26SVR/data
    ls -lh innatera_singularity.sif 2>/dev/null || echo "WARNING: no ~/innatera_singularity.sif; set SIF=<path> when you sbatch"'
  local sets=("$REPO/$OUT"); [ -d "$REPO/data/vocalset" ] && sets+=("$REPO/data/vocalset")
  rsync -a --info=progress2 "${sets[@]}" "$CLUSTER:MNH26SVR/data/"
  local d; d=$(basename "$OUT")
  cat <<EOF

On the cluster (ssh $CLUSTER), from ~/MNH26SVR. No venv: python, torch and talamo come from the Innatera image.
  sbatch --array=1 --export=ALL,EPOCHS=1,DAMP=$d --time=00:20:00 simon/dataprep/train_slurm.sh   # smoke test (B, 1 epoch)
  squeue --me ; cat data/runs/logs/singalong_*.out                                        # GPU check at the top
  sbatch --export=ALL,DAMP=$d simon/dataprep/train_slurm.sh      # A, B, C in parallel (only --array=1 without VocalSet)
Weights land in ~/MNH26SVR/data/runs/<name>/weights_int8.npz; copy them back for real_voice_check.py:
  rsync -a --include='*/' --include='weights_int8.npz' --include='metrics.json' --include='log.csv' --exclude='*' \\
    $CLUSTER:MNH26SVR/data/runs/ $REPO/data/runs/
EOF
}

[ $# -eq 0 ] && set -- all
for s in "$@"; do
  case "$s" in
    all) stage_download; stage_prep; stage_split; stage_vocalset; stage_check ;;
    download|prep|split|vocalset|check|upload) "stage_$s" ;;
    *) echo "unknown stage '$s' (download prep split vocalset check upload all)"; exit 2 ;;
  esac
done
