#!/usr/bin/env bash
# SLURM job array: hyper-parameter search for A (VocalSet only), pitch-only, inside the Innatera Singularity image.
# The network stays the chip one (48 -> 80 rec -> 24, enc thr 120, thr 127, int8); only training settings vary.
#
# Why: the best A so far is A_vocalset_P (lr 0.1, wd 0, --pitch-only, 40 epochs; eval_pitch vs_songs 0.530,
# vs_exer 0.599, damp250 0.364). Its val pitch was still creeping up at epoch 40 (0.631 -> 0.638), so the search
# trains longer and varies one setting at a time around it. Task 0 is the longer version of A_vocalset_P itself.
#
# Submit from the repo root on the cluster:
#   cd ~/MNH26SVR && sbatch simon/dataprep/search_vocalset.sh               # all 8 in parallel, ~30 min each (task 1: ~40 min)
#   sbatch --array=0,2 simon/dataprep/search_vocalset.sh                    # a subset
#   sbatch --array=0 --export=ALL,EPOCHS_OVERRIDE=1 --time=00:15:00 simon/dataprep/search_vocalset.sh   # smoke test
# Then rank them all with the one yardstick (vs_songs, vs_exer tie-break within 0.02, damp250 sanity):
#   sbatch --dependency=afterany:<jobid> --time=00:30:00 --gres=gpu:1 --mem=12G --wrap "singularity exec --nv \
#       --bind $HOME:$HOME --pwd $HOME/MNH26SVR/jan/singalong innatera_singularity.sif python3 ../../simon/eval_pitch.py \
#       ../../data/runs/A_vocalset_P ../../data/runs/A_vs_*"
# Watch: squeue --me   tail -f data/runs/logs/singalong-search_<job>_<task>.out
#SBATCH --job-name=singalong-search
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=data/runs/logs/%x_%A_%a.out
set -euo pipefail
REPO="${REPO:-$HOME/MNH26SVR}"
SIF="${SIF:-$REPO/innatera_singularity.sif}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
BASE=(--data ../../data/vocalset --out 24 --hidden 80 --pitch-only --wd 0 --compact --chunk 64)

#      name                  settings (everything else = train_torch.py defaults)
case "$TASK" in
  0) NAME=A_vs_lr0.1_e60;       SET=(--lr 0.1  --epochs 60 --qat-epochs 6) ;;           # A_vocalset_P, 50 % longer
  1) NAME=A_vs_lr0.1_e80;       SET=(--lr 0.1  --epochs 80 --qat-epochs 8) ;;
  2) NAME=A_vs_lr0.2_e60;       SET=(--lr 0.2  --epochs 60 --qat-epochs 6) ;;
  3) NAME=A_vs_lr0.05_e60;      SET=(--lr 0.05 --epochs 60 --qat-epochs 6) ;;
  4) NAME=A_vs_lr0.1_shift3;    SET=(--lr 0.1  --epochs 60 --qat-epochs 6 --shift 3) ;;  # less transposition
  5) NAME=A_vs_lr0.1_enc0.3;    SET=(--lr 0.1  --epochs 60 --qat-epochs 6 --lr-enc 0.3) ;; # calmer encoder thresholds
  6) NAME=A_vs_lr0.1_crop250;   SET=(--lr 0.1  --epochs 60 --qat-epochs 6 --crop 250) ;;  # 4 s context per crop
  7) NAME=A_vs_lr0.1_seed1;     SET=(--lr 0.1  --epochs 60 --qat-epochs 6 --seed 1) ;;   # seed spread of task 0
  *) echo "unknown array index $TASK (0-7)"; exit 2 ;;
esac
if [ -n "${EPOCHS_OVERRIDE:-}" ]; then SET+=(--epochs "$EPOCHS_OVERRIDE" --qat-epochs 1); NAME="${NAME}_smoke"; fi

export SINGULARITYENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
sif() { srun singularity exec --nv --bind "$HOME:$HOME" --pwd "$REPO/jan/singalong" "$SIF" "$@"; }

echo "$NAME on $(hostname), job ${SLURM_ARRAY_JOB_ID:-?}_${TASK}, image $SIF"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
sif python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "torch sees no GPU in the image")'
sif python3 train_torch.py "${BASE[@]}" "${SET[@]}" --device cuda --name "$NAME"
