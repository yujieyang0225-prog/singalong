#!/usr/bin/env bash
# SLURM job array for the three singalong trainings on the cluster (one GPU each, run in parallel),
# inside Innatera's Singularity image (python 3.12, torch, talamo), like the cluster's run_slurm.py template.
# Submit from the repo root on the cluster (build_datasets.sh upload copies it there and prints this):
#
#   cd ~/MNH26SVR && sbatch simon/dataprep/train_slurm.sh                                   # A, B and C
#   sbatch --array=1 simon/dataprep/train_slurm.sh                                          # only B
#   sbatch --array=0 --export=ALL,EPOCHS=1 --time=00:20:00 simon/dataprep/train_slurm.sh    # smoke test
#   sbatch -p <partition> ...                                                                # if there is no default partition
#
#   0 = A_vocalset   1 = B_damp   2 = C_both
# Watch: squeue --me   tail -f data/runs/logs/<job>_<task>.out   Cancel: scancel <job>
# Weights land in data/runs/<name>/weights_int8.npz.
#
# Env (sbatch passes your shell env through): DAMP=damp250  EPOCHS=40  HIDDEN=80
#                                             SIF=~/innatera_singularity.sif  REPO=~/MNH26SVR
#SBATCH --job-name=singalong
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=data/runs/logs/%x_%A_%a.out
set -euo pipefail
REPO="${REPO:-$HOME/MNH26SVR}"
SIF="${SIF:-$HOME/innatera_singularity.sif}"
DAMP="${DAMP:-damp250}"
EPOCHS="${EPOCHS:-40}"
HIDDEN="${HIDDEN:-80}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK" in
  0) NAME=A_vocalset; DATA=(--data ../../data/vocalset) ;;
  1) NAME=B_damp;     DATA=(--data "../../data/$DAMP") ;;
  2) NAME=C_both;     DATA=(--data "../../data/$DAMP" --data ../../data/vocalset) ;;
  *) echo "unknown array index $TASK (0-2)"; exit 2 ;;
esac

# crops and augmentation run in numpy on the CPU; SINGULARITYENV_ passes the variable into the container
export SINGULARITYENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
sif() { srun singularity exec --nv --bind "$HOME:$HOME" --pwd "$REPO/jan/singalong" "$SIF" "$@"; }

# fail in seconds, not after the queue wait plus a CPU-speed epoch, if torch in the image cannot see the GPU
echo "$NAME on $(hostname), job ${SLURM_ARRAY_JOB_ID:-?}_${TASK}, image $SIF"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
sif python3 -c '
import sys, torch
print(f"python {sys.version.split()[0]} · torch {torch.__version__} (built for CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False: the image torch needs a newer driver than nvidia-smi shows above")
print(f"cuda ok: {torch.cuda.get_device_name(0)}")'

sif python3 train_torch.py "${DATA[@]}" --out 24 --hidden "$HIDDEN" --epochs "$EPOCHS" --device cuda --name "$NAME"
