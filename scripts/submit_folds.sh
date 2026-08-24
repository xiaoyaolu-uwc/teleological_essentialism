#!/bin/bash
# Six-fold training: one junk gate + one stage 2 per fold = 12 jobs, submitted
# as a single burst so SLURM runs them in parallel rather than serially.
#
#   bash scripts/submit_folds.sh              # adopted config
#   bash scripts/submit_folds.sh 7            # a second seed, for the gate ensemble
#
# --holdout-fold reads the work titles from evaluation/folds.json by name.
# Do NOT pass a comma-delimited list of titles instead: three work titles
# contain commas ("The History of Creation, Vol. 1") and splitting on ',' will
# silently produce works that do not exist.
set -e
SEED=${1:-42}
SUFFIX=""; [ "$SEED" != "42" ] && SUFFIX="_s${SEED}"
module load slurm/slurm/25.05.2 >/dev/null 2>&1
cd ~/teleological_essentialism
mkdir -p logs
for N in 0 1 2 3 4 5; do
  sbatch --parsable scripts/train.slurm --run-name "gate_fold${N}${SUFFIX}" \
    --stage junk_gate --holdout-fold "fold${N}" --prompt-variant A_structured \
    --lora-r 32 --lora-alpha 64 --epochs 4 --seed "$SEED" \
    --text-column text --max-length 640 | xargs -I{} echo "{}  gate_fold${N}${SUFFIX}"
  sbatch --parsable scripts/train.slurm --run-name "s2_fold${N}${SUFFIX}" \
    --stage nonjunk_3way --holdout-fold "fold${N}" --prompt-variant S2_structured \
    --lora-r 32 --lora-alpha 64 --epochs 4 --seed "$SEED" \
    --text-column text --max-length 640 | xargs -I{} echo "{}  s2_fold${N}${SUFFIX}"
done
