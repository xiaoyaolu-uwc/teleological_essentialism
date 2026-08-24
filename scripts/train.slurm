#!/bin/bash
# Generic SLURM wrapper: every argument is forwarded straight to the LoRA
# trainer, so this one file covers every run (gate or stage 2, any fold, any
# hyperparameters) rather than needing a script per config.
#
#   sbatch scripts/train.slurm --run-name s2_fold4 --stage nonjunk_3way \
#       --holdout-fold fold4 --prompt-variant S2_structured \
#       --lora-r 32 --lora-alpha 64 --epochs 4 --seed 42 \
#       --text-column text --max-length 640
#
# 3h limit: at max_length 640 the ~11k-row gate job overruns a 20-minute limit
# partway through epoch 2. Do not lower this without re-checking that.
# NOTE the preempt partition can preempt and requeue a job mid-run; always
# validate a run's metrics.json config fields before consuming its checkpoint.
#SBATCH --job-name=teleo
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --account=marlowe-m000151
#SBATCH --output=logs/%x_%j.out

module load python3
source ~/envs/lora_gate/bin/activate
cd ~/teleological_essentialism

python3 models/lora/train.py "$@"
