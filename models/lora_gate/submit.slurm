#!/bin/bash
#SBATCH --job-name=lora_gate
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000151
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Generic submission wrapper for models/lora_gate/train.py -- every CLI arg
# passed to `sbatch submit.slurm ...` is forwarded straight through via "$@",
# so this one file covers every sweep run rather than needing a script per
# config. Submit multiple configs back to back (e.g. the LR sweep) and SLURM
# schedules each onto its own GPU on this partition's 8-GPU nodes, running
# them genuinely in parallel instead of contending for one GPU.
#
# Usage:
#   sbatch models/lora_gate/submit.slurm --run-name junk_gate_lora_lr1e4 --lr 1e-4

module load python3
source ~/envs/lora_gate/bin/activate
cd ~/teleological_essentialism

python3 models/lora_gate/train.py "$@"
