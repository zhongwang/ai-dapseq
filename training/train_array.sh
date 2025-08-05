#!/bin/bash
#SBATCH --job-name=model_wandb_sweep
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 48:0:0
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 8
#SBATCH -q es_normal
#SBATCH --gres=gpu:H100:8
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/wandb_sweep/outputs/wandb_sweep.out
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/wandb_sweep/errors/wandb_sweep.err

eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate

export WANDB_DISABLE_GPU_SETUP=true
wandb agent sallyliao-northwestern-university/transformer_coexpression/f3fenv7l