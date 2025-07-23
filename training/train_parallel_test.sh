#!/bin/bash
#SBATCH --job-name=test_wandb
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 0:20:0
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:2
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/wandb_sweep/outputs/wandb_sweep.out
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/wandb_sweep/errors/wandb_sweep.err

eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate

export WANDB_DISABLE_GPU_SETUP=true
wandb agent sallyliao-northwestern-university/transformer_coexpression/9u27jeig