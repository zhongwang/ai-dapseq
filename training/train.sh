#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 20:0:0
#SBATCH --ntasks 8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/output_train.log
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/error_train.log
eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate
python ./training/train.py