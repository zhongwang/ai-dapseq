#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 48:0:0
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 8
#SBATCH -q es_normal
#SBATCH --gres=gpu:H100:8
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/visualizations_long_run/output_train.log
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/visualizations_long_run/error_train.log
eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate
torchrun --nproc_per_node=8 ./training/train_siamese_transformer.py --pairs_file ./output/new_full_data/final_coexpressed_regression.txt --feature_dir ./output/new_full_data/feature_vectors --gene_info_file ./output/new_full_data/promoter_sequences.txt --save_path ./output/new_full_data/visualizations_long_run/best_siamese_model.pth --log_file ./output/new_full_data/visualizations_long_run/training_log.csv --epochs 30 --dropout 0.5 --regression_dropout 0.5 --num_encoder_layers 2 --d_model 128 --dim_feedforward 512 --batch_size 128 --early_stopping_patience 5