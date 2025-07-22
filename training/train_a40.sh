#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 22:0:0
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:4
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/visualizations_reduce_overfit/output_train.log
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/visualizations_reduce_overfit/error_train.log
eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate
torchrun --nproc_per_node=4 ./training/train_siamese_transformer.py --pairs_file ./output/new_full_data/final_coexpressed_regression.txt --feature_dir ./output/new_full_data/feature_vectors --gene_info_file ./output/new_full_data/promoter_sequences.txt --save_path ./output/new_full_data/visualizations_reduce_overfit/best_siamese_model.pth --log_file ./output/new_full_data/visualizations_reduce_overfit/training_log.csv --epochs 5 --dropout 0.5 --regression_dropout 0.5 --num_encoder_layers 2 --d_model 128 --dim_feedforward 512 --batch_size 128