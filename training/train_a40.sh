#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -t 4:0:0
#SBATCH -J 512_4_0.3_64_1e-4
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 8
#SBATCH -q es_normal
#SBATCH --gres=gpu:A40:2
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/tuning_on_small/512_4_0.3_64_1e-4/output_train.log
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/tuning_on_small/512_4_0.3_64_1e-4/error_train.log
eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate
torchrun --nproc_per_node=2 ./training/train_siamese_transformer.py --pairs_file ./output/new_full_data/final_coexpressed_regression_10bins.txt --feature_dir ./output/new_full_data/feature_vectors --gene_info_file ./output/new_full_data/promoter_sequences.txt --save_path /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/tuning_on_small/512_4_0.3_64_1e-4/best_siamese_model.pth --log_file /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/tuning_on_small/512_4_0.3_64_1e-4/training_log.csv --epochs 5 --d_model 512 --dim_feedforward 1024 --num_encoder_layers 4 --dropout 0.3 --regression_dropout 0.3 --batch_size 64 --learning_rate 0.0001