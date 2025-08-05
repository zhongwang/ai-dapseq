#!/bin/bash
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A pc_jgiga
#SBATCH -J train_undersample
#SBATCH -t 16:0:0
#SBATCH --ntasks 8
#SBATCH --cpus-per-task 8
#SBATCH -q es_normal
#SBATCH --gres=gpu:H100:8
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_undersample/test_run/output_train.log
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_undersample/test_run/error_train.log
eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate
torchrun --nproc_per_node=8 ./training/train_siamese_transformer.py --pairs_file ./output/new_full_undersample/final_coexpressed.txt --feature_dir ./output/new_full_data/feature_vectors --gene_info_file ./output/new_full_data/promoter_sequences.txt --save_path /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/tuning_on_small/512_4_0.3_64_1e-4/best_siamese_model.pth --log_file /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/tuning_on_small/512_4_0.3_64_1e-4/training_log.csv --epochs 10 --d_model 256 --dim_feedforward 512 --num_encoder_layers 2 --dropout 0.3 --regression_dropout 0.3 --batch_size 64