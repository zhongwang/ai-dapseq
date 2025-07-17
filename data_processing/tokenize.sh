#!/bin/bash
#SBATCH -N 1
#SBATCH -p lr7
#SBATCH -A pc_jgiga
#SBATCH -t 20:0:0
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH -q lr_normal
#SBATCH -o /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/output_tokenize.log
#SBATCH -e /global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/error_tokenize.log
eval "$(conda shell.bash hook)"
cd /global/scratch/users/sallyliao2027/aidapseq
source .venv/bin/activate
python ./data_processing/step4_create_tokenized_feature_vectors.py --dna_input_file ./output/new_full_data/promoter_sequences.txt --tf_signals_dir ./output/full_data/tf_signals --output_dir ./output/new_full_data/feature_vectors_hdbscan_cluster --expected_num_tfs 244
