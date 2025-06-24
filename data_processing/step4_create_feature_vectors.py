#!/usr/bin/env python3

"""
Step 4: Create Feature Vectors for Siamese Transformer Model

This script implements the feature engineering process described in Module 2.
It takes promoter DNA sequences and normalized per-base TF binding signals
as input and generates a feature vector for each base in the promoter regions.
The output is a set of .npy files, one for each gene, containing a matrix
of shape (MAX_PROMOTER_LENGTH, 4 + EXPECTED_NUM_TFS).
"""

import argparse
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

# Define constants for one-hot encoding
DNA_ONE_HOT_MAP = {
    'A': np.array([1, 0, 0, 0], dtype=np.float32),
    'C': np.array([0, 1, 0, 0], dtype=np.float32),
    'G': np.array([0, 0, 1, 0], dtype=np.float32),
    'T': np.array([0, 0, 0, 1], dtype=np.float32),
    'N': np.array([1, 1, 1, 1], dtype=np.float32), # As per implementation_plan_overview.md
}
PAD_DNA_VECTOR = np.array([0, 0, 0, 0], dtype=np.float32)


def load_dna_sequences(dna_file_path):
    """
    Loads promoter DNA sequences from a TSV file.
    The TSV file must contain 'gene_id' and 'promoter_dna_sequence' columns.
    Returns a dictionary mapping gene_id to its DNA sequence.
    """
    try:
        df = pd.read_csv(dna_file_path, sep='\t')
        if 'gene_id' not in df.columns or 'promoter_dna_sequence' not in df.columns:
            print(f"Error: DNA input file {dna_file_path} must contain 'gene_id' and 'promoter_dna_sequence' columns.")
            return {}
        return pd.Series(df.promoter_dna_sequence.values, index=df.gene_id).to_dict()
    except Exception as e:
        print(f"Error loading DNA sequences from {dna_file_path}: {e}")
        return {}


def load_tf_binding_signals(gene_id, tf_signals_dir, expected_num_tfs):
    """
    Loads the normalized per-base TF binding signal matrix for a given gene_id.
    Assumes .npy files named by gene_id in tf_signals_dir.
    Expected matrix shape: (expected_num_tfs, Promoter_Length).
    Validates the number of TFs against expected_num_tfs.
    Returns the loaded NumPy array or None if an error occurs.
    """
    file_path = os.path.join(tf_signals_dir, f"{gene_id}.npy")
    if not os.path.exists(file_path):
        print(f"Warning: TF signal file not found for {gene_id} at {file_path}")
        return None
    try:
        tf_matrix = np.load(file_path)
        if tf_matrix.shape[0] != expected_num_tfs:
            print(f"Error: TF signal file for {gene_id} has {tf_matrix.shape[0]} TFs, expected {expected_num_tfs}. Skipping gene.")
            return None
        return tf_matrix # Shape (expected_num_tfs, Promoter_Length)
    except Exception as e:
        print(f"Error loading TF signals for {gene_id} from {file_path}: {e}")
        return None


def one_hot_encode_sequence(dna_sequence, max_length):
    """
    One-hot encodes a DNA sequence and pads/truncates to max_length.
    Truncation keeps the beginning of the sequence.
    Padding adds PAD_DNA_VECTOR at the end.
    Returns a NumPy array of shape (max_length, 4).
    """
    encoded_sequence = []
    for base in dna_sequence.upper():
        encoded_sequence.append(DNA_ONE_HOT_MAP.get(base, DNA_ONE_HOT_MAP['N']))

    # Padding/Truncation
    if len(encoded_sequence) > max_length:
        # Truncate from the end (keep the first max_length elements)
        encoded_sequence = encoded_sequence[:max_length]
    elif len(encoded_sequence) < max_length:
        padding_needed = max_length - len(encoded_sequence)
        encoded_sequence.extend([PAD_DNA_VECTOR] * padding_needed)
    
    return np.array(encoded_sequence, dtype=np.float32) # Shape (max_length, 4)


def process_gene(args_tuple):
    """
    Worker function to process a single gene: load data, one-hot encode DNA,
    process TF signals, concatenate features, and save the result.
    """
    gene_id, dna_sequence, tf_signals_dir, output_dir, max_promoter_length, expected_num_tfs = args_tuple

    # 1. Load TF binding signals for the gene
    tf_signals = load_tf_binding_signals(gene_id, tf_signals_dir, expected_num_tfs)
    if tf_signals is None:
        # load_tf_binding_signals already prints a warning/error
        return f"Skipped {gene_id}: TF signals issue."

    # tf_signals shape is (expected_num_tfs, original_promoter_length)
    original_promoter_length = tf_signals.shape[1]

    # 2. One-hot encode DNA sequence
    one_hot_dna = one_hot_encode_sequence(dna_sequence, max_promoter_length) # Shape (max_promoter_length, 4)

    # 3. Prepare TF signals for concatenation
    # Transpose TF signals to (original_promoter_length, expected_num_tfs)
    tf_signals_transposed = tf_signals.T
    
    # Create a zero matrix for padded/truncated TF signals
    # Shape: (max_promoter_length, expected_num_tfs)
    tf_signals_processed = np.zeros((max_promoter_length, expected_num_tfs), dtype=np.float32)

    # Determine the length to copy from original TF signals to the processed matrix
    length_to_copy = min(original_promoter_length, max_promoter_length)
    
    # Copy the relevant part of TF signals
    tf_signals_processed[:length_to_copy, :] = tf_signals_transposed[:length_to_copy, :]
    # If original_promoter_length < max_promoter_length, the rest of tf_signals_processed remains zeros (padding)
    # If original_promoter_length > max_promoter_length, TF signals are truncated along with DNA sequence

    # 4. Concatenate features
    # final_features shape: (max_promoter_length, 4 + expected_num_tfs)
    final_features = np.concatenate((one_hot_dna, tf_signals_processed), axis=1)

    # 5. Save the final feature matrix
    try:
        output_path = os.path.join(output_dir, f"{gene_id}.npy")
        np.save(output_path, final_features)
        return f"Successfully processed {gene_id}. Output shape: {final_features.shape}. Saved to {output_path}"
    except Exception as e:
        return f"Error saving features for {gene_id}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Generate feature vectors for promoter sequences.")
    parser.add_argument("--dna_input_file", type=str, required=True,
                        help="Path to the TSV file containing promoter DNA sequences (columns: 'gene_id', 'promoter_dna_sequence').")
    parser.add_argument("--tf_signals_dir", type=str, required=True,
                        help="Path to the directory of .npy files containing normalized per-base TF binding signals.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the directory where final .npy feature files will be saved.")
    parser.add_argument("--max_promoter_length", type=int, required=True,
                        help="Maximum length for promoter sequences. Sequences will be padded or truncated to this length.")
    parser.add_argument("--expected_num_tfs", type=int, required=True,
                        help="Expected number of TFs (features) in each TF signal file. Used for validation and defining output shape.")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 1),
                        help="Number of worker processes for parallelization. Defaults to num CPUs - 1.")

    args = parser.parse_args()

    if args.num_workers <= 0:
        args.num_workers = 1

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load all DNA sequences
    print(f"Loading DNA sequences from {args.dna_input_file}...")
    dna_sequences_map = load_dna_sequences(args.dna_input_file) # gene_id -> sequence
    if not dna_sequences_map:
        print(f"No DNA sequences loaded or error occurred. Please check the input file: {args.dna_input_file}. Exiting.")
        return
    print(f"Loaded {len(dna_sequences_map)} DNA sequences.")

    # Prepare arguments for parallel processing
    processing_args = []
    for gene_id, dna_sequence in dna_sequences_map.items():
        if not isinstance(dna_sequence, str):
            print(f"Warning: DNA sequence for gene {gene_id} is not a string (type: {type(dna_sequence)}). Skipping.")
            continue
        processing_args.append((
            gene_id,
            dna_sequence,
            args.tf_signals_dir,
            args.output_dir,
            args.max_promoter_length,
            args.expected_num_tfs
        ))
    
    if not processing_args:
        print("No valid genes to process after filtering. Exiting.")
        return

    # 2. Process each gene in parallel
    print(f"Starting feature generation for {len(processing_args)} genes using {args.num_workers} worker(s)...")

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_gene, processing_args)

    success_count = 0
    failure_count = 0
    for result in results:
        print(result)
        if result.startswith("Successfully"):
            success_count +=1
        else:
            failure_count +=1
            
    print(f"\nFeature generation complete.")
    print(f"Total genes processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed/Skipped: {failure_count}")
    print(f"Outputs saved in {args.output_dir}")

if __name__ == "__main__":
    main()
