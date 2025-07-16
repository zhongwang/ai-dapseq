#!/usr/bin/env python3

"""
Step 4: Create Tokenized Feature Vectors for Siamese Transformer Model

This script implements a tokenization method for feature engineering. 
It takes promoter DNA sequences and normalized per-base TF binding signals
as input and generates a feature vector for each gene's promoter region.
The output is a set of .npy files, one for each gene, containing a matrix
of shape (# windows, ).
"""

import argparse
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


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


def get_aggregated_windows_from_features(features, window_size, stride, agg_method):
    """
    Generates aggregated features for each window from a feature matrix.
    Returns a numpy array of shape (num_windows, num_features).
    """
    num_windows = (features.shape[1] - window_size) // stride + 1
    if num_windows <= 0:
        return np.array([])

    # Shape: (num_features, num_windows)
    aggregated_features = np.zeros((features.shape[0], num_windows))

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        window = features[:, start:end]

        if agg_method == 'mean':
            aggregated_features[:, i] = np.mean(window, axis=1)
        elif agg_method == 'max':
            aggregated_features[:, i] = np.max(window, axis=1)
        elif agg_method == 'sum':
            aggregated_features[:, i] = np.sum(window, axis=1)
    
    # Transpose to get (num_windows, num_features)
    return aggregated_features.T


def prepare_gene_features(gene_id, dna_sequence, tf_signals_dir, max_promoter_length, expected_num_tfs):
    """
    Loads data for a single gene and prepares the concatenated feature matrix.
    """
    # 1. Load TF binding signals for the gene
    tf_signals = load_tf_binding_signals(gene_id, tf_signals_dir, expected_num_tfs)
    if tf_signals is None:
        # A warning is already printed in load_tf_binding_signals
        return None

    # 2. One-hot encode DNA sequence
    one_hot_dna = one_hot_encode_sequence(dna_sequence, max_promoter_length).T # Shape: (4, max_promoter_length)

    # 3. Pad/Truncate TF signals to match max_promoter_length
    tf_signals_processed = np.zeros((expected_num_tfs, max_promoter_length), dtype=np.float32)
    length_to_copy = min(tf_signals.shape[1], max_promoter_length)
    tf_signals_processed[:, :length_to_copy] = tf_signals[:, :length_to_copy]

    # 4. Concatenate DNA and TF features
    features = np.concatenate((one_hot_dna, tf_signals_processed), axis=0) # Shape: (4 + num_tfs, max_promoter_length)
    return features


def visualize_clusters(features, labels, output_dir):
    """
    Visualize clusters using UMAP for dimensionality reduction.
    Saves the plot to a file.
    """
    print("Reducing dimensionality for visualization using UMAP...")
    
    # UMAP for dimensionality reduction
    reducer = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)
    
    # Create a DataFrame for plotting
    df_viz = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df_viz['Cluster'] = labels
    
    # Number of clusters (excluding noise)
    n_clusters = len(pd.unique(labels[labels != -1]))
    
    print(f"Plotting {n_clusters} clusters and noise...")
    
    # Plotting
    plt.figure(figsize=(12, 10))
    # Plot noise points first (in gray)
    noise = df_viz[df_viz['Cluster'] == -1]
    plt.scatter(noise['UMAP1'], noise['UMAP2'], c='lightgray', label='Noise', s=5, alpha=0.5)
    
    # Plot clustered points
    clustered = df_viz[df_viz['Cluster'] != -1]
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', data=clustered,
                    palette=sns.color_palette("hsv", n_colors=n_clusters),
                    s=10, alpha=0.8, legend='full')
    
    plt.title(f'UMAP projection of the {n_clusters} clusters')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plot_path = os.path.join(output_dir, "cluster_visualization.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster visualization saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tokenized feature vectors for promoter sequences using global clustering.")
    parser.add_argument("--dna_input_file", type=str, required=True,
                        help="Path to the TSV file containing promoter DNA sequences (columns: 'gene_id', 'promoter_dna_sequence').")
    parser.add_argument("--tf_signals_dir", type=str, required=True,
                        help="Path to the directory of .npy files containing normalized per-base TF binding signals.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the directory where final .npy feature files will be saved.")
    parser.add_argument("--max_promoter_length", type=int, default=2501,
                        help="Maximum length for promoter sequences. Sequences will be padded or truncated to this length.")
    parser.add_argument("--expected_num_tfs", type=int, required=True,
                        help="Expected number of TFs (features) in each TF signal file.")
    parser.add_argument("--window_size", type=int, default=50, help="Size of the sliding window.")
    parser.add_argument("--stride", type=int, default=10, help="Stride of the sliding window.")
    parser.add_argument("--agg_method", type=str, default='mean', choices=['mean', 'max', 'sum'],
                        help="Aggregation method for the sliding window.")
    parser.add_argument("--min_cluster_size", type=int, default=100, help="Minimum cluster size for HDBSCAN.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load all DNA sequences
    print(f"Loading DNA sequences from {args.dna_input_file}...")
    dna_sequences_map = load_dna_sequences(args.dna_input_file)
    if not dna_sequences_map:
        print(f"No DNA sequences loaded or error occurred. Please check input file: {args.dna_input_file}. Exiting.")
        return
    print(f"Loaded {len(dna_sequences_map)} DNA sequences.")

    # 2. Extract windowed features from all genes
    print("Extracting windowed features from all genes...")
    all_windows_features = []
    gene_window_info = {}
    processed_genes = 0

    for gene_id, dna_sequence in dna_sequences_map.items():
        if not isinstance(dna_sequence, str):
            print(f"Warning: DNA sequence for gene {gene_id} is not a string. Skipping.")
            continue
        
        # Prepare feature matrix for the current gene
        features = prepare_gene_features(gene_id, dna_sequence, args.tf_signals_dir, args.max_promoter_length, args.expected_num_tfs)
        if features is None:
            continue

        # Generate aggregated features for each window
        gene_windows = get_aggregated_windows_from_features(features, args.window_size, args.stride, args.agg_method)

        if gene_windows.shape[0] > 0:
            start_index = len(all_windows_features)
            all_windows_features.extend(gene_windows)
            end_index = len(all_windows_features)
            gene_window_info[gene_id] = (start_index, end_index)
            processed_genes += 1
        else:
            print(f"Warning: No windows generated for gene {gene_id}. It might be shorter than the window size.")

    if not all_windows_features:
        print("No features were extracted from any gene. Exiting.")
        return

    print(f"Extracted a total of {len(all_windows_features)} windows from {processed_genes} genes.")

    # 3. Perform global clustering on all windows
    print(f"Performing global clustering on {len(all_windows_features)} windows...")
    all_windows_features_np = np.array(all_windows_features, dtype=np.float32)

    hdb = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                          gen_min_span_tree=True) # gen_min_span_tree can be useful for visualization/understanding
    global_cluster_labels = hdb.fit_predict(all_windows_features_np)

    # Noise points are labeled as -1. We can either keep them or handle them.
    # For now, we will keep them as a special token.
    num_clusters = len(set(global_cluster_labels)) - (1 if -1 in global_cluster_labels else 0)
    print(f"HDBSCAN found {num_clusters} clusters and {np.sum(global_cluster_labels == -1)} noise points.")

    # 4. Evaluate clustering performance and visualize
    if num_clusters > 1:
        # We need at least 2 clusters to calculate these scores.
        clustered_features = all_windows_features_np[global_cluster_labels != -1]
        clustered_labels = global_cluster_labels[global_cluster_labels != -1]
        
        # Performance metrics
        ch_score = calinski_harabasz_score(clustered_features, clustered_labels)
        db_score = davies_bouldin_score(clustered_features, clustered_labels)

        # Save performance report
        report_path = os.path.join(args.output_dir, "clustering_performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("Clustering Performance Report\n")
            f.write("="*30 + "\n")
            f.write(f"Number of clusters found: {num_clusters}\n")
            f.write(f"Number of noise points: {np.sum(global_cluster_labels == -1)}\n")
            f.write(f"Calinski-Harabasz Score: {ch_score:.4f}\n")
            f.write(f"Davies-Bouldin Score: {db_score:.4f}\n")
        
        print(f"Clustering performance report saved to {report_path}")

        # Visualize clusters
        visualize_clusters(all_windows_features_np, global_cluster_labels, args.output_dir)
    else:
        print("Skipping performance evaluation and visualization as less than 2 clusters were found.")

    # 5. Save tokenized vectors for each gene
    print("Saving tokenized feature vectors for each gene...")
    success_count = 0
    for gene_id, (start_idx, end_idx) in gene_window_info.items():
        gene_cluster_labels = global_cluster_labels[start_idx:end_idx]
        try:
            output_path = os.path.join(args.output_dir, f"{gene_id}.npy")
            np.save(output_path, gene_cluster_labels.astype(np.float32))
            success_count += 1
        except Exception as e:
            print(f"Error saving features for {gene_id}: {e}")
            
    print(f"\nFeature generation complete.")
    print(f"Total genes with extracted features: {processed_genes}")
    print(f"Tokenized vectors saved for {success_count} genes in {args.output_dir}")

if __name__ == "__main__":
    main()
