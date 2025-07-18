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
    The TSV file must contain 'gene_id', 'promoter_dna_sequence', and 'chromosome' columns.
    Returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(dna_file_path, sep='\t')
        required_cols = ['gene_id', 'promoter_dna_sequence', 'chromosome']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: DNA input file {dna_file_path} must contain {required_cols} columns.")
            return None
        return df
    except Exception as e:
        print(f"Error loading DNA sequences from {dna_file_path}: {e}")
        return None


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


def visualize_clusters(features, labels, output_dir, n_jobs=-1):
    """
    Visualize clusters using UMAP for dimensionality reduction.
    Saves the plot to a file.
    """
    print("Reducing dimensionality for visualization using UMAP...")
    
    # UMAP for dimensionality reduction
    reducer = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=2, random_state=42, n_jobs=n_jobs)
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
                        help="Path to the TSV file containing promoter DNA sequences (columns: 'gene_id', 'promoter_dna_sequence', 'chromosome').")
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
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing. -1 means using all available CPUs.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    print(f"Using {n_jobs} parallel jobs.")

    # 1. Load all DNA sequences
    print(f"Loading DNA sequences from {args.dna_input_file}...")
    dna_sequences_df = load_dna_sequences(args.dna_input_file)
    if dna_sequences_df is None:
        print(f"No DNA sequences loaded or error occurred. Please check input file: {args.dna_input_file}. Exiting.")
        return
    print(f"Loaded {len(dna_sequences_df)} DNA sequences.")

    # 2. Extract windowed features from all genes in parallel
    print("Extracting windowed features from all genes in parallel...")
    all_windows_features = []
    gene_window_info = {}
    
    tasks = [(row['gene_id'], row['promoter_dna_sequence'], args.tf_signals_dir, args.max_promoter_length,
              args.expected_num_tfs, args.window_size, args.stride, args.agg_method)
             for _, row in dna_sequences_df.iterrows()]

    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_gene_for_features_worker, tasks)

    processed_genes = 0
    for gene_id, gene_windows in results:
        if gene_windows is not None:
            start_index = len(all_windows_features)
            all_windows_features.extend(gene_windows)
            end_index = len(all_windows_features)
            gene_window_info[gene_id] = (start_index, end_index)
            processed_genes += 1

    if not all_windows_features:
        print("No features were extracted from any gene. Exiting.")
        return

    print(f"Extracted a total of {len(all_windows_features)} windows from {processed_genes} genes.")

    # 3. Separate training data and perform clustering
    print("Separating training data for clustering (genes not in Chr2 or Chr4)...")
    training_gene_ids = dna_sequences_df[~dna_sequences_df['chromosome'].isin(['Chr2', 'Chr4'])]['gene_id']
    
    training_windows_indices = []
    for gene_id in training_gene_ids:
        if gene_id in gene_window_info:
            start_index, end_index = gene_window_info[gene_id]
            training_windows_indices.extend(range(start_index, end_index))
            
    all_windows_features_np = np.array(all_windows_features, dtype=np.float32)
    training_windows_features = all_windows_features_np[training_windows_indices]

    if len(training_windows_features) == 0:
        print("No training features found. Check your data and chromosome filtering. Exiting.")
        return

    print(f"Performing clustering on {len(training_windows_features)} windows from training genes...")
    hdb = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                          gen_min_span_tree=True,
                          core_dist_n_jobs=n_jobs)
    hdb.fit(training_windows_features)

    # With the clusters defined, create embeddings for all genomes
    # Visualize clusters on all data
    visualize_clusters(all_windows_features_np, global_cluster_labels, args.output_dir, n_jobs=n_jobs)

    # 5. Save tokenized vectors for each gene in parallel
    print("Saving tokenized feature vectors for each gene in parallel...")
    
    save_tasks = [(gene_id, start_idx, end_idx, args.output_dir)
                  for gene_id, (start_idx, end_idx) in gene_window_info.items()]

    with Pool(processes=n_jobs, initializer=init_worker_saving, initargs=(global_cluster_labels,)) as pool:
        save_results = pool.map(save_gene_vector_worker, save_tasks)
    
    success_count = sum(save_results)
            
    print(f"\nFeature generation complete.")
    print(f"Predicting clusters for all {len(all_windows_features_np)} windows...")
    global_cluster_labels, _ = hdbscan.approximate_predict(hdb, all_windows_features_np)

    num_clusters = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
    print(f"HDBSCAN trained on training data found {num_clusters} clusters.")
    print(f"Total noise points across all data: {np.sum(global_cluster_labels == -1)}.")

    # 4. Evaluate clustering performance and visualize
    if num_clusters > 1:
        # We need at least 2 clusters to calculate these scores on the training data.
        training_clustered_features = training_windows_features[hdb.labels_ != -1]
        training_clustered_labels = hdb.labels_[hdb.labels_ != -1]

        if len(training_clustered_labels) > 0:
            # Performance metrics on training data
            ch_score = calinski_harabasz_score(training_clustered_features, training_clustered_labels)
            db_score = davies_bouldin_score(training_clustered_features, training_clustered_labels)

            # Save performance report
            report_path = os.path.join(args.output_dir, "clustering_performance_report.txt")
            with open(report_path, 'w') as f:
                f.write("Clustering Performance Report (based on training data)\n")
                f.write("="*50 + "\n")
                f.write(f"Number of clusters found: {num_clusters}\n")
                f.write(f"Number of noise points in training data: {np.sum(hdb.labels_ == -1)}\n")
                f.write(f"Calinski-Harabasz Score: {ch_score:.4f}\n")
                f.write(f"Davies-Bouldin Score: {db_score:.4f}\n")
            
            print(f"Clustering performance report saved to {report_path}")

        # Visualize clusters on all data
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

# Global variable for worker processes, to hold the large cluster labels array
_global_cluster_labels = None

def init_worker_saving(global_cluster_labels_arr):
    """Initializer for the saving worker pool."""
    global _global_cluster_labels
    _global_cluster_labels = global_cluster_labels_arr

def save_gene_vector_worker(params):
    """
    Worker function to save the tokenized vector for a single gene.
    """
    gene_id, start_idx, end_idx, output_dir = params
    gene_cluster_labels = _global_cluster_labels[start_idx:end_idx]
    try:
        output_path = os.path.join(output_dir, f"{gene_id}.npy")
        np.save(output_path, gene_cluster_labels.astype(np.float32))
        return True
    except Exception as e:
        print(f"Error saving features for {gene_id}: {e}")
        return False

def process_gene_for_features_worker(params):
    """
    Worker function to process a single gene.
    Loads data, prepares features, and generates windowed aggregates.
    Designed for use with multiprocessing.Pool.
    """
    gene_id, dna_sequence, tf_signals_dir, max_promoter_length, expected_num_tfs, window_size, stride, agg_method = params

    if not isinstance(dna_sequence, str):
        print(f"Warning: DNA sequence for gene {gene_id} is not a string. Skipping.")
        return gene_id, None
    
    # Prepare feature matrix for the current gene
    features = prepare_gene_features(gene_id, dna_sequence, tf_signals_dir, max_promoter_length, expected_num_tfs)
    if features is None:
        return gene_id, None

    # Generate aggregated features for each window
    gene_windows = get_aggregated_windows_from_features(features, window_size, stride, agg_method)

    if gene_windows.shape[0] > 0:
        return gene_id, gene_windows
    else:
        print(f"Warning: No windows generated for gene {gene_id}. It might be shorter than the window size.")
        return gene_id, None
if __name__ == "__main__":
    main()
