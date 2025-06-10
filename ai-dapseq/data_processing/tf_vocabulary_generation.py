import os
import numpy as np
import pandas as pd

# Placeholder paths - replace with actual paths
NORMALIZED_INPUT_DIR = "./tf_binding_profiles_normalized" # Directory containing normalized .npy files
OUTPUT_DIR = "./tf_vocabulary_features" # Output directory for TF vocabulary features (.npy files)

PROMOTER_LENGTH = 2501 # Should match previous scripts
WINDOW_SIZE = 50 # As specified in the research plan
STRIDE = 25 # Example stride, can be tuned (e.g., 10, 25, 50)

def load_normalized_data(input_dir):
    """
    Loads normalized TF binding data (.npy files) for all genes.
    Returns a dictionary mapping gene_id to its normalized signal matrix (num_tfs x promoter_length).
    """
    normalized_data = {}
    print(f"Loading normalized TF binding data from {input_dir}")
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return normalized_data # Return empty dict

    npy_files = [f for f in os.listdir(input_dir) if f.endswith("_tf_binding_normalized.npy")]
    print(f"Found {len(npy_files)} .npy files.")

    for npy_file in npy_files:
        gene_id = npy_file.replace("_tf_binding_normalized.npy", "")
        file_path = os.path.join(input_dir, npy_file)
        try:
            signal_matrix = np.load(file_path)
            # Expected shape: (num_tfs, promoter_length)
            normalized_data[gene_id] = signal_matrix
            # print(f"Loaded normalized data for {gene_id}") # Optional: print for each gene
        except Exception as e:
            print(f"Error loading normalized data for gene {gene_id} from {file_path}: {e}. Skipping.")

    print(f"Loaded normalized data for {len(normalized_data)} genes.")
    return normalized_data

def generate_tf_vocabulary_features(normalized_data, window_size, stride, promoter_length):
    """
    Generates TF vocabulary features by applying sliding windows and aggregating signals.
    Returns a dictionary mapping gene_id to its sequence of window vectors.
    """
    tf_vocabulary_features = {}
    print(f"Generating TF vocabulary features with window size {window_size} and stride {stride}...")

    if not normalized_data:
        print("No data to process.")
        return tf_vocabulary_features

    # Infer num_tfs from the first loaded gene's data
    first_gene_id = list(normalized_data.keys())[0]
    num_tfs = normalized_data[first_gene_id].shape[0]
    print(f"Inferred {num_tfs} TFs.")

    for gene_id, signal_matrix in normalized_data.items():
        # Ensure the signal matrix has the expected shape
        if signal_matrix.shape[1] != promoter_length:
            print(f"Warning: Signal matrix for gene {gene_id} has unexpected length {signal_matrix.shape[1]}, expected {promoter_length}. Skipping.")
            continue
        if signal_matrix.shape[0] != num_tfs:
             print(f"Warning: Signal matrix for gene {gene_id} has unexpected number of TFs {signal_matrix.shape[0]}, expected {num_tfs}. Skipping.")
             continue

        window_vectors = []
        # Slide the window across the promoter
        for start_pos in range(0, promoter_length - window_size + 1, stride):
            end_pos = start_pos + window_size

            # Extract the signal sub-matrix for the current window
            window_signal_matrix = signal_matrix[:, start_pos:end_pos]

            # Aggregate signals within the window for each TF
            # Using mean aggregation as recommended in the plan (Section III.B.5)
            # Handle potential NaNs within the window - mean will be NaN if all values are NaN
            aggregated_vector = np.nanmean(window_signal_matrix, axis=1) # Aggregate across the window length (axis=1)

            window_vectors.append(aggregated_vector)

        # Stack the window vectors into a sequence matrix (num_windows x num_tfs)
        if window_vectors:
            tf_vocabulary_features[gene_id] = np.vstack(window_vectors)
        else:
            print(f"Warning: No windows generated for gene {gene_id}. Skipping.")


    print(f"Generated features for {len(tf_vocabulary_features)} genes.")
    return tf_vocabulary_features

def save_tf_vocabulary_features(tf_vocabulary_features, output_dir):
    """
    Saves TF vocabulary features for each gene to separate files (.npy).
    """
    print(f"Saving TF vocabulary features to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for gene_id, features_matrix in tf_vocabulary_features.items():
        output_path = os.path.join(output_dir, f"{gene_id}_tf_vocabulary.npy")
        try:
            np.save(output_path, features_matrix)
            # print(f"Saved features for {gene_id}") # Optional: print for each gene
        except Exception as e:
            print(f"Error saving features for gene {gene_id} to {output_path}: {e}. Skipping save for this gene.")

    print("Saving complete.")

if __name__ == "__main__":
    # Step 1: Load normalized TF binding data
    normalized_tf_binding_data = load_normalized_data(NORMALIZED_INPUT_DIR)

    if normalized_tf_binding_data:
        # Step 2: Generate TF vocabulary features
        tf_vocabulary_features = generate_tf_vocabulary_features(
            normalized_tf_binding_data,
            WINDOW_SIZE,
            STRIDE,
            PROMOTER_LENGTH
        )

        # Step 3: Save generated features
        if tf_vocabulary_features:
            save_tf_vocabulary_features(tf_vocabulary_features, OUTPUT_DIR)
        else:
            print("No TF vocabulary features generated. Output directory not created/populated.")
    else:
        print("No normalized TF binding data loaded. Skipping feature generation.")