import os
import numpy as np
import pandas as pd # Needed if normalization factors are stored in a file

# Placeholder paths - replace with actual paths
TF_BINDING_INPUT_DIR = "./tf_binding_profiles" # Directory containing .npy files from tf_binding_extraction.py
NORMALIZED_OUTPUT_DIR = "./tf_binding_profiles_normalized" # Output directory for normalized .npy files
# Optional: Path to a file containing normalization factors or control signal data
# NORMALIZATION_FACTORS_FILE = "../path/to/normalization_factors.tsv"
# INPUT_CONTROL_BIGWIG_DIR = "../path/to/input_control_bigwigs" # If background subtraction is needed

PROMOTER_LENGTH = 2501 # Based on -2000 to +500, should match extraction script

def load_tf_binding_data(input_dir):
    """
    Loads raw TF binding data (.npy files) for all genes from the input directory.
    Returns a dictionary mapping gene_id to its signal matrix (num_tfs x promoter_length).
    """
    tf_binding_data = {}
    print(f"Loading raw TF binding data from {input_dir}")
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return tf_binding_data # Return empty dict

    npy_files = [f for f in os.listdir(input_dir) if f.endswith("_tf_binding.npy")]
    print(f"Found {len(npy_files)} .npy files.")

    for npy_file in npy_files:
        gene_id = npy_file.replace("_tf_binding.npy", "")
        file_path = os.path.join(input_dir, npy_file)
        try:
            # Load the numpy array
            signal_matrix = np.load(file_path)
            # Expected shape: (num_tfs, promoter_length)
            # We need to know num_tfs from the previous step or infer it.
            # For now, let's just store the matrix.
            tf_binding_data[gene_id] = signal_matrix
            # print(f"Loaded data for {gene_id}") # Optional: print for each gene
        except Exception as e:
            print(f"Error loading data for gene {gene_id} from {file_path}: {e}. Skipping.")

    print(f"Loaded data for {len(tf_binding_data)} genes.")
    return tf_binding_data

def apply_normalization(tf_binding_data):
    """
    Applies normalization and bias correction strategies to the TF binding data.
    Implements strategies from Section III.B.3 of the research plan.
    Returns a dictionary of normalized signal matrices.
    """
    normalized_data = {}
    print("Applying normalization and bias correction...")

    if not tf_binding_data:
        print("No data to normalize.")
        return normalized_data

    # Assuming all matrices have the same shape (num_tfs, promoter_length)
    # This needs to be consistent from the extraction step.
    # Infer num_tfs from the first loaded gene's data
    first_gene_id = list(tf_binding_data.keys())[0]
    num_tfs = tf_binding_data[first_gene_id].shape[0]
    print(f"Inferred {num_tfs} TFs.")

    # --- Normalization Strategy (based on Section III.B.3) ---
    # A multi-step approach is advisable:
    # 1. Library Size / Sequencing Depth Normalization (requires normalization factors per TF)
    # 2. Background/Input Control Subtraction (requires control bigwigs or pre-calculated control signals)
    # 3. Log Transformation (e.g., log2(x+c))
    # 4. Per-TF Scaling/Standardization (Z-score or Min-Max)

    # *** IMPLEMENTATION DETAILS NEEDED ***
    # This function needs the actual normalization factors per TF (for step 1),
    # and potentially control signal data (for step 2).
    # These would typically be calculated in a separate script or loaded from files.

    # Placeholder: Implement a simplified normalization (Log + Z-score per TF)
    # This assumes raw signals are loaded and we apply log and then Z-score per TF across all genes.

    # Collect all signals for each TF across all genes to calculate global stats for Z-score
    # This can be memory intensive for large datasets. Consider processing in batches if needed.
    all_tf_signals = [[] for _ in range(num_tfs)]
    for gene_id, signal_matrix in tf_binding_data.items():
        # Handle potential NaNs from extraction errors
        signal_matrix_masked = np.ma.masked_invalid(signal_matrix)
        for i in range(num_tfs):
             # Append only valid data points
             all_tf_signals[i].extend(signal_matrix_masked[i].compressed()) # .compressed() removes masked values

    # Calculate mean and std dev for each TF across all valid data points
    tf_means = [np.mean(signals) if signals else 0 for signals in all_tf_signals]
    tf_stds = [np.std(signals) if signals else 1 for signals in all_tf_signals] # Use 1 to avoid division by zero

    print("Applying Log2(x+1) and Z-score normalization per TF...")
    pseudocount = 1 # For log transformation

    for gene_id, signal_matrix in tf_binding_data.items():
        normalized_matrix = np.copy(signal_matrix) # Work on a copy

        # Handle NaNs: temporarily replace NaNs for log/z-score, then put them back
        nan_mask = np.isnan(normalized_matrix)
        normalized_matrix[nan_mask] = 0 # Replace NaN with 0 for transformation (or a small value)

        # Step 3: Log Transformation
        normalized_matrix = np.log2(normalized_matrix + pseudocount)

        # Step 4: Per-TF Z-score Standardization
        for i in range(num_tfs):
            mean = tf_means[i]
            std = tf_stds[i]
            if std > 0: # Avoid division by zero
                normalized_matrix[i] = (normalized_matrix[i] - mean) / std
            else:
                # If std is 0, all values for this TF were the same. Set to 0 after subtracting mean.
                normalized_matrix[i] = normalized_matrix[i] - mean # Should be all zeros if mean was correct

        # Put NaNs back
        normalized_matrix[nan_mask] = np.nan

        normalized_data[gene_id] = normalized_matrix

    print("Normalization complete.")
    return normalized_data

def save_normalized_data(normalized_data, output_dir):
    """
    Saves normalized TF binding data for each gene to separate files (.npy).
    """
    print(f"Saving normalized TF binding data to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for gene_id, signal_matrix in normalized_data.items():
        output_path = os.path.join(output_dir, f"{gene_id}_tf_binding_normalized.npy")
        try:
            np.save(output_path, signal_matrix)
            # print(f"Saved normalized data for {gene_id}") # Optional: print for each gene
        except Exception as e:
            print(f"Error saving normalized data for gene {gene_id} to {output_path}: {e}. Skipping save for this gene.")

    print("Saving complete.")

if __name__ == "__main__":
    # Step 1: Load raw TF binding data
    raw_tf_binding_data = load_tf_binding_data(TF_BINDING_INPUT_DIR)

    if raw_tf_binding_data:
        # Step 2: Apply normalization
        normalized_tf_binding_data = apply_normalization(raw_tf_binding_data)

        # Step 3: Save normalized data
        if normalized_tf_binding_data:
            save_normalized_data(normalized_tf_binding_data, NORMALIZED_OUTPUT_DIR)
        else:
            print("No normalized data generated. Output directory not created/populated.")
    else:
        print("No raw TF binding data loaded. Skipping normalization.")