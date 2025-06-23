import argparse
import numpy as np
import os
import logging
from multiprocessing import Pool, cpu_count
from glob import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_signal_matrix(args):
    """
    Worker function to normalize a single signal matrix.
    
    Args:
        args (tuple): A tuple containing (raw_signal_path, output_dir, pseudocount).
    
    Returns:
        str: The path to the saved normalized .npy file or None on error.
    """
    raw_signal_path, output_dir, pseudocount = args
    gene_id = os.path.basename(raw_signal_path).replace('.npy', '')
    
    try:
        # Load the raw signal matrix for one gene
        raw_matrix = np.load(raw_signal_path) # Shape: (num_tfs, 2501)

        # NOTE: This is a placeholder for a real normalization strategy.
        # A real implementation would require global statistics (e.g., library size,
        # mean/std per TF across all genes) which should be pre-calculated.
        # For this script, we will perform a simplified per-gene, per-TF normalization.

        # 1. Log Transformation
        log_matrix = np.log2(raw_matrix + pseudocount)

        # 2. Z-score Standardization (per TF)
        # Calculate mean and std for each TF's signal across the promoter
        mean_per_tf = np.mean(log_matrix, axis=1, keepdims=True)
        std_per_tf = np.std(log_matrix, axis=1, keepdims=True)
        
        # Avoid division by zero for TFs with no signal variation
        std_per_tf[std_per_tf == 0] = 1.0
        
        normalized_matrix = (log_matrix - mean_per_tf) / std_per_tf
        
        output_path = os.path.join(output_dir, f"{gene_id}.npy")
        np.save(output_path, normalized_matrix.astype(np.float32))
        return output_path

    except Exception as e:
        logging.error(f"Error normalizing {raw_signal_path}: {e}")
        return None

def main():
    """
    Main function to run the signal normalization script.
    """
    parser = argparse.ArgumentParser(description="Normalize TF binding signal matrices.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing the raw signal .npy files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the normalized .npy files.")
    parser.add_argument("--pseudocount", type=float, default=1.0, help="Pseudocount to add before log transformation.")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="Number of worker processes to use.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Searching for raw signal files in {args.input_dir}")
    raw_signal_files = glob(os.path.join(args.input_dir, '*.npy'))
    logging.info(f"Found {len(raw_signal_files)} files to normalize.")

    # Prepare arguments for multiprocessing
    tasks = [(f, args.output_dir, args.pseudocount) for f in raw_signal_files]

    logging.info(f"Starting normalization with {args.processes} processes.")
    with Pool(args.processes) as pool:
        results = pool.map(normalize_signal_matrix, tasks)
    
    successful_files = [res for res in results if res is not None]
    logging.info(f"Successfully normalized {len(successful_files)} out of {len(tasks)} files.")
    logging.info("Signal normalization complete.")

if __name__ == "__main__":
    main()