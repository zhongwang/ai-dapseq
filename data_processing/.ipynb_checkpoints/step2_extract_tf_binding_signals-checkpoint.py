import argparse
import pandas as pd
import numpy as np
import pyBigWig
import os
import glob
from multiprocessing import Pool, cpu_count

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract per-base TF binding signals for promoter regions and optionally normalize them.")
    parser.add_argument("--promoter_file", required=True, help="Path to the TSV file from Step 1 (gene_id, chromosome, promoter_start, promoter_end, strand).")
    parser.add_argument("--bigwig_dir", required=True, help="Directory containing TF binding bigWig files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output .npy files (raw or normalized signals).")
    parser.add_argument("--num_cores", type=int, default=max(1, cpu_count() - 1), help="Number of CPU cores to use for parallel processing.")

    # Normalization options (from Step 3.3)
    parser.add_argument("--log_transform", action='store_true', help="Apply log2(x+1) transformation to signals.")
    parser.add_argument("--zscore_normalize", action='store_true', help="Apply Z-score normalization per TF across all promoters.")
    parser.add_argument("--pseudocount", type=float, default=1.0, help="Pseudocount for log transformation (default: 1.0).")

    return parser.parse_args()

def process_gene(args_tuple):
    """
    Worker function to process a single gene: extract signals and optionally normalize.
    This function is designed to be used with multiprocessing.Pool.
    """
    gene_id, chrom, start, end, strand, bigwig_files, output_dir, log_transform, pseudocount = args_tuple

    promoter_length = end - start
    if promoter_length <= 0:
        print(f"Skipping gene {gene_id}: promoter length is non-positive ({promoter_length}).")
        return gene_id, None, "Invalid promoter length"

    # Matrix to store signals for this gene: (Number of TFs x Promoter Length)
    # Initialize with zeros, assuming if a TF has no signal or file is problematic, signal is 0.
    gene_signals_matrix = np.zeros((len(bigwig_files), promoter_length))

    for i, bw_file_path in enumerate(bigwig_files):
        tf_name = os.path.basename(bw_file_path).replace('.bigWig', '').replace('.bw', '')
        try:
            bw = pyBigWig.open(bw_file_path)
            if bw is None:
                print(f"Warning: Could not open bigWig file {bw_file_path} for gene {gene_id}. Skipping this TF.")
                continue # gene_signals_matrix[i, :] remains zeros

            # Ensure chromosome exists in the bigWig file
            if chrom not in bw.chroms():
                # print(f"Warning: Chromosome {chrom} not found in {bw_file_path} for gene {gene_id}. Assuming zero signal for this TF.")
                pass # gene_signals_matrix[i, :] remains zeros
            else:
                # pyBigWig.values returns NaN for regions with no data.
                # Fetch all values at once for efficiency.
                signals = bw.values(chrom, start, end, numpy=True)
                signals = np.nan_to_num(signals, nan=0.0) # Replace NaNs with 0

                if len(signals) == promoter_length:
                    gene_signals_matrix[i, :] = signals
                else:
                    # This case should ideally not happen if coordinates are correct and promoter_length matches signal length
                    print(f"Warning: Signal length mismatch for gene {gene_id}, TF {tf_name}. Expected {promoter_length}, got {len(signals)}. Padding/truncating.")
                    # Pad with 0 or truncate if necessary
                    if len(signals) > promoter_length:
                        gene_signals_matrix[i, :] = signals[:promoter_length]
                    else:
                        gene_signals_matrix[i, :len(signals)] = signals
            bw.close()
        except Exception as e:
            print(f"Error processing TF {tf_name} for gene {gene_id} (file: {bw_file_path}): {e}. Assuming zero signal.")
            # gene_signals_matrix[i, :] remains zeros

    # Optional Log Transformation (applied before Z-score if both are chosen)
    if log_transform:
        gene_signals_matrix = np.log2(gene_signals_matrix + pseudocount)

    # Output file path for this gene
    # Note: Z-score normalization needs to be done *after* collecting all raw/log-transformed signals.
    # So, this function will return the raw or log-transformed matrix. Z-scoring will be handled in main.
    output_file_path = os.path.join(output_dir, f"{gene_id}.npy")
    # np.save(output_file_path, gene_signals_matrix) # Saving will be done after potential Z-scoring

    return gene_id, gene_signals_matrix, None # Return None for error status if successful

def extract_tf_binding_signals(promoter_file_path, bigwig_dir_path, output_dir_path, num_cores, log_transform_flag, zscore_normalize_flag, pseudocount_val):
    """
    Main function to orchestrate TF binding signal extraction and normalization.
    """
    try:
        promoter_df = pd.read_csv(promoter_file_path, sep='\t')
        required_cols = ['gene_id', 'chromosome', 'promoter_start', 'promoter_end', 'strand']
        if not all(col in promoter_df.columns for col in required_cols):
            print(f"Error: Promoter file {promoter_file_path} must contain columns: {required_cols}")
            return
        # Ensure correct types
        promoter_df['promoter_start'] = promoter_df['promoter_start'].astype(int)
        promoter_df['promoter_end'] = promoter_df['promoter_end'].astype(int)

        print(f"Successfully parsed promoter file: {promoter_file_path}. Found {len(promoter_df)} genes.")
    except FileNotFoundError:
        print(f"Error: Promoter file not found at {promoter_file_path}")
        return
    except Exception as e:
        print(f"Error parsing promoter file {promoter_file_path}: {e}")
        return

    bigwig_files = sorted(glob.glob(os.path.join(bigwig_dir_path, "*.bigWig"))) + \
                     sorted(glob.glob(os.path.join(bigwig_dir_path, "*.bw")))
    if not bigwig_files:
        print(f"Error: No bigWig files (.bigWig or .bw) found in directory {bigwig_dir_path}")
        return
    print(f"Found {len(bigwig_files)} bigWig files in {bigwig_dir_path}.")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"Created output directory: {output_dir_path}")

    # Prepare arguments for multiprocessing
    tasks = []
    for _, row in promoter_df.iterrows():
        tasks.append((
            row['gene_id'], str(row['chromosome']), int(row['promoter_start']), int(row['promoter_end']),
            row['strand'], bigwig_files, output_dir_path, # Pass output_dir_path but actual saving handled later
            log_transform_flag, pseudocount_val
        ))

    print(f"Starting signal extraction for {len(tasks)} genes using {num_cores} cores...")

    all_gene_signals = {} # gene_id -> signal_matrix (raw or log-transformed)
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_gene, tasks)

    successful_extractions = 0
    failed_extractions = 0
    for gene_id, signal_matrix, error_msg in results:
        if error_msg:
            print(f"Extraction failed for {gene_id}: {error_msg}")
            failed_extractions += 1
        elif signal_matrix is not None:
            all_gene_signals[gene_id] = signal_matrix
            successful_extractions += 1
        else:
            # This case implies an issue not caught by error_msg but matrix is None
            print(f"Extraction yielded no data for {gene_id}, but no specific error message.")
            failed_extractions += 1

    print(f"Signal extraction phase complete. Successfully processed: {successful_extractions}, Failed: {failed_extractions}")

    if not all_gene_signals:
        print("No signals were extracted. Aborting further processing.")
        return

    # Step 3.3: Signal Normalization (continued - Z-score)
    if zscore_normalize_flag:
        print("Applying Z-score normalization across promoters for each TF...")
        num_tfs = len(bigwig_files)
        # For Z-scoring across all bases of all promoters for each TF:
        # 1. All signals for a given TF (across all promoters and all positions) are collected.
        # 2. The global mean and std deviation are computed for that TF's collected signals.
        # 3. Z-score transformation is then applied to each promoter's signal for that TF using these global statistics.
        # This requires collecting all signals for each TF, which can be memory intensive but provides accurate global normalization.
        # Let's collect all data for each TF first.
        tf_all_signals_list = [[] for _ in range(num_tfs)] # List of lists, one per TF

        for gene_id in promoter_df['gene_id']:
            if gene_id in all_gene_signals:
                signal_matrix = all_gene_signals[gene_id]
                if signal_matrix.shape[0] == num_tfs: # Ensure matrix has expected num TFs
                    for tf_idx in range(num_tfs):
                        tf_all_signals_list[tf_idx].extend(signal_matrix[tf_idx, :].tolist())
                else:
                    print(f"Warning: Signal matrix for gene {gene_id} has {signal_matrix.shape[0]} TFs, expected {num_tfs}. Skipping for Z-score calculation of this gene.")

        tf_means = np.zeros(num_tfs)
        tf_stds = np.ones(num_tfs) # Use 1 for std if data is constant or count is too low

        for tf_idx in range(num_tfs):
            if tf_all_signals_list[tf_idx]:
                flat_signals = np.array(tf_all_signals_list[tf_idx])
                tf_means[tf_idx] = np.mean(flat_signals)
                std_val = np.std(flat_signals)
                tf_stds[tf_idx] = std_val if std_val > 1e-6 else 1.0 # Avoid division by zero or very small std
            else:
                print(f"Warning: No signals collected for TF index {tf_idx} for Z-score normalization.")

        # Apply Z-score normalization to the stored matrices
        for gene_id in all_gene_signals:
            signal_matrix = all_gene_signals[gene_id]
            if signal_matrix.shape[0] == num_tfs:
                normalized_matrix = (signal_matrix - tf_means[:, np.newaxis]) / tf_stds[:, np.newaxis]
                all_gene_signals[gene_id] = normalized_matrix
            # else: already warned above
        print("Z-score normalization applied.")

    # Save the (potentially normalized) matrices
    print(f"Saving {len(all_gene_signals)} signal matrices to {output_dir_path}...")
    saved_count = 0
    for gene_id, signal_matrix in all_gene_signals.items():
        output_file = os.path.join(output_dir_path, f"{gene_id}.npy")
        try:
            np.save(output_file, signal_matrix)
            saved_count +=1
        except Exception as e:
            print(f"Error saving {output_file}: {e}")

    print(f"Successfully saved {saved_count} .npy files.")


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.isdir(args.bigwig_dir):
        print(f"Error: bigWig directory not found or is not a directory: {args.bigwig_dir}")
    elif not os.path.exists(args.promoter_file):
        print(f"Error: Promoter file not found: {args.promoter_file}")
    else:
        extract_tf_binding_signals(
            args.promoter_file,
            args.bigwig_dir,
            args.output_dir,
            args.num_cores,
            args.log_transform,
            args.zscore_normalize,
            args.pseudocount
        )
        print("Step 2 & 3: TF binding signal extraction and optional normalization finished.")

# Example usage (comment out or remove before running as a script):
# python step2_extract_tf_binding_signals.py \
#   --promoter_file "/path/to/your/output_promoter_sequences.tsv" \
#   --bigwig_dir "/path/to/your/bigwig_files/" \
#   --output_dir "/path/to/your/output_npy_signals/" \
#   --num_cores 4 \
#   --log_transform \
#   --zscore_normalize
