import argparse
import pandas as pd
import numpy as np
import pyBigWig
import os
import glob
from multiprocessing import Pool, cpu_count
import time

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract per-base TF binding signals for promoter regions and optionally normalize them.")
    parser.add_argument("--promoter_file", required=True, help="Path to the TSV file from Step 1 (gene_id, chromosome, promoter_start, promoter_end, strand).")
    parser.add_argument("--bigwig_dir", required=True, help="Directory containing TF binding bigWig files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output .npy files (raw or normalized signals).")
    parser.add_argument("--num_cores", type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())), help="Number of CPU cores to use for parallel processing.")
    #parser.add_argument("--num_cores", type=int, default=max(1, cpu_count() - 1), help="Number of CPU cores to use for parallel processing.")

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
    start_time = time.time() # Record start time for this gene

    gene_id, chrom, start, end, strand, bigwig_files, output_dir, log_transform, pseudocount = args_tuple

    promoter_length = end - start
    if promoter_length <= 0:
        print(f"Skipping gene {gene_id}: promoter length is non-positive ({promoter_length}).")
        # Return duration even if skipping
        end_time = time.time()
        duration = end_time - start_time
        return gene_id, None, "Invalid promoter length", duration

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

    # Return the result along with the processing duration
    end_time = time.time()
    duration = end_time - start_time
    return gene_id, gene_signals_matrix, None, duration

def extract_tf_binding_signals(promoter_file_path, bigwig_dir_path, output_dir_path, num_cores, log_transform_flag, zscore_normalize_flag, pseudocount_val):
    """
    Main function to orchestrate TF binding signal extraction and normalization.
    """
    total_start_time = time.time() # Record start time for overall process

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
    successful_extractions = 0
    failed_extractions = 0
    processed_count = 0
    total_duration_processed = 0
    num_tasks = len(tasks)

    with Pool(processes=num_cores) as pool:
        # Use imap_unordered for results as they complete
        results_iterator = pool.imap_unordered(process_gene, tasks)

        for result in results_iterator:
            # Ensure the result tuple has 4 elements (gene_id, signal_matrix, error_msg, duration)
            if len(result) == 4:
                 gene_id, signal_matrix, error_msg, duration = result
            else:
                 # Handle unexpected result format (e.g., from old code or error)
                 print(f"Warning: Received unexpected result format for a gene. Skipping timing for this result: {result}")
                 if len(result) == 3: # Assume old format (gene_id, signal_matrix, error_msg)
                      gene_id, signal_matrix, error_msg = result
                      duration = 0 # Assign 0 duration or handle as appropriate
                 else:
                      print(f"Error: Unexpected result format: {result}. Skipping this result.")
                      failed_extractions += 1
                      continue # Skip to the next result

            processed_count += 1
            total_duration_processed += duration

            if error_msg:
                print(f"Extraction failed for {gene_id}: {error_msg}")
                failed_extractions += 1
            elif signal_matrix is not None:
                all_gene_signals[gene_id] = signal_matrix
                successful_extractions += 1
                print(f"Processed gene {gene_id} in {duration:.4f} seconds. Completed {processed_count}/{num_tasks}.")
            else:
                # This case implies an issue not caught by error_msg but matrix is None
                print(f"Extraction yielded no data for {gene_id}, but no specific error message.")
                failed_extractions += 1

            # Provide estimated finish time periodically
            if processed_count > 0 and (processed_count % 100 == 0 or processed_count == num_tasks):
                elapsed_time = time.time() - total_start_time
                # Avoid division by zero if no duration recorded yet or processed_count is 0
                if processed_count > 0:
                    avg_time_per_gene = total_duration_processed / processed_count
                    remaining_genes = num_tasks - processed_count
                    estimated_remaining_time = avg_time_per_gene * remaining_genes
                    estimated_finish_time = time.time() + estimated_remaining_time
                    print(f"Elapsed time: {elapsed_time:.2f}s. Processed {processed_count}/{num_tasks}. Estimated remaining time: {estimated_remaining_time:.2f}s. Estimated finish time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(estimated_finish_time))}")
                else:
                     print(f"Elapsed time: {elapsed_time:.2f}s. Processed {processed_count}/{num_tasks}. Cannot estimate finish time yet.")


    # After pool is done and results are collected:
    print(f"Signal extraction phase complete. Successfully processed: {successful_extractions}, Failed: {failed_extractions}")

    if not all_gene_signals:
        print("No signals were extracted. Aborting further processing.")
        return

    # Save the (raw or log-transformed) matrices to individual .npy files
    # This happens regardless of whether z-score normalization is requested later
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

    # Clear the dictionary to free up memory
    del all_gene_signals
    import gc
    gc.collect() # Suggest garbage collection

    # Step 3.3: Signal Normalization (Z-score) - Memory Optimized Two-Pass Approach
    if zscore_normalize_flag:
        print("Applying Z-score normalization across promoters for each TF (memory optimized)...")
        num_tfs = len(bigwig_files)
        gene_files = glob.glob(os.path.join(output_dir_path, "*.npy"))
        if not gene_files:
            print("No gene .npy files found for normalization.")
            # This case should ideally not happen if saving was successful, but good to check
            print("Error: No .npy files found in the output directory after saving.")
            return

        # Pass 1: Calculate sum and sum of squares for each TF
        print("Pass 1: Calculating sums and sum of squares...")
        tf_sum = np.zeros(num_tfs)
        tf_sum_sq = np.zeros(num_tfs)
        tf_counts = np.zeros(num_tfs, dtype=int) # To handle potential missing data for a TF across all genes or issues loading files

        for gene_file in gene_files:
            try:
                # Use mmap_mode='r' to read file without loading the entire content into memory
                signal_matrix = np.load(gene_file, mmap_mode='r')
                if signal_matrix.shape[0] == num_tfs:
                     # Reshape to flatten all signals for a TF across positions in this gene
                    flat_signals = signal_matrix.reshape(num_tfs, -1)
                    # Perform calculation on mmap'd array - should be memory efficient
                    tf_sum += np.sum(flat_signals, axis=1)
                    tf_sum_sq += np.sum(flat_signals**2, axis=1)
                    tf_counts += flat_signals.shape[1]
                else:
                    print(f"Warning: Skipping {os.path.basename(gene_file)} in normalization passes due to unexpected TF count ({signal_matrix.shape[0]} vs {num_tfs}).")
            except Exception as e:
                print(f"Error reading {gene_file} in Pass 1: {e}")


        # Calculate global mean and std deviation for each TF
        tf_means = np.zeros(num_tfs)
        tf_stds = np.ones(num_tfs) # Default to std=1.0

        for tf_idx in range(num_tfs):
            count = tf_counts[tf_idx]
            if count > 1: # Need at least two data points for a meaningful std deviation
                tf_means[tf_idx] = tf_sum[tf_idx] / count
                # Calculate variance: E[x^2] - (E[x])^2
                mean_sq = tf_sum_sq[tf_idx] / count
                variance = mean_sq - tf_means[tf_idx]**2
                # Ensure non-negative variance due to potential floating point inaccuracies
                variance = max(0, variance)
                std_val = np.sqrt(variance)
                tf_stds[tf_idx] = std_val if std_val > 1e-6 else 1.0
            elif count == 1:
                 tf_means[tf_idx] = tf_sum[tf_idx] # Mean is just the single value
                 tf_stds[tf_idx] = 1.0 # Std deviation is undefined or zero, use 1.0 to avoid division by zero
                 print(f"Warning: Only one data point for TF index {tf_idx}. Std set to 1.0.")
            else: # count == 0
                 print(f"Warning: No data counted for TF index {tf_idx}. Mean=0, Std=1.")


        print("Pass 1 complete. Global means and std deviations calculated.")

        # Pass 2: Apply Z-score normalization and save
        print("Pass 2: Applying normalization and saving...")
        normalized_count = 0
        for gene_file in gene_files:
            try:
                # Load the data fully into memory for modification and saving
                signal_matrix = np.load(gene_file)
                if signal_matrix.shape[0] == num_tfs:
                    # Apply Z-score normalization
                    normalized_matrix = (signal_matrix - tf_means[:, np.newaxis]) / tf_stds[:, np.newaxis]

                    # Overwrite the existing .npy file with normalized data
                    np.save(gene_file, normalized_matrix)
                    normalized_count += 1
                # else: already warned in Pass 1
            except Exception as e:
                print(f"Error processing or saving normalized data for {gene_file} in Pass 2: {e}")

        print(f"Normalization and saving complete. Successfully normalized {normalized_count} files.")


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
    total_end_time = time.time()
    print(f"Step 2 & 3: TF binding signal extraction and optional normalization finished.") #Total time: {total_end_time - total_start_time:.2f} seconds.")

# Example usage (comment out or remove before running as a script):
# python step2_extract_tf_binding_signals.py \
#   --promoter_file "/path/to/your/output_promoter_sequences.tsv" \
#   --bigwig_dir "/path/to/your/bigwig_files/" \
#   --output_dir "/path/to/your/output_npy_signals/" \
#   --num_cores 4 \
#   --log_transform \
#   --zscore_normalize
