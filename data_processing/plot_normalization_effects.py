import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot the effects of normalization on TF binding signals.")
    parser.add_argument("--npy_dir", required=True, help="Directory containing RAW .npy signal files (genes x (TFs x promoter_length) matrices).")
    parser.add_argument("--bigwig_dir", required=True, help="Directory containing original TF bigWig files (to determine TF order and names).")
    parser.add_argument("--output_plot_dir", required=True, help="Directory to save the output plots.")
    parser.add_argument("--tfs_to_plot", required=True, help="Comma-separated list of TF names (e.g., 'TF1,TF_ABC') or 0-based indices (e.g., '0,5,10') to plot.")
    parser.add_argument("--pseudocount", type=float, default=1.0, help="Pseudocount for log transformation (default: 1.0).")
    parser.add_argument("--num_bins", type=int, default=100, help="Number of bins for histograms.")
    return parser.parse_args()

def get_tf_list_and_names(bigwig_dir_path):
    """Gets a sorted list of TF file paths and extracts names."""
    bigwig_files = sorted(glob.glob(os.path.join(bigwig_dir_path, "*.bigWig"))) + \
                     sorted(glob.glob(os.path.join(bigwig_dir_path, "*.bw")))
    if not bigwig_files:
        raise ValueError(f"No bigWig files (.bigWig or .bw) found in directory {bigwig_dir_path}")
    
    tf_names = [os.path.basename(f).replace('.bigWig', '').replace('.bw', '') for f in bigwig_files]
    return bigwig_files, tf_names

def load_signals_for_tf(npy_dir_path, tf_index, num_total_tfs):
    """Loads all signals for a specific TF index from all .npy files."""
    all_signals_for_tf = []
    npy_files = glob.glob(os.path.join(npy_dir_path, "*.npy"))
    if not npy_files:
        print(f"Warning: No .npy files found in {npy_dir_path}.")
        return np.array([])

    for npy_file in npy_files:
        try:
            gene_matrix = np.load(npy_file)
            if gene_matrix.ndim == 2 and gene_matrix.shape[0] == num_total_tfs:
                if 0 <= tf_index < num_total_tfs:
                    all_signals_for_tf.extend(gene_matrix[tf_index, :].flatten())
                else:
                    print(f"Warning: TF index {tf_index} is out of bounds for matrix {npy_file} with shape {gene_matrix.shape}. Skipping this file for this TF.")
                    continue
            elif gene_matrix.ndim == 1 and num_total_tfs == 1 and tf_index == 0 : # Case where npy stores only one TF data
                 all_signals_for_tf.extend(gene_matrix.flatten())
            else:
                print(f"Warning: Matrix in {npy_file} has unexpected shape {gene_matrix.shape} or TF count mismatch (expected {num_total_tfs} TFs). Skipping this file.")
                continue
        except Exception as e:
            print(f"Error loading or processing {npy_file}: {e}. Skipping.")
            continue
    
    return np.array(all_signals_for_tf)

def main():
    args = parse_arguments()

    if not os.path.isdir(args.npy_dir):
        print(f"Error: NPY directory not found: {args.npy_dir}")
        return
    if not os.path.isdir(args.bigwig_dir):
        print(f"Error: bigWig directory not found: {args.bigwig_dir}")
        return

    if not os.path.exists(args.output_plot_dir):
        os.makedirs(args.output_plot_dir)
        print(f"Created output plot directory: {args.output_plot_dir}")

    try:
        _, tf_names = get_tf_list_and_names(args.bigwig_dir)
        num_total_tfs = len(tf_names)
        if num_total_tfs == 0:
            print("Error: No TFs found in the bigwig directory.")
            return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Parse TFs to plot
    selected_tf_indices = []
    selected_tf_names_for_plot = []
    
    tf_parts = args.tfs_to_plot.split(',')
    for part in tf_parts:
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < num_total_tfs:
                selected_tf_indices.append(idx)
                selected_tf_names_for_plot.append(tf_names[idx])
            else:
                print(f"Warning: TF index {idx} is out of range (0-{num_total_tfs-1}). Skipping.")
        else: # Attempt to match by name
            try:
                idx = tf_names.index(part)
                selected_tf_indices.append(idx)
                selected_tf_names_for_plot.append(part)
            except ValueError:
                print(f"Warning: TF name '{part}' not found in bigWig directory. Skipping.")

    if not selected_tf_indices:
        print("No valid TFs selected for plotting. Exiting.")
        return

    print(f"Selected TFs for plotting: {selected_tf_names_for_plot}")

    for tf_idx, tf_name_for_plot in zip(selected_tf_indices, selected_tf_names_for_plot):
        print(f"Processing TF: {tf_name_for_plot} (Index: {tf_idx})")
        
        raw_signals = load_signals_for_tf(args.npy_dir, tf_idx, num_total_tfs)

        if raw_signals.size == 0:
            print(f"No signals found for TF {tf_name_for_plot}. Skipping plot generation.")
            continue
        
        print(f"  Loaded {raw_signals.size} signal points for TF {tf_name_for_plot}.")

        # Perform transformations
        log_transformed_signals = np.log2(raw_signals + args.pseudocount)
        
        mean_raw = np.mean(raw_signals)
        std_raw = np.std(raw_signals)
        if std_raw < 1e-9: # Avoid division by zero or very small std
            print(f"  Warning: Standard deviation of raw signals for TF {tf_name_for_plot} is very small ({std_raw}). Z-scores may not be meaningful.")
            zscore_from_raw = np.zeros_like(raw_signals) if std_raw == 0 else (raw_signals - mean_raw) / 1.0
        else:
            zscore_from_raw = (raw_signals - mean_raw) / std_raw

        log_then_zscore_signals = np.log2(raw_signals + args.pseudocount) # re-calculate for clarity, though same as log_transformed_signals
        mean_log = np.mean(log_then_zscore_signals)
        std_log = np.std(log_then_zscore_signals)
        if std_log < 1e-9:
            print(f"  Warning: Standard deviation of log-transformed signals for TF {tf_name_for_plot} is very small ({std_log}). Z-scores may not be meaningful.")
            zscore_from_log = np.zeros_like(log_then_zscore_signals) if std_log == 0 else (log_then_zscore_signals - mean_log) / 1.0
        else:
            zscore_from_log = (log_then_zscore_signals - mean_log) / std_log

        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Normalization Effects for TF: {tf_name_for_plot} (all promoters)", fontsize=16)

        # Raw signals
        axs[0, 0].hist(raw_signals, bins=args.num_bins, color='blue', alpha=0.7, density=True)
        axs[0, 0].set_title("Raw Signals")
        axs[0, 0].set_xlabel("Signal Value")
        axs[0, 0].set_ylabel("Density")

        # Log-transformed signals
        axs[0, 1].hist(log_transformed_signals, bins=args.num_bins, color='green', alpha=0.7, density=True)
        axs[0, 1].set_title(f"Log2(x + {args.pseudocount}) Transformed")
        axs[0, 1].set_xlabel("Log2(Signal Value + Pseudocount)")
        axs[0, 1].set_ylabel("Density")

        # Z-score from raw signals
        axs[1, 0].hist(zscore_from_raw, bins=args.num_bins, color='red', alpha=0.7, density=True)
        axs[1, 0].set_title("Z-score Standardized (from Raw)")
        axs[1, 0].set_xlabel("Z-score")
        axs[1, 0].set_ylabel("Density")

        # Log-transformed then Z-score standardized
        axs[1, 1].hist(zscore_from_log, bins=args.num_bins, color='purple', alpha=0.7, density=True)
        axs[1, 1].set_title("Log2 Transform then Z-score")
        axs[1, 1].set_xlabel("Z-score of Log-Transformed Data")
        axs[1, 1].set_ylabel("Density")

        for ax_row in axs:
            for ax in ax_row:
                ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        
        # Sanitize TF name for filename
        safe_tf_name = re.sub(r'[^a-zA-Z0-9_\\-\\.]', '_', tf_name_for_plot)
        plot_filename = os.path.join(args.output_plot_dir, f"{safe_tf_name}_normalization_effects.png")
        try:
            plt.savefig(plot_filename)
            print(f"  Plot saved to {plot_filename}")
        except Exception as e:
            print(f"  Error saving plot {plot_filename}: {e}")
        plt.close(fig)

    print("Plotting script finished.")

if __name__ == "__main__":
    # This try-except block is for environments where Matplotlib might
    # try to use a display server even if not available (e.g., in some remote execution).
    # Using 'Agg' backend for non-interactive plotting.
    try:
        import matplotlib
        matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
        main()
    except ImportError:
        print("Matplotlib is not installed. Please install it to run this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage (comment out or remove before running as a script):
# python plot_normalization_effects.py \\
#   --npy_dir "/path/to/your/raw_npy_signals/" \\
#   --bigwig_dir "/path/to/your/bigwig_files/" \\
#   --output_plot_dir "/path/to/your/normalization_plots/" \\
#   --tfs_to_plot "TF_NAME_1,2,TF_XYZ" \\
#   --pseudocount 1.0 \\
#   --num_bins 100
