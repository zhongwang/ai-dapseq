import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_target_distribution(df, target, output_dir, pseudocount, bins = 20):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey = True, sharex = True)
    fig.suptitle(f"Transformation effects on target", fontsize=16)

    # No transformation
    axs[0].hist(df[target], bins=bins, color='blue', alpha=0.7, density=True)
    axs[0].set_title("No transformation")
    axs[0].set_xlabel(target)
    axs[0].set_ylabel("Density")

    # Log-transformation
    axs[1].hist(np.log2(df[target] + pseudocount), bins=bins, color='green', alpha=0.7, density=True)
    axs[1].set_title(f"Log2({target} + {pseudocount}) transformation")
    axs[1].set_xlabel(f"Log2({target} + pseudocount)")
    axs[1].set_ylabel("Density")

    # Fisher's Z-transformation
    z_trans = 0.5 * np.log((1 + df[target]) / (1 - df[target]))
    axs[2].hist(z_trans, bins=bins, color='orange', alpha=0.7, density=True)
    axs[2].set_title(f"Fisher's Z-transformation")
    axs[2].set_xlabel(f"Fisher's Z-transformation")
    axs[2].set_ylabel("Density")

    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

    plot_filename = os.path.join(output_dir, f"{target}_normalization_effects.png")
    try:
        plt.savefig(plot_filename)
        print(f"  Plot saved to {plot_filename}")
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")
    plt.close(fig)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Plots the transformation on the target variable")
    parser.add_argument("--data_tsv", required=True, help="TSV containing the data to plot")
    parser.add_argument("--target_var", required = True, help = "Target variable to create histogram of")
    parser.add_argument("--output_plot_dir", required=True, help="Directory to save the output plots.")
    parser.add_argument("--pseudocount", type=float, default=1.0, help="Pseudocount for log transformation (default: 1.0).")
    parser.add_argument("--num_bins", type=int, default=20, help="Number of bins for histograms.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.isdir(args.output_plot_dir):
        print(f"Error: NPY directory not found: {args.output_plot_dir}")
        return

    df = pd.read_csv(args.data_tsv, sep='\t')

    plot_target_distribution(df, args.target_var, args.output_plot_dir, args.pseudocount, args.num_bins)

if __name__ == "__main__":
    # This try-except block is for environments where Matplotlib might
    # try to use a display server even if not available (e.g., in some remote execution).
    # Using 'Agg' backend for non-interactive plotting.
    main()