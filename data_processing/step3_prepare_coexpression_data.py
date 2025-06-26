
import argparse
import pandas as pd
import os

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare co-expression correlation coefficient dataset.")
    parser.add_argument("--coexpression_file", required=True, help="Path to the input TSV file with gene pairs and correlations (e.g., Gene1, Gene2, Correlation).")
    parser.add_argument("--processed_signals_dir", help="Path to the directory containing .npy files from Step 3.2/3.3 (used for validation if --validate_genes is set).")
    parser.add_argument("--output_file", required=True, help="Path to the output final co-expression TSV file.")
    parser.add_argument("--validate_genes", action='store_true', help="If set, validate that gene IDs in the coexpression file have corresponding processed signal files in --processed_signals_dir.")
    return parser.parse_args()

def prepare_coexpression_data(coexpression_file_path, output_file_path, processed_signals_dir_path=None, validate_genes_flag=False):
    """
    Loads, optionally validates, and saves co-expression data.

    Args:
        coexpression_file_path (str): Path to the input co-expression TSV.
        output_file_path (str): Path for the output TSV.
        processed_signals_dir_path (str, optional): Directory of processed .npy signal files.
        validate_genes_flag (bool): Whether to validate gene IDs.
    """
    try:
        coexp_df = pd.read_csv(coexpression_file_path, sep='\t')
        print(f"Successfully loaded co-expression data from {coexpression_file_path}. Found {len(coexp_df)} pairs.")

        # Expecting columns like 'Gene1', 'Gene2', 'Correlation'
        # Adjust column names if they are different in the actual input file.
        expected_cols = ['Gene1', 'Gene2', 'Correlation'] # Example names
        if not all(col in coexp_df.columns for col in expected_cols):
            print(f"Warning: Expected columns {expected_cols} not all found in {coexpression_file_path}. Current columns: {coexp_df.columns.tolist()}")
            # Proceeding, but user should verify column names.

    except FileNotFoundError:
        print(f"Error: Co-expression file not found at {coexpression_file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Co-expression file {coexpression_file_path} is empty.")
        return
    except Exception as e:
        print(f"Error parsing co-expression file {coexpression_file_path}: {e}")
        return

    if validate_genes_flag:
        if not processed_signals_dir_path:
            print("Error: --processed_signals_dir must be provided if --validate_genes is set.")
            return
        if not os.path.isdir(processed_signals_dir_path):
            print(f"Error: Processed signals directory not found or is not a directory: {processed_signals_dir_path}")
            return

        print(f"Validating gene pairs against processed signals in {processed_signals_dir_path}...")
        initial_pairs_count = len(coexp_df)

        def check_gene_processed(gene_id):
            # Assumes .npy files are named <gene_id>.npy
            return os.path.exists(os.path.join(processed_signals_dir_path, f"{gene_id}.npy"))

        # Assuming column names are 'Gene1' and 'Gene2'
        # Adjust if your column names are different
        try:
            mask_gene1_exists = coexp_df['Gene1'].apply(check_gene_processed)
            mask_gene2_exists = coexp_df['Gene2'].apply(check_gene_processed)
            coexp_df = coexp_df[mask_gene1_exists & mask_gene2_exists]
            validated_pairs_count = len(coexp_df)
            filtered_count = initial_pairs_count - validated_pairs_count
            print(f"Validation complete. {validated_pairs_count} pairs remain after filtering. {filtered_count} pairs were removed.")
        except KeyError as e:
            print(f"Error during validation: Column {e} not found in the co-expression file. Please ensure column names are 'Gene1' and 'Gene2' or adjust script.")
            return
        except Exception as e:
            print(f"An unexpected error occurred during gene validation: {e}")
            return

    if coexp_df.empty:
        print("No data remains after processing/validation. Output file will not be created or will be empty.")

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory for output file: {output_dir}")

        coexp_df.to_csv(output_file_path, sep='\t', index=False)
        print(f"Successfully saved final co-expression data to {output_file_path}. Contains {len(coexp_df)} pairs.")
    except Exception as e:
        print(f"Error writing output file {output_file_path}: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    prepare_coexpression_data(
        args.coexpression_file,
        args.output_file,
        args.processed_signals_dir,
        args.validate_genes
    )
    print("Step 3.4: Co-expression data preparation finished.")

# Example usage (comment out or remove before running as a script):
# python step3_prepare_coexpression_data.py \
#   --coexpression_file "/path/to/your/gene_pairs_correlations.tsv" \
#   --output_file "/path/to/your/final_coexpression_data.tsv" \
#   --processed_signals_dir "/path/to/your/output_npy_signals/" \
#   --validate_genes
