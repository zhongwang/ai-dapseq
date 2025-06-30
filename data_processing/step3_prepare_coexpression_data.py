import argparse
import pandas as pd
import os
import numpy as np

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare co-expression correlation coefficient dataset.")
    parser.add_argument("--coexpression_file", required=True, help="Path to the input TSV file with gene pairs and correlations (e.g., Gene1, Gene2, Correlation).")
    parser.add_argument("--gene_chromosome_map_file", required=True, help="Path to the TSV file mapping gene IDs to chromosomes (with 'gene_id' and 'chromosome' columns).")
    parser.add_argument("--processed_signals_dir", help="Path to the directory containing .npy files from Step 3.2/3.3 (used for validation if --validate_genes is set).")
    parser.add_argument("--output_file", required=True, help="Path to the output final co-expression TSV file.")
    parser.add_argument("--validate_genes", action='store_true', help="If set, validate that gene IDs in the coexpression file have corresponding processed signal files in --processed_signals_dir.")
    return parser.parse_args()

def prepare_coexpression_data(coexpression_file_path, gene_chromosome_map_file_path, output_file_path, processed_signals_dir_path=None, validate_genes_flag=False):
    """
    Loads, optionally validates, splits by chromosome, samples by correlation bin, and saves co-expression data.

    Args:
        coexpression_file_path (str): Path to the input co-expression TSV.
        gene_chromosome_map_file_path (str): Path to the gene-chromosome mapping TSV.
        output_file_path (str): Path for the output TSV.
        processed_signals_dir_path (str, optional): Directory of processed .npy signal files.
        validate_genes_flag (bool): Whether to validate gene IDs.
    """
    try:
        coexp_df = pd.read_csv(coexpression_file_path, sep='\t')
        print(f"Successfully loaded co-expression data from {coexpression_file_path}. Found {len(coexp_df)} pairs.")

        # Expecting columns like 'Gene1', 'Gene2', 'Correlation'
        expected_cols = ['Gene1', 'Gene2', 'Correlation'] # Example names
        if not all(col in coexp_df.columns for col in expected_cols):
            print(f"Error: Expected columns {expected_cols} not all found in {coexpression_file_path}. Current columns: {coexp_df.columns.tolist()}")
            return # Stop if essential columns are missing.

    except FileNotFoundError:
        print(f"Error: Co-expression file not found at {coexpression_file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Co-expression file {coexpression_file_path} is empty.")
        return
    except Exception as e:
        print(f"Error parsing co-expression file {coexpression_file_path}: {e}")
        return

    try:
        gene_chr_df = pd.read_csv(gene_chromosome_map_file_path, sep='\t')
        print(f"Successfully loaded gene-chromosome map from {gene_chromosome_map_file_path}. Found {len(gene_chr_df)} entries.")

        # Expecting columns 'gene_id' and 'chromosome'
        if not all(col in gene_chr_df.columns for col in ['gene_id', 'chromosome']):
             print(f"Error: Gene-chromosome map file must contain 'gene_id' and 'chromosome' columns. Current columns: {gene_chr_df.columns.tolist()}")
             return

        gene_to_chr = gene_chr_df.set_index('gene_id')['chromosome'].to_dict()

    except FileNotFoundError:
        print(f"Error: Gene-chromosome map file not found at {gene_chromosome_map_file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Gene-chromosome map file {gene_chromosome_map_file_path} is empty.")
        return
    except Exception as e:
        print(f"Error parsing gene-chromosome map file {gene_chromosome_map_file_path}: {e}")
        return

    # Add chromosome information to the co-expression dataframe
    # Handle cases where a gene might not be in the mapping file
    coexp_df['Gene1_chr'] = coexp_df['Gene1'].map(gene_to_chr)
    coexp_df['Gene2_chr'] = coexp_df['Gene2'].map(gene_to_chr)

    # Drop pairs where chromosome information is missing for either gene
    initial_pairs_count = len(coexp_df)
    coexp_df.dropna(subset=['Gene1_chr', 'Gene2_chr'], inplace=True)
    if len(coexp_df) < initial_pairs_count:
        print(f"Removed {initial_pairs_count - len(coexp_df)} pairs due to missing chromosome information.")


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
        return

    # Chromosome-based splitting
    test_set = coexp_df[(coexp_df['Gene1_chr'] == 'Chr2') & (coexp_df['Gene2_chr'] == 'Chr2')].copy()
    val_set = coexp_df[(coexp_df['Gene1_chr'] == 'Chr4') & (coexp_df['Gene2_chr'] == 'Chr4')].copy()
    # The training set includes all pairs NOT in the test or validation sets
    train_set = coexp_df[
        (~coexp_df['Gene1_chr'].isin(['Chr2', 'Chr4'])) &
        (~coexp_df['Gene2_chr'].isin(['Chr2', 'Chr4']))
    ].copy()

    print(f"Split data: Train={len(train_set)} pairs, Validation={len(val_set)} pairs, Test={len(test_set)} pairs.")

    if train_set.empty:
        print("Error: Training set is empty. Cannot perform binning and sampling.")
        # As a fallback, save the unsampled validation and test sets along with the empty train set structure
        final_coexp_df = pd.concat([train_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore') if 'Gene1_chr' in train_set.columns else train_set,
                                     val_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore') if 'Gene1_chr' in val_set.columns else val_set,
                                     test_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore') if 'Gene1_chr' in test_set.columns else test_set])

        if final_coexp_df.empty:
             print("No data remains after splitting. Output file will not be created or will be empty.")
             return

        try:
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory for output file: {output_dir}")

            final_coexp_df.to_csv(output_file_path, sep='\t', index=False)
            print(f"Successfully saved unsampled split data to {output_file_path}. Contains {len(final_coexp_df)} pairs.")
        except Exception as e_save:
            print(f"Error writing output file {output_file_path}: {e_save}")
        return


    # Bin training data by correlation using fixed bin widths
    try:
        min_corr = train_set['Correlation'].min()
        max_corr = train_set['Correlation'].max()
        # Create 5 equally spaced bins between min and max correlation
        bins = np.linspace(min_corr, max_corr, 6) # 11 edges for 10 bins

        # Use pd.cut for fixed-width binning. `include_lowest=True` ensures min value is included.
        # Labels=False assigns integer bin labels (0 to 9 for 10 bins).
        # right=True means intervals are closed on the right (e.g., (a, b]). The first interval includes the left edge.
        train_set['Correlation_Bin'] = pd.cut(train_set['Correlation'], bins=bins, labels=False, include_lowest=True, right=True)

        # Determine the actual number of bins created. pd.cut might create fewer if there aren't enough unique values.
        num_actual_bins = train_set['Correlation_Bin'].nunique() if 'Correlation_Bin' in train_set.columns else 0

        print(f"Created {num_actual_bins} correlation bins based on the training set using fixed widths.")

        # Check if binning resulted in fewer than 2 bins (meaningful for sampling)
        if num_actual_bins < 2 and len(train_set) > 0:
             print(f"Warning: Only {num_actual_bins} unique correlation bins created from training data with fixed widths. Cannot perform meaningful sampling across bins.")
             # Fallback to saving unsampled split data
             print("Saving unsampled data after splitting due to insufficient bins for fixed-width binning.")
             final_coexp_df = pd.concat([train_set.drop(columns=['Correlation_Bin', 'Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                         val_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                         test_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore')])

             if final_coexp_df.empty:
                print("No data remains after splitting. Output file will not be created or will be empty.")
                return

             try:
                 output_dir = os.path.dirname(output_file_path)
                 if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created directory for output file: {output_dir}")

                 final_coexp_df.to_csv(output_file_path, sep='\t', index=False)
                 print(f"Successfully saved unsampled split data to {output_file_path}. Contains {len(final_coexp_df)} pairs.")
             except Exception as e_save:
                 print(f"Error writing output file {output_file_path}: {e_save}")
             return # Exit after saving unsampled data


    except Exception as e: # Catch potential errors during binning (e.g., train_set empty, though handled above, or issues with data types)
        print(f"Could not create bins for training data using fixed widths: {e}. This can happen if there's an issue with correlation values or data structure.")
        # Fallback to saving unsampled split data
        print("Saving unsampled data after splitting due to binning error.")
        final_coexp_df = pd.concat([train_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                     val_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                     test_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore')])

        if final_coexp_df.empty:
            print("No data remains after splitting. Output file will not be created or will be empty.")
            return

        try:
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory for output file: {output_dir}")

            final_coexp_df.to_csv(output_file_path, sep='\t', index=False)
            print(f"Successfully saved unsampled split data to {output_file_path}. Contains {len(final_coexp_df)} pairs.")
        except Exception as e_save:
            print(f"Error writing output file {output_file_path}: {e_save}")
        return # Exit after saving unsampled data


    # Find the size of the smallest training bin
    # Filter out potential NaN bin counts before finding the minimum
    train_bin_counts = train_set['Correlation_Bin'].value_counts().dropna()

    if train_bin_counts.empty:
         print("Error: No data in any training bin after fixed-width binning. Cannot perform sampling.")
         # Fallback to saving unsampled split data
         print("Saving unsampled data after splitting due to empty bins.")
         final_coexp_df = pd.concat([train_set.drop(columns=['Correlation_Bin', 'Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                     val_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                     test_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore')])

         if final_coexp_df.empty:
             print("No data remains after splitting. Output file will not be created or will be empty.")
             return

         try:
             output_dir = os.path.dirname(output_file_path)
             if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)
                 print(f"Created directory for output file: {output_dir}")

             final_coexp_df.to_csv(output_file_path, sep='\t', index=False)
             print(f"Successfully saved unsampled split data to {output_file_path}. Contains {len(final_coexp_df)} pairs.")
         except Exception as e_save:
             print(f"Error writing output file {output_file_path}: {e_save}")
         return # Exit after saving unsampled data

    min_train_bin_size = train_bin_counts.min()
    print(f"Smallest training bin size: {min_train_bin_size}")

    if min_train_bin_size == 0:
         print("Error: Smallest training bin size is 0 after fixed-width binning. Cannot perform sampling evenly.")
         # Fallback to saving unsampled split data
         print("Saving unsampled data after splitting due to empty bin.")
         final_coexp_df = pd.concat([train_set.drop(columns=['Correlation_Bin', 'Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                     val_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore'),
                                     test_set.drop(columns=['Gene1_chr', 'Gene2_chr'], errors='ignore')])

         if final_coexp_df.empty:
             print("No data remains after splitting. Output file will not be created or will be empty.")
             return

         try:
             output_dir = os.path.dirname(output_file_path)
             if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)
                 print(f"Created directory for output file: {output_dir}")

             final_coexp_df.to_csv(output_file_path, sep='\t', index=False)
             print(f"Successfully saved unsampled split data to {output_file_path}. Contains {len(final_coexp_df)} pairs.")
         except Exception as e_save:
             print(f"Error writing output file {output_file_path}: {e_save}")
         return # Exit after saving unsampled data


    # Function to sample from bins
    def sample_from_bins(df, bins, min_samples_per_bin):
        if df.empty:
            # Return empty df with relevant columns, dropping chromosome columns
            cols = ['Gene1', 'Gene2', 'Correlation']
            return pd.DataFrame(columns=cols)

        # Assign bins based on the train set\'s bin edges (fixed widths)
        # Use include_lowest=True and right=True to match pd.cut behavior
        # pd.cut can return NaN for values outside the bins; dropna handles this
        df['Correlation_Bin'] = pd.cut(df['Correlation'], bins=bins, labels=False, include_lowest=True, right=True)

        sampled_df = pd.DataFrame()
        # Iterate through all possible bin labels (0 to num_actual_bins-1) to ensure all bins are considered
        # even if a specific set (val/test) doesn\'t have data in a particular bin initially
        # Use the number of bins defined by the 'bins' edges (len(bins) - 1)
        num_expected_bins = len(bins) - 1
        for bin_label in range(num_expected_bins):
             # Ensure the bin_label exists in the current DataFrame\'s Correlation_Bin column before filtering
             if bin_label in df['Correlation_Bin'].unique():
                 bin_data = df[df['Correlation_Bin'] == bin_label]
                 if not bin_data.empty:
                    # Sample up to min_samples_per_bin, or all if the bin is smaller
                    sampled_bin = bin_data.sample(n=min(len(bin_data), min_samples_per_bin), replace=False, random_state=42) # Use random_state for reproducibility
                    sampled_df = pd.concat([sampled_df, sampled_bin])


        # Drop the temporary bin column
        # Drop chromosome columns here as they are no longer needed after splitting
        sampled_df = sampled_df.drop(columns=['Correlation_Bin', 'Gene1_chr', 'Gene2_chr'], errors='ignore')
        return sampled_df

    print("Sampling from bins for train, validation, and test sets...")
    # Pass the original train_set (which has the \'Correlation_Bin\' column for reference) but drop it inside the function
    # Also pass the bins calculated from the training set
    sampled_train_set = sample_from_bins(train_set, bins, min_train_bin_size)
    sampled_val_set = sample_from_bins(val_set, bins, min_train_bin_size)
    sampled_test_set = sample_from_bins(test_set, bins, min_train_bin_size)

    print(f"Sampled data: Train={len(sampled_train_set)} pairs, Validation={len(sampled_val_set)} pairs, Test={len(sampled_test_set)} pairs.")

    # Combine sampled dataframes
    final_coexp_df = pd.concat([sampled_train_set, sampled_val_set, sampled_test_set])

    # Temporary chromosome columns and Correlation_Bin were dropped during sampling.

    if final_coexp_df.empty:
        print("No data remains after splitting and sampling. Output file will not be created or will be empty.")
        return

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory for output file: {output_dir}")

        final_coexp_df.to_csv(output_file_path, sep='\t', index=False)
        print(f"Successfully saved final co-expression data to {output_file_path}. Contains {len(final_coexp_df)} pairs.")
    except Exception as e:
        print(f"Error writing output file {output_file_path}: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    prepare_coexpression_data(
        args.coexpression_file,
        args.gene_chromosome_map_file,
        args.output_file,
        args.processed_signals_dir,
        args.validate_genes
    )
    print("Step 3.4: Co-expression data preparation finished.")

# Example usage (comment out or remove before running as a script):
# python step3_prepare_coexpression_data.py \
#   --coexpression_file "/path/to/your/gene_pairs_correlations.tsv" \
#   --gene_chromosome_map_file "/path/to/your/gene_chromosome_map.tsv" \
#   --output_file "/path/to/your/final_coexpression_data.tsv" \
#   --processed_signals_dir "/path/to/your/output_npy_signals/" \
#   --validate_genes``