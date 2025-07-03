import argparse
import pandas as pd
import os
import numpy as np

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare co-expression correlation coefficient dataset.")
    parser.add_argument("--coexpression_file", required=True, help="Path to the input TSV file with gene pairs and correlations (e.g., Gene1, Gene2, Correlation).\nExpected columns: 'Gene1', 'Gene2', 'Correlation'.")
    parser.add_argument("--gene_chromosome_map_file", required=True, help="Path to the TSV file mapping gene IDs to chromosomes (with 'gene_id' and 'chromosome' columns).")
    parser.add_argument("--processed_signals_dir", help="Path to the directory containing .npy files from Step 3.2/3.3 (used for validation if --validate_genes is set).\nAssumes .npy files are named '<gene_id>.npy'.")
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

        print(f"Validating gene pairs against processed signals in {processed_signals_dir_path}...\nAssuming .npy files are named '<gene_id>.npy'.")
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
    # Ensure to copy the slices to avoid SettingWithCopyWarning and ensure independent dataframes
    test_set = coexp_df[(coexp_df['Gene1_chr'] == 'Chr2') & (coexp_df['Gene2_chr'] == 'Chr2')].copy()
    val_set = coexp_df[(coexp_df['Gene1_chr'] == 'Chr4') & (coexp_df['Gene2_chr'] == 'Chr4')].copy()
    # The training set includes all pairs NOT in the test or validation sets
    # Use index.isin to correctly exclude rows present in test_set or val_set
    #train_set = coexp_df[~coexp_df.index.isin(test_set.index) & ~coexp_df.index.isin(val_set.index)].copy()

    train_set = coexp_df[
        (~coexp_df['Gene1_chr'].isin(['Chr2', 'Chr4'])) &
        (~coexp_df['Gene2_chr'].isin(['Chr2', 'Chr4']))
    ].copy()


    print(f"Split data: Train={len(train_set)} pairs, Validation={len(val_set)} pairs, Test={len(test_set)} pairs.")

    if train_set.empty:
        print("Error: Training set is empty. Cannot perform binning and sampling.")
        # As a fallback, save the unsampled validation and test sets along with the empty train set structure
        # Ensure to drop temporary columns before saving
        cols_to_drop = ['Gene1_chr', 'Gene2_chr']
        train_save = train_set.drop(columns=[col for col in cols_to_drop if col in train_set.columns], errors='ignore')
        val_save = val_set.drop(columns=[col for col in cols_to_drop if col in val_set.columns], errors='ignore')
        test_save = test_set.drop(columns=[col for col in cols_to_drop if col in test_set.columns], errors='ignore')

        final_coexp_df = pd.concat([train_save, val_save, test_save])

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
        # Create 3 equally spaced bins between min and max correlation
        # Use 4 edges for 10 bins
        bins = np.linspace(min_corr, max_corr, 4)


    except Exception as e: # Catch potential errors during binning (e.g., train_set empty, though handled above, or issues with data types)
        print(f"Could not create bins for training data using fixed widths: {e}. This can happen if there's an issue with correlation values or data structure.")
        # Fallback to saving unsampled split data
        print("Saving unsampled data after splitting due to binning error.")
        cols_to_drop = ['Gene1_chr', 'Gene2_chr']
        train_save = train_set.drop(columns=[col for col in cols_to_drop if col in train_set.columns], errors='ignore')
        val_save = val_set.drop(columns=[col for col in cols_to_drop if col in val_set.columns], errors='ignore')
        test_save = test_set.drop(columns=[col for col in cols_to_drop if col in test_set.columns], errors='ignore')
        final_coexp_df = pd.concat([train_save, val_save, test_save])

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


    # Calculate and find the size of the smallest bin for each set based on the training set's bins

    # Function to calculate min bin size for a given dataframe using the defined bins
    def calculate_and_assign_bins_and_min_size(df, bins):
        if df.empty:
            # Return empty df with the expected columns including original ones and Correlation_Bin for consistency
            cols = ['Gene1', 'Gene2', 'Correlation', 'Gene1_chr', 'Gene2_chr', 'Correlation_Bin']
            return pd.DataFrame(columns=cols), 0

        try:
            # Ensure the 'Correlation' column exists before attempting to bin
            if 'Correlation' not in df.columns:
                print("Warning: 'Correlation' column not found for bin size calculation.")
                # Return original df and 0 min size if Correlation column is missing
                return df, 0

            # Apply binning to the current dataframe for calculating its bin sizes
            # Note: This adds 'Correlation_Bin' to the df in place.
            # Use pd.cut to assign initial bins. Values outside the bin range will be NaN.
            df['Correlation_Bin'], cutoff_bins = pd.cut(df['Correlation'], bins=bins, labels=False, include_lowest=True, retbins=True, right=True)

            # Handle values outside the defined bins
            nan_mask = df['Correlation_Bin'].isna()
            if nan_mask.any():
                print(f"Found {nan_mask.sum()} values outside bin range. Assigning to boundary bins.")
                # Values less than the minimum bin edge get assigned to the first bin (bin 0)
                min_edge = bins[0]
                df.loc[nan_mask & (df['Correlation'] < min_edge), 'Correlation_Bin'] = 0

                # Values greater than the maximum bin edge get assigned to the last bin
                # The last bin index is num_bins - 1
                max_edge = bins[-1]
                num_bins = len(bins) - 1  # Number of actual bins
                df.loc[nan_mask & (df['Correlation'] > max_edge), 'Correlation_Bin'] = num_bins - 1

            # After handling NaNs, convert the column to integer type
            # Use errors='coerce' to turn any remaining non-integer values into NaN, though ideally there shouldn't be any.
            df['Correlation_Bin'] = df['Correlation_Bin'].astype(int, errors='ignore')

            print(f"Bin cutoffs: {cutoff_bins}")

            # Filter out NaN bin labels before counting (should be none after handling)
            bin_counts = df['Correlation_Bin'].value_counts().dropna()

            if bin_counts.empty:
                return df, 0

            # Ensure min_samples_per_bin is an integer
            min_samples_per_bin = int(bin_counts.min())

            return df, min_samples_per_bin
        except Exception as e:
            print(f"Error calculating min bin size or assigning bins for a set: {e}")
            # Return original df and 0 min size in case of other errors during binning
            return df, 0

    # Apply binning and calculate minimum bin sizes for each set
    # The 'Correlation_Bin' column will be added to each set by calculate_and_assign_bins_and_min_size
    train_set, min_train_bin_size = calculate_and_assign_bins_and_min_size(train_set, bins)
    val_set, min_val_bin_size = calculate_and_assign_bins_and_min_size(val_set, bins)
    test_set, min_test_bin_size = calculate_and_assign_bins_and_min_size(test_set, bins)

    print(f"Smallest bin size: Train={min_train_bin_size}, Validation={min_val_bin_size}, Test={min_test_bin_size}")

    # Function to sample from bins using the set's minimum bin size
    # This function no longer needs the 'bins' argument as binning is done beforehand.
    def sample_from_bins(df, min_samples_per_bin):
        if df.empty or min_samples_per_bin == 0:
            # Return empty df with relevant columns, dropping chromosome and bin columns
            cols = ['Gene1', 'Gene2', 'Correlation']
            return pd.DataFrame(columns=cols)

        # The 'Correlation_Bin' column is expected to exist in the dataframe at this point
        # after calculate_and_assign_bins_and_min_size has been called on it.

        sampled_df = pd.DataFrame()
        # Iterate through all unique bin labels in the current dataframe
        # Use .dropna().unique() to handle potential NaN values in the bin column
        # Also sort the unique labels to ensure consistent iteration order
        # Only iterate through the bins that actually exist in the current dataframe
        for bin_label in sorted(df['Correlation_Bin'].dropna().unique()):
             # Ensure the bin_data is not empty before attempting to sample
             bin_data = df[df['Correlation_Bin'] == bin_label]
             if not bin_data.empty:
                # Sample up to min_samples_per_bin for this specific set
                # Ensure the number of samples does not exceed the available data in the bin
                n_samples = min(len(bin_data), min_samples_per_bin)
                if n_samples > 0:
                    sampled_bin = bin_data.sample(n=n_samples, replace=False, random_state=42) # Use random_state for reproducibility
                    sampled_df = pd.concat([sampled_df, sampled_bin])


        # Drop the Correlation bin, and temporary chromosome columns
        cols_to_drop = ['Correlation_Bin', 'Gene1_chr', 'Gene2_chr']
        sampled_df = sampled_df.drop(columns=[col for col in cols_to_drop if col in sampled_df.columns], errors='ignore')
        return sampled_df

    print("Sampling from bins for train, validation, and test sets using respective min bin sizes...")
    # Pass the calculated minimum bin size for each set
    # The 'Correlation_Bin' column is expected to exist in train_set, val_set, and test_set
    # after the min bin size calculations.
    sampled_train_set = sample_from_bins(train_set, min_train_bin_size)
    sampled_val_set = sample_from_bins(val_set, min_val_bin_size)
    sampled_test_set = sample_from_bins(test_set, min_test_bin_size)

    print(f"Sampled data: Train={len(sampled_train_set)} pairs, Validation={len(sampled_val_set)} pairs, Test={len(sampled_test_set)} pairs.")

    # Combine sampled dataframes
    final_coexp_df = pd.concat([sampled_train_set, sampled_val_set, sampled_test_set])

    # Temporary columns were dropped during sampling.

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