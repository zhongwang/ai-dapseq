import pandas as pd
import sys
from collections import Counter
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count within-chromosome gene pairs using a correlation file and a gene-to-chromosome map file.')
    parser.add_argument('--correlation_file', help='Path to the input TSV file with columns Gene1, Gene2, and Correlation.')
    parser.add_argument('--gene_chromosome_map_file', help='Path to the TSV file mapping gene IDs to chromosomes. Must have columns for Gene ID and Chromosome.')
    args = parser.parse_args()

    correlation_file_path = args.correlation_file
    gene_chromosome_map_file_path = args.gene_chromosome_map_file

    try:
        # Read the gene-chromosome mapping file
        df_map = pd.read_csv(gene_chromosome_map_file_path, sep='\t')

        # Assuming the map file has columns like 'GeneID' and 'Chromosome'
        # You might need to adjust these column names based on your actual file
        gene_id_col_map = 'gene_id' # *** IMPORTANT: Change this to the actual Gene ID column name in your map file ***
        chromosome_col_map = 'chromosome' # *** IMPORTANT: Change this to the actual Chromosome column name in your map file ***

        if gene_id_col_map not in df_map.columns or chromosome_col_map not in df_map.columns:
            print(f"Error: Gene-chromosome map file must contain \'{gene_id_col_map}\' and \'{chromosome_col_map}\' columns. Please check the script and update the column names if necessary.")
            sys.exit(1)
            
        # Create a dictionary for quick lookup: GeneID -> Chromosome
        gene_to_chromosome_map = pd.Series(df_map[chromosome_col_map].values, index=df_map[gene_id_col_map]).to_dict()

        # Read the correlation file
        df_corr = pd.read_csv(correlation_file_path, sep='\t')

        if 'Gene1' not in df_corr.columns or 'Gene2' not in df_corr.columns:
            print("Error: Correlation file must contain 'Gene1' and 'Gene2' columns.")
            sys.exit(1)

        # Map Gene1 and Gene2 to their chromosomes using the map file
        # Use .get(gene_id, None) to handle potential gene IDs not found in the map file
        df_corr['Chr1'] = df_corr['Gene1'].apply(lambda gene_id: gene_to_chromosome_map.get(gene_id, None))
        df_corr['Chr2'] = df_corr['Gene2'].apply(lambda gene_id: gene_to_chromosome_map.get(gene_id, None))

        # Filter out rows where chromosome mapping failed or chromosomes are different
        df_within_chrom = df_corr.dropna(subset=['Chr1', 'Chr2'])
        df_within_chrom = df_within_chrom[df_within_chrom['Chr1'] == df_within_chrom['Chr2']].copy()

        # Filter for correlation values greater than 0.48650343
        # Ensure 'Correlation' column exists and is numeric
        if 'Correlation' not in df_within_chrom.columns:
             print("Warning: 'Correlation' column not found in the correlation file. Cannot filter by correlation value.")
             df_filtered_corr = df_within_chrom # Proceed without correlation filter
        else:
            try:
                df_within_chrom['Correlation'] = pd.to_numeric(df_within_chrom['Correlation'])
                df_filtered_corr = df_within_chrom[df_within_chrom['Correlation'] > 0.48650343].copy()
            except ValueError:
                print("Warning: 'Correlation' column contains non-numeric values. Cannot filter by correlation value.")
                df_filtered_corr = df_within_chrom # Proceed without correlation filter

        # Count the occurrences of each chromosome for the filtered within-chromosome pairs
        if not df_filtered_corr.empty:
            # We only need to count one of the columns, as Chr1 and Chr2 are the same
            within_chrom_counts = df_filtered_corr['Chr1'].value_counts().sort_index()

            # Print the results
            print("Within-Chromosome Pair Counts (Correlation > 0.48650343):")
            for chrom, count in within_chrom_counts.items():
                print(f"{chrom}: {count}")
        else:
            print("No within-chromosome pairs found with correlation > 0.48650343.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
