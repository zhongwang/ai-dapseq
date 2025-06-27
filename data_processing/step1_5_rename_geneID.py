import pandas as pd
import sys
import os

def clean_gene_id(gene_id):
    """
    Cleans the gene_id string from format like AT1G01020.1.Araport11.447_promoter
    to AT1G01020.
    """
    if isinstance(gene_id, str):
        # Split by '.' and take the first part
        return gene_id.split('.')[0]
    return gene_id # Return as is if not a string

def main():
    if len(sys.argv) != 2:
        print("Usage: python step1_5_rename_geneid.py <input_tsv_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        sys.exit(1)

    try:
        # Read the TSV file
        df = pd.read_csv(input_file_path, sep='\t')

        # Check if 'gene_id' column exists
        if 'gene_id' not in df.columns:
            print(f"Error: 'gene_id' column not found in {input_file_path}")
            sys.exit(1)

        # Apply the cleaning function to the 'gene_id' column
        df['gene_id'] = df['gene_id'].apply(clean_gene_id)

        # Construct the output file path
        base, ext = os.path.splitext(input_file_path)
        output_file_path = f"{base}_cleaned{ext}"

        # Save the processed data to a new TSV file
        df.to_csv(output_file_path, sep='\t', index=False)

        print(f"Successfully processed {input_file_path}. Cleaned data saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
