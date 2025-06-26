import pandas as pd
import numpy as np
import os

# Define file paths
input_bed = "/global/scratch/users/sallyliao2027/aidapseq/test_data/Arabidopsis_thaliana_Col-0_promoter_regions_short_geneID.bed"
output_bed = "/global/scratch/users/sallyliao2027/aidapseq/test_data/selected_promoters_by_chromosome.bed"

# Define BED columns (assuming standard BED format with geneID in the 4th column)
# Ref: https://genome.ucsc.edu/FAQ/FAQmr#convert1
# Chromosome, Start, End, Name, Score, Strand
bed_columns = ['chrom', 'start', 'end', 'name', 'score', 'strand']

try:
    # Read the BED file into a pandas DataFrame
    # Using sep='\t' for tab-separated values
    # header=None because BED files typically don't have a header row
    data = pd.read_csv(input_bed, sep='\t', header=None, names=bed_columns)

    print(f"Successfully read {input_bed}. Total promoters: {len(data)}")

    # Group by chromosome and sample 10 promoters from each
    selected_promoters = []
    for chrom, group in data.groupby('chrom'):
        if len(group) <= 10:
            # If a chromosome has 10 or fewer promoters, select all of them
            selected_promoters.append(group)
            print(f"Chromosome {chrom}: Selected all {len(group)} promoters (<= 10).")
        else:
            # Otherwise, randomly sample 10 promoters
            sampled_group = group.sample(n=10, random_state=42) # Using random_state for reproducibility
            selected_promoters.append(sampled_group)
            print(f"Chromosome {chrom}: Sampled 10 promoters from {len(group)}.")

    # Concatenate the sampled promoters into a single DataFrame
    if selected_promoters:
        selected_df = pd.concat(selected_promoters)
        print(f"Total selected promoters: {len(selected_df)}")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_bed)
        os.makedirs(output_dir, exist_ok=True)

        # Save the selected promoters to a new BED file
        # index=False to prevent writing the DataFrame index as a column
        # sep='\t' for tab-separated values
        # header=False because BED files typically don't have a header row
        selected_df.to_csv(output_bed, sep='\t', index=False, header=False)

        print(f"Successfully saved selected promoters to {output_bed}")
    else:
        print("No promoters were selected.")


except FileNotFoundError:
    print(f"Error: Input file not found at {input_bed}")
except Exception as e:
    print(f"An error occurred: {e}")