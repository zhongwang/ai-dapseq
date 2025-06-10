import os
import pandas as pd
import numpy as np
import random

# Placeholder paths - replace with actual paths
COEXPRESSION_TSV = "../path/to/coexpressed_gene_pairs.tsv" # Input TSV with known co-expressed pairs
TF_VOCABULARY_DIR = "./tf_vocabulary_features" # Directory containing TF vocabulary features (.npy files)
OUTPUT_FILE = "./coexpression_dataset.tsv" # Output TSV for the final labeled dataset

# Negative sampling ratio (e.g., 1:1 means equal number of negatives as positives)
NEGATIVE_SAMPLING_RATIO = 1 # Can be tuned (e.g., 1, 3, 5, 10)

def load_coexpressed_pairs(tsv_path):
    """
    Loads known co-expressed gene pairs from a TSV file.
    Assumes the TSV has at least 'Gene1_ID' and 'Gene2_ID' columns.
    Returns a set of frozensets for unique, order-independent pairs.
    Also returns a list of all unique gene IDs involved in positive pairs.
    """
    coexpressed_pairs = set()
    all_positive_genes = set()
    print(f"Loading known co-expressed gene pairs from {tsv_path}")
    try:
        # Assuming columns are named 'Gene1_ID' and 'Gene2_ID'
        # Adjust column names if necessary based on the actual file
        coexp_df = pd.read_csv(tsv_path, sep='\t')
        print(f"Loaded {len(coexp_df)} potential co-expressed pairs.")

        # Process pairs: ensure order doesn't matter and store as frozensets
        for index, row in coexp_df.iterrows():
            gene1 = str(row['Gene1_ID'])
            gene2 = str(row['Gene2_ID'])
            if gene1 != gene2: # Exclude self-pairs
                pair = frozenset({gene1, gene2})
                coexpressed_pairs.add(pair)
                all_positive_genes.add(gene1)
                all_positive_genes.add(gene2)

        print(f"Identified {len(coexpressed_pairs)} unique co-expressed pairs.")
        print(f"Involved {len(all_positive_genes)} unique genes in positive pairs.")
        return coexpressed_pairs, list(all_positive_genes)

    except FileNotFoundError:
        print(f"Error: Co-expression TSV file not found at {tsv_path}")
        return set(), []
    except KeyError as e:
        print(f"Error: Missing expected column in TSV: {e}. Please ensure the file has 'Gene1_ID' and 'Gene2_ID' columns.")
        return set(), []
    except Exception as e:
        print(f"Error loading or processing co-expression TSV: {e}")
        return set(), []

def get_available_genes(tf_vocabulary_dir):
    """
    Gets a list of all gene IDs for which TF vocabulary features are available.
    Assumes file names are like GENE_ID_tf_vocabulary.npy.
    """
    available_genes = []
    print(f"Scanning directory for available TF vocabulary features: {tf_vocabulary_dir}")
    if not os.path.exists(tf_vocabulary_dir):
        print(f"Error: TF vocabulary directory not found: {tf_vocabulary_dir}")
        return available_genes

    npy_files = [f for f in os.listdir(tf_vocabulary_dir) if f.endswith("_tf_vocabulary.npy")]
    available_genes = [f.replace("_tf_vocabulary.npy", "") for f in npy_files]
    print(f"Found features for {len(available_genes)} available genes.")
    return available_genes

def generate_negative_samples(coexpressed_pairs, available_genes, num_positive_samples, ratio):
    """
    Generates negative gene pairs by randomly sampling from available genes.
    Ensures generated negatives are not in the positive set.
    """
    negative_samples = []
    num_negative_to_generate = int(num_positive_samples * ratio)
    print(f"Generating {num_negative_to_generate} negative samples...")

    # Create a list of all possible gene pairs from available genes
    # This can be very large, so we'll sample iteratively instead of generating all.
    available_genes_list = list(available_genes)
    num_available_genes = len(available_genes_list)

    if num_available_genes < 2:
        print("Warning: Not enough available genes to generate negative samples.")
        return negative_samples

    attempts = 0
    max_attempts = num_negative_to_generate * 10 # Prevent infinite loops

    while len(negative_samples) < num_negative_to_generate and attempts < max_attempts:
        # Randomly pick two distinct genes
        gene1, gene2 = random.sample(available_genes_list, 2)

        # Ensure the pair is not in the positive set
        pair = frozenset({gene1, gene2})
        if pair not in coexpressed_pairs:
            negative_samples.append((gene1, gene2))
        attempts += 1

    if attempts >= max_attempts:
        print(f"Warning: Reached max attempts ({max_attempts}) for negative sampling. Generated {len(negative_samples)} negatives.")
    else:
         print(f"Generated {len(negative_samples)} negative samples.")

    return negative_samples

def create_labeled_dataset(coexpressed_pairs, negative_samples):
    """
    Combines positive and negative samples into a labeled dataset DataFrame.
    """
    dataset = []
    # Add positive samples
    for pair in coexpressed_pairs:
        gene1, gene2 = tuple(pair) # Convert frozenset back to tuple
        dataset.append({'Gene1_ID': gene1, 'Gene2_ID': gene2, 'Label': 1})

    # Add negative samples
    for gene1, gene2 in negative_samples:
        dataset.append({'Gene1_ID': gene1, 'Gene2_ID': gene2, 'Label': 0})

    # Create DataFrame and shuffle
    dataset_df = pd.DataFrame(dataset)
    dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle rows

    print(f"Created labeled dataset with {len(dataset_df)} pairs ({len(coexpressed_pairs)} positives, {len(negative_samples)} negatives).")
    return dataset_df

def save_dataset(dataset_df, output_path):
    """
    Saves the labeled dataset DataFrame to a TSV file.
    """
    print(f"Saving labeled dataset to {output_path}")
    try:
        dataset_df.to_csv(output_path, sep='\t', index=False)
        print("Saving complete.")
    except Exception as e:
        print(f"Error saving dataset file: {e}")

if __name__ == "__main__":
    # Step 1: Load known co-expressed gene pairs
    coexpressed_pairs, positive_genes_list = load_coexpressed_pairs(COEXPRESSION_TSV)

    if not coexpressed_pairs:
        print("No co-expressed pairs loaded. Cannot proceed.")
    else:
        # Step 2: Get list of all genes with available TF vocabulary features
        available_genes = get_available_genes(TF_VOCABULARY_DIR)

        if not available_genes:
            print("No available genes with TF vocabulary features found. Cannot proceed.")
        else:
            # Filter positive pairs to only include genes for which we have features
            # This is important if the co-expression TSV contains genes not in our feature set.
            filtered_coexpressed_pairs = set()
            genes_with_features_set = set(available_genes)
            for pair in coexpressed_pairs:
                gene1, gene2 = tuple(pair)
                if gene1 in genes_with_features_set and gene2 in genes_with_features_set:
                    filtered_coexpressed_pairs.add(pair)
                else:
                    # print(f"Warning: Skipping positive pair ({gene1}, {gene2}) as features are missing for one or both genes.")
                    pass # Optional: print warning

            print(f"Using {len(filtered_coexpressed_pairs)} positive pairs with available features.")

            if not filtered_coexpressed_pairs:
                 print("No positive pairs with available features after filtering. Cannot proceed.")
            else:
                # Step 3: Generate negative samples from available genes
                negative_samples = generate_negative_samples(
                    filtered_coexpressed_pairs,
                    available_genes,
                    len(filtered_coexpressed_pairs), # Match number of positives for 1:1 ratio initially
                    NEGATIVE_SAMPLING_RATIO
                )

                if not negative_samples:
                    print("No negative samples generated. Cannot proceed.")
                else:
                    # Step 4: Create the final labeled dataset
                    labeled_dataset_df = create_labeled_dataset(filtered_coexpressed_pairs, negative_samples)

                    # Step 5: Save the dataset
                    if not labeled_dataset_df.empty:
                        # Ensure output directory exists
                        output_dir = os.path.dirname(OUTPUT_FILE)
                        if output_dir and not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            print(f"Created output directory: {output_dir}")

                        save_dataset(labeled_dataset_df, OUTPUT_FILE)
                    else:
                        print("Labeled dataset is empty. Output file not created.")