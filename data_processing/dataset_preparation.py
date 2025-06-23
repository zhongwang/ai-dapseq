import argparse
import pandas as pd
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_negative_samples(positive_pairs_df, all_genes, num_negative_samples):
    """
    Creates a set of negative samples for training.

    Args:
        positive_pairs_df (pd.DataFrame): DataFrame of positive pairs.
        all_genes (list): A list of all unique gene IDs.
        num_negative_samples (int): The number of negative samples to generate.

    Returns:
        pd.DataFrame: A DataFrame of negatively sampled gene pairs.
    """
    positive_pairs_set = set(map(tuple, positive_pairs_df[['gene1_id', 'gene2_id']].apply(sorted, axis=1).values))
    
    negative_pairs = set()
    while len(negative_pairs) < num_negative_samples:
        gene1, gene2 = np.random.choice(all_genes, 2, replace=False)
        pair = tuple(sorted((gene1, gene2)))
        
        if pair not in positive_pairs_set and pair not in negative_pairs:
            negative_pairs.add(pair)
            
    return pd.DataFrame(list(negative_pairs), columns=['gene1_id', 'gene2_id'])

def main():
    """
    Main function to prepare the final co-expression dataset.
    """
    parser = argparse.ArgumentParser(description="Prepare co-expression dataset with negative sampling.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input TSV file of positive co-expressed pairs.")
    parser.add_argument("-g", "--genes_dir", required=True, help="Directory containing the processed .npy files, used to get the list of all genes.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the train, validation, and test sets.")
    parser.add_argument("--neg_ratio", type=float, default=1.0, help="Ratio of negative to positive samples.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of the dataset to use for validation.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion of the dataset to use for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Loading positive pairs from {args.input}")
    positive_df = pd.read_csv(args.input, sep='\t')
    positive_df['label'] = 1
    # Assuming columns are named 'gene1_id' and 'gene2_id' or similar
    positive_df.columns = ['gene1_id', 'gene2_id', 'label']


    logging.info(f"Getting all gene IDs from {args.genes_dir}")
    all_genes = [os.path.basename(f).replace('.npy', '') for f in os.listdir(args.genes_dir) if f.endswith('.npy')]
    
    num_positive = len(positive_df)
    num_negative = int(num_positive * args.neg_ratio)
    logging.info(f"Generating {num_negative} negative samples.")
    negative_df = create_negative_samples(positive_df, all_genes, num_negative)
    negative_df['label'] = 0

    # Combine and shuffle
    full_dataset = pd.concat([positive_df, negative_df]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    logging.info(f"Full dataset created with {len(full_dataset)} pairs ({num_positive} positive, {num_negative} negative).")

    # Gene-disjoint split
    unique_genes = pd.unique(full_dataset[['gene1_id', 'gene2_id']].values.ravel('K'))
    np.random.shuffle(unique_genes)
    
    test_genes_size = int(len(unique_genes) * args.test_size)
    val_genes_size = int(len(unique_genes) * args.val_size)
    
    test_genes = set(unique_genes[:test_genes_size])
    val_genes = set(unique_genes[test_genes_size : test_genes_size + val_genes_size])
    train_genes = set(unique_genes[test_genes_size + val_genes_size:])

    logging.info(f"Splitting genes: {len(train_genes)} train, {len(val_genes)} validation, {len(test_genes)} test.")

    train_df = full_dataset[full_dataset.gene1_id.isin(train_genes) & full_dataset.gene2_id.isin(train_genes)]
    val_df = full_dataset[full_dataset.gene1_id.isin(val_genes) & full_dataset.gene2_id.isin(val_genes)]
    test_df = full_dataset[full_dataset.gene1_id.isin(test_genes) & full_dataset.gene2_id.isin(test_genes)]

    logging.info(f"Final split sizes: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test pairs.")

    # Save datasets
    train_df.to_csv(os.path.join(args.output_dir, 'train_pairs.tsv'), sep='\t', index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'val_pairs.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_pairs.tsv'), sep='\t', index=False)

    logging.info(f"Datasets saved to {args.output_dir}")

if __name__ == "__main__":
    main()