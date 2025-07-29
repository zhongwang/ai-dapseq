
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm

# Add model directory to Python path to allow direct import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.siamese_transformer import SiameseGeneTransformer


# --- Custom PyTorch Dataset (adapted from train_siamese_transformer.py) ---
class GenePairDataset(Dataset):
    """
    Dataset for loading gene pairs, their correlation, and corresponding
    feature vectors based on chromosome-based splits.
    """
    def __init__(self, pairs_df, gene_chromosome_map, feature_dir, mode='test'):
        self.feature_dir = feature_dir
        
        # Map gene_id to chromosome
        self.gene_to_chrom = pd.Series(gene_chromosome_map.chromosome.values, index=gene_chromosome_map.gene_id).to_dict()

        # Filter pairs based on chromosome assignment for train/val/test splits
        def get_chrom(gene_id):
            return self.gene_to_chrom.get(gene_id)

        pairs_df['Chr1'] = pairs_df['Gene1'].apply(get_chrom)
        pairs_df['Chr2'] = pairs_df['Gene2'].apply(get_chrom)

        if mode == 'train':
            # Training: Not on Chr2 or Chr4
            self.pairs = pairs_df[
                (~pairs_df['Chr1'].isin(['Chr2', 'Chr4'])) &
                (~pairs_df['Chr2'].isin(['Chr2', 'Chr4']))
            ].reset_index(drop=True)
        elif mode == 'val':
            # Validation: Both on Chr4
            self.pairs = pairs_df[
                (pairs_df['Chr1'] == 'Chr4') & (pairs_df['Chr2'] == 'Chr4')
            ].reset_index(drop=True)
        elif mode == 'test':
            # Test: Both on Chr2
            self.pairs = pairs_df[
                (pairs_df['Chr1'] == 'Chr2') & (pairs_df['Chr2'] == 'Chr2')
            ].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]
        gene1_id, gene2_id = pair['Gene1'], pair['Gene2']
        correlation = torch.tensor(pair['Correlation'], dtype=torch.float)

        # Load feature vectors
        try:
            vec1 = np.load(os.path.join(self.feature_dir, f"{gene1_id}.npy"))
            vec2 = np.load(os.path.join(self.feature_dir, f"{gene2_id}.npy"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Feature vector not found for a gene in pair ({gene1_id}, {gene2_id}). Error: {e}")

        return {
            'vec1': torch.from_numpy(vec1).float(),
            'vec2': torch.from_numpy(vec2).float(),
            'correlation': correlation,
            'gene1_id': gene1_id,
            'gene2_id': gene2_id
        }

def collate_fn(batch):
    """
    Custom collate function to handle padding for variable-length sequences.
    """
    vec1s = [item['vec1'] for item in batch]
    vec2s = [item['vec2'] for item in batch]
    correlations = torch.stack([item['correlation'] for item in batch])
    gene1_ids = [item['gene1_id'] for item in batch]
    gene2_ids = [item['gene2_id'] for item in batch]

    padded_vec1s = torch.nn.utils.rnn.pad_sequence(vec1s, batch_first=True, padding_value=0)
    padded_vec2s = torch.nn.utils.rnn.pad_sequence(vec2s, batch_first=True, padding_value=0)
    
    mask1 = (padded_vec1s.sum(dim=-1) == 0)
    mask2 = (padded_vec2s.sum(dim=-1) == 0)

    return {
        'promoter_sequence_A': padded_vec1s,
        'promoter_sequence_B': padded_vec2s,
        'key_padding_mask_A': mask1,
        'key_padding_mask_B': mask2,
        'labels': correlations.unsqueeze(1),
        'gene1_ids': gene1_ids,
        'gene2_ids': gene2_ids
    }

def validate_model(args):
    """Main function for validation."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    print("Loading data...")
    all_pairs_df = pd.read_csv(args.pairs_file, sep='\t')
    gene_chrom_map = pd.read_csv(args.gene_info_file, sep='\t')

    # Create test dataset
    test_dataset = GenePairDataset(all_pairs_df, gene_chrom_map, args.feature_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    if len(test_dataset) == 0:
        print("Test dataset is empty. Check chromosome filtering and data files.")
        return

    print(f"Test dataset created with {len(test_dataset)} samples.")

    # --- Model Initialization ---
    try:
        sample_vec = test_dataset[0]['vec1']
        input_feature_dim = sample_vec.shape[1]
    except IndexError:
        print("Could not get a sample from the dataset to infer input_feature_dim.")
        return

    model = SiameseGeneTransformer(
        input_feature_dim=input_feature_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        aggregation_method=args.aggregation_method,
        max_seq_len=args.max_seq_len,
        regression_hidden_dim=args.regression_hidden_dim,
        regression_dropout=args.regression_dropout
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.eval()
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    # --- Predictions ---
    all_preds = []
    all_labels = []
    all_gene1s = []
    all_gene2s = []

    print("Making predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(DEVICE)
            
            model_inputs = {
                'promoter_sequence_A': batch['promoter_sequence_A'],
                'promoter_sequence_B': batch['promoter_sequence_B'],
                'key_padding_mask_A': batch['key_padding_mask_A'],
                'key_padding_mask_B': batch['key_padding_mask_B']
            }
            
            outputs = model(**model_inputs)
            
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(batch['labels'].squeeze().cpu().numpy())
            all_gene1s.extend(batch['gene1_ids'])
            all_gene2s.extend(batch['gene2_ids'])

    # --- Save Predictions and Calculate Metrics ---
    if not all_preds:
        print("No predictions were made.")
        return
        
    predictions_df = pd.DataFrame({
        'Gene1': all_gene1s,
        'Gene2': all_gene2s,
        'Actual_Correlation': all_labels,
        'Predicted_Correlation': all_preds
    })
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    predictions_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    pearson_r, _ = pearsonr(all_labels, all_preds)
    
    print("\n--- Test Set Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Pearson Correlation Coefficient: {pearson_r:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Validate a Siamese Transformer for Gene Co-expression Prediction.")
    
    # Data and Path Arguments
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model state_dict (.pth file).")
    parser.add_argument('--pairs_file', type=str, required=True, help="Path to the TSV file with all gene pairs and correlations.")
    parser.add_argument('--feature_dir', type=str, required=True, help="Directory containing gene feature vectors (.npy files).")
    parser.add_argument('--gene_info_file', type=str, required=True, help="Path to the TSV file mapping gene IDs to chromosomes.")
    parser.add_argument('--output_csv', type=str, default='test_predictions.csv', help="Path to save the output CSV with predictions.")
    
    # Model Hyperparameters (must match the trained model)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--aggregation_method', type=str, default='cls', choices=['cls', 'mean'])
    parser.add_argument('--max_seq_len', type=int, default=2501)
    parser.add_argument('--regression_hidden_dim', type=int, default=128)
    parser.add_argument('--regression_dropout', type=float, default=0.3)
    
    # Validation Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for validation.")

    args = parser.parse_args()
    validate_model(args)

if __name__ == '__main__':
    main()
