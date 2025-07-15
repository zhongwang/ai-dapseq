import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from tqdm import tqdm
import csv
from scipy.stats import pearsonr

# It's good practice to import the model from its source file
# Add model directory to Python path to allow direct import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.siamese_transformer import SiameseGeneTransformer

# --- Distributed Training Setup ---
def setup():
    """
    Initializes the distributed process group.
    Assumes `torchrun` sets the necessary environment variables.
    """
    dist.init_process_group("nccl")

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

# --- Logging Setup ---
def setup_logging(rank):
    """Sets up logging for each process."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Custom PyTorch Dataset ---
class GenePairDataset(Dataset):
    """
    Dataset for loading gene pairs, their correlation, and corresponding
    feature vectors based on chromosome-based splits.
    """
    def __init__(self, pairs_df, gene_chromosome_map, feature_dir, mode='train'):
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

        # Load feature vectors (assuming they are saved as .npy files)
        try:
            vec1 = np.load(os.path.join(self.feature_dir, f"{gene1_id}.npy"))
            vec2 = np.load(os.path.join(self.feature_dir, f"{gene2_id}.npy"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Feature vector not found for a gene in pair ({gene1_id}, {gene2_id}). Error: {e}")

        return {
            'vec1': torch.from_numpy(vec1).float(),
            'vec2': torch.from_numpy(vec2).float(),
            'correlation': correlation
        }

def collate_fn(batch):
    """
    Custom collate function to handle padding for variable-length sequences.
    """
    vec1s = [item['vec1'] for item in batch]
    vec2s = [item['vec2'] for item in batch]
    correlations = torch.stack([item['correlation'] for item in batch])

    # Pad sequences to the max length in the batch
    padded_vec1s = torch.nn.utils.rnn.pad_sequence(vec1s, batch_first=True, padding_value=0)
    padded_vec2s = torch.nn.utils.rnn.pad_sequence(vec2s, batch_first=True, padding_value=0)
    
    # Create padding masks: True for padded elements
    mask1 = (padded_vec1s.sum(dim=-1) == 0)
    mask2 = (padded_vec2s.sum(dim=-1) == 0)

    return {
        'promoter_sequence_A': padded_vec1s,
        'promoter_sequence_B': padded_vec2s,
        'key_padding_mask_A': mask1,
        'key_padding_mask_B': mask2,
        'labels': correlations.unsqueeze(1) # Ensure shape is (batch_size, 1)
    }

# --- Main Training Function ---
def train_model(args):
    """Main function for training, validation, and testing."""
    # DDP setup: Get rank and world size from environment variables
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    setup()
    setup_logging(rank)

    # --- Data Loading ---
    if rank == 0:
        logging.info("Loading data...")
        # Setup CSV log file
        with open(args.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_pearson_r', 'val_mae'])
            
    all_pairs_df = pd.read_csv(args.pairs_file, sep='\t')
    gene_chrom_map = pd.read_csv(args.gene_info_file, sep='\t')
    
    # Ensure required columns exist
    required_pair_cols = ['Gene1', 'Gene2', 'Correlation']
    if not all(col in all_pairs_df.columns for col in required_pair_cols):
        raise ValueError(f"Pairs file must contain columns: {required_pair_cols}")
        
    required_gene_cols = ['gene_id', 'chromosome']
    if not all(col in gene_chrom_map.columns for col in required_gene_cols):
        raise ValueError(f"Gene info file must contain columns: {required_gene_cols}")

    # Create datasets for each split
    train_dataset = GenePairDataset(all_pairs_df, gene_chrom_map, args.feature_dir, mode='train')
    val_dataset = GenePairDataset(all_pairs_df, gene_chrom_map, args.feature_dir, mode='val')
    test_dataset = GenePairDataset(all_pairs_df, gene_chrom_map, args.feature_dir, mode='test')

    if rank == 0:
        logging.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
             logging.warning("One or more dataset splits is empty. Check chromosome filtering logic and data.")

    # --- Dataloaders with DistributedSampler ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=args.num_workers)
    
    # Validation sampler needs to be distributed as well
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)
    
    # Test dataset is only used on rank 0, so it doesn't need a distributed sampler
    test_loader = None
    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # --- Model Initialization ---
    # Infer input_feature_dim from a sample
    sample_vec = train_dataset[0]['vec1']
    input_feature_dim = sample_vec.shape[1]
    
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
    ).to(local_rank)

    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # --- Training Loop ---
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch) # Ensure shuffling is different each epoch
        ddp_model.train()
        total_train_loss = 0
        
        # Add tqdm for progress bar, only on the main rank
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", disable=(rank != 0))
        for batch in train_loader_tqdm:
            # Move data to the correct device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(local_rank)
            
            optimizer.zero_grad()
            
            model_inputs = {
                'promoter_sequence_A': batch['promoter_sequence_A'],
                'promoter_sequence_B': batch['promoter_sequence_B'],
                'key_padding_mask_A': batch['key_padding_mask_A'],
                'key_padding_mask_B': batch['key_padding_mask_B']
            }

            outputs = ddp_model(**model_inputs) # Pass dictionary directly
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Loop ---
        ddp_model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc="Validation", disable=(rank != 0))
            for batch in val_loader_tqdm:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(local_rank)
                
                model_inputs = {
                    'promoter_sequence_A': batch['promoter_sequence_A'],
                    'promoter_sequence_B': batch['promoter_sequence_B'],
                    'key_padding_mask_A': batch['key_padding_mask_A'],
                    'key_padding_mask_B': batch['key_padding_mask_B']
                }

                outputs = ddp_model(**model_inputs)
                loss = criterion(outputs, batch['labels'])
                total_val_loss += loss.item()

                # Gather predictions and labels from all GPUs
                all_val_preds.append(outputs.cpu())
                all_val_labels.append(batch['labels'].cpu())

        # Combine predictions and labels from all batches
        all_val_preds = torch.cat(all_val_preds)
        all_val_labels = torch.cat(all_val_labels)
        
        # Collect results from all processes
        if world_size > 1:
            # Create tensors to store gathered data on each process
            gathered_preds = [torch.zeros_like(all_val_preds) for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(all_val_labels) for _ in range(world_size)]
            
            dist.all_gather(gathered_preds, all_val_preds)
            dist.all_gather(gathered_labels, all_val_labels)

            if rank == 0:
                all_val_preds = torch.cat(gathered_preds)
                all_val_labels = torch.cat(gathered_labels)
        
        avg_val_loss = total_val_loss / len(val_loader)

        # Log, save model, and check for early stopping only on the main process
        stop_signal = torch.tensor(0.0, device=local_rank)
        if rank == 0:
            # Calculate metrics on rank 0 with all data
            val_mae = nn.L1Loss()(all_val_preds, all_val_labels).item()
            val_pearson_r, _ = pearsonr(all_val_preds.numpy().flatten(), all_val_labels.numpy().flatten())
            
            logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Pearson R: {val_pearson_r:.4f} | Val MAE: {val_mae:.4f}")

            # Log to CSV
            with open(args.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_pearson_r, val_mae])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(ddp_model.module.state_dict(), args.save_path)
                logging.info(f"Model saved to {args.save_path} with new best validation loss.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                logging.info(f"Early stopping triggered after {args.early_stopping_patience} epochs with no improvement.")
                stop_signal.fill_(1.0)
        
        # Broadcast the stop signal to all processes. If it's 1, everyone stops.
        dist.broadcast(stop_signal, src=0)
        
        if stop_signal.item() == 1.0:
            break

    # --- Final Test Evaluation (on rank 0) ---
    if rank == 0:
        logging.info("Starting final evaluation on the test set...")
        # Load the best model onto the correct device for the main process
        # Use map_location to load onto the CPU first, then move to the GPU
        model.load_state_dict(torch.load(args.save_path, map_location='cpu'))
        model.to(local_rank)
        model.eval()
        
        total_test_loss = 0
        with torch.no_grad():
            test_loader_tqdm = tqdm(test_loader, desc="Final Testing")
            for batch in test_loader_tqdm:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(local_rank)
                
                outputs = model(**batch)
                loss = criterion(outputs, batch['labels'])
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader_final)
        logging.info(f"Final Test Loss: {avg_test_loss:.4f}")

    cleanup()

# --- Argument Parser ---
def main():
    parser = argparse.ArgumentParser(description="Train a Siamese Transformer for Gene Co-expression Prediction using torchrun.")
    
    # Data and Path Arguments
    parser.add_argument('--pairs_file', type=str, required=True, help="Path to the TSV file with gene pairs and correlation.")
    parser.add_argument('--feature_dir', type=str, required=True, help="Directory containing gene feature vectors (.npy files).")
    parser.add_argument('--gene_info_file', type=str, required=True, help="Path to the TSV file mapping gene IDs to chromosomes.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the final trained model.")
    parser.add_argument('--log_file', type=str, default='training_log.csv', help="Path to save the training log CSV.")
    
    # Model Hyperparameters
    parser.add_argument('--d_model', type=int, default=256, help="Internal dimension of the transformer model.")
    parser.add_argument('--nhead', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--num_encoder_layers', type=int, default=4, help="Number of transformer encoder layers.")
    parser.add_argument('--dim_feedforward', type=int, default=1024, help="Dimension of the feed-forward network.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate for the transformer.")
    parser.add_argument('--aggregation_method', type=str, default='cls', choices=['cls', 'mean'], help="Aggregation method for promoter embeddings.")
    parser.add_argument('--max_seq_len', type=int, default=2501, help="Maximum sequence length for positional encoding.")
    parser.add_argument('--regression_hidden_dim', type=int, default=128, help="Hidden dimension of the regression head.")
    parser.add_argument('--regression_dropout', type=float, default=0.3, help="Dropout rate for the regression head.")
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per GPU.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for the dataloader.")
    parser.add_argument('--early_stopping_patience', type=int, default=3, help="Epochs with no improvement to wait before stopping. 0 to disable.")

    args = parser.parse_args()

    # WORLD_SIZE is set by torchrun. If it's not set, we can't proceed.
    if 'WORLD_SIZE' not in os.environ:
        sys.exit("This script is designed to be run with 'torchrun'.\n"
                 "Please launch it using: torchrun --nproc_per_node=NUM_GPUS train_siamese_transformer.py [ARGS]")

    train_model(args)

if __name__ == '__main__':
    main()
