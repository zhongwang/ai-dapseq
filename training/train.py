import torch
import torch.nn as nn # Added
import torch.optim as optim # Added
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import time # Added
import copy # Added
import csv # Added for logging
from scipy.stats import pearsonr # Added for validation metric
from sklearn.metrics import mean_absolute_error # Added for validation metric
import torch.cuda.amp as amp # Added for mixed precision training
import torch.distributed as dist # Added for DDP
from torch.utils.data.distributed import DistributedSampler # Added for DDP
from torch.nn.parallel import DistributedDataParallel # Added for DDP
import argparse # Added for DDP rank/world size
from tqdm import tqdm # Added for progress bar

# Attempt to import the model from the parent directory's 'model' folder
try:
    from ..model.siamese_transformer import SiameseGeneTransformer
except ImportError:
    # Fallback for running the script directly from the 'training' directory for testing
    # This assumes 'model' and 'training' are sibling directories.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from model.siamese_transformer import SiameseGeneTransformer


# --- Configuration (Placeholder paths - adjust as needed) ---
GENE_PAIRS_TSV_PATH = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/final_coexpressed_regression_15bins.txt"
FEATURE_VECTOR_DIR = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/feature_vectors/"
MODEL_SAVE_PATH = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/best_siamese_model.pth"
GENE_CHROMOSOME_MAPPING_PATH = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/promoter_sequences.txt"

# --- Model Hyperparameters (Example values, tune as needed) ---
INPUT_FEATURE_DIM = 248      # Example: 4 (DNA one-hot) + 80 (TF affinities)
D_MODEL = 256               # Hidden size of the model (transformer embedding dim)
NHEAD = 8                   # Number of attention heads
NUM_ENCODER_LAYERS = 4      # Number of layers in the transformer towers
DIM_FEEDFORWARD = 1024      # Dimension of the feed-forward network in transformer
DROPOUT = 0.1               # Dropout rate in transformer
AGGREGATION_METHOD = 'cls'  # 'cls' or 'mean'
MAX_SEQ_LEN = 2501          # Max promoter sequence length for positional encoding
REGRESSION_HIDDEN_DIM = 128 # Hidden dimension in the regression head
REGRESSION_DROPOUT = 0.5   # Dropout in the regression head

# --- Training Hyperparameters ---
BATCH_SIZE = 32 # Can be small for large models / long sequences
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 10 # Adjust as needed
EARLY_STOPPING_PATIENCE = 3
LR_SCHEDULER_PATIENCE = 5 # Patience for ReduceLROnPlateau

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class GenePairDataset(Dataset):
    def __init__(self, feature_dir, gene_pairs_df):
        self.gene_pairs = gene_pairs_df.reset_index(drop=True)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.gene_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Gene1 = self.gene_pairs.iloc[idx]['Gene1']
        Gene2 = self.gene_pairs.iloc[idx]['Gene2']
        correlation = float(self.gene_pairs.iloc[idx]['Correlation'])

        try:
            promoter_seq1_path = os.path.join(self.feature_dir, f"{Gene1}.npy")
            promoter_seq2_path = os.path.join(self.feature_dir, f"{Gene2}.npy")
            promoter_seq1 = np.load(promoter_seq1_path)
            promoter_seq2 = np.load(promoter_seq2_path)
        except FileNotFoundError as e:
            print(f"Error loading .npy file for gene pair: ({Gene1}, {Gene2}) at index {idx}.")
            print(f"File not found: {e.filename}")
            # Propagate error to be handled by training loop or user
            raise

        promoter_seq1 = torch.from_numpy(promoter_seq1).float()
        promoter_seq2 = torch.from_numpy(promoter_seq2).float()
        correlation = torch.tensor(correlation, dtype=torch.float32)

        # Assuming promoter_seq1 and promoter_seq2 are already padded/truncated to MAX_SEQ_LEN by data_processing Step 4.
        # The mask indicates non-padded elements (False for real, True for padded).
        # Since they are pre-processed to a fixed length, there's no *additional* padding here,
        # so the mask is all False.
        mask1 = torch.zeros(promoter_seq1.shape[0], dtype=torch.bool) # seq_len is dim 0 for numpy array
        mask2 = torch.zeros(promoter_seq2.shape[0], dtype=torch.bool)

        return promoter_seq1, mask1, promoter_seq2, mask2, correlation

# --- Training Loop Definition (Step 3.2) ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience, model_save_path, log_file_path, rank, world_size):
    """
    Trains the model, implements early stopping, and logs metrics to a CSV file.
    Adjusted for DistributedDataParallel.
    """
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_pearson_r': [], 'val_mae': []}

    # Initialize CSV log file - only on rank 0
    if rank == 0:
        with open(log_file_path, 'w', newline='') as logfile:
            logwriter = csv.writer(logfile)
            logwriter.writerow(['epoch', 'train_loss', 'val_loss', 'val_pearson_r', 'val_mae'])

    # Initialize GradScaler for mixed precision training
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        # Ensure samplers are set to the correct epoch in distributed training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if val_loader and hasattr(val_loader.sampler, 'set_epoch'):
             val_loader.sampler.set_epoch(epoch)

        start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        # Use tqdm for progress bar only on rank 0
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Rank {rank}] Training", leave=False, disable=(rank != 0))

        for i, (seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch) in enumerate(train_loader_iter):
            seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch = \
                seq_a_batch.to(device), mask_a_batch.to(device), \
                seq_b_batch.to(device), mask_b_batch.to(device), \
                corr_batch.to(device)

            optimizer.zero_grad()

            try:
                # Forward pass
                predicted_correlations = model(seq_a_batch, seq_b_batch, key_padding_mask_A=mask_a_batch, key_padding_mask_B=mask_b_batch)

                # Calculate loss
                loss = criterion(predicted_correlations.squeeze(), corr_batch.float())

                # Backward pass and optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Aggregate loss from all processes
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                # Divide by world size to get the average loss across all processes for this batch
                loss = loss / world_size

                running_train_loss += loss.item() * seq_a_batch.size(0) # loss.item() is avg loss for batch across processes

            except Exception as e:
                print(f"Error during training batch {i+1}/{len(train_loader)} on rank {rank}: {e}")
                print(f"Shapes: seq_a: {seq_a_batch.shape}, seq_b: {seq_b_batch.shape}, corr: {corr_batch.shape}")
                # Potentially skip batch or raise error depending on desired robustness
                continue

        # Calculate average training loss for the epoch across all processes
        # running_train_loss already has sum of losses scaled by batch size and divided by world_size
        # So we just divide by the total number of samples processed by this rank
        # To get the global average, we would need to sum running_train_loss across all ranks and divide by total dataset size
        # Let's just calculate the average per rank for now, or better, aggregate the total loss.

        # Sum running_train_loss across all processes
        total_train_loss = torch.tensor(running_train_loss).to(device)
        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
        # Calculate global average epoch loss
        epoch_train_loss = total_train_loss.item() / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0

        if rank == 0:
             history['train_loss'].append(epoch_train_loss)


        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        if val_loader and len(val_loader.dataset) > 0: # Ensure val_loader is provided and not empty
            with torch.no_grad():
                with amp.autocast(): # Autocast for mixed precision
                    for seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch in val_loader:
                        seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch = \
                            seq_a_batch.to(device), mask_a_batch.to(device), \
                            seq_b_batch.to(device), mask_b_batch.to(device), \
                            corr_batch.to(device)

                        try:
                            predicted_correlations = model(seq_a_batch, seq_b_batch, key_padding_mask_A=mask_a_batch, key_padding_mask_B=mask_b_batch).squeeze()
                            loss = criterion(predicted_correlations, corr_batch.float())
                            running_val_loss += loss.item() * seq_a_batch.size(0)

                            all_val_preds.extend(predicted_correlations.cpu().numpy())
                            all_val_targets.extend(corr_batch.cpu().numpy())

                        except Exception as e:
                            print(f"Error during validation batch on rank {rank}: {e}")
                            continue

            # Aggregate validation loss across processes
            total_val_loss = torch.tensor(running_val_loss).to(device)
            dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
            epoch_val_loss = total_val_loss.item() / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0

            if rank == 0:
                history['val_loss'].append(epoch_val_loss)

                # Gather predictions and targets from all processes to rank 0
                gathered_preds = [None] * world_size
                gathered_targets = [None] * world_size
                dist.gather_object(all_val_preds, gathered_preds if rank == 0 else None, dst=0)
                dist.gather_object(all_val_targets, gathered_targets if rank == 0 else None, dst=0)

                if rank == 0:
                    # Concatenate lists from all ranks
                    all_val_preds_global = [item for sublist in gathered_preds for item in sublist]
                    all_val_targets_global = [item for sublist in gathered_targets for item in sublist]

                    # Calculate validation metrics on rank 0 using global data
                    val_pearson_r, _ = pearsonr(all_val_targets_global, all_val_preds_global)
                    val_mae = mean_absolute_error(all_val_targets_global, all_val_preds_global)
                    history['val_pearson_r'].append(val_pearson_r)
                    history['val_mae'].append(val_mae)

                    # Log metrics to CSV
                    with open(log_file_path, 'a', newline='') as logfile:
                        logwriter = csv.writer(logfile)
                        logwriter.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, val_pearson_r, val_mae])

                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {epoch_train_loss:.4f}, Global Val Loss: {epoch_val_loss:.4f}, "
                          f"Global Val Pearson R: {val_pearson_r:.4f}, Global Val MAE: {val_mae:.4f} - "
                          f"Duration: {time.time() - start_time:.2f}s")


            # Broadcast the global validation loss from rank 0 to all other ranks for scheduler and early stopping
            global_val_loss_tensor = torch.tensor(epoch_val_loss).to(device)
            dist.broadcast(global_val_loss_tensor, src=0)
            epoch_val_loss_broadcasted = global_val_loss_tensor.item()

            # Learning rate scheduler step - use broadcasted global loss
            if scheduler:
                scheduler.step(epoch_val_loss_broadcasted)


        else: # No validation loader or it's empty
            epoch_val_loss = float('inf') # Or some other indicator
            if rank == 0:
                history['val_loss'].append(None)
                history['val_pearson_r'].append(None)
                history['val_mae'].append(None)
                # Log only train loss if no validation
                with open(log_file_path, 'a', newline='') as logfile:
                    logwriter = csv.writer(logfile)
                    logwriter.writerow([epoch + 1, epoch_train_loss, None, None, None])

                print("Validation loader not available or empty, skipping validation phase.")


        epoch_duration = time.time() - start_time
        # Early stopping - use broadcasted global loss
        if val_loader and len(val_loader.dataset) > 0: # Only if validation was performed
            if epoch_val_loss_broadcasted < best_val_loss:
                best_val_loss = epoch_val_loss_broadcasted
                if rank == 0:
                    # Save model only on rank 0
                    best_model_state = copy.deepcopy(model.module.state_dict()) # Save unwrapped model state_dict
                    torch.save(best_model_state, model_save_path)
                    print(f"Validation loss improved. Saved new best model to {model_save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if rank == 0:
                     print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
                # Broadcast epochs_no_improve to all ranks
                epochs_no_improve_tensor = torch.tensor(epochs_no_improve).to(device)
                dist.broadcast(epochs_no_improve_tensor, src=0)
                epochs_no_improve = epochs_no_improve_tensor.item()

                if epochs_no_improve >= patience:
                    if rank == 0:
                         print(f"Early stopping triggered after {epoch+1} epochs.")
                    # Load best model weights on all ranks
                    # Need to load the state_dict from the saved file (saved by rank 0)
                    # Since only rank 0 saves, other ranks need to wait or load from the file
                    # A simpler approach in DDP is to save/load the DDP model's state_dict directly if continuing distributed training.
                    # If saving for inference or non-distributed training, save model.module.state_dict().
                    # For early stopping and continuing distributed training, loading the best state_dict back to DDP model is complex.
                    # A common pattern is to save the best state_dict on rank 0, and then load it on all ranks.
                    # Let's load the saved state dict on all ranks.
                    # Ensure all ranks wait for rank 0 to save.
                    if rank == 0:
                        torch.save(best_model_state, model_save_path) # Resave if it was updated
                    dist.barrier() # Wait for rank 0 to save
                    # Load state dict on all ranks
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} # Map saved weights to current rank's device
                    model.module.load_state_dict(torch.load(model_save_path, map_location=map_location))
                    return model, history # Return the model with best weights
        elif epoch == num_epochs -1: # If no validation, save last model
             if rank == 0:
                 best_model_state = copy.deepcopy(model.module.state_dict())
                 torch.save(best_model_state, model_save_path)
                 print(f"Training complete. Saved final model to {MODEL_SAVE_PATH}")

    # After training, load the best model state if it exists and validation was performed
    if val_loader and len(val_loader.dataset) > 0 and best_model_state:
         # Load state dict on all ranks from the best model file saved by rank 0
         dist.barrier() # Ensure rank 0 has saved the best model
         map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} # Map saved weights to current rank's device
         model.module.load_state_dict(torch.load(model_save_path, map_location=map_location))
    elif epoch == num_epochs -1 and not (val_loader and len(val_loader.dataset) > 0): # If no validation and training finished
         # The last model was saved by rank 0 in the loop, load it back to module if needed (though it's already loaded)
         pass # No need to load again, model is already the last trained state

    # Synchronize at the end of training
    dist.barrier()

    # Return the DDP model (or model.module if you only need the core model)
    # Returning model.module is often more convenient if you don't need DDP wrapper after training
    return model.module, history



if __name__ == '__main__':
    # Add argparse for rank and world size
    parser = argparse.ArgumentParser(description='Distributed Training Script')
    # parser.add_argument('--rank', type=int, help='Rank of the current process (0 to world_size - 1)')
    # parser.add_argument('--world_size', type=int, help='Total number of processes')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    args = parser.parse_args()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        # world_size=args.world_size,
        # rank=args.rank
    )

    local_rank = dist.get_rank() #args.rank
    world_size = dist.get_world_size()

    # Initialize the distributed environment
    print(f"Initializing process group with rank {local_rank} and world size {world_size}")

    # Set up the device for this process
    # The device should be specific to the rank
    DEVICE = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    print(f"--- Training Script (Rank {local_rank}/{world_size}) ---")

    # 1. Load and Prepare Data
    print("\n[Phase 1: Data Loading and Preparation]")
    dummy_tsv_path = "dummy_gene_pairs.tsv"
    dummy_feature_dir = "dummy_feature_vectors/"
    actual_data_mode = True
    if os.path.exists(GENE_PAIRS_TSV_PATH) and os.path.isdir(FEATURE_VECTOR_DIR):
        print(f"Attempting to use actual data paths: {GENE_PAIRS_TSV_PATH}, {FEATURE_VECTOR_DIR}")
        try:
            if not any(f.endswith(".npy") for f in os.listdir(FEATURE_VECTOR_DIR)):
                print(f"Warning: {FEATURE_VECTOR_DIR} is empty or contains no .npy files.")
                raise FileNotFoundError("Feature directory empty")
            all_pairs_df = pd.read_csv(GENE_PAIRS_TSV_PATH, sep='\t')
            required_cols = ['Gene1', 'Gene2', 'Correlation']
            if not all(col in all_pairs_df.columns for col in required_cols):
                print(f"Actual TSV file {GENE_PAIRS_TSV_PATH} is missing required columns.")
                raise ValueError("Missing columns")
            print(f"Successfully loaded actual gene pairs data: {len(all_pairs_df)} pairs.")
            current_feature_dir = FEATURE_VECTOR_DIR
            actual_data_mode = True
        except Exception as e:
            print(f"Error loading actual data ({e}). Falling back to dummy data.")
            actual_data_mode = False

    if not actual_data_mode:
        print(f"Using dummy data. Generating in {os.getcwd()} if not present.")
        os.makedirs(dummy_feature_dir, exist_ok=True)
        num_dummy_genes = 50
        dummy_gene_pool = [f'gene{i}' for i in range(num_dummy_genes)]
        num_dummy_pairs = 200 # Increased for more realistic split
        Gene1s = np.random.choice(dummy_gene_pool, num_dummy_pairs)
        Gene2s = np.random.choice(dummy_gene_pool, num_dummy_pairs)
        valid_pairs = []
        seen_pair_hashes = set()
        for g1, g2 in zip(Gene1s, Gene2s):
            if g1 == g2: continue
            pair_hash = tuple(sorted((g1, g2)))
            if pair_hash not in seen_pair_hashes:
                valid_pairs.append({'Gene1': g1, 'Gene2': g2, 'Correlation': np.random.rand()})
                seen_pair_hashes.add(pair_hash)
        all_pairs_df = pd.DataFrame(valid_pairs)
        if all_pairs_df.empty: # Fallback
             all_pairs_df = pd.DataFrame({
                'Gene1': [f'g{i}' for i in range(10)], 'Gene2': [f'g{i+10}' for i in range(10)],
                'Correlation': np.random.rand(10)
            })
             # Add some pairs with shared genes for splitting test
             all_pairs_df = pd.concat([all_pairs_df, pd.DataFrame({
                 'Gene1': ['g0', 'g1', 'g2'], 'Gene2': ['g1', 'g2', 'g0'],
                 'Correlation': [0.5,0.6,0.7]})], ignore_index=True)


        all_pairs_df.to_csv(dummy_tsv_path, sep='\t', index=False)
        print(f"Created dummy TSV: {dummy_tsv_path} with {len(all_pairs_df)} pairs.")
        all_genes_in_dummy_df = set(all_pairs_df['Gene1']).union(set(all_pairs_df['Gene2']))
        for gene_id in all_genes_in_dummy_df:
            dummy_array_path = os.path.join(dummy_feature_dir, f"{gene_id}.npy")
            if not os.path.exists(dummy_array_path):
                # Dummy sequences should match the expected fixed length (MAX_SEQ_LEN after padding/truncation from Step 4)
                # and feature dimension (INPUT_FEATURE_DIM).
                dummy_array = np.random.rand(MAX_SEQ_LEN, INPUT_FEATURE_DIM).astype(np.float32)
                np.save(dummy_array_path, dummy_array)
        print(f"Ensured dummy .npy files (shape: ({MAX_SEQ_LEN}, {INPUT_FEATURE_DIM})) exist in: {dummy_feature_dir}")
        current_feature_dir = dummy_feature_dir

    # Load the gene-to-chromosome mapping file
    print(f"Loading gene-to-chromosome mapping from: {GENE_CHROMOSOME_MAPPING_PATH}")
    try:
        gene_chromosome_map_df = pd.read_csv(GENE_CHROMOSOME_MAPPING_PATH, sep='\t')
        if 'gene_id' not in gene_chromosome_map_df.columns or 'chromosome' not in gene_chromosome_map_df.columns:
             raise ValueError("Mapping file must contain 'gene_id' and 'chromosome' columns.")
        print(f"Successfully loaded chromosome mapping for {len(gene_chromosome_map_df)} genes.")
    except FileNotFoundError:
        print(f"Error: Gene-to-chromosome mapping file not found at {GENE_CHROMOSOME_MAPPING_PATH}.")
        print("Cannot perform chromosome-based split without this file.")
        exit() # Exit if mapping file is not found
    except Exception as e:
        print(f"Error loading gene-to-chromosome mapping file: {e}")
        exit()


    # Create a mapping dictionary from gene_id to chromosome
    gene_to_chr = gene_chromosome_map_df.set_index('gene_id')['chromosome'].to_dict()

    # Add chromosome information to the all_pairs_df
    print("Adding chromosome information to gene pairs.")
    all_pairs_df['Gene1_chromosome'] = all_pairs_df['Gene1'].map(gene_to_chr)
    all_pairs_df['Gene2_chromosome'] = all_pairs_df['Gene2'].map(gene_to_chr)

    # Handle cases where gene IDs in all_pairs_df are not in the mapping file
    if all_pairs_df['Gene1_chromosome'].isnull().any() or all_pairs_df['Gene2_chromosome'].isnull().any():
        print("Warning: Some gene IDs in gene pairs were not found in the chromosome mapping file.")
        print("These pairs will be excluded from all datasets.")
        all_pairs_df.dropna(subset=['Gene1_chromosome', 'Gene2_chromosome'], inplace=True)
        print(f"Remaining pairs after removing those with missing chromosome info: {len(all_pairs_df)}")


    # Perform chromosome-based split
    print("\nPerforming chromosome-based data split.")

    # Test set: pairs where both genes are on Chr2
    test_df = all_pairs_df[
        (all_pairs_df['Gene1_chromosome'] == 'Chr2') &
        (all_pairs_df['Gene2_chromosome'] == 'Chr2')
    ].copy()

    # Validation set: pairs where both genes are on Chr4
    val_df = all_pairs_df[
        (all_pairs_df['Gene1_chromosome'] == 'Chr4') &
        (all_pairs_df['Gene2_chromosome'] == 'Chr4')
    ].copy()

    # Train set: pairs where *neither* gene is on Chr2 or Chr4
    train_df = all_pairs_df[
        (~all_pairs_df['Gene1_chromosome'].isin(['Chr2', 'Chr4'])) &
        (~all_pairs_df['Gene2_chromosome'].isin(['Chr2', 'Chr4']))
    ].copy()


    print(f"Total pairs loaded: {len(all_pairs_df)}")
    print(f"Train pairs: {len(train_df)}")
    print(f"Validation pairs (Chr4-Chr4): {len(val_df)}")
    print(f"Test pairs (Chr2-Chr2): {len(test_df)}")


    # Save the test data to a file for later evaluation
    test_data_save_path = './data/test_gene_pairs.csv'
    if not test_df.empty:
        try:
            # Ensure the data directory exists
            os.makedirs('./data', exist_ok=True)
            test_df.to_csv(test_data_save_path, index=False)
            print(f"Test data saved to {test_data_save_path}")
        except Exception as e:
            print(f"Error saving test data: {e}")

    if train_df.empty:
        print("CRITICAL: Training DataFrame is empty after split. Cannot proceed with training.")
        # Exit or raise error if train_df is empty as training is not possible
    else:
        # Use DistributedSampler for distributed training
        train_dataset = GenePairDataset(feature_dir=current_feature_dir, gene_pairs_df=train_df)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, collate_fn=None) # shuffle=False with sampler
        print(f"Train DataLoader created for rank {local_rank}. Batches: {len(train_loader)}, Samples: {len(train_dataset)}")

        val_loader = None
        if not val_df.empty:
            val_dataset = GenePairDataset(feature_dir=current_feature_dir, gene_pairs_df=val_df)
            if len(val_dataset) > 0:
                # Use DistributedSampler for validation data as well
                val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0, collate_fn=None) # shuffle=False with sampler
                print(f"Validation DataLoader created for rank {local_rank}. Batches: {len(val_loader)}, Samples: {len(val_dataset)}")
            else:
                print("Validation dataset created but is empty.")
        else:
            print("Validation DataFrame is empty. No validation loader will be used.")

        # Test loader (not used in training loop directly here, but good to check)
        # For evaluation after training, you might want a non-distributed sampler or handle aggregation
        # For now, keep as is for basic structure but be aware of potential issues in distributed evaluation without proper aggregation.
        if not test_df.empty:
            test_dataset = GenePairDataset(feature_dir=current_feature_dir, gene_pairs_df=test_df)
            if len(test_dataset) > 0:
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
                print(f"Test DataLoader created. Batches: {len(test_loader)}, Samples: {len(test_dataset)}")
            else:
                print("Test dataset created but is empty.")


        # 2. Model, Loss, Optimizer, Scheduler Setup
        print("\n[Phase 2: Model Initialization]")
        model = SiameseGeneTransformer(
            input_feature_dim=INPUT_FEATURE_DIM,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            aggregation_method=AGGREGATION_METHOD,
            max_seq_len=MAX_SEQ_LEN, # Important for PositionalEncoding
            regression_hidden_dim=REGRESSION_HIDDEN_DIM,
            regression_dropout=REGRESSION_DROPOUT
        ).to(DEVICE)
        print(f"SiameseGeneTransformer model instantiated on {DEVICE}.")
        print(f"Model total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # Wrap the model with DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[local_rank])
        print("Model wrapped with DistributedDataParallel.")

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Scheduler will only be used if val_loader is available
        scheduler = None
        if val_loader:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=LR_SCHEDULER_PATIENCE, factor=0.1)
            print("ReduceLROnPlateau LR scheduler configured.")
        else:
            print("No validation data, so ReduceLROnPlateau scheduler will not be used.")


        # 3. Train the Model
        print("\n[Phase 3: Model Training]")

        # Define log file path and ensure directory exists
        log_dir = "./training_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "training_metrics.csv")
        print(f"Training metrics will be logged to: {log_file_path}")

        try:
            # Check if train_loader has data before starting training
            if len(train_loader.dataset) == 0:
                print("Train dataset is empty. Skipping training.")
            else:
                print(f"Starting training for {NUM_EPOCHS} epochs with early stopping patience {EARLY_STOPPING_PATIENCE}...")
                trained_model, history = train_model(
                    model, train_loader, val_loader, criterion, optimizer, scheduler,
                    NUM_EPOCHS, DEVICE, EARLY_STOPPING_PATIENCE, MODEL_SAVE_PATH, log_file_path
                )
                print("Training finished.")
                print(f"Best model saved to: {MODEL_SAVE_PATH}")
                # Optional: Plot history['train_loss'] and history['val_loss']
        except FileNotFoundError as e:
            print(f"\nCRITICAL ERROR DURING TRAINING: A required .npy file was not found: {e.filename}")
            print("This likely means that some gene IDs in your TSV file do not have corresponding .npy files in the feature directory.")
            print("Please check your data generation steps (Module 1 & 2) and ensure all necessary files are present.")
        except Exception as e:
            import traceback
            print(f"\nAn unexpected error occurred during model training: {e}")
            traceback.print_exc()


    print("\n--- End of Full Training Script Simulation ---")