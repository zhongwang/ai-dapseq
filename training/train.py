import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import time
import copy
import csv
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import torch.amp as amp # Using torch.amp, not torch.cuda.amp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import argparse
from tqdm import tqdm
import sys # Import sys for sys.exit()

# Attempt to import the model from the parent directory's 'model' folder
try:
    from ..model.siamese_transformer import SiameseGeneTransformer
except ImportError:
    # Fallback for running the script directly from the 'training' directory for testing
    # This assumes 'model' and 'training' are sibling directories.
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from model.siamese_transformer import SiameseGeneTransformer


# --- Configuration (Placeholder paths - adjust as needed) ---
GENE_PAIRS_TSV_PATH = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/final_coexpressed_regression_15bins.txt"
FEATURE_VECTOR_DIR = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/feature_vectors/"
MODEL_SAVE_PATH = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/best_siamese_model.pth"
GENE_CHROMOSOME_MAPPING_PATH = "/global/scratch/users/sallyliao2027/aidapseq/output/new_full_data/promoter_sequences.txt"

# --- Model Hyperparameters (Example values, tune as needed) ---
INPUT_FEATURE_DIM = 248
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
AGGREGATION_METHOD = 'cls'
MAX_SEQ_LEN = 2501
REGRESSION_HIDDEN_DIM = 128
REGRESSION_DROPOUT = 0.5

# --- Training Hyperparameters ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
LR_SCHEDULER_PATIENCE = 5

# --- Device Configuration ---
# Initial DEVICE print removed as it's not truly representative before DDP setup.
# The actual device assignment will happen inside __main__ based on local_rank.


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

            promoter_seq1 = torch.from_numpy(promoter_seq1).float()
            promoter_seq2 = torch.from_numpy(promoter_seq2).float()
            correlation = torch.tensor(correlation, dtype=torch.float32)

            mask1 = torch.zeros(promoter_seq1.shape[0], dtype=torch.bool)
            mask2 = torch.zeros(promoter_seq2.shape[0], dtype=torch.bool)

            return promoter_seq1, mask1, promoter_seq2, mask2, correlation

        except FileNotFoundError as e:
            print(f"[Rank {dist.get_rank()}] Warning: File not found for gene pair: ({Gene1}, {Gene2}) at index {idx}. File: {e.filename}. Skipping this sample.")
            return None # Return None for invalid samples
        except Exception as e:
            print(f"[Rank {dist.get_rank()}] Warning: Unexpected error loading data for gene pair: ({Gene1}, {Gene2}) at index {idx}. Error: {e}. Skipping this sample.")
            return None # Return None for other errors

# --- Training Loop Definition ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience, model_save_path, log_file_path, rank, world_size):
    """
    Trains the model, implements early stopping, and logs metrics to a CSV file.
    Adjusted for DistributedDataParallel and handling potentially empty or NaN/Inf batches.
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
    scaler = amp.GradScaler(device.type)


    for epoch in range(num_epochs):
        # Ensure samplers are set to the correct epoch in distributed training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if val_loader and hasattr(val_loader.sampler, 'set_epoch'):
             val_loader.sampler.set_epoch(epoch)

        start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_train_loss_sum = 0.0
        running_train_samples = 0

        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Rank {rank}] Training", leave=False, disable=(rank != 0))

        for i, batch_data in enumerate(train_loader_iter):
            # Initialize local tensors for loss sum and batch size for this batch
            local_batch_loss_sum_tensor = torch.tensor(0.0, device=device)
            local_batch_size_tensor = torch.tensor(0, dtype=torch.long, device=device)
            current_batch_loss_tensor = None

            if batch_data is not None:
                seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch = batch_data
                seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch = \
                    seq_a_batch.to(device), mask_a_batch.to(device), \
                    seq_b_batch.to(device), mask_b_batch.to(device), \
                    corr_batch.to(device)

                optimizer.zero_grad()

                try:
                    with amp.autocast(device.type): # Autocast for mixed precision during forward pass
                        predicted_correlations = model(seq_a_batch, seq_b_batch, key_padding_mask_A=mask_a_batch, key_padding_mask_B=mask_b_batch)
                        loss = criterion(predicted_correlations.squeeze(), corr_batch.float())

                    # --- Check for NaN/Inf in loss immediately after calculation ---
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[Rank {rank}] Warning: Detected NaN/Inf loss in training batch {i+1}/{len(train_loader)}. Skipping backward pass for this batch.")
                        # local_batch_loss_sum_tensor and local_batch_size_tensor remain zero from initialization
                        current_batch_loss_tensor = None # Ensure no backward pass
                    else:
                        local_batch_loss_sum_tensor = loss * seq_a_batch.size(0)
                        local_batch_size_tensor = torch.tensor(seq_a_batch.size(0), dtype=torch.long, device=device)
                        current_batch_loss_tensor = loss

                except Exception as e:
                    print(f"[Rank {rank}] Error during training batch {i+1}/{len(train_loader)}: {e}")
                    print(f"[Rank {rank}] Skipping processing for this batch.")
                    local_batch_loss_sum_tensor = torch.tensor(0.0, device=device)
                    local_batch_size_tensor = torch.tensor(0, dtype=torch.long, device=device)
                    current_batch_loss_tensor = None


            # --- Synchronize loss sum and batch size across all ranks ---
            # Perform all_reduce to sum up local batch loss sums and batch sizes from all ranks
            # This acts as a synchronization point. It's crucial this happens for EVERY batch.
            # Add NaN/Inf checks right before the all_reduce
            if torch.isnan(local_batch_loss_sum_tensor).any() or torch.isinf(local_batch_loss_sum_tensor).any():
                print(f"[Rank {rank}] CRITICAL ERROR: NaN/Inf detected in local_batch_loss_sum_tensor for batch {i+1}. Value: {local_batch_loss_sum_tensor.item()}")
                sys.exit(1) # Force exit to get full traceback
            if torch.isnan(local_batch_size_tensor).any() or torch.isinf(local_batch_size_tensor).any():
                print(f"[Rank {rank}] CRITICAL ERROR: NaN/Inf detected in local_batch_size_tensor for batch {i+1}. Value: {local_batch_size_tensor.item()}")
                sys.exit(1) # Force exit to get full traceback

            dist.all_reduce(local_batch_loss_sum_tensor, op=dist.ReduceOp.SUM, async_op=False)
            dist.all_reduce(local_batch_size_tensor, op=dist.ReduceOp.SUM, async_op=False)

            global_batch_loss_sum = local_batch_loss_sum_tensor.item()
            global_batch_size_sum = local_batch_size_tensor.item()

            # Accumulate global sums for epoch average
            running_train_loss_sum += global_batch_loss_sum
            running_train_samples += global_batch_size_sum

            # --- Backward Pass and Optimizer Step ---
            # Only perform backward and step if there were *any* valid samples in this batch across all ranks
            # and the local forward pass was successful and produced a valid loss (current_batch_loss_tensor is not None)
            if global_batch_size_sum > 0 and current_batch_loss_tensor is not None:
                try:
                    scaler.scale(current_batch_loss_tensor).backward()
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as e:
                    print(f"[Rank {rank}] Error during backward/optimizer step for batch {i+1}/{len(train_loader)}: {e}")
                    # This indicates a deeper issue if backward fails after a successful forward.
                    # Training might become unstable, but the all_reduce for metrics will keep ranks synced.


        # Calculate average training loss for the epoch across all processes
        epoch_train_loss = running_train_loss_sum / running_train_samples if running_train_samples > 0 else 0.0

        if rank == 0:
            history['train_loss'].append(epoch_train_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Duration: {time.time() - start_time:.2f}s")


        # --- Validation Phase ---
        model.eval()
        running_val_loss_sum = 0.0
        running_val_samples = 0
        all_val_preds = []
        all_val_targets = []

        if val_loader and len(val_loader.dataset) > 0:
            print(f"[{rank}] val_loader has {len(val_loader)} batches (local).")
            print(f"[{rank}] val_dataset has {len(val_loader.dataset)} samples (local).")
            dist.barrier() # Ensure all ranks are ready for validation loop

            with torch.no_grad():
                with amp.autocast(device.type): # Autocast for mixed precision
                    for batch_data in val_loader:
                        local_batch_loss_sum_tensor = torch.tensor(0.0, device=device)
                        local_batch_size_tensor = torch.tensor(0, dtype=torch.long, device=device)

                        if batch_data is not None:
                            seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch = batch_data
                            seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch = \
                                seq_a_batch.to(device), mask_a_batch.to(device), \
                                seq_b_batch.to(device), mask_b_batch.to(device), \
                                corr_batch.to(device)

                            try:
                                predicted_correlations = model(seq_a_batch, seq_b_batch, key_padding_mask_A=mask_a_batch, key_padding_mask_B=mask_b_batch).squeeze()
                                loss = criterion(predicted_correlations, corr_batch.float())

                                if torch.isnan(loss) or torch.isinf(loss):
                                    print(f"[Rank {rank}] Warning: Detected NaN/Inf loss in validation batch. Skipping this batch for loss calculation and metric gathering.")
                                else:
                                    local_batch_loss_sum_tensor = loss * seq_a_batch.size(0)
                                    local_batch_size_tensor = torch.tensor(seq_a_batch.size(0), dtype=torch.long, device=device)
                                    all_val_preds.extend(predicted_correlations.cpu().numpy())
                                    all_val_targets.extend(corr_batch.cpu().numpy())

                            except Exception as e:
                                print(f"[Rank {rank}] Error during validation batch: {e}")
                                print(f"[Rank {rank}] Skipping processing for this validation batch.")

                            try:
                                predicted_correlations = model(seq_a_batch, seq_b_batch, key_padding_mask_A=mask_a_batch, key_padding_mask_B=mask_b_batch).squeeze()
                                loss = criterion(predicted_correlations, corr_batch.float())

                                if torch.isnan(loss) or torch.isinf(loss):
                                    print(f"[Rank {rank}] Warning: Detected NaN/Inf loss in validation batch. Skipping this batch for loss calculation and metric gathering.")
                                else:
                                    # Accumulate local loss and sample count for this batch
                                    running_val_loss_sum += (loss * seq_a_batch.size(0)).item()
                                    running_val_samples += seq_a_batch.size(0)
                                    all_val_preds.extend(predicted_correlations.cpu().numpy())
                                    all_val_targets.extend(corr_batch.cpu().numpy())

                            except Exception as e:
                                print(f"[Rank {rank}] Error during validation batch: {e}")
                                print(f"[Rank {rank}] Skipping processing for this validation batch.")

                        # The batch-level all_reduces and subsequent accumulations are removed here.
                        # The accumulation of local batch loss and sample count is now within the 'else' block above.

            print(f"[{rank}] Finished local validation loop.")
            dist.barrier() # Ensure all ranks finished their val loop

            print(f"[{rank}] Before all_reduce for total_val_loss. Local sum: {running_val_loss_sum:.6f}")
            print(f"[{rank}] Before all_reduce for total_val_samples. Local count: {running_val_samples}")
            dist.barrier() # Barrier to ensure both ranks print before the all_reduce

            # Calculate average validation loss for the epoch across all processes
            total_val_loss = torch.tensor(running_val_loss_sum).to(device)
            total_val_samples_tensor = torch.tensor(running_val_samples, dtype=torch.long, device=device)

            # --- NaN/Inf checks before final validation all_reduce ---
            if torch.isnan(total_val_loss).any() or torch.isinf(total_val_loss).any():
                print(f"[{rank}] CRITICAL ERROR: NaN/Inf detected in total_val_loss before final all_reduce. Value: {total_val_loss.item()}")
                sys.exit(1)
            if torch.isnan(total_val_samples_tensor).any() or torch.isinf(total_val_samples_tensor).any():
                print(f"[{rank}] CRITICAL ERROR: NaN/Inf detected in total_val_samples_tensor before final all_reduce. Value: {total_val_samples_tensor.item()}")
                sys.exit(1)

            dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_samples_tensor, op=dist.ReduceOp.SUM)

            epoch_val_loss = total_val_loss.item() / total_val_samples_tensor.item() if total_val_samples_tensor.item() > 0 else float('inf')

            if rank == 0:
                history['val_loss'].append(epoch_val_loss)

                gathered_preds_list = [None] * world_size
                gathered_targets_list = [None] * world_size

                # Ensure all ranks participate in the gather call, but only rank 0 receives the data
                dist.gather_object(all_val_preds, gathered_preds_list if rank == 0 else None, dst=0)
                dist.gather_object(all_val_targets, gathered_targets_list if rank == 0 else None, dst=0)

                if total_val_samples_tensor.item() > 0: # Check global samples before calculating metrics
                    all_val_preds_global = [item for sublist in gathered_preds_list if sublist is not None for item in sublist]
                    all_val_targets_global = [item for sublist in gathered_targets_list if sublist is not None for item in sublist]

                    if len(all_val_targets_global) > 0:
                        val_pearson_r, _ = pearsonr(all_val_targets_global, all_val_preds_global)
                        val_mae = mean_absolute_error(all_val_targets_global, all_val_preds_global)
                        history['val_pearson_r'].append(val_pearson_r)
                        history['val_mae'].append(val_mae)

                        with open(log_file_path, 'a', newline='') as logfile:
                            logwriter = csv.writer(logfile)
                            logwriter.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, val_pearson_r, val_mae])

                        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Global Val Loss: {epoch_val_loss:.4f}, "
                                  f"Global Val Pearson R: {val_pearson_r:.4f}, Global Val MAE: {val_mae:.4f} - Duration: {time.time() - start_time:.2f}s")
                    else:
                        history['val_pearson_r'].append(None)
                        history['val_mae'].append(None)
                        with open(log_file_path, 'a', newline='') as logfile:
                            logwriter = csv.writer(logfile)
                            logwriter.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, None, None])
                        print(f"Epoch {epoch+1}/{num_epochs} - No valid validation samples found globally. Train Loss: {epoch_train_loss:.4f}, Global Val Loss: {epoch_val_loss:.4f} - Duration: {time.time() - start_time:.2f}s")
                else: # No samples globally for validation
                    history['val_pearson_r'].append(None)
                    history['val_mae'].append(None)
                    with open(log_file_path, 'a', newline='') as logfile:
                        logwriter = csv.writer(logfile)
                        logwriter.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, None, None])
                    print(f"Epoch {epoch+1}/{num_epochs} - No valid validation samples found globally. Train Loss: {epoch_train_loss:.4f}, Global Val Loss: {epoch_val_loss:.4f} - Duration: {time.time() - start_time:.2f}s")
            
            dist.barrier() # Barrier after gathering and logging on rank 0
            # Broadcast the global validation loss from rank 0 to all other ranks for scheduler and early stopping
            global_val_loss_tensor = torch.tensor(epoch_val_loss).to(device)
            # Add NaN/Inf check before broadcast
            if torch.isnan(global_val_loss_tensor).any() or torch.isinf(global_val_loss_tensor).any():
                print(f"[{rank}] CRITICAL ERROR: NaN/Inf detected in global_val_loss_tensor before broadcast. Value: {global_val_loss_tensor.item()}")
                sys.exit(1)
            dist.broadcast(global_val_loss_tensor, src=0)
            epoch_val_loss_broadcasted = global_val_loss_tensor.item()

            # Learning rate scheduler step - use broadcasted global loss
            if scheduler:
                scheduler.step(epoch_val_loss_broadcasted)


        else: # No validation loader or it's empty
            epoch_val_loss = float('inf') # Use inf if no validation data
            if rank == 0:
                history['val_loss'].append(epoch_val_loss)
                history['val_pearson_r'].append(None)
                history['val_mae'].append(None)
                with open(log_file_path, 'a', newline='') as logfile:
                    logwriter = csv.writer(logfile)
                    logwriter.writerow([epoch + 1, epoch_train_loss, epoch_val_loss, None, None])

                print(f"Epoch {epoch+1}/{num_epochs} - No validation data available. Train Loss: {epoch_train_loss:.4f} - Duration: {time.time() - start_time:.2f}s")

            # This branch MUST also participate in collective operations that others (in the 'if' branch) do
            # For now, it seems like the scheduler step broadcast is the only one.
            # Ensure epoch_val_loss is broadcasted even if no validation data
            global_val_loss_tensor = torch.tensor(epoch_val_loss).to(device)
            # Add NaN/Inf check before broadcast
            if torch.isnan(global_val_loss_tensor).any() or torch.isinf(global_val_loss_tensor).any():
                print(f"[{rank}] CRITICAL ERROR: NaN/Inf detected in global_val_loss_tensor (empty val branch) before broadcast. Value: {global_val_loss_tensor.item()}")
                sys.exit(1)
            dist.broadcast(global_val_loss_tensor, src=0)
            epoch_val_loss_broadcasted = global_val_loss_tensor.item() # Update broadcasted value for early stopping check

            # Learning rate scheduler step - it's okay for scheduler to be None here
            if scheduler: # Only run if scheduler was actually initialized
                scheduler.step(epoch_val_loss_broadcasted)


        epoch_duration = time.time() - start_time
        # Early stopping - use broadcasted global loss
        if val_loader and len(val_loader.dataset) > 0: # Only if validation was performed
            if epoch_val_loss_broadcasted < best_val_loss:
                best_val_loss = epoch_val_loss_broadcasted
                if rank == 0:
                    best_model_state = copy.deepcopy(model.module.state_dict())
                    torch.save(best_model_state, model_save_path)
                    print(f"Validation loss improved. Saved new best model to {model_save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if rank == 0:
                    print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

                epochs_no_improve_tensor = torch.tensor(epochs_no_improve).to(device)
                # Add NaN/Inf check before broadcast
                if torch.isnan(epochs_no_improve_tensor).any() or torch.isinf(epochs_no_improve_tensor).any():
                    print(f"[{rank}] CRITICAL ERROR: NaN/Inf detected in epochs_no_improve_tensor before broadcast. Value: {epochs_no_improve_tensor.item()}")
                    sys.exit(1)
                dist.broadcast(epochs_no_improve_tensor, src=0)
                epochs_no_improve = epochs_no_improve_tensor.item()

                if epochs_no_improve >= patience:
                    if rank == 0:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                    dist.barrier() # Ensure all ranks are synced before attempting to load the model
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                    try:
                        model.module.load_state_dict(torch.load(model_save_path, map_location=map_location))
                        print(f"[Rank {rank}] Loaded best model state dict for early stopping.")
                    except FileNotFoundError:
                        print(f"[Rank {rank}] Warning: Best model file not found at {model_save_path}. Could not load state dict for early stopping.")
                    except Exception as e:
                        print(f"[Rank {rank}] Error loading best model state dict: {e}")

                    return model, history
        elif epoch == num_epochs -1: # If no validation, save last model
             if rank == 0:
                 best_model_state = copy.deepcopy(model.module.state_dict())
                 torch.save(best_model_state, model_save_path)
                 print(f"Training complete. Saved final model to {MODEL_SAVE_PATH}")

    # After training, load the best model state if it exists and validation was performed
    if val_loader and len(val_loader.dataset) > 0 and best_model_state:
        dist.barrier() # Ensure rank 0 has saved the best model
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        try:
            model.module.load_state_dict(torch.load(model_save_path, map_location=map_location))
            if rank == 0:
                print("Loaded best model state dict after training.")
        except FileNotFoundError:
            if rank == 0:
                print(f"Warning: Best model file not found at {model_save_path}. Could not load state dict after training.")
        except Exception as e:
            if rank == 0:
                print(f"Error loading best model state dict after training: {e}")

    elif epoch == num_epochs -1 and not (val_loader and len(val_loader.dataset) > 0):
        pass

    dist.barrier()
    return model.module, history


def custom_collate_fn(batch):
    # Filter out None values (invalid samples)
    batch = [item for item in batch if item is not None]

    if not batch:
        return None # Return None for an empty batch

    # Transpose the batch (list of tuples to tuple of lists)
    seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch = zip(*batch)

    # Stack the tensors
    seq_a_batch = torch.stack(seq_a_batch)
    mask_a_batch = torch.stack(mask_a_batch)
    seq_b_batch = torch.stack(seq_b_batch)
    mask_b_batch = torch.stack(mask_b_batch)
    corr_batch = torch.stack(corr_batch)

    return seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch


if __name__ == '__main__':
    # Add argparse for rank and world size
    parser = argparse.ArgumentParser(description='Distributed Training Script')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    args = parser.parse_args()

    # Initial print statement BEFORE DDP initialization (will be sequential)
    # Moved to a more appropriate place after DDP init
    # print(f"PID: {os.getpid()} | Rank: {local_rank} | [Timestamp: {time.time()}]")

    # CRITICAL: Initialize DDP first thing
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
    )

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set up the device for this process
    DEVICE = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    # Now print with correct rank and ensure concurrency is expected
    print(f"PID: {os.getpid()} | Rank: {local_rank} | World Size: {world_size} | Device: {DEVICE} | [Timestamp: {time.time()}]")
    print(f"PID: {os.getpid()} | Rank {local_rank}: Passed initial DDP setup.")
    dist.barrier() # Ensure all processes are fully initialized and ready before data loading

    print(f"\nPID: {os.getpid()} | Rank {local_rank}: [Phase 1: Data Loading and Preparation]")
    
    # --- Data Loading and Preparation ---
    dummy_tsv_path = "dummy_gene_pairs.tsv"
    dummy_feature_dir = "dummy_feature_vectors/"
    actual_data_mode = True
    all_pairs_df = None # Initialize to None

    if os.path.exists(GENE_PAIRS_TSV_PATH) and os.path.isdir(FEATURE_VECTOR_DIR):
        print(f"PID: {os.getpid()} | Rank {local_rank}: Attempting to use actual data paths: {GENE_PAIRS_TSV_PATH}, {FEATURE_VECTOR_DIR}")
        try:
            if not any(f.endswith(".npy") for f in os.listdir(FEATURE_VECTOR_DIR)):
                print(f"PID: {os.getpid()} | Rank {local_rank}: Warning: {FEATURE_VECTOR_DIR} is empty or contains no .npy files.")
                raise FileNotFoundError("Feature directory empty")
            all_pairs_df = pd.read_csv(GENE_PAIRS_TSV_PATH, sep='\t')
            required_cols = ['Gene1', 'Gene2', 'Correlation']
            if not all(col in all_pairs_df.columns for col in required_cols):
                print(f"PID: {os.getpid()} | Rank {local_rank}: Actual TSV file {GENE_PAIRS_TSV_PATH} is missing required columns.")
                raise ValueError("Missing columns")
            print(f"PID: {os.getpid()} | Rank {local_rank}: Successfully loaded actual gene pairs data: {len(all_pairs_df)} pairs.")
            current_feature_dir = FEATURE_VECTOR_DIR
            actual_data_mode = True
        except Exception as e:
            print(f"PID: {os.getpid()} | Rank {local_rank}: Error loading actual data ({e}). Falling back to dummy data.")
            actual_data_mode = False

    if not actual_data_mode:
        print(f"PID: {os.getpid()} | Rank {local_rank}: Using dummy data. Generating in {os.getcwd()} if not present.")
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
        print(f"PID: {os.getpid()} | Rank {local_rank}: Created dummy TSV: {dummy_tsv_path} with {len(all_pairs_df)} pairs.")
        all_genes_in_dummy_df = set(all_pairs_df['Gene1']).union(set(all_pairs_df['Gene2']))
        for gene_id in all_genes_in_dummy_df:
            dummy_array_path = os.path.join(dummy_feature_dir, f"{gene_id}.npy")
            if not os.path.exists(dummy_array_path):
                dummy_array = np.random.rand(MAX_SEQ_LEN, INPUT_FEATURE_DIM).astype(np.float32)
                np.save(dummy_array_path, dummy_array)
        print(f"PID: {os.getpid()} | Rank {local_rank}: Ensured dummy .npy files (shape: ({MAX_SEQ_LEN}, {INPUT_FEATURE_DIM})) exist in: {dummy_feature_dir}")
        current_feature_dir = dummy_feature_dir

    # Load the gene-to-chromosome mapping file
    print(f"PID: {os.getpid()} | Rank {local_rank}: Loading gene-to-chromosome mapping from: {GENE_CHROMOSOME_MAPPING_PATH}")
    try:
        gene_chromosome_map_df = pd.read_csv(GENE_CHROMOSOME_MAPPING_PATH, sep='\t')
        if 'gene_id' not in gene_chromosome_map_df.columns or 'chromosome' not in gene_chromosome_map_df.columns:
            raise ValueError("Mapping file must contain 'gene_id' and 'chromosome' columns.")
        print(f"PID: {os.getpid()} | Rank {local_rank}: Successfully loaded chromosome mapping for {len(gene_chromosome_map_df)} genes.")
    except FileNotFoundError:
        print(f"PID: {os.getpid()} | Rank {local_rank}: Error: Gene-to-chromosome mapping file not found at {GENE_CHROMOSOME_MAPPING_PATH}.")
        print(f"PID: {os.getpid()} | Rank {local_rank}: Cannot perform chromosome-based split without this file.")
        sys.exit(1) # Use sys.exit for DDP context
    except Exception as e:
        print(f"PID: {os.getpid()} | Rank {local_rank}: Error loading gene-to-chromosome mapping file: {e}")
        sys.exit(1) # Use sys.exit for DDP context


    # Create a mapping dictionary from gene_id to chromosome
    gene_to_chr = gene_chromosome_map_df.set_index('gene_id')['chromosome'].to_dict()

    # Add chromosome information to the all_pairs_df
    print(f"PID: {os.getpid()} | Rank {local_rank}: Adding chromosome information to gene pairs.")
    all_pairs_df['Gene1_chromosome'] = all_pairs_df['Gene1'].map(gene_to_chr)
    all_pairs_df['Gene2_chromosome'] = all_pairs_df['Gene2'].map(gene_to_chr)

    # Handle cases where gene IDs in all_pairs_df are not in the mapping file
    if all_pairs_df['Gene1_chromosome'].isnull().any() or all_pairs_df['Gene2_chromosome'].isnull().any():
        print(f"PID: {os.getpid()} | Rank {local_rank}: Warning: Some gene IDs in gene pairs were not found in the chromosome mapping file.")
        print(f"PID: {os.getpid()} | Rank {local_rank}: These pairs will be excluded from all datasets.")
        all_pairs_df.dropna(subset=['Gene1_chromosome', 'Gene2_chromosome'], inplace=True)
        print(f"PID: {os.getpid()} | Rank {local_rank}: Remaining pairs after removing those with missing chromosome info: {len(all_pairs_df)}")


    # Perform chromosome-based split
    print(f"\nPID: {os.getpid()} | Rank {local_rank}: Performing chromosome-based data split.")

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


    print(f"PID: {os.getpid()} | Rank {local_rank}: Total pairs loaded: {len(all_pairs_df)}")
    print(f"PID: {os.getpid()} | Rank {local_rank}: Train pairs: {len(train_df)}")
    print(f"PID: {os.getpid()} | Rank {local_rank}: Validation pairs (Chr4-Chr4): {len(val_df)}")
    print(f"PID: {os.getpid()} | Rank {local_rank}: Test pairs (Chr2-Chr2): {len(test_df)}")


    # Save the test data to a file for later evaluation (only on rank 0)
    test_data_save_path = './data/test_gene_pairs.csv'
    if local_rank == 0 and not test_df.empty:
        try:
            os.makedirs('./data', exist_ok=True)
            test_df.to_csv(test_data_save_path, index=False)
            print(f"PID: {os.getpid()} | Rank {local_rank}: Test data saved to {test_data_save_path}")
        except Exception as e:
            print(f"PID: {os.getpid()} | Rank {local_rank}: Error saving test data: {e}")
    dist.barrier() # Ensure all ranks wait for rank 0 to save

    if train_df.empty:
        print(f"PID: {os.getpid()} | Rank {local_rank}: CRITICAL: Training DataFrame is empty after split. Cannot proceed with training.")
        sys.exit(1) # Force exit if train_df is empty
    else:
        train_dataset = GenePairDataset(feature_dir=current_feature_dir, gene_pairs_df=train_df)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, collate_fn=custom_collate_fn)
        print(f"PID: {os.getpid()} | Rank {local_rank}: Train DataLoader created. Batches: {len(train_loader)}, Samples: {len(train_dataset)}")

        val_loader = None
        if not val_df.empty:
            val_dataset = GenePairDataset(feature_dir=current_feature_dir, gene_pairs_df=val_df)
            if len(val_dataset) > 0:
                val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0, collate_fn=custom_collate_fn, drop_last=True)
                print(f"PID: {os.getpid()} | Rank {local_rank}: Validation DataLoader created. Batches: {len(val_loader)}, Samples: {len(val_dataset)}")
            else:
                print(f"PID: {os.getpid()} | Rank {local_rank}: Validation dataset created but is empty.")
        else:
            print(f"PID: {os.getpid()} | Rank {local_rank}: Validation DataFrame is empty. No validation loader will be used.")

        # Test loader (only for evaluation, should not use DistributedSampler if only evaluated by rank 0)
        test_loader = None # Initialize to None
        if not test_df.empty:
            test_dataset = GenePairDataset(feature_dir=current_feature_dir, gene_pairs_df=test_df)
            if len(test_dataset) > 0:
                # If test is only evaluated by rank 0 later, no sampler is needed.
                # If all ranks evaluate a portion, use DistributedSampler here too.
                # For now, assuming only rank 0 will use it.
                if local_rank == 0:
                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
                    print(f"PID: {os.getpid()} | Rank {local_rank}: Test DataLoader created. Batches: {len(test_loader)}, Samples: {len(test_dataset)}")
            else:
                print(f"PID: {os.getpid()} | Rank {local_rank}: Test dataset created but is empty.")

    print(f"PID: {os.getpid()} | Rank {local_rank}: Finished DataLoaders setup.")
    dist.barrier() # Ensure all ranks have created their DataLoaders

    # --- 2. Model, Loss, Optimizer, Scheduler Setup ---
    print(f"\nPID: {os.getpid()} | Rank {local_rank}: [Phase 2: Model Initialization]")
    print(f"PID: {os.getpid()} | Rank: {local_rank} | [Timestamp: {time.time()}]")
    model = SiameseGeneTransformer(
        input_feature_dim=INPUT_FEATURE_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        aggregation_method=AGGREGATION_METHOD,
        max_seq_len=MAX_SEQ_LEN,
        regression_hidden_dim=REGRESSION_HIDDEN_DIM,
        regression_dropout=REGRESSION_DROPOUT
    ).to(DEVICE)
    print(f"PID: {os.getpid()} | Rank {local_rank}: SiameseGeneTransformer model instantiated on {DEVICE}.")
    print(f"PID: {os.getpid()} | Rank {local_rank}: Model total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model = DistributedDataParallel(model, device_ids=[local_rank])
    print(f"PID: {os.getpid()} | Rank {local_rank}: Model wrapped with DistributedDataParallel.")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = None
    if val_loader:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=LR_SCHEDULER_PATIENCE, factor=0.1)
        print(f"PID: {os.getpid()} | Rank {local_rank}: ReduceLROnPlateau LR scheduler configured.")
    else:
        print(f"PID: {os.getpid()} | Rank {local_rank}: No validation data, so ReduceLROnPlateau scheduler will not be used.")

    print(f"PID: {os.getpid()} | Rank {local_rank}: Model and optimizer setup complete.")
    dist.barrier() # Ensure all ranks finish model setup

    # --- 3. Train the Model ---
    print(f"\nPID: {os.getpid()} | Rank {local_rank}: [Phase 3: Model Training]")
    print(f"PID: {os.getpid()} | Rank: {local_rank} | [Timestamp: {time.time()}]")

    log_dir = "./training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training_metrics.csv")
    print(f"PID: {os.getpid()} | Rank {local_rank}: Training metrics will be logged to: {log_file_path}")
    dist.barrier() # Ensure logging setup is complete

    try:
        if len(train_loader.dataset) == 0:
            print(f"PID: {os.getpid()} | Rank {local_rank}: Train dataset is empty. Skipping training.")
        else:
            print(f"PID: {os.getpid()} | Rank {local_rank}: Starting training for {NUM_EPOCHS} epochs with early stopping patience {EARLY_STOPPING_PATIENCE}...")
            trained_model, history = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                NUM_EPOCHS, DEVICE, EARLY_STOPPING_PATIENCE, MODEL_SAVE_PATH, log_file_path,
                local_rank, world_size
            )
            print(f"PID: {os.getpid()} | Rank {local_rank}: Training finished.")
            print(f"PID: {os.getpid()} | Rank {local_rank}: Best model saved to: {MODEL_SAVE_PATH}")
    except FileNotFoundError as e:
        print(f"\nPID: {os.getpid()} | Rank {local_rank}: CRITICAL ERROR DURING TRAINING: A required .npy file was not found: {e.filename}")
        print(f"PID: {os.getpid()} | Rank {local_rank}: This likely means that some gene IDs in your TSV file do not have corresponding .npy files in the feature directory.")
        print(f"PID: {os.getpid()} | Rank {local_rank}: Please check your data generation steps (Module 1 & 2) and ensure all necessary files are present.")
        sys.exit(1) # Ensure graceful exit on critical data error
    except Exception as e:
        import traceback
        print(f"\nPID: {os.getpid()} | Rank {local_rank}: An unexpected error occurred during model training: {e}")
        traceback.print_exc()
        sys.exit(1) # Ensure graceful exit on unexpected error


    print(f"\nPID: {os.getpid()} | Rank {local_rank}: --- End of Full Training Script Simulation ---")
    dist.barrier() # Final barrier before process destruction
    dist.destroy_process_group()