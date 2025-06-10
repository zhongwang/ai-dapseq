import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import sys

# Add the parent directory to the sys.path to import modules from data_processing and model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.siamese_transformer import SiameseTransformer # Import the model
# from data_processing.tf_vocabulary_generation import load_tf_vocabulary_features # Might need a function to load features for specific genes
# from data_processing.dataset_preparation import load_dataset # Assuming dataset_preparation saves a final TSV

# Placeholder paths - replace with actual paths
LABELED_DATASET_FILE = "../data_processing/coexpression_dataset.tsv" # Output from dataset_preparation.py
TF_VOCABULARY_DIR = "../data_processing/tf_vocabulary_features" # Output from tf_vocabulary_generation.py
MODEL_OUTPUT_DIR = "./trained_models" # Directory to save trained models

# Model Hyperparameters (Example values, should match siamese_transformer.py or be tuned)
NUM_TFS = 300 # Example number of TFs
D_MODEL = 256 # Embedding dimension
NHEAD = 8 # Number of attention heads
NUM_ENCODER_LAYERS = 4 # Number of transformer encoder layers
DIM_FEEDFORWARD = 1024 # Dimension of the feedforward network
DROPOUT = 0.1 # Dropout rate
MAX_PROMOTER_WINDOWS = 100 # Max sequence length (number of 50bp windows) - needs to match generation script output

# Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100 # Example number of epochs
WEIGHT_DECAY = 1e-3 # L2 regularization
PATIENCE = 10 # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoexpressionDataset(Dataset):
    """
    Custom Dataset for loading gene pair data and their TF vocabulary features.
    Loads features on demand to save memory.
    """
    def __init__(self, labeled_pairs_df, tf_vocabulary_dir, max_seq_len):
        self.labeled_pairs_df = labeled_pairs_df
        self.tf_vocabulary_dir = tf_vocabulary_dir
        self.max_seq_len = max_seq_len

        # Optional: Pre-load a mapping of gene_id to feature file path
        self.feature_files = {
            f.replace("_tf_vocabulary.npy", ""): os.path.join(tf_vocabulary_dir, f)
            for f in os.listdir(tf_vocabulary_dir) if f.endswith("_tf_vocabulary.npy")
        }

    def __len__(self):
        return len(self.labeled_pairs_df)

    def __getitem__(self, idx):
        gene1_id = self.labeled_pairs_df.iloc[idx]['Gene1_ID']
        gene2_id = self.labeled_pairs_df.iloc[idx]['Gene2_ID']
        label = self.labeled_pairs_df.iloc[idx]['Label'] # 0 or 1

        # Load TF vocabulary features for gene 1
        gene1_features = self._load_features(gene1_id)
        # Load TF vocabulary features for gene 2
        gene2_features = self._load_features(gene2_id)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return gene1_features, gene2_features, label_tensor

    def _load_features(self, gene_id):
        """Loads TF vocabulary features for a single gene."""
        file_path = self.feature_files.get(gene_id)
        if file_path and os.path.exists(file_path):
            features = np.load(file_path) # Shape: (num_windows, num_tfs)
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32)

            # Handle sequence length: pad or truncate to max_seq_len
            seq_len = features_tensor.size(0)
            if seq_len > self.max_seq_len:
                # Truncate
                features_tensor = features_tensor[:self.max_seq_len, :]
                # print(f"Warning: Truncating features for {gene_id} from {seq_len} to {self.max_seq_len}")
            elif seq_len < self.max_seq_len:
                # Pad
                padding_needed = self.max_seq_len - seq_len
                padding_tensor = torch.zeros(padding_needed, features_tensor.size(1), dtype=torch.float32)
                features_tensor = torch.cat((features_tensor, padding_tensor), dim=0)
                # print(f"Padding features for {gene_id} from {seq_len} to {self.max_seq_len}")

            return features_tensor
        else:
            # Return a tensor of zeros or NaNs if features are missing
            # Returning zeros assuming padding handles missing files implicitly
            print(f"Warning: Features not found for gene {gene_id}. Returning zeros.")
            return torch.zeros(self.max_seq_len, NUM_TFS, dtype=torch.float32) # Use NUM_TFS from config

# Collate function for DataLoader to handle batching and padding (if needed for variable lengths)
# Our Dataset already pads/truncates to max_seq_len, so a default collate should work,
# but a custom one is useful if we want dynamic padding in batches.
def collate_fn(batch):
    gene1_features, gene2_features, labels = zip(*batch)

    # Stack tensors
    gene1_features = torch.stack(gene1_features)
    gene2_features = torch.stack(gene2_features)
    labels = torch.stack(labels)

    # No need for padding masks if all sequences are padded/truncated to max_seq_len in __getitem__
    # If using dynamic padding, this is where you'd pad and create masks.

    return gene1_features, gene2_features, labels

def train(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    for gene1_features, gene2_features, labels in dataloader:
        gene1_features, gene2_features, labels = gene1_features.to(device), gene2_features.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # The model returns probability and attention weights (optional)
        outputs, _ = model(gene1_features, gene2_features)
        outputs = outputs.squeeze() # Remove the last dimension if it's 1

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for gene1_features, gene2_features, labels in dataloader:
            gene1_features, gene2_features, labels = gene1_features.to(device), gene2_features.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(gene1_features, gene2_features)
            outputs = outputs.squeeze()

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

            # Store labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            # Convert probabilities to binary predictions (threshold 0.5)
            all_predictions.extend((outputs.cpu().numpy() > 0.5).astype(int))

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Handle cases where there are no positive or negative samples in the batch/dataset
    # This can happen with small datasets or specific data splits
    try:
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auprc = np.nan # Or 0.0, depending on desired behavior for no positive samples

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = np.nan # Or 0.0 for no positive/negative samples

    # F1, Precision, Recall, Accuracy require both positive and negative predictions/labels
    # If only one class is present, these metrics might raise errors or be misleading.
    # We can calculate them only if both classes are present in the evaluation set.
    if len(np.unique(all_labels)) > 1:
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, all_predictions)
    else:
        f1, precision, recall, accuracy = np.nan, np.nan, np.nan, np.nan
        if len(all_labels) > 0:
             print(f"Warning: Only one class ({np.unique(all_labels)[0]}) present in evaluation set. Skipping F1, Precision, Recall, Accuracy.")


    metrics = {
        'loss': epoch_loss,
        'auprc': auprc,
        'auc_roc': auc_roc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

    return metrics

def main():
    print(f"Using device: {DEVICE}")

    # --- Data Loading and Splitting ---
    print("Loading labeled dataset...")
    try:
        labeled_pairs_df = pd.read_csv(LABELED_DATASET_FILE, sep='\t')
        print(f"Loaded {len(labeled_pairs_df)} labeled gene pairs.")
    except FileNotFoundError:
        print(f"Error: Labeled dataset file not found at {LABELED_DATASET_FILE}. Please run dataset_preparation.py first.")
        return
    except Exception as e:
        print(f"Error loading labeled dataset: {e}")
        return

    if labeled_pairs_df.empty:
        print("Labeled dataset is empty. Cannot proceed with training.")
        return

    # Crucial Data Splitting: Ensure no gene overlap between train/val/test sets
    # This is more complex than simple row-wise splitting.
    # Need to get all unique genes, split genes, then create pairs based on gene splits.

    all_genes_in_dataset = pd.unique(labeled_pairs_df[['Gene1_ID', 'Gene2_ID']].values.ravel())
    print(f"Total unique genes in dataset: {len(all_genes_in_dataset)}")

    # Split genes into train, val, test sets
    train_genes, temp_genes = train_test_split(all_genes_in_dataset, test_size=0.3, random_state=42) # 70% train, 30% temp
    val_genes, test_genes = train_test_split(temp_genes, test_size=0.5, random_state=42) # 15% val, 15% test

    print(f"Train genes: {len(train_genes)}")
    print(f"Validation genes: {len(val_genes)}")
    print(f"Test genes: {len(test_genes)}")

    # Create sets for faster lookup
    train_genes_set = set(train_genes)
    val_genes_set = set(val_genes)
    test_genes_set = set(test_genes)

    # Filter the labeled pairs DataFrame based on gene splits
    train_df = labeled_pairs_df[
        labeled_pairs_df.apply(lambda row: row['Gene1_ID'] in train_genes_set and row['Gene2_ID'] in train_genes_set, axis=1)
    ].reset_index(drop=True)

    val_df = labeled_pairs_df[
        labeled_pairs_df.apply(lambda row: row['Gene1_ID'] in val_genes_set and row['Gene2_ID'] in val_genes_set, axis=1)
    ].reset_index(drop=True)

    test_df = labeled_pairs_df[
        labeled_pairs_df.apply(lambda row: row['Gene1_ID'] in test_genes_set and row['Gene2_ID'] in test_genes_set, axis=1)
    ].reset_index(drop=True)

    print(f"Train pairs: {len(train_df)}")
    print(f"Validation pairs: {len(val_df)}")
    print(f"Test pairs: {len(test_df)}")

    if train_df.empty or val_df.empty or test_df.empty:
        print("One or more datasets are empty after splitting. This might happen with small input data or if genes are not well-distributed.")
        print("Consider adjusting the splitting strategy or using a larger dataset.")
        return

    # Create Dataset objects
    train_dataset = CoexpressionDataset(train_df, TF_VOCABULARY_DIR, MAX_PROMOTER_WINDOWS)
    val_dataset = CoexpressionDataset(val_df, TF_VOCABULARY_DIR, MAX_PROMOTER_WINDOWS)
    test_dataset = CoexpressionDataset(test_df, TF_VOCABULARY_DIR, MAX_PROMOTER_WINDOWS)

    # Create DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count() // 2 or 1) # Use half CPU cores for data loading
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 2 or 1)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count() // 2 or 1)

    # --- Model, Loss, and Optimizer ---
    model = SiameseTransformer(
        num_tfs=NUM_TFS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_promoter_windows=MAX_PROMOTER_WINDOWS # Pass max_len to positional encoding
    ).to(DEVICE)

    # Loss Function: Binary Cross-Entropy with potential weighting
    # Calculate class weights if there's class imbalance
    positive_count = train_df['Label'].sum()
    negative_count = len(train_df) - positive_count
    if positive_count > 0 and negative_count > 0:
        pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Use BCEWithLogitsLoss for numerical stability
        print(f"Using weighted BCE loss with pos_weight: {pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using unweighted BCE loss.")


    # Optimizer: AdamW recommended for transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning Rate Scheduler (Optional but recommended)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    # Using a simple step scheduler for now
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    # --- Training Loop ---
    print("Starting training...")
    best_val_metric = -float('inf') # Monitor AUPRC or F1-score
    epochs_without_improvement = 0
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model.pth")

    # Ensure model output directory exists
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        print(f"Created model output directory: {MODEL_OUTPUT_DIR}")


    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss = train(model, train_dataloader, optimizer, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, criterion, DEVICE)
        print(f"Validation Loss: {val_metrics['loss']:.4f}, AUPRC: {val_metrics['auprc']:.4f}, F1: {val_metrics['f1']:.4f}, AUC-ROC: {val_metrics['auc_roc']:.4f}")

        # Step the scheduler
        # scheduler.step(val_metrics['auprc']) # For ReduceLROnPlateau
        scheduler.step() # For StepLR

        # Check for early stopping
        current_val_metric = val_metrics['auprc'] # Or val_metrics['f1']
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path} with AUPRC: {best_val_metric:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"Validation metric did not improve. Patience: {epochs_without_improvement}/{PATIENCE}")
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    print("\nTraining finished.")

    # --- Final Evaluation on Test Set ---
    print("Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(DEVICE)
        print("Best model loaded.")

        print("\nEvaluating on test set...")
        test_metrics = evaluate(model, test_dataloader, criterion, DEVICE)
        print("--- Test Set Metrics ---")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test AUPRC: {test_metrics['auprc']:.4f}")
        print(f"Test F1-score: {test_metrics['f1']:.4f}")
        print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print("------------------------")
    else:
        print(f"Best model not found at {best_model_path}. Skipping final test evaluation.")


if __name__ == "__main__":
    main()