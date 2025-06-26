
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Attempt to import the model from the parent directory's 'model' folder
try:
    from ..model.siamese_transformer import SiameseGeneTransformer
except ImportError:
    # Fallback for running the script directly from the 'training' directory for testing
    # This assumes 'model' and 'training' are sibling directories.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from model.siamese_transformer import SiameseGeneTransformer

# --- Configuration ---
model_path = './output/across_chromosome_test/best_siamese_model.pth'
test_data_path = './data/test_gene_pairs.csv'
output_dir = "./output"
output_csv_path = os.path.join(output_dir, 'test_predictions.csv')
# Assuming feature vectors are in a directory relative to the workspace root or data dir
# Adjust FEATURE_VECTOR_DIR as necessary based on your project structure
FEATURE_VECTOR_DIR = "/global/scratch/users/sallyliao2027/aidapseq/output/across_chromosome_test/feature_vectors_by_chrom/"

# --- Model Hyperparameters (Must match training script) ---
# Example values, ensure these match the model you are loading
INPUT_FEATURE_DIM = 14
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
AGGREGATION_METHOD = 'cls'
MAX_SEQ_LEN = 2500
REGRESSION_HIDDEN_DIM = 128
REGRESSION_DROPOUT = 0.15
BATCH_SIZE = 32 # Use a reasonable batch size for evaluation

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- Dataset Class (Copied/Adapted from train.py) ---
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
            # In validation, we might skip this pair or raise an error
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


# --- Load Model and Data ---
try:
    # Instantiate the model with the same hyperparameters as training
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
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Set model to evaluation mode
    print(f"Successfully loaded model state dictionary from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the training script saves the best model to this location.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    # Load the test gene pairs data
    test_gene_pairs_df = pd.read_csv(test_data_path)
    print(f"Successfully loaded test gene pairs data from {test_data_path}")
    print(f"Test gene pairs data shape: {test_gene_pairs_df.shape}")
    
    # Create Dataset and DataLoader
    test_dataset = GenePairDataset(feature_dir=FEATURE_VECTOR_DIR, gene_pairs_df=test_gene_pairs_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # No shuffle for evaluation
    print(f"Test DataLoader created. Batches: {len(test_loader)}, Samples: {len(test_dataset)}")

except FileNotFoundError:
    print(f"Error: Test data file not found at {test_data_path}")
    print("Please ensure train.py is run first to generate this file.")
    exit()
except Exception as e:
    print(f"Error loading test data or creating DataLoader: {e}")
    exit()

# --- Make Predictions ---
predictions = []
true_values = []

print("Making predictions on test data...")
with torch.no_grad():
    for seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch, corr_batch in test_loader:
        seq_a_batch, mask_a_batch, seq_b_batch, mask_b_batch = \
            seq_a_batch.to(DEVICE), mask_a_batch.to(DEVICE), \
            seq_b_batch.to(DEVICE), mask_b_batch.to(DEVICE)
        
        try:
            predicted_correlations = model(seq_a_batch, seq_b_batch, key_padding_mask_A=mask_a_batch, key_padding_mask_B=mask_b_batch)
            predictions.append(predicted_correlations.squeeze().cpu().numpy())
            true_values.append(corr_batch.cpu().numpy())
        except Exception as e:
             print(f"Error during prediction batch: {e}")
             # Decide how to handle errors here - skip batch or raise
             continue

# Concatenate results
y_pred = np.concatenate(predictions)
y_test = np.concatenate(true_values)
print("Finished making predictions.")

# Save actual and predicted correlations to CSV
predictions_df = pd.DataFrame({
    'actual_correlation': y_test,
    'predicted_correlation': y_pred
})

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

predictions_df.to_csv(output_csv_path, index=False)
print(f"Saved actual and predicted correlations to {output_csv_path}")

# --- Calculate Metrics ---

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Metrics on Test Data ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# You can add more metrics or visualizations here as needed
# print(f"y_pred: {y_pred}")
# print(f"y_test: {y_test}")

