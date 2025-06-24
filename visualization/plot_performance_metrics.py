import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Placeholder for loading training logs (e.g., from a CSV file)
def load_training_logs(log_file_path):
    """
    Loads training logs from a CSV file.
    Expected columns: 'epoch', 'train_loss', 'val_loss', 
                      'val_pearson_r', 'val_mae', etc.
    """
    # Example: return pd.read_csv(log_file_path)
    print(f"INFO: Attempting to load training logs from: {log_file_path}")
    # Replace with actual loading logic
    data = {
        'epoch': range(1, 51),
        'train_loss': np.random.rand(50) * 0.1 + 0.05,
        'val_loss': np.random.rand(50) * 0.1 + 0.08,
        'val_pearson_r': np.random.rand(50) * 0.3 + 0.5,
        'val_mae': np.random.rand(50) * 0.1 + 0.1
    }
    return pd.DataFrame(data)

# Placeholder for loading model predictions (actual vs. predicted)
def load_model_predictions(predictions_file_path):
    """
    Loads model predictions from a CSV file.
    Expected columns: 'actual_correlation', 'predicted_correlation'
    """
    # Example: return pd.read_csv(predictions_file_path)
    print(f"INFO: Attempting to load model predictions from: {predictions_file_path}")
    # Replace with actual loading logic
    data = {
        'actual_correlation': np.random.rand(100) * 2 - 1,
        'predicted_correlation': np.random.rand(100) * 2 - 1
    }
    # Ensure some correlation for a more realistic plot
    data['predicted_correlation'] = data['actual_correlation'] * 0.7 + (np.random.rand(100) * 0.6 - 0.3)
    return pd.DataFrame(data)

def plot_training_curves(logs_df, output_path="training_loss_curves.png"):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(logs_df['epoch'], logs_df['train_loss'], label='Training Loss (MSE)')
    plt.plot(logs_df['epoch'], logs_df['val_loss'], label='Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"SUCCESS: Training curves plot saved to {output_path}")

def plot_regression_metric_curves(logs_df, output_path_prefix="validation_metrics"):
    """Plots validation regression metrics (Pearson R, MAE) over epochs."""
    # Pearson Correlation Coefficient
    plt.figure(figsize=(10, 6))
    plt.plot(logs_df['epoch'], logs_df['val_pearson_r'], label='Validation Pearson Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Validation Pearson Correlation Over Epochs')
    plt.legend()
    plt.grid(True)
    pearson_plot_path = f"{output_path_prefix}_pearson_correlation.png"
    plt.savefig(pearson_plot_path)
    plt.close()
    print(f"SUCCESS: Pearson correlation plot saved to {pearson_plot_path}")

    # Mean Absolute Error (MAE)
    plt.figure(figsize=(10, 6))
    plt.plot(logs_df['epoch'], logs_df['val_mae'], label='Validation MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Validation MAE Over Epochs')
    plt.legend()
    plt.grid(True)
    mae_plot_path = f"{output_path_prefix}_mae.png"
    plt.savefig(mae_plot_path)
    plt.close()
    print(f"SUCCESS: MAE plot saved to {mae_plot_path}")

def plot_predicted_vs_actual(predictions_df, output_path="predicted_vs_actual.png"):
    """Creates a scatter plot of predicted vs. actual correlation coefficients."""
    actual = predictions_df['actual_correlation']
    predicted = predictions_df['predicted_correlation']
    
    pearson_r, _ = pearsonr(actual, predicted)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=actual, y=predicted, alpha=0.6)
    plt.plot([min(actual.min(), predicted.min()), max(actual.max(), predicted.max())],
             [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())],
             'k--', lw=2, label="y=x (Ideal)") # y=x line
    plt.xlabel('Actual Correlation Coefficient')
    plt.ylabel('Predicted Correlation Coefficient')
    plt.title(f'Predicted vs. Actual Correlation Coefficients\nPearson R: {pearson_r:.3f}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.savefig(output_path)
    plt.close()
    print(f"SUCCESS: Predicted vs. Actual scatter plot saved to {output_path}")

def plot_residual_plot(predictions_df, output_path="residual_plot.png"):
    """Plots residuals against predicted values."""
    actual = predictions_df['actual_correlation']
    predicted = predictions_df['predicted_correlation']
    residuals = actual - predicted
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predicted, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Correlation Coefficient')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"SUCCESS: Residual plot saved to {output_path}")

def main():
    print("--- Running Performance Visualization Script ---")
    # Define file paths for data (these will need to be actual paths from Module 4 outputs)
    training_logs_path = "path/to/your/training_logs.csv" # TODO: Update this path
    predictions_path = "path/to/your/test_predictions.csv" # TODO: Update this path
    
    # Define output directory for plots
    output_dir = "/Users/zhongwang/scratch/ai-dapseq/visualization/plots" # Store plots in a sub-directory

    # Create output directory if it doesn't exist (Python 3.5+)
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    # For now, using placeholder data. Replace with actual data loading.
    logs_df = load_training_logs(training_logs_path) # Assumes this function is updated to load real data
    predictions_df = load_model_predictions(predictions_path) # Assumes this function is updated to load real data

    if logs_df is not None:
        plot_training_curves(logs_df, output_path=os.path.join(output_dir, "training_loss_curves.png"))
        plot_regression_metric_curves(logs_df, output_path_prefix=os.path.join(output_dir, "validation_metrics"))
    else:
        print(f"WARNING: Could not load training logs from {training_logs_path}. Skipping training plots.")

    if predictions_df is not None:
        plot_predicted_vs_actual(predictions_df, output_path=os.path.join(output_dir, "predicted_vs_actual.png"))
        plot_residual_plot(predictions_df, output_path=os.path.join(output_dir, "residual_plot.png"))
    else:
        print(f"WARNING: Could not load predictions from {predictions_path}. Skipping prediction plots.")
    
    print("--- Performance Visualization Script Finished ---")

if __name__ == "__main__":
    main()
