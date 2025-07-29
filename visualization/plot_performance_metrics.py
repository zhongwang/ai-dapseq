import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import argparse # Added

# Placeholder for loading training logs (e.g., from a CSV file)
def load_training_logs(log_file_path):
    """
    Loads training logs from a CSV file.
    Expected columns: 'epoch', 'train_loss', 'val_loss', 
                      'val_pearson_r', 'val_mae', etc.
    """
    # Example: return pd.read_csv(log_file_path)
    print(f"INFO: Attempting to load training logs from: {log_file_path}")

    if log_file_path == "placeholder_logs":
        # Replace with actual loading logic
        data = {
            'epoch': range(1, 51),
            'train_loss': np.random.rand(50) * 0.1 + 0.05,
            'val_loss': np.random.rand(50) * 0.1 + 0.08,
            'val_pearson_r': np.random.rand(50) * 0.3 + 0.5,
            'val_mae': np.random.rand(50) * 0.1 + 0.1
        }
        return pd.DataFrame(data)

    return pd.read_csv(log_file_path)

# Placeholder for loading model predictions (actual vs. predicted)
def load_model_predictions(predictions_file_path):
    """
    Loads model predictions from a CSV file.
    Expected columns: 'actual_correlation', 'predicted_correlation'
    """
    # Example: return pd.read_csv(predictions_file_path)
    print(f"INFO: Attempting to load model predictions from: {predictions_file_path}")

    if predictions_file_path == "placeholder_predictions":
        # Replace with actual loading logic
        data = {
            'actual_correlation': np.random.rand(100) * 2 - 1,
            'predicted_correlation': np.random.rand(100) * 2 - 1
        }
        # Ensure some correlation for a more realistic plot
        data['predicted_correlation'] = data['actual_correlation'] * 0.7 + (np.random.rand(100) * 0.6 - 0.3)
        return pd.DataFrame(data)

    return pd.read_csv(predictions_file_path)

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

def plot_predicted_vs_actual(predictions_df: pd.DataFrame, output_path="predicted_vs_actual.png", sample_size: int = None):
    """
    Creates a scatter plot of predicted vs. actual correlation coefficients.
    The Pearson R in the title is calculated for the *entire* dataset,
    while the plot itself can show a sample.

    Args:
        predictions_df (pd.DataFrame): DataFrame with 'Actual_Correlation' and 'Predicted_Correlation' columns.
        output_path (str): Path to save the plot image.
        sample_size (int, optional): The number of data points to sample for plotting.
                                     If None, all data points are plotted.
                                     If the DataFrame has fewer rows than sample_size, all rows are used.
    """
    # --- Calculate Pearson R for the ENTIRE dataset FIRST ---
    actual_full = predictions_df['Actual_Correlation']
    predicted_full = predictions_df['Predicted_Correlation']
    pearson_r_full, _ = pearsonr(actual_full, predicted_full)
    print(f"Pearson R for the entire dataset: {pearson_r_full:.3f}")

    # --- Determine the data to plot (full or sampled) ---
    if sample_size is not None and len(predictions_df) > sample_size:
        # Sample the DataFrame if sample_size is provided and data is larger
        # Using random_state for reproducibility of the sample
        data_to_plot = predictions_df.sample(n=sample_size, random_state=42)
        print(f"Plotting a sample of {len(data_to_plot)} data points.")
    else:
        data_to_plot = predictions_df
        if sample_size is not None:
            print(f"Sample size ({sample_size}) is greater than or equal to total data points ({len(predictions_df)}). Plotting all data.")
        else:
            print(f"Plotting all {len(data_to_plot)} data points.")

    actual_plot = data_to_plot['Actual_Correlation']
    predicted_plot = data_to_plot['Predicted_Correlation']

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=actual_plot, y=predicted_plot, alpha=0.6)
    plt.plot([min(actual_plot.min(), predicted_plot.min()), max(actual_plot.max(), predicted_plot.max())],
             [min(actual_plot.min(), predicted_plot.min()), max(actual_plot.max(), predicted_plot.max())],
             'k--', lw=2, label="y=x (Ideal)") # y=x line
    plt.xlabel('Actual Correlation Coefficient')
    plt.ylabel('Predicted Correlation Coefficient')
    # --- Use the pearson_r_full in the title ---
    plt.title(f'Predicted vs. Actual Correlation Coefficients\nPearson R: {pearson_r_full:.3f}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.savefig(output_path)
    plt.close()
    print(f"SUCCESS: Predicted vs. Actual scatter plot saved to {output_path}")


def plot_residual_plot(predictions_df, output_path="residual_plot.png"):
    """Plots residuals against predicted values."""
    actual = predictions_df['Actual_Correlation']
    predicted = predictions_df['Predicted_Correlation']
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

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot model performance metrics from training logs and prediction files.")
    parser.add_argument("--training_logs_file", type=str, required=False,
                        help="Path to the CSV file containing training logs (epoch, train_loss, val_loss, etc.).")
    parser.add_argument("--predictions_file", type=str, required=False,
                        help="Path to the CSV file containing actual vs. predicted correlations.")
    parser.add_argument("--output_dir", type=str, default="./visualization_plots",
                        help="Directory to save the output plots (default: ./visualization_plots).")
    parser.add_argument("--sample_size", type = int, default = 1000, required = False)
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("--- Running Performance Visualization Script ---")
    
    # Use arguments for file paths and output directory
    training_logs_path = args.training_logs_file
    predictions_path = args.predictions_file
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    print(f"INFO: Plots will be saved to: {os.path.abspath(output_dir)}")

    # Load data
    logs_df = None
    if training_logs_path:
        try:
            # User should replace load_training_logs with actual loading logic if this placeholder is insufficient
            logs_df = load_training_logs(training_logs_path) 
        except FileNotFoundError:
            print(f"WARNING: Training logs file not found at {training_logs_path}. Using placeholder data for logs.")
            logs_df = load_training_logs("placeholder_logs") # Call with a dummy path to trigger placeholder
        except Exception as e:
            print(f"ERROR: Could not load or parse training logs from {training_logs_path}: {e}. Using placeholder data.")
            logs_df = load_training_logs("placeholder_logs")
    else:
        print("INFO: --training_logs_file not provided. Using placeholder data for logs.")
        logs_df = load_training_logs("placeholder_logs") # Placeholder data

    predictions_df = None
    if predictions_path:
        try:
            # User should replace load_model_predictions with actual loading logic
            predictions_df = load_model_predictions(predictions_path)
        except FileNotFoundError:
            print(f"WARNING: Predictions file not found at {predictions_path}. Using placeholder data for predictions.")
            predictions_df = load_model_predictions("placeholder_predictions") # Placeholder data
        except Exception as e:
            print(f"ERROR: Could not load or parse predictions from {predictions_path}: {e}. Using placeholder data.")
            predictions_df = load_model_predictions("placeholder_predictions")
    else:
        print("INFO: --predictions_file not provided. Using placeholder data for predictions.")
        predictions_df = load_model_predictions("placeholder_predictions") # Placeholder data

    if logs_df is not None and not logs_df.empty:
        plot_training_curves(logs_df, output_path=os.path.join(output_dir, "training_loss_curves.png"))
        # Check if metric columns exist before plotting
        if 'val_pearson_r' in logs_df.columns and 'val_mae' in logs_df.columns:
            plot_regression_metric_curves(logs_df, output_path_prefix=os.path.join(output_dir, "validation_metrics"))
        else:
            print("WARNING: 'val_pearson_r' or 'val_mae' not found in training logs. Skipping regression metric plots.")
    else:
        print(f"WARNING: No training log data to plot. Logs path was: {training_logs_path}")

    if predictions_df is not None and not predictions_df.empty:
        plot_predicted_vs_actual(predictions_df, output_path=os.path.join(output_dir, "predicted_vs_actual.png"), sample_size = args.sample_size)
        plot_residual_plot(predictions_df, output_path=os.path.join(output_dir, "residual_plot.png"))
    else:
        print(f"WARNING: No prediction data to plot. Predictions path was: {predictions_path}")
    
    print("--- Performance Visualization Script Finished ---")

if __name__ == "__main__":
    main()
