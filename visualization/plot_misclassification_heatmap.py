import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

def plot_misclassification_heatmap(csv_filepath, true_col='true_labels', pred_col='predicted_labels', output_filepath=None):
    """
    Generates a confusion matrix heatmap from a CSV file and either displays it or saves it to a file.

    Args:
        csv_filepath (str): The full path to the input CSV file.
        true_col (str): The name of the column containing true labels.
        pred_col (str): The name of the column containing predicted labels.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: File not found at {csv_filepath}")
        return

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if true_col not in df.columns or pred_col not in df.columns:
        print(f"Error: CSV must contain columns '{true_col}' and '{pred_col}'")
        return

    true_labels = df[true_col].astype(int)
    predicted_labels = df[pred_col].astype(int)

    # Ensure labels are integers and cover the full range 0, 1, 2...
    # Determine the unique classes and sort them to ensure correct order for ordinal data
    all_labels = sorted(np.unique(np.concatenate((true_labels, predicted_labels))))
    class_names = [str(c) for c in all_labels]

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')

    if output_filepath:
        try:
            plt.savefig(output_filepath, bbox_inches='tight')
            print(f"Heatmap saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving heatmap to {output_filepath}: {e}")
    else:
        plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    # Specify the column names for true and predicted labels if different from defaults
    true_label_column = 'actual_correlation'
    predicted_label_column = 'predicted_correlation'
    # ---------------------

    # Check for command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_misclassification_heatmap.py <path_to_your_predictions.csv> [output_image_path]")
        sys.exit(1) # Exit with an error code

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None # Get output path if provided

    print(f"Attempting to plot heatmap for file: {csv_file}")
    if output_file:
        print(f"Output will be saved to: {output_file}")
    else:
        print("Heatmap will be displayed on screen.")

    # Pass the output_file argument to the function
    plot_misclassification_heatmap(csv_file, true_label_column, predicted_label_column, output_filepath=output_file)
    print("Script finished.")