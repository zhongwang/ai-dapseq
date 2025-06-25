
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration --- 
model_path = './training/models/best_model.pkl'
validation_data_path = './data/validation_data.csv'

# --- Load Model and Data ---
try:
    model = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the training script saves the best model to this location.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    # Assuming the last column is the target variable
    validation_data = pd.read_csv(validation_data_path)
    X_val = validation_data.iloc[:, :-1]
    y_val = validation_data.iloc[:, -1]
    print(f"Successfully loaded validation data from {validation_data_path}")
    print(f"Validation data shape: Features {X_val.shape}, Target {y_val.shape}")
except FileNotFoundError:
    print(f"Error: Validation data file not found at {validation_data_path}")
    print("Please ensure validation data is available at this location.")
    exit()
except Exception as e:
    print(f"Error loading validation data: {e}")
    exit()

# --- Make Predictions ---
try:
    y_pred = model.predict(X_val)
    print("Successfully made predictions on validation data.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# --- Calculate Metrics ---

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\n--- Regression Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# You can add more metrics or visualizations here as needed

