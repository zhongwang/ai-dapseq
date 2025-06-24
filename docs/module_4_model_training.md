# Module 4: Model Training and Evaluation

## 1. Objective

The objective of this module is to take the implemented Siamese transformer model from Module 3 and the feature-engineered data (sequences of concatenated per-base DNA and TF affinity vectors) from Module 2, and then train, validate, and test it to achieve the best possible predictive performance for co-expression correlation. This involves creating a robust training pipeline, systematically evaluating regression performance, and tuning hyperparameters.

## 2. Key Responsibilities

-   Develop the complete end-to-end training and validation script for the regression task.
-   Implement a data loading and splitting strategy (gene-disjoint) that prevents data leakage.
-   Configure the appropriate regression loss function, optimizer, and learning rate scheduler.
-   Systematically track and log performance metrics relevant to regression (e.g., MSE, Pearson correlation).
-   Implement early stopping to prevent overfitting based on a validation regression metric.

## 3. Detailed Implementation Steps

### Step 3.1: Data Loading and Splitting

-   **Inputs:**
    -   The TSV file of gene pairs with their **co-expression correlation coefficients** from Module 1.
    -   The directory of `.npy` files from Module 2, where each file contains a promoter's sequence of **concatenated per-base feature vectors** (DNA one-hot + TF affinities).
-   **Process:**
    1.  **Create a Custom Dataset Class:** In PyTorch, create a `torch.utils.data.Dataset` subclass.
        -   The `__init__` method will load the gene pairs TSV (containing `gene1_id`, `gene2_id`, `co_expression_correlation`).
        -   The `__len__` method will return the total number of pairs.
        -   The `__getitem__` method will take an index, retrieve the corresponding `gene1_id`, `gene2_id`, and its `co_expression_correlation` (float value), and then load the associated `.npy` files (concatenated per-base feature sequences) from disk for each gene. This implements lazy loading.
    2.  **Gene-Disjoint Data Splitting:** This is critical.
        -   Collect a unique list of all genes present in the co-expression dataset.
        -   Split this **list of genes** into training, validation, and testing sets (e.g., 70%/15%/15%).
        -   Create the final train/validation/test sets of **gene pairs** based on this gene split. A pair `(A, B)` is in the test set only if *both* gene A and gene B are in the test set list of genes.
    3.  **Create DataLoaders:** Use `torch.utils.data.DataLoader` for training, validation, and testing sets. This handles batching, shuffling (for training), and parallel data loading.

### Step 3.2: The Training Loop

-   **Components:**
    1.  **Instantiate Model, Optimizer, and Loss Function:**
        -   Model: `model = SiameseGeneTransformer(...)` (from Module 3, configured with appropriate input feature dimension).
        -   Loss Function: `criterion = torch.nn.MSELoss()` (Mean Squared Error Loss), suitable for regressing against correlation coefficients.
        -   Optimizer: `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)` (or other appropriate optimizer).
    2.  **Learning Rate Scheduler (Optional but Recommended):**
        -   Implement a scheduler like `torch.optim.lr_scheduler.ReduceLROnPlateau` (monitoring validation MSE loss or another regression metric) or `torch.optim.lr_scheduler.CosineAnnealingLR`.
    3.  **The Loop:**
        -   Iterate for a fixed number of `epochs`.
        -   **Training Phase:**
            -   Set the model to training mode: `model.train()`.
            -   Iterate through the training `DataLoader`.
            -   For each batch, retrieve `promoter_sequence_A`, `promoter_sequence_B`, and `true_correlations`. Move data to the GPU.
            -   Zero the gradients: `optimizer.zero_grad()`.
            -   Perform the forward pass: `predicted_correlations = model(promoter_sequence_A, promoter_sequence_B)`. The output shape should be `(batch_size, 1)`.
            -   Calculate the loss: `loss = criterion(predicted_correlations.squeeze(), true_correlations.float())`. (Ensure dimensions match for loss calculation).
            -   Perform backpropagation: `loss.backward()`.
            -   Update the weights: `optimizer.step()`.
        -   **Validation Phase:**
            -   Set the model to evaluation mode: `model.eval()`.
            -   Disable gradient calculations: `with torch.no_grad():`.
            -   Iterate through the validation `DataLoader`.
            -   Calculate validation loss and performance metrics (see Step 3.3).
            -   If using `ReduceLROnPlateau`, update the scheduler: `scheduler.step(validation_mse_loss)`.

### Step 3.3: Performance Evaluation

-   **Metrics for Regression:**
    -   **Primary Metrics:**
        -   **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values.
        -   **Pearson Correlation Coefficient:** Measures the linear correlation between predicted and actual co-expression coefficients. This is very relevant to the task.
    -   **Secondary Metrics:**
        -   **Mean Absolute Error (MAE):** Measures the average absolute difference.
        -   **R-squared (Coefficient of Determination):** Represents the proportion of variance in the dependent variable that is predictable from the independent variable.
-   **Implementation:** Use `scikit-learn.metrics` (for R-squared, MAE) and `scipy.stats.pearsonr` (for Pearson correlation) or implement MSE directly. Calculate these during validation and final testing.

### Step 3.4: Experiment Tracking and Early Stopping

-   **Experiment Tracking:** Use a tool like `TensorBoard` or `Weights & Biases` to log training/validation loss, all performance metrics (MSE, Pearson correlation, MAE, R-squared), and model hyperparameters for each run.
-   **Early Stopping:**
    -   Monitor a key validation metric (e.g., validation MSE loss - lower is better, or validation Pearson correlation - higher is better).
    -   Keep track of the best performing model weights according to this metric.
    -   If the metric does not improve for a set number of "patience" epochs (e.g., 10-20), stop the training process to prevent overfitting and save the best model weights.

## 4. Recommended Libraries

-   **`PyTorch`**: For the entire training pipeline.
-   **`scikit-learn`**: For data splitting (gene-level) and calculating some performance metrics (MAE, R-squared).
-   **`SciPy`**: For Pearson correlation (`scipy.stats.pearsonr`).
-   **`NumPy`**: For data manipulation.
-   **`pandas`**: For loading the initial gene pair correlation data.
-   **`TensorBoard` / `wandb`**: For experiment tracking and visualization.

## 5. Success Criteria

This module is complete when a robust, reproducible training script is created. The script should be able to train the model from start to finish, perform validation using appropriate regression metrics, log results comprehensively, and save the best performing model based on a clear early stopping criterion (e.g., best validation Pearson correlation or lowest validation MSE). The final output is the trained model weights and a comprehensive log of its performance, demonstrating the model's ability to predict co-expression correlation coefficients.
