# Module 4: Model Training and Evaluation

## 1. Objective

The objective of this module is to take the implemented model from Module 3 and the feature-engineered data from Module 2, and then train, validate, and test it to achieve the best possible predictive performance. This involves creating a robust training pipeline, systematically evaluating performance, and tuning hyperparameters.

## 2. Key Responsibilities

-   Develop the complete end-to-end training and validation script.
-   Implement a data loading and splitting strategy that prevents data leakage.
-   Configure the loss function, optimizer, and learning rate scheduler.
-   Systematically track and log performance metrics, with a focus on those robust to class imbalance.
-   Implement early stopping to prevent overfitting.

## 3. Detailed Implementation Steps

### Step 3.1: Data Loading and Splitting

-   **Inputs:**
    -   The labeled gene pairs TSV file from Module 1.
    -   The directory of "TF vocabulary" sequence `.npy` files from Module 2.
-   **Process:**
    1.  **Create a Custom Dataset Class:** In PyTorch, create a `torch.utils.data.Dataset` subclass.
        -   The `__init__` method will load the gene pairs TSV.
        -   The `__len__` method will return the total number of pairs.
        -   The `__getitem__` method will take an index, retrieve the corresponding gene pair (`gene1_id`, `gene2_id`) and its `label`, and then load the associated "TF vocabulary" `.npy` files from disk for each gene. This implements lazy loading, which is crucial for memory efficiency.
    2.  **Gene-Disjoint Data Splitting:** This is a critical step to ensure the model generalizes to unseen genes.
        -   First, collect a unique list of all genes present in the co-expression dataset.
        -   Split this **list of genes** into training, validation, and testing sets (e.g., 70%/15%/15%).
        -   Create the final train/validation/test sets of **gene pairs** based on this gene split. For example, a pair `(A, B)` is in the test set only if *both* gene A and gene B are in the test set list of genes. This prevents any gene seen during training from appearing in the validation or test sets.
    3.  **Create DataLoaders:** Use `torch.utils.data.DataLoader` to create iterable data loaders for the training, validation, and testing sets. This will handle batching, shuffling (for the training set), and parallel data loading.

### Step 3.2: The Training Loop

-   **Components:**
    1.  **Instantiate Model, Optimizer, and Loss Function:**
        -   Model: `model = SiameseTransformer(...)`
        -   Loss Function: `criterion = torch.nn.BCELoss()` (Binary Cross-Entropy Loss). Consider using `pos_weight` to handle class imbalance if necessary.
        -   Optimizer: `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)`
    2.  **Learning Rate Scheduler:**
        -   Implement a scheduler to adjust the learning rate during training. `ReduceLROnPlateau` (monitoring validation AUPRC) or `CosineAnnealingLR` are good choices.
    3.  **The Loop:**
        -   Iterate for a fixed number of `epochs`.
        -   **Training Phase:**
            -   Set the model to training mode: `model.train()`.
            -   Iterate through the training `DataLoader`.
            -   For each batch, move data to the GPU.
            -   Zero the gradients: `optimizer.zero_grad()`.
            -   Perform the forward pass: `outputs = model(promoter_A, promoter_B)`.
            -   Calculate the loss: `loss = criterion(outputs, labels)`.
            -   Perform backpropagation: `loss.backward()`.
            -   Update the weights: `optimizer.step()`.
        -   **Validation Phase:**
            -   Set the model to evaluation mode: `model.eval()`.
            -   Disable gradient calculations: `with torch.no_grad():`.
            -   Iterate through the validation `DataLoader`.
            -   Calculate validation loss and performance metrics.
            -   Update the learning rate scheduler: `scheduler.step(validation_auprc)`.

### Step 3.3: Performance Evaluation

-   **Metrics:**
    -   **Primary Metric:** **Area Under the Precision-Recall Curve (AUPRC)**. This is the most important metric given the likely class imbalance and the "possibly incomplete" nature of the positive labels.
    -   **Secondary Metrics:** F1-score, Precision, Recall, and AUC-ROC.
-   **Implementation:** Use `scikit-learn.metrics` to calculate these scores during the validation and final testing phases.

### Step 3.4: Experiment Tracking and Early Stopping

-   **Experiment Tracking:** Use a tool like `TensorBoard` or `Weights & Biases` to log training/validation loss, all performance metrics, and model hyperparameters for each run. This is essential for comparing experiments.
-   **Early Stopping:**
    -   Monitor a key validation metric (e.g., validation AUPRC or validation loss).
    -   Keep track of the best performing model weights.
    -   If the metric does not improve for a set number of "patience" epochs (e.g., 5-10), stop the training process to prevent overfitting and save the best model weights.

## 4. Recommended Libraries

-   **`PyTorch`** or **`TensorFlow`**: For the entire training pipeline.
-   **`scikit-learn`**: For data splitting and calculating performance metrics.
-   **`NumPy`**: For data manipulation.
-   **`TensorBoard` / `wandb`**: For experiment tracking and visualization.

## 5. Success Criteria

This module is complete when a robust, reproducible training script is created. The script should be able to train the model from start to finish, perform validation, log results, and save the best performing model based on a clear early stopping criterion. The final output is the trained model weights and a comprehensive log of its performance.