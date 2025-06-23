# Module 3: Siamese Transformer Model Implementation

## 1. Objective

The objective of this module is to implement the core predictive engine of the project: the Siamese transformer model. This involves translating the architectural design specified in the research plan into clean, efficient, and well-documented code using a modern deep learning framework.

## 2. Key Responsibilities

-   Build the overall Siamese network structure, which processes two gene promoter profiles in parallel.
-   Implement the transformer encoder blocks, which are the heart of the feature extraction process.
-   Construct the final classifier head that takes the outputs from the Siamese towers and makes a co-expression prediction.
-   Ensure the model code is modular, reusable, and adheres to best practices for the chosen framework.

## 3. Detailed Architectural and Implementation Steps

### Step 3.1: The Overall Siamese Architecture

-   **Concept:** The model will be a "Siamese network," meaning it will have two identical sub-networks (towers) that share the same weights.
-   **Process:**
    1.  Create a main model class (e.g., `SiameseTransformer`).
    2.  This class will instantiate a single `TransformerEncoder` module.
    3.  In the `forward` pass, the model will accept two inputs: `promoter_A` and `promoter_B`.
    4.  `promoter_A` will be passed through the `TransformerEncoder` to produce an embedding vector `embedding_A`.
    5.  `promoter_B` will be passed through the **same** `TransformerEncoder` instance to produce `embedding_B`.
    6.  These two embeddings will then be passed to the classifier head.

### Step 3.2: The Transformer Encoder Tower

This is the core component that processes a single promoter's "TF vocabulary" sequence.

-   **Input:** A tensor of shape `(batch_size, num_windows, num_tfs)`, e.g., `(64, 99, ~300)`.
-   **Components:**
    1.  **Input Projection Layer (Optional but Recommended):** A linear layer that projects the input `num_tfs` dimension to the model's hidden dimension (`d_model`). This allows the model to learn a richer initial representation.
        -   `Linear(in_features=num_tfs, out_features=d_model)`
    2.  **Positional Encoding:** The transformer needs explicit positional information. Add a standard sinusoidal or a learned positional encoding to the sequence embeddings. The encoding should have the same dimension as `d_model`.
    3.  **Transformer Encoder Layers:** The main body of the tower. This will be a stack of `N` identical encoder blocks.
        -   Each block contains:
            -   A **Multi-Head Self-Attention** layer.
            -   A **Feed-Forward Network** (typically two linear layers with a ReLU or GeLU activation).
            -   Layer normalization and residual connections (dropout is also applied here for regularization).
    4.  **Output Aggregation:** The encoder will output a sequence of hidden states. Aggregate these into a single fixed-size vector for the entire promoter.
        -   **Recommendation:** Prepend a special `[CLS]` token to the input sequence and use the corresponding output hidden state as the final promoter embedding. Alternatively, use mean pooling over all output hidden states.
-   **Output:** A tensor of shape `(batch_size, d_model)`, representing the learned embedding for the promoter.

### Step 3.3: The Classifier Head

-   **Input:** The two promoter embeddings from the towers, `embedding_A` and `embedding_B`.
-   **Process:**
    1.  **Combine Embeddings:** Combine the two embeddings to capture their relationship. A robust strategy is to concatenate the two vectors along with their element-wise difference:
        -   `combined_vector = concat(embedding_A, embedding_B, abs(embedding_A - embedding_B))`
    2.  **Classification Layers:** Pass the `combined_vector` through a small multi-layer perceptron (MLP).
        -   For example: `Linear -> ReLU -> Dropout -> Linear -> Sigmoid`.
        -   The final layer must have a single output neuron and a sigmoid activation function to produce a probability score between 0 and 1.
-   **Output:** A single value per input pair, representing the predicted probability of co-expression.

## 4. Recommended Libraries and Frameworks

-   **`PyTorch` (Recommended)** or **`TensorFlow`**: Choose one framework and stick with it. PyTorch is often favored for its flexibility in research and custom model development.
    -   Use `torch.nn.TransformerEncoder` and `torch.nn.TransformerEncoderLayer` as building blocks in PyTorch.
-   **`NumPy`**: For any necessary numerical operations.

## 5. Key Hyperparameters to Expose

Your model implementation should make it easy to configure the following hyperparameters:

-   `d_model`: The hidden size of the model (e.g., 256).
-   `nhead`: The number of attention heads (e.g., 8).
-   `num_encoder_layers`: The number of layers in the transformer towers (e.g., 4).
-   `dim_feedforward`: The dimension of the feed-forward network (e.g., 1024).
-   `dropout`: The dropout rate (e.g., 0.1).

## 6. Success Criteria

This module is complete when there is a fully implemented, well-documented Python script containing the `SiameseTransformer` model. The script should be runnable, and it should be possible to instantiate the model and perform a forward pass with dummy data of the correct shape without errors.