# Module 2: Feature Engineering

## 1. Objective

The objective of this module is to implement the feature engineering process. This involves creating tokenized feature vectors for each gene's promoter region. These vectors are generated by applying a sliding window to the concatenated DNA and TF binding signal data, followed by aggregation and clustering.

## 2. Key Responsibilities

-   Ingest the promoter DNA sequences and normalized per-base TF binding signals generated by Module 1.
-   Develop a robust pipeline to generate a tokenized feature vector for each promoter region.
-   Implement one-hot encoding for DNA sequences.
-   Concatenate one-hot encoded DNA with TF binding signals.
-   Apply a sliding window to the concatenated data.
-   Aggregate features within each window.
-   Cluster the aggregated window features to create a final tokenized vector.

## 3. Detailed Implementation Steps

### Step 3.1: Load Input Data from Module 1

-   **Inputs:**
    1.  **Promoter DNA Sequences:** A TSV file containing `gene_id` and `promoter_dna_sequence`.
    2.  **Normalized Per-Base TF Binding Signals:** A directory of `.npy` files, each named by `gene_id`, containing a NumPy matrix of shape `(Number_of_TFs, Promoter_Length)`.
-   **Process:**
    -   Load DNA sequences into a dictionary mapping `gene_id` to sequence.
    -   Load TF binding signals for each gene.

### Step 3.2: Prepare and Concatenate Features

-   **Process:**
    1.  **One-Hot Encode DNA:** Convert each DNA sequence into a one-hot encoded matrix of shape `(4, max_promoter_length)`.
    2.  **Process TF Signals:** Pad or truncate the TF binding signals to `max_promoter_length`, resulting in a matrix of shape `(Number_of_TFs, max_promoter_length)`.
    3.  **Concatenate:** Combine the one-hot DNA and TF signal matrices into a single feature matrix of shape `(4 + Number_of_TFs, max_promoter_length)`.

### Step 3.3: Tokenize Features

-   **Process:**
    1.  **Sliding Window:** Apply a sliding window to the feature matrix from Step 3.2.
    2.  **Aggregation:** For each window, aggregate the features using a specified method (mean, max, or sum). This results in a matrix of shape `(4 + Number_of_TFs, num_windows)`.
    3.  **Clustering:** Cluster the aggregated window features using KMeans to generate a tokenized feature vector of shape `(num_windows,)`.

### Step 3.4: Generate Final Promoter Representation

-   **Process:** The tokenized feature vector generated in Step 3.3 for each gene is its final feature-engineered representation.
-   **Store Data:** Save this final vector for each gene as a `.npy` file, named by `gene_id`.
-   **Output:** A directory of `.npy` files, where each file contains a vector of shape `(num_windows,)`.

## 4. Recommended Libraries

-   **`NumPy`**: For all numerical and matrix/vector operations.
-   **`pandas`**: For loading initial gene lists or DNA sequences.
-   **`scikit-learn`**: For KMeans clustering.
-   **`multiprocessing`**: For parallel processing of genes.

## 5. Deliverables and Success Criteria

This module is complete when, for all relevant genes:

1.  Input DNA sequences and normalized per-base TF binding signals from Module 1 have been successfully ingested.
2.  The data has been processed through the tokenization pipeline (one-hot encoding, concatenation, sliding window, aggregation, clustering).
3.  The resulting tokenized feature vector for each promoter is saved as a `.npy` file.

The primary deliverable is **a dataset (directory of `.npy` files) where each gene's promoter is represented as a 1D tokenized vector, ready for Module 3.**
