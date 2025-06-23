# Module 1: Data Curation and Preprocessing

## 1. Objective

The primary objective of this module is to take the raw, disparate data sources and transform them into a clean, normalized, and structured format that can be directly consumed by the feature engineering and modeling pipelines. The quality and consistency of this output are critical for the success of the entire project.

## 2. Key Responsibilities

- Parse genomic annotation and sequence files to define promoter regions.
- Extract quantitative TF binding signals from a large set of bigWig files for each promoter.
- Implement a robust signal normalization pipeline to ensure comparability across TFs.
- Prepare the final labeled dataset of gene pairs for training, including a sound negative sampling strategy.

## 3. Detailed Implementation Steps

### Step 3.1: Promoter Region Definition and Extraction

-   **Inputs:**
    -   *A. thaliana* genome annotation file (GFF3 format).
    -   *A. thaliana* genome sequence file (FASTA format).
-   **Process:**
    1.  **Parse GFF3:** Use `Biopython` to parse the GFF3 file. Iterate through entries and identify all protein-coding genes. For each gene, extract its unique ID, chromosome, strand, and the start coordinate of the start codon.
    2.  **Define Promoter Coordinates:** For each gene, calculate the promoter region coordinates: **-2000bp to +500bp** relative to the start codon. You must correctly handle strand specificity.
    3.  **Extract Promoter Sequences (Optional but Recommended):** Extract the DNA sequence for each promoter. This is not strictly required for the model but is useful for validation and potential future analysis.
-   **Output:** A structured file (e.g., TSV) mapping each `gene_id` to its `chromosome`, `strand`, `promoter_start`, and `promoter_end`.

### Step 3.2: TF Binding Profile Extraction

-   **Inputs:**
    -   The promoter coordinates file from Step 3.1.
    -   Approximately 300 DAP-seq bigWig files.
-   **Process:**
    1.  **Iterate:** For each gene's promoter region, iterate through all ~300 TF bigWig files.
    2.  **Extract Signal:** Use the `pyBigWig` library to extract the base-pair resolution signal for the promoter interval (`chromosome`, `promoter_start`, `promoter_end`). The `pyBigWig.values()` function is recommended.
    3.  **Store Data:** This will generate a large amount of data. For each gene, you will have a matrix of size `(Number of TFs x 2501)`. Store these matrices efficiently, for example, in NumPy's `.npy` format, one file per gene.
-   **Output:** A directory of `.npy` files, where each file contains the raw TF binding signal matrix for a single gene's promoter.

### Step 3.3: Signal Normalization

-   **Rationale:** Raw DAP-seq signals are not directly comparable due to technical biases. This step is crucial for correcting these biases.
-   **Process:**
    1.  **Sequencing Depth Normalization:** Calculate a scaling factor for each TF's bigWig file (e.g., based on total signal across all promoters) and scale the signals accordingly.
    2.  **Background Subtraction (If control data is available):** If an input control bigWig file exists, subtract its normalized signal from the TF signals.
    3.  **Log Transformation:** Apply a `log2(x + c)` transformation (where `c` is a small pseudocount) to the signals to stabilize variance.
    4.  **Z-score Standardization:** For each TF, calculate the mean and standard deviation of its signal across all promoters and apply Z-score normalization.
-   **Output:** A directory of `.npy` files containing the *normalized* TF binding signal matrices.

### Step 3.4: Co-expression Dataset Preparation

-   **Input:** The TSV file of known co-expressed gene pairs.
-   **Process:**
    1.  **Load Positive Pairs:** Load the TSV file using `pandas`. These are your positive training examples (label = 1).
    2.  **Negative Sampling:**
        -   Randomly sample pairs of genes that are **not** in the positive set.
        -   The ratio of negative to positive samples is a key hyperparameter. Start with a 1:1 ratio.
        -   Ensure that sampled negative pairs do not include a gene paired with itself and are not just reversed versions of positive pairs.
    3.  **Create Final Dataset:** Combine the positive and sampled negative pairs into a single DataFrame with columns: `gene1_id`, `gene2_id`, `label`.
-   **Output:** A final TSV file containing the labeled gene pairs for training, validation, and testing.

## 4. Recommended Libraries

-   **`Biopython`**: For GFF3 and FASTA parsing.
-   **`pyBigWig`**: For efficient reading of bigWig files.
-   **`pandas`**: For handling tabular data (gene lists, co-expression pairs).
-   **`NumPy` / `SciPy`**: For numerical operations and signal processing.
-   **`multiprocessing` / `joblib`**: To parallelize the data extraction across multiple CPU cores.

## 5. Success Criteria

This module is complete when all raw data has been processed into a set of clean, normalized files that are ready for the next stage of the pipeline. The final deliverables should be well-documented and easily accessible to the engineers working on subsequent modules.