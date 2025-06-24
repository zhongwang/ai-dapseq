# Module 1: Data Curation and Preprocessing

## 1. Objective

The primary objective of this module is to process the raw genomic, transcriptomic, and co-expression data into a clean, normalized, and structured format suitable for the per-base feature generation in Module 2. The quality and consistency of this output are critical for the success of the entire project.

## 2. Key Responsibilities

- Ingest and parse all raw data files (promoter BED, FASTA, bigWig, TSV).
- Process promoter regions using the provided promoter BED file and extract corresponding DNA sequences from FASTA.
- Extract and normalize TF binding signals (e.g., from bigWig files) per base within the defined promoter regions.
- Prepare the dataset of gene pairs with their associated co-expression correlation coefficients (from the provided TSV).

## 3. Detailed Implementation Steps

### Step 3.1: Promoter Region Definition and DNA Sequence Extraction

-   **Inputs:**
    -   Promoter regions file (`promoter.bed` - BED format). This file should contain `chromosome`, `start`, `end`, `gene_id`, and `strand`.
    -   *A. thaliana* genome sequence file (FASTA format).
-   **Process:**
    1.  **Parse BED:** Read the `promoter.bed` file (e.g., using `pandas` or a dedicated BED parsing library) to get the coordinates and strand information for each gene's promoter.
    2.  **Extract DNA Sequences:** For each promoter region defined in the BED file, use `Biopython` (e.g., `SeqIO.index` for the FASTA file and then slicing/reverse-complementing based on strand) to extract the corresponding DNA sequence.
-   **Output:** A structured file (e.g., TSV or FASTA) containing `gene_id`, `chromosome`, `promoter_start`, `promoter_end`, `strand`, and the extracted `promoter_dna_sequence` for each gene.

### Step 3.2: Per-Base TF Binding Signal Extraction

-   **Inputs:**
    -   The file containing promoter coordinates and gene IDs from Step 3.1.
    -   Approximately 300 DAP-seq bigWig files.
-   **Process:**
    1.  **Iterate:** For each gene's promoter region (defined by `chromosome`, `promoter_start`, `promoter_end` from Step 3.1 output), iterate through all ~300 TF bigWig files.
    2.  **Extract Per-Base Signal:** Use the `pyBigWig` library to extract the signal for each base within the promoter interval. The `pyBigWig.values(chromosome, start, end)` function, called for each base or in segments as appropriate, can be used. Handle regions with no signal (NaNs) if necessary (e.g., by replacing with 0). The length of the signal vector will correspond to the length of the promoter.
    3.  **Store Data:** For each gene, store the TF binding signals. This could be a 2D array (matrix) of shape `(Number of TFs x Promoter Length)` where each row corresponds to a TF and each column to a base in the promoter. Store these matrices efficiently, for example, in NumPy's `.npy` format, one file per gene, using the `gene_id` for naming.
-   **Output:** A directory of `.npy` files, where each file contains the raw per-base TF binding signal matrix for a single gene's promoter.

### Step 3.3: Signal Normalization

-   **Rationale:** Raw DAP-seq signals are not directly comparable across different TFs or experiments due to technical biases (e.g., sequencing depth, antibody efficiency). This step is crucial for correcting these biases.
-   **Process (applied to the per-base signals from Step 3.2):**
    1.  **Log Transformation (Optional but common):** Apply a transformation like `log2(x + c)` (where `c` is a small pseudocount, e.g., 1) to the signals to stabilize variance and handle skewed distributions.
    2.  **Z-score Standardization (Across promoters for each TF):** For each TF, calculate the mean and standard deviation of its signal values across all bases of all promoters (or on a per-promoter basis, depending on strategy). Then, standardize the signals: `(signal - mean) / std_dev`.
-   **Output:** A directory of `.npy` files, analogous to the output of Step 3.2, but containing the *normalized* per-base TF binding signal matrices.

### Step 3.4: Co-expression Correlation Coefficient Dataset Preparation

-   **Input:** A TSV file containing gene pairs and their experimentally determined co-expression correlation coefficients (e.g., `gene_pairs_correlations.tsv` with columns `gene1_id`, `gene2_id`, `correlation_coefficient`).
-   **Process:**
    1.  **Load Data:** Load the TSV file using `pandas` into a DataFrame.
    2.  **Validation (Optional but Recommended):** Ensure that `gene1_id` and `gene2_id` in this file correspond to gene IDs for which promoter sequences and TF binding data have been processed in the previous steps. Filter pairs if necessary.
-   **Output:** A final TSV file (e.g., `final_coexpression_data.tsv`) with columns: `gene1_id`, `gene2_id`, `co_expression_correlation`. This file is ready for use in model training and evaluation.

## 4. Recommended Libraries

-   **`Biopython`**: For GFF3 and FASTA parsing.
-   **`pyBigWig`**: For efficient reading of bigWig files.
-   **`pandas`**: For handling tabular data (gene lists, co-expression pairs).
-   **`NumPy` / `SciPy`**: For numerical operations and signal processing.
-   **`multiprocessing` / `joblib`**: To parallelize the data extraction across multiple CPU cores.

## 5. Deliverables and Success Criteria

This module is complete when the following curated data files are generated, well-documented, and ready for Module 2:

1.  **Promoter DNA Sequences:** A file (e.g., FASTA or TSV) containing the DNA sequence for each promoter region, linked to `gene_id`. (Output of Step 3.1)
2.  **Normalized Per-Base TF Binding Signals:** A directory of `.npy` files, where each file (named by `gene_id`) contains a matrix of normalized TF binding affinity values (Number of TFs x Promoter Length) for the corresponding promoter. (Output of Step 3.3)
3.  **Gene Pair Co-expression Data:** A TSV file listing gene pairs (`gene1_id`, `gene2_id`) and their associated co-expression correlation coefficients. (Output of Step 3.4)

The successful generation and validation of these three components signify the completion of Module 1.