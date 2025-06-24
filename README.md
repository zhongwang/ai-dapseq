# Predicting Gene Co-expression in Arabidopsis thaliana

This project aims to develop a machine learning model to predict gene co-expression in *Arabidopsis thaliana*. The core of the project is a Siamese transformer model that learns a "TF vocabulary" from DAP-seq data to predict co-expression between gene pairs, with the goal of providing novel insights into gene regulatory networks.

## Project Overview

The project follows a modular pipeline:
1.  **Data Preprocessing:** Ingests raw genomic, transcriptomic, and co-expression data and prepares it for feature engineering.
2.  **TF Vocabulary Feature Engineering:** Creates input sequences by concatenating per-base DNA sequence vectors and TF-DAPseq vectors.
3.  **Siamese Transformer Implementation:** Builds the Siamese transformer model in PyTorch.
4.  **Model Training & Evaluation:** Trains, tunes, and evaluates the model's performance.
5.  **Visualization & Interpretation:** Creates tools for visualizing model performance and interpreting predictions.

## Installation

1.  Clone the repository:
    ```shell
    git clone <repository-url>
    cd ai-dapseq
    ```
2.  Install the required dependencies:
    ```shell
    pip install -r requirements.txt
    ```

## Usage

The project is structured into several steps, primarily executed by Python scripts.

### 1. Data Processing

The scripts in the `data_processing/` directory should generally be run in the following order:

*   `python data_processing/step1_extract_promoter_sequences.py`: Extracts promoter sequences.
*   `python data_processing/step2_extract_tf_binding_signals.py`: Extracts TF binding signals.
*   `python data_processing/step3_prepare_coexpression_data.py`: Prepares gene co-expression data.
*   `python data_processing/step4_create_feature_vectors.py`: Creates feature vectors for the model.

Auxiliary scripts:
*   `python data_processing/plot_normalization_effects.py`: Plots the effects of normalization (can be run after relevant data processing steps).

### 2. Model Training

*   `python training/train.py`: Trains the Siamese transformer model using the processed data.

### 3. Visualization

*   `python visualization/plot_performance_metrics.py`: Plots performance metrics of the trained model.

*Note: You may need to adjust paths and parameters within the scripts or provide them as command-line arguments if the scripts are designed to accept them.*

## Project Structure

```
ai-dapseq/
├── .git/
├── README.md
├── data_processing/
│   ├── plot_normalization_effects.py
│   ├── step1_extract_promoter_sequences.py
│   ├── step2_extract_tf_binding_signals.py
│   ├── step3_prepare_coexpression_data.py
│   └── step4_create_feature_vectors.py
├── docs/
│   ├── implementation_plan_overview.md
│   ├── module_1_data_preprocessing.md
│   ├── module_2_tf_vocabulary.md
│   ├── module_3_model_implementation.md
│   ├── module_4_model_training.md
│   ├── module_5_visualization.md
│   └── research_plan.md
├── model/
│   └── siamese_transformer.py
├── requirements.txt
├── training/
│   └── train.py
└── visualization/
    └── plot_performance_metrics.py
```

## Modules

The project is divided into the following key modules:

*   **Module 1: Data Curation and Preprocessing:** Processes raw data into a structured format.
*   **Module 2: "TF Vocabulary" Feature Engineering:** Generates feature vectors for promoter regions by concatenating DNA sequence information and TF binding affinities.
*   **Module 3: Siamese Transformer Model Implementation:** Implements the Siamese transformer model architecture.
*   **Module 4: Model Training and Evaluation:** Manages the training, validation, and testing pipelines.
*   **Module 5: Visualization and Interpretation:** Develops scripts for visualizing results and interpreting model predictions.

Refer to the `docs/` directory for more detailed documentation on each module.

## Collaboration and Version Control

All code is managed using Git. Development should occur in separate feature branches, with pull requests for review to ensure code quality and smooth integration.
