# Predicting Gene Co-expression in Arabidopsis thaliana

This project aims to develop a machine learning model to predict gene co-expression in *Arabidopsis thaliana*. The core of the project is a Siamese transformer model that learns a "TF vocabulary" from DAP-seq data to predict co-expression between gene pairs, with the goal of providing novel insights into gene regulatory networks.

## Project Overview

The project follows a modular pipeline:
1.  **Data Preprocessing:** Ingests raw genomic, transcriptomic, and co-expression data and prepares it for feature engineering.
2.  **Feature Engineering:** Creates tokenized feature vectors for each gene's promoter region using a sliding window, aggregation, and clustering approach.
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
*   `python data_processing/count_chromosomal_pairs.py`: Prints out the total number of pairs within a chromosome over a set threshold. Change this threshold by changing line 55. 
*   `python data_processing/step1_5_rename_geneID.py`: If step 1 was done without cleaned gene IDs, run this script in between steps 1 and 2 to clean the IDs.
*   `python data_processing/step2_5_rename_signals.py`: If step 2 was done without cleaned gene IDs, run this script in between steps 2 and 3 to clean the IDs.
*   `python data_processing/step4_create_clustered_feature_vectors.py`: An alternate tokenization method involving creating clusters with base pair windows. Run instead of data_processing/step4_create_feature_vectors.py if you want to use this clustering method. 
*   `sbatch data_processing/tokenize.sh`: Run this shell script to use the clustered tokenization method, changing relevant sbatch parameters.

### 2. Model Training

To run the model without hyperparameter tuning, deploy a script similar to the following scripts to a HPC:
*   `sbatch training/train.sh` 
*   `sbatch training/train_a40.sh`
These two scripts do the same thing; the only difference is that the first uses H100 GPUs, while the second uses A40 GPUs. 

To run the model with a wandb sweep, which helps tune the hyperparameters, first sign into the wandb library. If you're new to wandb, refer to the following link to set up a wandb account and syncing: https://docs.wandb.ai/quickstart/. Once you set up your wandb account, run scripts in the following order, modifying relevant paths, parameters, and the unique wandb agent code. 
*   `wandb sweep training/sweep.yaml`: Creates the wandb sweep such that you can track metrics on the wandb website. Also generates a unique wandb agent code, which should be added to the next sbatch script. 
*   `sbatch training/train_array.sh`: Submits the job to run the wandb sweep. Ensure that you update the wandb agent code, relevant hyperparameters, and paths. 

If you do not have access to distributed GPUs, you may run `python data_processing/train.py` instead. 

### 3. Visualization
To get the test predictions and performance plots, run the following
*   `python data_processing/validate.py`: Gets the test predictions the trained model.
*   `python visualization/plot_performance_metrics.py`: Uses the test predictions to generate performance plots.

Auxiliary scripts: 
*   `python visualization/plot_target_distribution.py`: Plots the distribution of the target variable (in our case, co-expression correlation).

*Note: You may need to adjust paths and parameters within the scripts or provide them as command-line arguments if the scripts are designed to accept them.*

## Project Structure

```
ai-dapseq/
├── .git/
├── README.md
├── data_processing/
│   ├── plot_normalization_effects.py
│   ├── step1_extract_promoter_sequences.py
│   ├── step1_5_rename_geneID.py
│   ├── step2_extract_tf_binding_signals.py
│   ├── step2_5_rename_signals.py
│   ├── step3_prepare_coexpression_data.py
│   └── step4_create_clustered_feature_vectors.py
│   └── step4_create_feature_vectors.py
├── docs/
│   ├── implementation_plan_overview.md
│   ├── module_1_data_preprocessing.md
│   ├── module_2_feature_engineering.md
│   ├── module_3_model_implementation.md
│   ├── module_4_model_training.md
│   ├── module_5_visualization.md
│   └── research_plan.md
├── model/
│   └── siamese_transformer.py
├── requirements.txt
├── training/
│   └── sweep.yaml
│   └── train_a40.sh
│   └── train_array.sh
│   └── train_parallel_test.sh
│   └── train_siamese_transformer_wandb.py
│   └── train_siamese_transformer.py
│   └── train.py
│   └── train.sh
│   └── validate.py
└── visualization/
    └── plot_performance_metrics.py
    └── plot_target_distribution.py
```

## Modules

The project is divided into the following key modules:

*   **Module 1: Data Curation and Preprocessing:** Processes raw data into a structured format.
*   **Module 2: Feature Engineering:** Generates tokenized feature vectors for promoter regions using a sliding window, aggregation, and clustering approach.
*   **Module 3: Siamese Transformer Model Implementation:** Implements the Siamese transformer model architecture.
*   **Module 4: Model Training and Evaluation:** Manages the training, validation, and testing pipelines.
*   **Module 5: Visualization and Interpretation:** Develops scripts for visualizing results and interpreting model predictions.

Refer to the `docs/` directory for more detailed documentation on each module.

## Collaboration and Version Control

All code is managed using Git. Development should occur in separate feature branches, with pull requests for review to ensure code quality and smooth integration.
