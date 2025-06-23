# Implementation Plan: Predicting Gene Co-expression in *Arabidopsis thaliana*

## 1. Project Vision and Goals

This document outlines the implementation plan for developing a machine learning model to predict gene co-expression in *Arabidopsis thaliana*. The project is based on the detailed research plan and is designed to be executed by a team of software engineers.

The primary goal is to build a Siamese transformer model that learns a "TF vocabulary" from DAP-seq data to predict co-expression between gene pairs. This will provide novel insights into gene regulatory networks.

## 2. High-Level Architecture

The project is divided into five distinct modules that form a sequential data processing and modeling pipeline. Each module has a clear set of responsibilities and deliverables, allowing for parallel development and clear ownership.

### Project Workflow

The overall workflow is as follows:

```mermaid
graph TD
    A[Raw Data <br/> GFF3, FASTA, bigWig, TSV] --> B{Module 1: Data Preprocessing};
    B --> C[Normalized TF Profiles & Labeled Gene Pairs];
    C --> D{Module 2: TF Vocabulary Feature Engineering};
    D --> E[Promoter "TF Vocabulary" Sequences];
    E --> F{Module 3: Siamese Transformer Implementation};
    F --> G[Trainable Model];
    G --> H{Module 4: Model Training & Evaluation};
    H --> I[Trained Model & Performance Metrics];
    I --> J{Module 5: Visualization & Interpretation};
    J --> K[Biological Insights & Co-expression Network];
```

## 3. Module Breakdown and Responsibilities

### Module 1: Data Curation and Preprocessing
- **Lead Engineer:** TBD
- **Objective:** To process the raw genomic and co-expression data into a clean, structured format suitable for downstream analysis.
- **Key Responsibilities:**
    - Ingest and parse all raw data files (GFF3, FASTA, bigWig, TSV).
    - Define and extract promoter regions for all genes.
    - Extract and normalize TF binding signals from DAP-seq data.
    - Prepare the labeled dataset of co-expressed and non-co-expressed gene pairs.
- **Deliverable:** A set of curated data files containing normalized TF binding profiles and labeled gene pairs.

### Module 2: "TF Vocabulary" Feature Engineering
- **Lead Engineer:** TBD
- **Objective:** To implement the core feature engineering step of creating the "TF Vocabulary".
- **Key Responsibilities:**
    - Develop a robust pipeline to transform normalized TF binding profiles into sequences of "TF vocabulary" summary vectors.
    - Implement the 50bp sliding window mechanism.
    - Aggregate TF signals within each window to create summary vectors.
- **Deliverable:** A dataset where each gene's promoter is represented as a sequence of TF summary vectors.

### Module 3: Siamese Transformer Model Implementation
- **Lead Engineer:** TBD
- **Objective:** To build the Siamese transformer model in PyTorch or TensorFlow.
- **Key Responsibilities:**
    - Implement the Siamese architecture with weight-sharing transformer encoders.
    - Build the core transformer encoder blocks (self-attention, feed-forward networks).
    - Implement the classifier head for predicting co-expression.
- **Deliverable:** A well-structured, documented, and trainable model script.

### Module 4: Model Training and Evaluation
- **Lead Engineer:** TBD
- **Objective:** To train, tune, and rigorously evaluate the model's performance.
- **Key Responsibilities:**
    - Develop the complete training, validation, and testing pipeline.
    - Implement data loaders and a gene-disjoint data splitting strategy.
    - Manage hyperparameter tuning experiments.
    - Track and report key performance metrics (AUPRC, F1-score).
- **Deliverable:** A trained model, performance metrics, and a report on the results of the training experiments.

### Module 5: Visualization and Interpretation
- **Lead Engineer:** TBD
- **Objective:** To create tools for visualizing model performance and interpreting its predictions.
- **Key Responsibilities:**
    - Develop scripts to plot performance curves (loss, AUPRC).
    - Implement methods to extract and visualize attention maps from the model.
    - Generate and visualize a predicted co-expression network.
- **Deliverable:** A suite of visualization scripts and a final report with interpreted results and biological insights.

## 4. Collaboration and Version Control

All code will be managed using Git. Engineers are expected to work in separate feature branches for their respective modules and submit pull requests for review. This will ensure code quality and a smooth integration process. Regular team meetings will be held to discuss progress, challenges, and ensure alignment across modules.