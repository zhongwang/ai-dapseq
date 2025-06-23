# Module 5: Visualization and Interpretation

## 1. Objective

A predictive model is only as useful as our ability to understand and trust it. The objective of this module is to move beyond raw performance metrics and create visualizations and analyses that provide deep insights into the model's behavior, its predictions, and the biological patterns it has learned.

## 2. Key Responsibilities

-   Visualize the model's training process and final performance.
-   Implement techniques to interpret the model's decision-making process, focusing on the transformer's attention mechanism.
-   Use the trained model to generate a genome-wide co-expression network.
-   Perform biological validation of the model's novel predictions.

## 3. Detailed Implementation Steps

### Step 3.1: Performance Visualization

-   **Inputs:** The training logs (e.g., from TensorBoard or a CSV file) and the saved model predictions from Module 4.
-   **Process:**
    1.  **Training Curves:** Plot the training and validation loss curves over epochs. This is essential for diagnosing issues like overfitting or underfitting.
    2.  **Metric Curves:** Plot the primary validation metric (AUPRC) and key secondary metrics (F1-score, etc.) over epochs. This shows how performance evolved during training.
    3.  **Precision-Recall and ROC Curves:** For the final test set predictions, generate and plot the full Precision-Recall (PR) curve and the ROC curve. This gives a complete picture of the model's trade-offs between precision and recall at different probability thresholds.
-   **Output:** A set of publication-quality plots summarizing the model's performance.

### Step 3.2: Attention Map Interpretation

-   **Rationale:** The self-attention mechanism in the transformer learns to assign importance scores (attention weights) to different parts of the input sequence. By visualizing these weights, we can see which promoter regions the model "focused on" when making a prediction.
-   **Process:**
    1.  **Hook into the Model:** Modify the `SiameseTransformer` model from Module 3 to output the attention weights from its encoder layers. This can be done by adding a flag to the `forward` method.
    2.  **Select Interesting Pairs:** Choose some high-confidence true positive predictions from the test set.
    3.  **Extract and Aggregate Weights:** For a given gene, pass its "TF vocabulary" sequence through the encoder and extract the attention weights from one or more layers. You will need to aggregate the weights from the different attention heads (e.g., by averaging).
    4.  **Visualize:** Create a heatmap where the x-axis represents the promoter (divided into the 99 windows) and the y-axis could represent different layers or heads. The color intensity at each position shows the attention score. This will create a visual map of "important" regions in the promoter.
-   **Output:** A script that can take a gene ID as input and generate a plot of its promoter attention map.

### Step 3.3: Co-expression Network Generation

-   **Rationale:** The model is trained to predict pairwise co-expression. We can leverage this to build a full co-expression network for a large set of genes.
-   **Process:**
    1.  **Large-Scale Prediction:** Create a large set of candidate gene pairs (e.g., all-vs-all for a subset of genes, or all genes against a set of known TFs).
    2.  **Run Inference:** Use the trained model to predict the co-expression probability for all these pairs.
    3.  **Filter and Format:** Keep only the pairs with a predicted probability above a certain confidence threshold (e.g., > 0.8). Format this list as a network file (e.g., a simple edge list: `gene1 gene2 probability`).
    4.  **Visualize:** Import the network file into a dedicated network visualization tool.
        -   **Recommendation:** **`Cytoscape`** is the industry standard for biological network visualization. It allows for rich, interactive exploration of the network structure.
-   **Output:** A network file (e.g., `.sif` or `.csv`) and a high-quality visualization of the predicted co-expression network.

### Step 3.4: Biological Validation

-   **Rationale:** To confirm that the model has learned biologically meaningful patterns, we need to compare its novel predictions against existing biological knowledge.
-   **Process:**
    1.  **Identify Novel Predictions:** Find gene pairs that the model predicts as co-expressed with high confidence but were **not** in the original training set.
    2.  **Identify Network Modules:** In the network generated in Step 3.3, use an algorithm (e.g., MCODE within Cytoscape) to find densely connected clusters or modules of genes.
    3.  **Perform GO Term Enrichment:** For these novel pairs or modules, perform a Gene Ontology (GO) term enrichment analysis. If the genes in a predicted module are significantly enriched for specific biological processes (e.g., "response to heat stress" or "flower development"), it provides strong evidence that the model's predictions are biologically relevant.
-   **Output:** A report summarizing the results of the GO term enrichment analysis, providing biological validation for the model's findings.

## 4. Recommended Libraries and Tools

-   **`Matplotlib` / `Seaborn`**: For static plotting (performance curves, heatmaps).
-   **`scikit-learn`**: For calculating PR and ROC curves.
-   **`Cytoscape`**: For network visualization and analysis.
-   **GO Enrichment Tools**: Web-based tools like PANTHER or programmatic libraries in Python or R.

## 5. Success Criteria

This module is complete when there is a suite of scripts and reports that clearly visualize the model's performance, interpret its key learned features (attention), and provide biological validation for its novel predictions. The final output should be a compelling story about what the model learned and what it can tell us about gene regulation in *A. thaliana*.