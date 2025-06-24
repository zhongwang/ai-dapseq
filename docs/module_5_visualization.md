# Module 5: Visualization and Interpretation

## 1. Objective

A predictive model is only as useful as our ability to understand and trust it. The objective of this module is to move beyond raw performance metrics (like MSE or Pearson correlation from Module 4) and create visualizations and analyses that provide deep insights into the model's behavior, its predictions of co-expression correlation coefficients, and the biological patterns it has learned from the per-base DNA and TF affinity features.

## 2. Key Responsibilities

-   Visualize the model's training process and final regression performance.
-   Implement techniques to interpret the model's decision-making process, focusing on the transformer's attention mechanism applied to the concatenated per-base input sequences.
-   Use the trained model to generate a genome-wide co-expression network where edge weights represent predicted correlation coefficients.
-   Perform biological validation of the model's novel high-confidence correlation predictions.

## 3. Detailed Implementation Steps

### Step 3.1: Performance Visualization

-   **Inputs:** The training logs (e.g., from TensorBoard or a CSV file) and the saved model predictions (actual vs. predicted correlation coefficients) from Module 4.
-   **Process:**
    1.  **Training Curves:** Plot the training and validation loss curves (e.g., MSE loss) over epochs. This is essential for diagnosing issues like overfitting or underfitting.
    2.  **Regression Metric Curves:** Plot key validation regression metrics (e.g., Pearson Correlation Coefficient, MAE) over epochs. This shows how performance evolved during training.
    3.  **Predicted vs. Actual Scatter Plot:** For the final test set predictions, create a scatter plot of predicted co-expression correlation coefficients versus the true (actual) correlation coefficients. Ideally, points should cluster around the y=x line. Calculate and display the Pearson correlation on this plot.
    4.  **Residual Plots:** Plot the residuals (errors: `actual_correlation - predicted_correlation`) against the predicted values. This helps check for homoscedasticity, outliers, and systematic errors in predictions.
-   **Output:** A set of publication-quality plots summarizing the model's regression performance and diagnostic characteristics.

### Step 3.2: Attention Map Interpretation

-   **Rationale:** The self-attention mechanism in the transformer learns to assign importance scores (attention weights) to different parts of the input sequence (which are per-base concatenated DNA and TF affinity vectors). By visualizing these weights, we can see which base positions and associated features the model "focused on" when making a prediction for a promoter.
-   **Process:**
    1.  **Hook into the Model:** Modify the `SiameseGeneTransformer` model from Module 3 to output the attention weights from its encoder layers. This can be done by adding a flag to the `forward` method or by using PyTorch hooks.
    2.  **Select Interesting Pairs/Genes:** Choose some gene pairs from the test set for which the model made accurate (or interestingly inaccurate) predictions of correlation coefficients.
    3.  **Extract and Aggregate Weights:** For a given gene's promoter sequence (concatenated per-base features), pass it through the encoder and extract the attention weights from one or more layers. You will need to aggregate the weights from the different attention heads (e.g., by averaging or taking the max).
    4.  **Visualize:** Create a heatmap where the x-axis represents the base positions along the promoter sequence, and the y-axis could represent different layers or aggregated head attention. The color intensity at each position shows the attention score for that specific base. This will create a visual map of "important" base positions within the promoter that contributed to its final embedding.
-   **Output:** A script that can take a gene ID (and potentially its partner in a pair) as input and generate a plot of its promoter attention map(s).

### Step 3.3: Co-expression Network Generation

-   **Rationale:** The model is trained to predict pairwise co-expression correlation coefficients. We can leverage this to build a co-expression network for a large set of genes, where edges represent the strength and direction (positive/negative) of the predicted correlation.
-   **Process:**
    1.  **Large-Scale Prediction:** Create a large set of candidate gene pairs (e.g., all-vs-all for genes of interest, or all genes against a set of known TFs).
    2.  **Run Inference:** Use the trained model to predict the co-expression **correlation coefficient** for all these pairs.
    3.  **Filter and Format:** Keep only the pairs with a predicted absolute correlation coefficient above a certain confidence threshold (e.g., `|correlation| > 0.7`, or top/bottom X% of predictions). Format this list as a network file (e.g., an edge list: `gene1 gene2 predicted_correlation`).
    4.  **Visualize:** Import the network file into a dedicated network visualization tool.
        -   **Recommendation:** **`Cytoscape`** is the industry standard for biological network visualization. It allows for rich, interactive exploration, and edge weights can be mapped to thickness or color to represent correlation strength/sign.
-   **Output:** A network file (e.g., `.sif` or `.csv` with edge attributes for correlation) and a high-quality visualization of the predicted co-expression network.

### Step 3.4: Biological Validation

-   **Rationale:** To confirm that the model has learned biologically meaningful patterns, we need to compare its novel high-confidence correlation predictions against existing biological knowledge.
-   **Process:**
    1.  **Identify Novel High-Confidence Predictions:** Find gene pairs that the model predicts as strongly co-expressed (high positive or negative correlation) but were **not** in the original training set or had weak/unknown prior evidence.
    2.  **Identify Network Modules/Communities:** In the network generated in Step 3.3, use community detection algorithms (e.g., Louvain, MCODE within Cytoscape) to find densely connected clusters or modules of genes based on the predicted correlations.
    3.  **Perform GO Term Enrichment:** For these novel pairs or modules, perform a Gene Ontology (GO) term enrichment analysis. If the genes in a predicted module (e.g., a set of genes with strong positive predicted correlations among them) are significantly enriched for specific biological processes, it provides strong evidence that the model's predictions are biologically relevant.
-   **Output:** A report summarizing the results of the GO term enrichment analysis, providing biological validation for the model's findings.

## 4. Recommended Libraries and Tools

-   **`Matplotlib` / `Seaborn`**: For static plotting (performance curves, scatter plots, heatmaps).
-   **`scikit-learn.metrics`**: For calculating regression metrics like MSE, MAE, R-squared if not directly computed.
-   **`SciPy.stats`**: For Pearson/Spearman correlation if needed for analysis.
-   **`Cytoscape`**: For network visualization and analysis (e.g., MCODE for module detection).
-   **GO Enrichment Tools**: Web-based tools like PANTHER, DAVID, Metascape, or programmatic libraries in Python (e.g., `goatools`) or R (e.g., `clusterProfiler`).

## 5. Success Criteria

This module is complete when there is a suite of scripts and reports that clearly visualize the model's regression performance, interpret its key learned features (attention on per-base inputs), and provide biological validation for its novel predicted co-expression correlations. The final output should be a compelling story about what the model learned about gene regulation in *A. thaliana* through the lens of predicted co-expression relationships.
