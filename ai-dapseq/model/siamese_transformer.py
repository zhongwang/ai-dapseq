import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Based on Section III.C of the research plan

class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer Encoder layer.
    Based on the standard Transformer architecture.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu # Using ReLU as a common choice, GeLU is also an option

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights # Optionally return attention weights for interpretability

class TransformerEncoder(nn.Module):
    """
    A stack of TransformerEncoderLayers.
    Forms one 'tower' of the Siamese network.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        all_attn_weights = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            all_attn_weights.append(attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, all_attn_weights # Return final output and attention weights from all layers

class PositionalEncoding(nn.Module):
    """
    Basic sinusoidal positional encoding.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe) # Register as buffer so it's part of the state_dict but not a learnable parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input embeddings
        # Ensure seq_len of input is not greater than max_len
        x = x + self.pe[:, :x.size(1), :]
        return x

class SiameseTransformer(nn.Module):
    """
    Siamese Transformer model for predicting co-expression between gene pairs.
    Each tower processes the TF vocabulary sequence for one gene's promoter.
    """
    def __init__(self, num_tfs, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_promoter_windows):
        super().__init__()

        # Input projection layer: maps num_TFs dimensional window vector to d_model
        self.input_projection = nn.Linear(num_tfs, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_promoter_windows)

        # Transformer Encoder Layer definition (shared weights between towers)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        # Transformer Encoder (the 'tower' - weights are shared by using the same instance)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm=nn.LayerNorm(d_model))

        # Classifier Head
        # Input to classifier head depends on how the two tower outputs are combined.
        # Using concatenation: input dim is 2 * d_model (if using CLS token or pooling)
        # Using concatenation + difference: input dim is 3 * d_model
        # Let's use concatenation of pooled outputs for simplicity initially.
        # Assuming we use mean pooling over the sequence dimension to get a fixed-size promoter embedding.
        classifier_input_dim = 2 * d_model # For concatenated mean-pooled embeddings

        # Classifier layers (example: two dense layers)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, dim_feedforward // 2), # Example hidden size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1), # Output size 1 for binary classification
            nn.Sigmoid() # Sigmoid for probability output
        )

    def forward_one_tower(self, x, mask=None, src_key_padding_mask=None):
        """
        Forward pass through one transformer tower.
        Args:
            x: Input sequence of TF vocabulary vectors [batch_size, seq_len, num_tfs]
        Returns:
            Promoter embedding [batch_size, d_model]
            Attention weights from all layers
        """
        # Project input features to d_model
        x = self.input_projection(x) # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = self.positional_encoding(x) # [batch_size, seq_len, d_model]

        # Pass through transformer encoder
        encoder_output, all_attn_weights = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask) # [batch_size, seq_len, d_model]

        # Obtain fixed-size promoter embedding from encoder output
        # Option 1: Use a CLS token (requires adding a token to input sequence)
        # Option 2: Mean pooling over the sequence length
        # Let's use mean pooling for simplicity
        promoter_embedding = torch.mean(encoder_output, dim=1) # [batch_size, d_model]

        return promoter_embedding, all_attn_weights

    def forward(self, gene1_features, gene2_features, gene1_mask=None, gene2_mask=None, gene1_key_padding_mask=None, gene2_key_padding_mask=None):
        """
        Forward pass through the Siamese network.
        Args:
            gene1_features: TF vocabulary sequence for gene 1 [batch_size, seq_len1, num_tfs]
            gene2_features: TF vocabulary sequence for gene 2 [batch_size, seq_len2, num_tfs]
            masks/key_padding_masks: Optional masks for handling variable sequence lengths or padding
        Returns:
            Co-expression probability [batch_size, 1]
            Attention weights from both towers (for interpretability)
        """
        # Process gene 1 features through the first tower
        # The forward_one_tower method uses the shared self.transformer_encoder
        gene1_embedding, gene1_attn = self.forward_one_tower(gene1_features, mask=gene1_mask, src_key_padding_mask=gene1_key_padding_mask)

        # Process gene 2 features through the second tower (uses the SAME shared weights)
        gene2_embedding, gene2_attn = self.forward_one_tower(gene2_features, mask=gene2_mask, src_key_padding_mask=gene2_key_padding_mask)

        # Combine the two promoter embeddings
        # Using concatenation as defined in __init__
        combined_embedding = torch.cat((gene1_embedding, gene2_embedding), dim=1) # [batch_size, 2 * d_model]

        # Pass combined embedding through the classifier head
        coexpression_prob = self.classifier(combined_embedding) # [batch_size, 1]

        return coexpression_prob, (gene1_attn, gene2_attn)

# Example Usage (for testing the model definition)
if __name__ == "__main__":
    # Define model hyperparameters (example values)
    NUM_TFS = 300 # Example number of TFs
    D_MODEL = 256 # Embedding dimension
    NHEAD = 8 # Number of attention heads
    NUM_ENCODER_LAYERS = 4 # Number of transformer encoder layers
    DIM_FEEDFORWARD = 1024 # Dimension of the feedforward network
    DROPOUT = 0.1 # Dropout rate
    MAX_PROMOTER_WINDOWS = 100 # Max sequence length (number of 50bp windows)

    # Instantiate the model
    model = SiameseTransformer(
        num_tfs=NUM_TFS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_promoter_windows=MAX_PROMOTER_WINDOWS
    )

    print("Model Architecture:")
    print(model)

    # Create dummy input data (batch size = 2)
    # Assume sequence length (number of windows) can vary per gene in a batch,
    # but for simplicity in this example, let's use fixed length.
    # In a real scenario, padding would be needed.
    batch_size = 2
    seq_len_gene1 = 99 # Example number of windows
    seq_len_gene2 = 99 # Example number of windows

    # Dummy TF vocabulary features: [batch_size, seq_len, num_tfs]
    dummy_gene1_features = torch.randn(batch_size, seq_len_gene1, NUM_TFS)
    dummy_gene2_features = torch.randn(batch_size, seq_len_gene2, NUM_TFS)

    # Perform a forward pass
    print("\nPerforming a dummy forward pass...")
    try:
        coexpression_probs, attention_weights = model(dummy_gene1_features, dummy_gene2_features)
        print(f"Output co-expression probabilities shape: {coexpression_probs.shape}")
        # print(f"Output attention weights (Gene 1, Layer 0) shape: {attention_weights[0][0].shape}") # Example shape
        # print(f"Output attention weights (Gene 2, Layer 0) shape: {attention_weights[1][0].shape}") # Example shape
        print("Dummy forward pass successful.")
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")

    # Example with different sequence lengths (requires padding and masks in a real scenario)
    # This dummy example does not handle padding, so seq_len must be the same.
    # To handle variable lengths, you would pad the shorter sequence and use key_padding_mask.
    # Example of how you *would* use masks if implemented:
    # gene1_key_padding_mask = torch.zeros(batch_size, seq_len_gene1, dtype=torch.bool) # True for padded positions
    # gene2_key_padding_mask = torch.zeros(batch_size, seq_len_gene2, dtype=torch.bool)
    # coexpression_probs, attention_weights = model(dummy_gene1_features, dummy_gene2_features,
    #                                               gene1_key_padding_mask=gene1_key_padding_mask,
    #                                               gene2_key_padding_mask=gene2_key_padding_mask)