\
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is (batch_size, seq_len, d_model)
        # self.pe is (1, max_len, d_model)
        # We need to select the relevant part of pe: self.pe[:, :x.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PromoterTransformerEncoder(nn.Module):
    """
    Core component that processes a single promoter's sequence of concatenated per-base feature vectors.
    """
    def __init__(self, input_feature_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, dropout, aggregation_method='cls', max_seq_len=2000):
        super(PromoterTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.aggregation_method = aggregation_method.lower()

        # 1. Input Projection Layer
        self.input_projection = nn.Linear(input_feature_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: input tensors are (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Output Aggregation
        if self.aggregation_method == 'cls':
            # Learnable CLS token embedding
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            # Initialize CLS token (e.g., Xavier initialization)
            nn.init.xavier_uniform_(self.cls_token)
        elif self.aggregation_method not in ['mean']:
            raise ValueError(f"Unsupported aggregation_method: {aggregation_method}. Choose 'cls' or 'mean'.")

    def forward(self, src):
        # src shape: (batch_size, sequence_length, feature_dimension)

        # 1. Input Projection
        # projected_src shape: (batch_size, sequence_length, d_model)
        projected_src = self.input_projection(src)

        # 2. Add CLS token if using 'cls' aggregation
        if self.aggregation_method == 'cls':
            batch_size = src.size(0)
            # Repeat CLS token for each item in the batch
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Prepend CLS token to the projected sequence
            # projected_src shape: (batch_size, sequence_length + 1, d_model)
            projected_src = torch.cat((cls_tokens, projected_src), dim=1)

        # 3. Add Positional Encoding
        # sequence_embed shape: (batch_size, sequence_length or sequence_length+1, d_model)
        sequence_embed = self.pos_encoder(projected_src * math.sqrt(self.d_model)) # Scale before pos encoding

        # 4. Pass through Transformer Encoder
        # transformer_output shape: (batch_size, sequence_length or sequence_length+1, d_model)
        transformer_output = self.transformer_encoder(sequence_embed)

        # 5. Output Aggregation
        if self.aggregation_method == 'cls':
            # Use the output hidden state of the CLS token
            # promoter_embedding shape: (batch_size, d_model)
            promoter_embedding = transformer_output[:, 0, :] # CLS token is at index 0
        elif self.aggregation_method == 'mean':
            # Mean pooling over all output hidden states of the actual sequence
            # promoter_embedding shape: (batch_size, d_model)
            promoter_embedding = transformer_output.mean(dim=1)
        
        return promoter_embedding


class SiameseGeneTransformer(nn.Module):
    """
    Siamese network with two PromoterTransformerEncoder towers and a regression head.
    """
    def __init__(self, input_feature_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, dropout, aggregation_method='cls', max_seq_len=2000,
                 regression_hidden_dim=None, regression_dropout=0.1):
        super(SiameseGeneTransformer, self).__init__()

        self.d_model = d_model

        # Instantiate a single PromoterTransformerEncoder (shared weights)
        self.transformer_encoder_tower = PromoterTransformerEncoder(
            input_feature_dim=input_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            aggregation_method=aggregation_method,
            max_seq_len=max_seq_len
        )

        # Regression Head
        if regression_hidden_dim is None:
            regression_hidden_dim = d_model # Default to d_model if not specified

        # The input to the regression head is concat(emb_A, emb_B, abs(emb_A - emb_B))
        regression_input_dim = 3 * d_model
        
        self.regression_head = nn.Sequential(
            nn.Linear(regression_input_dim, regression_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=regression_dropout),
            nn.Linear(regression_hidden_dim, 1) # Output a single scalar value
            # No activation for standard regression, or nn.Tanh() if output scaled to [-1, 1]
        )

    def forward(self, promoter_sequence_A, promoter_sequence_B):
        # promoter_sequence_A shape: (batch_size, seq_len_A, feature_dim)
        # promoter_sequence_B shape: (batch_size, seq_len_B, feature_dim)

        # Pass each promoter sequence through the shared transformer encoder tower
        # embedding_A shape: (batch_size, d_model)
        embedding_A = self.transformer_encoder_tower(promoter_sequence_A)
        # embedding_B shape: (batch_size, d_model)
        embedding_B = self.transformer_encoder_tower(promoter_sequence_B)

        # Combine embeddings for the regression head
        abs_diff = torch.abs(embedding_A - embedding_B)
        # combined_vector shape: (batch_size, 3 * d_model)
        combined_vector = torch.cat((embedding_A, embedding_B, abs_diff), dim=1)

        # Pass the combined vector through the regression head
        # predicted_correlation shape: (batch_size, 1)
        predicted_correlation = self.regression_head(combined_vector)

        return predicted_correlation

if __name__ == '__main__':
    # 5. Key Hyperparameters to Expose
    input_feature_dim = 4 + 300  # Example: 4 (DNA one-hot) + 300 TFs
    d_model = 256              # Hidden size of the model
    nhead = 8                  # Number of attention heads
    num_encoder_layers = 4     # Number of layers in the transformer towers
    dim_feedforward = 1024     # Dimension of the feed-forward network
    dropout = 0.1              # Dropout rate
    aggregation_method = 'cls' # 'cls' or 'mean'
    max_seq_len = 1000         # Max promoter sequence length for positional encoding
    
    # Regression head specific params
    regression_hidden_dim = 128
    regression_dropout = 0.15

    # Instantiate the model
    print("Instantiating SiameseGeneTransformer...")
    model = SiameseGeneTransformer(
        input_feature_dim=input_feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        aggregation_method=aggregation_method,
        max_seq_len=max_seq_len,
        regression_hidden_dim=regression_hidden_dim,
        regression_dropout=regression_dropout
    )
    print("Model instantiated successfully.")
    print(f"Model uses '{aggregation_method}' aggregation for promoter embeddings.")

    # Create dummy data for a forward pass
    batch_size = 32
    seq_len_A = 800  # Can be different for A and B if model handles variable length
    seq_len_B = 750
    
    # Ensure dummy data matches expected input shape from Module 2
    # (batch_size, sequence_length, feature_dimension)
    dummy_promoter_A = torch.randn(batch_size, seq_len_A, input_feature_dim)
    dummy_promoter_B = torch.randn(batch_size, seq_len_B, input_feature_dim)

    print(f"\\nPerforming forward pass with dummy data:")
    print(f"  Promoter A shape: {dummy_promoter_A.shape}")
    print(f"  Promoter B shape: {dummy_promoter_B.shape}")

    # Perform a forward pass
    try:
        output = model(dummy_promoter_A, dummy_promoter_B)
        print("\\nForward pass successful!")
        print(f"  Output shape: {output.shape} (expected: (batch_size, 1))")
        assert output.shape == (batch_size, 1), f"Output shape mismatch! Expected ({batch_size}, 1), got {output.shape}"
        print("Output shape assertion passed.")

        # Test with 'mean' aggregation if 'cls' was default
        if aggregation_method == 'cls':
            print("\\nTesting with 'mean' aggregation method...")
            model_mean_agg = SiameseGeneTransformer(
                input_feature_dim=input_feature_dim,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                aggregation_method='mean', # Switch to mean
                max_seq_len=max_seq_len,
                regression_hidden_dim=regression_hidden_dim,
                regression_dropout=regression_dropout
            )
            output_mean_agg = model_mean_agg(dummy_promoter_A, dummy_promoter_B)
            print("Forward pass with 'mean' aggregation successful!")
            print(f"  Output shape (mean_agg): {output_mean_agg.shape}")
            assert output_mean_agg.shape == (batch_size, 1), f"Output shape mismatch (mean_agg)! Expected ({batch_size}, 1), got {output_mean_agg.shape}"
            print("Output shape assertion passed for 'mean' aggregation.")

    except Exception as e:
        print(f"\\nError during forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\\nScript finished.")
