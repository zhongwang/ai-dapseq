\
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len is the max number of tokens (seq_len including CLS)
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
                 dim_feedforward, dropout, aggregation_method='cls', max_seq_len=2000): # max_seq_len for input biological sequence
        super(PromoterTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.aggregation_method = aggregation_method.lower()

        # 1. Input Projection Layer
        self.input_projection = nn.Linear(input_feature_dim, d_model)

        # 2. Positional Encoding
        # Max length for PE should account for potential CLS token (+1)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 1)


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

    def forward(self, src, src_key_padding_mask=None):
        # src shape: (batch_size, sequence_length, feature_dimension)
        # src_key_padding_mask shape: (batch_size, sequence_length), True if padded

        # 1. Input Projection
        # projected_src shape: (batch_size, sequence_length, d_model)
        projected_src = self.input_projection(src)
        
        current_mask = src_key_padding_mask

        # 2. Add CLS token if using 'cls' aggregation
        if self.aggregation_method == 'cls':
            batch_size = src.size(0)
            # Repeat CLS token for each item in the batch
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) # shape (batch_size, 1, d_model)
            # Prepend CLS token to the projected sequence
            # projected_src shape: (batch_size, sequence_length + 1, d_model)
            projected_src = torch.cat((cls_tokens, projected_src), dim=1)
            
            # Adjust mask for CLS token: CLS token is never padded
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src.device)
                current_mask = torch.cat((cls_mask, src_key_padding_mask), dim=1)
            # If src_key_padding_mask was None, current_mask remains None (no padding for any token including CLS)
            # If aggregation is CLS and there was no mask, all tokens (incl CLS) are treated as unpadded.

        # 3. Add Positional Encoding
        # sequence_embed shape: (batch_size, effective_sequence_length, d_model)
        # effective_sequence_length is sequence_length or sequence_length + 1 (if CLS)
        # PositionalEncoding max_len was set to max_seq_len + 1 to handle this.
        sequence_embed = self.pos_encoder(projected_src * math.sqrt(self.d_model)) # Scale before pos encoding

        # 4. Pass through Transformer Encoder
        # transformer_output shape: (batch_size, effective_sequence_length, d_model)
        # current_mask shape: (batch_size, effective_sequence_length) or None
        transformer_output = self.transformer_encoder(sequence_embed, src_key_padding_mask=current_mask)

        # 5. Output Aggregation
        if self.aggregation_method == 'cls':
            # Use the output hidden state of the CLS token (assumed to be at index 0)
            # promoter_embedding shape: (batch_size, d_model)
            promoter_embedding = transformer_output[:, 0, :] 
        elif self.aggregation_method == 'mean':
            # Mean pooling over all output hidden states of the actual sequence (respecting padding)
            # For 'mean' aggregation, the CLS token should not have been added. So current_mask refers to original sequence.
            # transformer_output shape: (batch_size, sequence_length, d_model)
            # current_mask shape: (batch_size, sequence_length), True for padded
            
            if current_mask is None:
                # No padding mask provided, so mean over all tokens
                promoter_embedding = transformer_output.mean(dim=1)
            else:
                # Create a mask for values to keep (inverse of padding mask)
                # keep_mask shape: (batch_size, sequence_length, 1)
                keep_mask = (~current_mask).unsqueeze(-1).float()
                
                # Mask the output (zero out padded positions)
                masked_transformer_output = transformer_output * keep_mask
                
                # Sum the kept values
                summed_output = masked_transformer_output.sum(dim=1) # shape: (batch_size, d_model)
                
                # Count the number of kept values per sequence
                num_kept_elements = keep_mask.sum(dim=1) # shape: (batch_size, 1)
                num_kept_elements = torch.clamp(num_kept_elements, min=1e-9) # Avoid division by zero
                
                promoter_embedding = summed_output / num_kept_elements # shape: (batch_size, d_model)
        
        return promoter_embedding


class SiameseGeneTransformer(nn.Module):
    """
    Siamese network with two PromoterTransformerEncoder towers and a regression head.
    """
    def __init__(self, input_feature_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, dropout, aggregation_method='cls', max_seq_len=2000,
                 regression_hidden_dim=None, regression_dropout=0.1):
        super(SiameseGeneTransformer, self).__init__()

        self.d_model = d_model # d_model is the embedding dimension AFTER input projection by the tower

        # Instantiate a single PromoterTransformerEncoder (shared weights)
        self.transformer_encoder_tower = PromoterTransformerEncoder(
            input_feature_dim=input_feature_dim, # This is the dimension of the raw input features per base for the tower
            d_model=d_model, # This is the internal dimension of the transformer tower (output embedding dim)
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            aggregation_method=aggregation_method,
            max_seq_len=max_seq_len # Max length of the biological sequence itself (before CLS if used by tower)
        )

        # Regression Head
        if regression_hidden_dim is None:
            regression_hidden_dim = d_model # Default to d_model if not specified

        # The input to the regression head is concat(emb_A, emb_B, abs(emb_A - emb_B))
        # Each embedding (emb_A, emb_B) has dimension d_model (output of the tower)
        regression_input_dim = 3 * d_model
        
        self.regression_head = nn.Sequential(
            nn.Linear(regression_input_dim, regression_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=regression_dropout),
            nn.Linear(regression_hidden_dim, 1) # Output a single scalar value
            # No activation for standard regression, or nn.Tanh() if output scaled to [-1, 1]
        )

    def forward(self, promoter_sequence_A, promoter_sequence_B, key_padding_mask_A=None, key_padding_mask_B=None):
        # promoter_sequence_A shape: (batch_size, seq_len_A, feature_dim)
        # promoter_sequence_B shape: (batch_size, seq_len_B, feature_dim)
        # key_padding_mask_A/B shape: (batch_size, seq_len_A/B), True if padded (mask for the original biological sequence)

        # Pass each promoter sequence through the shared transformer encoder tower
        # embedding_A shape: (batch_size, d_model)
        embedding_A = self.transformer_encoder_tower(promoter_sequence_A, src_key_padding_mask=key_padding_mask_A)
        # embedding_B shape: (batch_size, d_model)
        embedding_B = self.transformer_encoder_tower(promoter_sequence_B, src_key_padding_mask=key_padding_mask_B)

        # Combine embeddings for the regression head
        abs_diff = torch.abs(embedding_A - embedding_B)
        # combined_vector shape: (batch_size, 3 * d_model)
        combined_vector = torch.cat((embedding_A, embedding_B, abs_diff), dim=1)

        # Pass the combined vector through the regression head
        # predicted_correlation shape: (batch_size, 1)
        predicted_correlation = self.regression_head(combined_vector)

        return predicted_correlation

if __name__ == '__main__':
    # Hyperparameters
    input_feature_dim = 4 + 300  # Example: 4 (DNA one-hot) + 300 TFs
    d_model = 256              # Hidden size of the model (embedding dimension from tower)
    nhead = 8                  # Number of attention heads
    num_encoder_layers = 4     # Number of layers in the transformer towers
    dim_feedforward = 1024     # Dimension of the feed-forward network
    dropout = 0.1              # Dropout rate
    aggregation_method_default = 'cls' # 'cls' or 'mean' for the first model instantiation
    max_seq_len_bio = 1000     # Max biological promoter sequence length.
                               # PositionalEncoding in tower will use max_seq_len_bio + 1.
    
    # Regression head specific params
    regression_hidden_dim = 128
    regression_dropout = 0.15

    # Instantiate the model (using aggregation_method_default)
    print(f"Instantiating SiameseGeneTransformer with '{aggregation_method_default}' aggregation...")
    model = SiameseGeneTransformer(
        input_feature_dim=input_feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        aggregation_method=aggregation_method_default,
        max_seq_len=max_seq_len_bio, # Pass biological max length
        regression_hidden_dim=regression_hidden_dim,
        regression_dropout=regression_dropout
    )
    print("Model instantiated successfully.")
    print(f"Model uses '{aggregation_method_default}' aggregation for promoter embeddings.")

    # Create dummy data for a forward pass
    batch_size = 32
    # Define sequence lengths for the biological part of the sequence for dummy data
    # These lengths must be <= max_seq_len_bio
    seq_len_A_bio_data = 800  
    seq_len_B_bio_data = 750
    
    dummy_promoter_A = torch.randn(batch_size, seq_len_A_bio_data, input_feature_dim)
    dummy_promoter_B = torch.randn(batch_size, seq_len_B_bio_data, input_feature_dim)
    
    # Create dummy key_padding_masks (all False, i.e., no padding for this basic test)
    # These masks are for the biological sequence part.
    dummy_mask_A = torch.zeros(batch_size, seq_len_A_bio_data, dtype=torch.bool)
    dummy_mask_B = torch.zeros(batch_size, seq_len_B_bio_data, dtype=torch.bool)

    print(f"\nPerforming forward pass with dummy data (aggregation: '{aggregation_method_default}'):")
    print(f"  Promoter A (bio) shape: {dummy_promoter_A.shape}, Mask A shape: {dummy_mask_A.shape}")
    print(f"  Promoter B (bio) shape: {dummy_promoter_B.shape}, Mask B shape: {dummy_mask_B.shape}")

    # Perform a forward pass
    try:
        output = model(dummy_promoter_A, dummy_promoter_B, key_padding_mask_A=dummy_mask_A, key_padding_mask_B=dummy_mask_B)
        print("\nForward pass successful!")
        print(f"  Output shape: {output.shape} (expected: (batch_size, 1))")
        assert output.shape == (batch_size, 1), f"Output shape mismatch! Expected ({batch_size}, 1), got {output.shape}"
        print("Output shape assertion passed.")

        # Test with 'mean' aggregation explicitly, including a case with padding
        print("\nTesting with 'mean' aggregation method...")
        model_mean_agg = SiameseGeneTransformer(
            input_feature_dim=input_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            aggregation_method='mean', # Explicitly use mean
            max_seq_len=max_seq_len_bio, 
            regression_hidden_dim=regression_hidden_dim,
            regression_dropout=regression_dropout
        )
        
        # Test with some padding for mean aggregation on Promoter A
        # seq_len_A_tensor_for_pad_test is the full length of the tensor passed to the model.
        # For this test, let's use the same tensor length as seq_len_A_bio_data.
        seq_len_A_tensor_for_pad_test = seq_len_A_bio_data 
        actual_content_len_A = seq_len_A_tensor_for_pad_test - 50 if seq_len_A_tensor_for_pad_test > 50 else seq_len_A_tensor_for_pad_test // 2
        
        dummy_promoter_A_padded = torch.randn(batch_size, seq_len_A_tensor_for_pad_test, input_feature_dim)
        dummy_mask_A_padded = torch.zeros(batch_size, seq_len_A_tensor_for_pad_test, dtype=torch.bool)
        if actual_content_len_A < seq_len_A_tensor_for_pad_test : # Ensure we are actually padding
             dummy_mask_A_padded[:, actual_content_len_A:] = True # Mark the end tokens as padding

        # Promoter B uses its original full length (seq_len_B_bio_data) and no explicit padding for this test case
        dummy_promoter_B_no_pad = dummy_promoter_B # from previous setup (length seq_len_B_bio_data)
        dummy_mask_B_no_pad = dummy_mask_B # from previous setup (all False, length seq_len_B_bio_data)


        print(f"  Mean Agg Test: Promoter A (tensor length {seq_len_A_tensor_for_pad_test}, content length {actual_content_len_A}) shape: {dummy_promoter_A_padded.shape}, Mask A (padded) shape: {dummy_mask_A_padded.shape}")
        print(f"  Mean Agg Test: Promoter B (tensor length {seq_len_B_bio_data}, content length {seq_len_B_bio_data}) shape: {dummy_promoter_B_no_pad.shape}, Mask B shape: {dummy_mask_B_no_pad.shape}")

        output_mean_agg = model_mean_agg(dummy_promoter_A_padded, dummy_promoter_B_no_pad, key_padding_mask_A=dummy_mask_A_padded, key_padding_mask_B=dummy_mask_B_no_pad)
        print("Forward pass with 'mean' aggregation (and padding on A) successful!")
        print(f"  Output shape (mean_agg): {output_mean_agg.shape}")
        assert output_mean_agg.shape == (batch_size, 1), f"Output shape mismatch (mean_agg)! Expected ({batch_size}, 1), got {output_mean_agg.shape}"
        print("Output shape assertion passed for 'mean' aggregation with padding.")

    except Exception as e:
        print(f"\nError during forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\nScript finished.")
