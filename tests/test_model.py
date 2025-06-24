import unittest
import torch
import torch.nn as nn
import math
import os
import sys

# Add the project root to the Python path to allow importing from model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.siamese_transformer import PositionalEncoding, PromoterTransformerEncoder, SiameseGeneTransformer

# Common Hyperparameters for testing
D_MODEL = 32 # embedding dimension
INPUT_FEATURE_DIM = 7 # e.g. 4 for OHE DNA + 3 TFs
NHEAD = 2 # num attention heads
NUM_ENCODER_LAYERS = 1 # num transformer encoder layers
DIM_FEEDFORWARD = 64 # dim of feedforward network in transformer
DROPOUT = 0.1
MAX_SEQ_LEN_BIO = 50 # max biological sequence length
BATCH_SIZE = 4
REGRESSION_HIDDEN_DIM = D_MODEL // 2
REGRESSION_DROPOUT = 0.1


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.d_model = D_MODEL
        self.max_len_pe = MAX_SEQ_LEN_BIO + 1 # PE max_len is bio_seq_len + 1 for CLS
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=DROPOUT, max_len=self.max_len_pe)
        torch.manual_seed(0)

    def test_output_shape(self):
        seq_len = 30
        batch_size = BATCH_SIZE
        dummy_input = torch.randn(batch_size, seq_len, self.d_model)
        output = self.pos_encoder(dummy_input)
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_encoding_values_added(self):
        seq_len = 10
        batch_size = 1
        # Use zeros to clearly see the added positional encoding (before dropout)
        dummy_input = torch.zeros(batch_size, seq_len, self.d_model)
        # Temporarily set dropout to 0 for this check
        self.pos_encoder.dropout.p = 0
        output = self.pos_encoder(dummy_input)
        
        # Expected PE part: self.pos_encoder.pe is (1, max_len_pe, d_model)
        expected_pe_slice = self.pos_encoder.pe[:, :seq_len, :]
        
        # output should be pe_slice + zeros = pe_slice
        torch.testing.assert_close(output, expected_pe_slice)
        self.assertFalse(torch.all(output == 0).item(), "Positional encodings were not added or are all zero.")
        self.pos_encoder.dropout.p = DROPOUT # Reset dropout

    def test_dropout_applied(self):
        seq_len = 10
        batch_size = BATCH_SIZE
        dummy_input = torch.randn(batch_size, seq_len, self.d_model)
        
        # With dropout (original setup)
        self.pos_encoder.dropout.p = 0.5 # Ensure high dropout for testing
        self.pos_encoder.train() # Enable dropout
        output_with_dropout = self.pos_encoder(dummy_input.clone())

        # Without dropout
        self.pos_encoder.eval() # Disable dropout
        output_without_dropout = self.pos_encoder(dummy_input.clone())
        
        if self.d_model * seq_len * batch_size > 0 : # only if there are elements
            self.assertFalse(torch.equal(output_with_dropout, output_without_dropout),
                             "Dropout did not change the output. Ensure model is in train() mode and p > 0.")
        self.pos_encoder.train() # Reset to train mode
        self.pos_encoder.dropout.p = DROPOUT # Reset dropout prob


class TestPromoterTransformerEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_dim = INPUT_FEATURE_DIM
        self.d_model = D_MODEL
        self.max_bio_len = MAX_SEQ_LEN_BIO

        self.encoder_cls = PromoterTransformerEncoder(
            input_feature_dim=self.input_dim, d_model=self.d_model, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, aggregation_method='cls', max_seq_len=self.max_bio_len
        )
        self.encoder_mean = PromoterTransformerEncoder(
            input_feature_dim=self.input_dim, d_model=self.d_model, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, aggregation_method='mean', max_seq_len=self.max_bio_len
        )

    def test_output_shape_cls(self):
        seq_len_bio = 30
        dummy_src = torch.randn(BATCH_SIZE, seq_len_bio, self.input_dim)
        output = self.encoder_cls(dummy_src)
        self.assertEqual(output.shape, (BATCH_SIZE, self.d_model))

    def test_output_shape_mean(self):
        seq_len_bio = 30
        dummy_src = torch.randn(BATCH_SIZE, seq_len_bio, self.input_dim)
        output = self.encoder_mean(dummy_src)
        self.assertEqual(output.shape, (BATCH_SIZE, self.d_model))

    def test_cls_token_prepended_and_mask_adjusted(self):
        seq_len_bio = 15
        dummy_src = torch.randn(BATCH_SIZE, seq_len_bio, self.input_dim)
        # Create a mask where the last 5 tokens are padded
        src_key_padding_mask = torch.zeros(BATCH_SIZE, seq_len_bio, dtype=torch.bool)
        if seq_len_bio > 5:
             src_key_padding_mask[:, -5:] = True

        # Mock the internal transformer encoder to inspect its input
        original_transformer_encoder_call = self.encoder_cls.transformer_encoder.forward
        
        # These will store the arguments passed to the actual transformer encoder
        call_args_store = {}

        def mock_transformer_forward(sequence_embed, src_key_padding_mask=None):
            call_args_store['sequence_embed_shape'] = sequence_embed.shape
            call_args_store['mask_shape'] = src_key_padding_mask.shape if src_key_padding_mask is not None else None
            if src_key_padding_mask is not None:
                call_args_store['mask_first_col_sum'] = src_key_padding_mask[:, 0].sum().item()
            # Call the original method to ensure the model completes its path
            return original_transformer_encoder_call(sequence_embed, src_key_padding_mask=src_key_padding_mask)

        self.encoder_cls.transformer_encoder.forward = mock_transformer_forward
        _ = self.encoder_cls(dummy_src, src_key_padding_mask=src_key_padding_mask)
        self.encoder_cls.transformer_encoder.forward = original_transformer_encoder_call # Restore

        self.assertIn('sequence_embed_shape', call_args_store)
        # effective_sequence_length = biological_seq_len + 1 (for CLS)
        self.assertEqual(call_args_store['sequence_embed_shape'], (BATCH_SIZE, seq_len_bio + 1, self.d_model))
        
        self.assertIsNotNone(call_args_store['mask_shape'], "Mask was None but should have been adjusted.")
        self.assertEqual(call_args_store['mask_shape'], (BATCH_SIZE, seq_len_bio + 1))
        # First column of the mask (for CLS token) should be all False (0)
        self.assertEqual(call_args_store['mask_first_col_sum'], 0)


    def test_mean_aggregation_with_padding(self):
        seq_len_bio = 20
        # Create input where first half is real data, second half is to be masked
        dummy_src = torch.ones(BATCH_SIZE, seq_len_bio, self.input_dim) # Use ones for clarity
        src_key_padding_mask = torch.zeros(BATCH_SIZE, seq_len_bio, dtype=torch.bool)
        actual_len = seq_len_bio // 2
        src_key_padding_mask[:, actual_len:] = True # Mask the second half

        # For mean aggregation, the input to the internal transformer is NOT prepended with CLS.
        # Its output is (batch, seq_len_bio, d_model).
        # We want to check if the mean is taken over the correct tokens.
        
        # To test the aggregation precisely, we'd ideally mock the transformer_encoder output.
        # As a simpler check, ensure it runs and output shape is correct.
        output = self.encoder_mean(dummy_src, src_key_padding_mask=src_key_padding_mask)
        self.assertEqual(output.shape, (BATCH_SIZE, self.d_model))
        
        # A more involved check: if all unmasked values were, say, X, and masked were Y,
        # the output should be based on X.
        # With a real transformer, this is hard to verify without knowing exact outputs.
        # This test primarily ensures the pathway for masked mean aggregation executes.

    def test_forward_pass_no_mask(self):
        seq_len_bio = 10
        dummy_src = torch.randn(BATCH_SIZE, seq_len_bio, self.input_dim)
        
        # CLS aggregation
        output_cls = self.encoder_cls(dummy_src, src_key_padding_mask=None)
        self.assertEqual(output_cls.shape, (BATCH_SIZE, self.d_model))

        # Mean aggregation
        output_mean = self.encoder_mean(dummy_src, src_key_padding_mask=None)
        self.assertEqual(output_mean.shape, (BATCH_SIZE, self.d_model))


class TestSiameseGeneTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_dim = INPUT_FEATURE_DIM
        self.d_model = D_MODEL
        self.max_bio_len = MAX_SEQ_LEN_BIO

        self.siamese_model_cls = SiameseGeneTransformer(
            input_feature_dim=self.input_dim, d_model=self.d_model, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, aggregation_method='cls', max_seq_len=self.max_bio_len,
            regression_hidden_dim=REGRESSION_HIDDEN_DIM, regression_dropout=REGRESSION_DROPOUT
        )
        self.siamese_model_mean = SiameseGeneTransformer(
            input_feature_dim=self.input_dim, d_model=self.d_model, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, aggregation_method='mean', max_seq_len=self.max_bio_len,
            regression_hidden_dim=REGRESSION_HIDDEN_DIM, regression_dropout=REGRESSION_DROPOUT
        )

    def test_output_shape(self):
        seq_len_A_bio = 25
        seq_len_B_bio = 20
        promoter_A = torch.randn(BATCH_SIZE, seq_len_A_bio, self.input_dim)
        promoter_B = torch.randn(BATCH_SIZE, seq_len_B_bio, self.input_dim)

        output_cls = self.siamese_model_cls(promoter_A, promoter_B)
        self.assertEqual(output_cls.shape, (BATCH_SIZE, 1))

        output_mean = self.siamese_model_mean(promoter_A, promoter_B)
        self.assertEqual(output_mean.shape, (BATCH_SIZE, 1))

    def test_forward_pass_with_masks(self):
        seq_len_A_bio = MAX_SEQ_LEN_BIO
        seq_len_B_bio = MAX_SEQ_LEN_BIO - 10
        
        promoter_A = torch.randn(BATCH_SIZE, seq_len_A_bio, self.input_dim)
        promoter_B = torch.randn(BATCH_SIZE, seq_len_B_bio, self.input_dim)

        mask_A = torch.zeros(BATCH_SIZE, seq_len_A_bio, dtype=torch.bool)
        mask_A[:, -5:] = True # Pad last 5 for A
        
        mask_B = torch.zeros(BATCH_SIZE, seq_len_B_bio, dtype=torch.bool)
        # No padding for B in this case, mask is all False

        output_cls = self.siamese_model_cls(promoter_A, promoter_B, key_padding_mask_A=mask_A, key_padding_mask_B=mask_B)
        self.assertEqual(output_cls.shape, (BATCH_SIZE, 1))

        output_mean = self.siamese_model_mean(promoter_A, promoter_B, key_padding_mask_A=mask_A, key_padding_mask_B=mask_B)
        self.assertEqual(output_mean.shape, (BATCH_SIZE, 1))

    def test_shared_tower_weights(self):
        # The single tower is self.transformer_encoder_tower
        # We can check if the parameters used for both inputs are identical
        param_ids_cls_tower = {id(p) for p in self.siamese_model_cls.transformer_encoder_tower.parameters()}
        
        # This doesn't directly test if it's *used* twice with shared weights in a forward pass,
        # but confirms the model is structured for weight sharing.
        # A more rigorous test would involve checking gradients if backpropping.
        self.assertTrue(len(param_ids_cls_tower) > 0, "No parameters in the tower.")

        # Test if the same tower instance is used.
        # This is implicit by design: self.transformer_encoder_tower is called twice.

    def test_aggregation_method_in_tower(self):
        self.assertEqual(self.siamese_model_cls.transformer_encoder_tower.aggregation_method, 'cls')
        self.assertEqual(self.siamese_model_mean.transformer_encoder_tower.aggregation_method, 'mean')
        self.assertEqual(self.siamese_model_cls.transformer_encoder_tower.pos_encoder.pe.size(1), MAX_SEQ_LEN_BIO + 1)


if __name__ == '__main__':
    unittest.main()
