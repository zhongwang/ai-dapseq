import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, call # Added call for checking multiple calls
import shutil # For cleaning up temp dirs if needed, though TemporaryDirectory handles it

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.siamese_transformer import SiameseGeneTransformer
from training.train import GenePairDataset, split_data_gene_disjoint, train_model

# Test-specific Constants (can be smaller than script's defaults for speed)
TEST_INPUT_FEATURE_DIM = 10 # 4 OHE + 6 TFs
TEST_D_MODEL = 16
TEST_NHEAD = 1
TEST_NUM_ENCODER_LAYERS = 1
TEST_DIM_FEEDFORWARD = 32
TEST_MAX_SEQ_LEN = 20 # Max promoter sequence length for feature vectors
TEST_BATCH_SIZE = 2
TEST_LEARNING_RATE = 1e-3
TEST_NUM_EPOCHS = 3
TEST_EARLY_STOPPING_PATIENCE = 2


class TestGenePairDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.feature_dir = os.path.join(self.temp_dir.name, "features")
        os.makedirs(self.feature_dir, exist_ok=True)

        # Create dummy .npy files
        self.gene_ids = ["gene1", "gene2", "gene3", "gene4_missing"]
        self.dummy_feature_shape = (TEST_MAX_SEQ_LEN, TEST_INPUT_FEATURE_DIM)

        for gene_id in self.gene_ids[:3]: # gene4_missing.npy will not be created
            np.save(os.path.join(self.feature_dir, f"{gene_id}.npy"), np.random.rand(*self.dummy_feature_shape).astype(np.float32))

        self.gene_pairs_data = {
            'gene1_id': ["gene1", "gene2", "gene1"],
            'gene2_id': ["gene2", "gene3", "gene4_missing"], # Last pair will cause FileNotFoundError
            'co_expression_correlation': [0.8, 0.5, 0.9]
        }
        self.gene_pairs_df = pd.DataFrame(self.gene_pairs_data)
        self.dataset = GenePairDataset(feature_dir=self.feature_dir, gene_pairs_df=self.gene_pairs_df)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.gene_pairs_df))

    def test_getitem_success(self):
        seq_a, mask_a, seq_b, mask_b, corr = self.dataset[0]

        self.assertIsInstance(seq_a, torch.Tensor)
        self.assertIsInstance(mask_a, torch.Tensor)
        self.assertIsInstance(seq_b, torch.Tensor)
        self.assertIsInstance(mask_b, torch.Tensor)
        self.assertIsInstance(corr, torch.Tensor)

        self.assertEqual(seq_a.shape, self.dummy_feature_shape)
        self.assertEqual(seq_b.shape, self.dummy_feature_shape)
        self.assertEqual(mask_a.shape, (TEST_MAX_SEQ_LEN,))
        self.assertEqual(mask_b.shape, (TEST_MAX_SEQ_LEN,))
        self.assertEqual(corr.ndim, 0) # Scalar tensor

        self.assertTrue(torch.all(mask_a == False)) # Masks should be all False
        self.assertTrue(torch.all(mask_b == False))
        
        self.assertEqual(corr.item(), self.gene_pairs_data['co_expression_correlation'][0])

    def test_getitem_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            _ = self.dataset[2] # This pair involves "gene4_missing"

class TestSplitDataGeneDisjoint(unittest.TestCase):
    def setUp(self):
        self.all_pairs_data = {
            'gene1_id': ["G1", "G1", "G2", "G3", "G4", "G5", "G6", "G1", "G7"],
            'gene2_id': ["G2", "G3", "G3", "G4", "G1", "G6", "G1", "G7", "G2"],
            'co_expression_correlation': [0.1] * 9
        }
        self.all_pairs_df = pd.DataFrame(self.all_pairs_data)
        self.unique_genes = set(self.all_pairs_df['gene1_id']).union(set(self.all_pairs_df['gene2_id']))

    def test_basic_split(self):
        train_df, val_df, test_df = split_data_gene_disjoint(self.all_pairs_df, train_frac=0.6, val_frac=0.2, random_state=42)

        self.assertFalse(train_df.empty)
        self.assertFalse(val_df.empty)
        # Test df might be empty depending on exact split of few unique genes
        # self.assertFalse(test_df.empty)

        train_genes = set(train_df['gene1_id']).union(set(train_df['gene2_id']))
        val_genes = set(val_df['gene1_id']).union(set(val_df['gene2_id']))
        test_genes = set(test_df['gene1_id']).union(set(test_df['gene2_id']))
        
        self.assertTrue(train_genes.isdisjoint(val_genes))
        self.assertTrue(train_genes.isdisjoint(test_genes))
        self.assertTrue(val_genes.isdisjoint(test_genes))

        for _, row in train_df.iterrows():
            self.assertIn(row['gene1_id'], train_genes)
            self.assertIn(row['gene2_id'], train_genes)
        for _, row in val_df.iterrows():
            self.assertIn(row['gene1_id'], val_genes)
            self.assertIn(row['gene2_id'], val_genes)
        if not test_df.empty:
            for _, row in test_df.iterrows():
                self.assertIn(row['gene1_id'], test_genes)
                self.assertIn(row['gene2_id'], test_genes)
    
    def test_split_few_genes(self):
        few_genes_data = {'gene1_id': ["G1","G1"], 'gene2_id': ["G2","G2"], 'co_expression_correlation':[0.1,0.2]}
        few_genes_df = pd.DataFrame(few_genes_data)
        train_df, val_df, test_df = split_data_gene_disjoint(few_genes_df, train_frac=0.5, val_frac=0.5, random_state=42)
        # With 2 genes, default is one for train, one for val.
        self.assertEqual(len(set(train_df['gene1_id']).union(set(train_df['gene2_id']))), 1)
        self.assertEqual(len(set(val_df['gene1_id']).union(set(val_df['gene2_id']))), 1)
        self.assertTrue(test_df.empty)

    def test_empty_input(self):
        empty_df = pd.DataFrame(columns=['gene1_id', 'gene2_id', 'co_expression_correlation'])
        train_df, val_df, test_df = split_data_gene_disjoint(empty_df)
        self.assertTrue(train_df.empty)
        self.assertTrue(val_df.empty)
        self.assertTrue(test_df.empty)

    def test_reproducibility(self):
        train1, val1, test1 = split_data_gene_disjoint(self.all_pairs_df, random_state=42)
        train2, val2, test2 = split_data_gene_disjoint(self.all_pairs_df, random_state=42)
        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(val1.reset_index(drop=True), val2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))

        train3, _, _ = split_data_gene_disjoint(self.all_pairs_df, random_state=123)
        # It's probabilistic, but with different seeds, the DFs should ideally differ if enough data
        # This is hard to assert strictly without knowing the exact split logic outcomes.
        # Check if gene sets differ at least.
        genes_t1 = set(train1['gene1_id']).union(set(train1['gene2_id']))
        genes_t3 = set(train3['gene1_id']).union(set(train3['gene2_id']))
        if len(self.unique_genes) > 3 : # Only if there's enough diversity for splits to vary
             self.assertNotEqual(genes_t1, genes_t3, "Splits with different random states were identical.")


class TestTrainModelLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0) # For reproducibility of model init and any torch random ops
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_save_path = os.path.join(self.temp_dir.name, "test_model.pth")
        
        # Create dummy feature dir and files for DataLoaders
        self.feature_dir = os.path.join(self.temp_dir.name, "train_features")
        os.makedirs(self.feature_dir, exist_ok=True)
        self.train_genes = ["tr_g1", "tr_g2", "tr_g3"]
        self.val_genes = ["val_g1", "val_g2"]
        
        for g_id in self.train_genes + self.val_genes:
            np.save(os.path.join(self.feature_dir, f"{g_id}.npy"), 
                    np.random.rand(TEST_MAX_SEQ_LEN, TEST_INPUT_FEATURE_DIM).astype(np.float32))

        train_pairs = pd.DataFrame({
            'gene1_id': [self.train_genes[0], self.train_genes[1]],
            'gene2_id': [self.train_genes[1], self.train_genes[2]],
            'co_expression_correlation': [0.7, 0.6]
        })
        val_pairs = pd.DataFrame({
            'gene1_id': [self.val_genes[0]],
            'gene2_id': [self.val_genes[1]],
            'co_expression_correlation': [0.5]
        })

        train_dataset = GenePairDataset(self.feature_dir, train_pairs)
        val_dataset = GenePairDataset(self.feature_dir, val_pairs)
        self.train_loader = DataLoader(train_dataset, batch_size=TEST_BATCH_SIZE)
        self.val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE)

        self.model = SiameseGeneTransformer(
            input_feature_dim=TEST_INPUT_FEATURE_DIM, d_model=TEST_D_MODEL, nhead=TEST_NHEAD,
            num_encoder_layers=TEST_NUM_ENCODER_LAYERS, dim_feedforward=TEST_DIM_FEEDFORWARD,
            dropout=0.1, aggregation_method='cls', max_seq_len=TEST_MAX_SEQ_LEN,
            regression_hidden_dim=TEST_D_MODEL // 2, regression_dropout=0.1
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=TEST_LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=1, factor=0.1)
        self.device = torch.device("cpu") # Force CPU for tests
        self.model.to(self.device)


    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_one_epoch(self):
        _, history = train_model(self.model, self.train_loader, self.val_loader, self.criterion, 
                                 self.optimizer, self.scheduler, num_epochs=1, device=self.device, 
                                 patience=TEST_EARLY_STOPPING_PATIENCE, model_save_path=self.model_save_path)
        self.assertEqual(len(history['train_loss']), 1)
        self.assertEqual(len(history['val_loss']), 1)
        self.assertIsNotNone(history['train_loss'][0])
        self.assertIsNotNone(history['val_loss'][0])
        self.assertTrue(os.path.exists(self.model_save_path)) # Model should be saved even after 1 epoch if val loss is best

    @patch('training.train.copy.deepcopy') # To verify model state copying
    def test_early_stopping(self, mock_deepcopy):
        # Mock validation loss to trigger early stopping
        # We need to make val_loss increase after the first epoch.
        # This is tricky without direct access to the loss variable inside train_model.
        # An alternative: make the val_loader return data that leads to higher loss, or mock criterion.
        
        # Simpler approach: Mock parts of the validation phase or model output directly.
        # For now, let's assume validation loss will not improve for `patience` epochs.
        # We can't easily control the loss without deeper mocks.
        # Let's check if it runs for `patience + 1` epochs then stops.

        # To ensure val_loss doesn't improve, we can mock the model's output during validation
        # or mock the criterion to return increasing values.
        # Let's try to mock the model's output for validation only.
        
        original_model_forward = self.model.forward
        val_call_count = 0

        def mocked_forward_for_early_stop(*args, **kwargs):
            nonlocal val_call_count
            if not self.model.training: # If in eval mode (validation)
                val_call_count +=1
                if val_call_count == 1: # First validation epoch
                    return torch.tensor([[0.5]] * TEST_BATCH_SIZE) # Good loss
                else: # Subsequent validation epochs
                    return torch.tensor([[10.0]] * TEST_BATCH_SIZE) # Bad loss to trigger early stopping
            return original_model_forward(*args, **kwargs)

        self.model.forward = mocked_forward_for_early_stop

        _, history = train_model(self.model, self.train_loader, self.val_loader, self.criterion,
                                 self.optimizer, self.scheduler, num_epochs=5, device=self.device,
                                 patience=TEST_EARLY_STOPPING_PATIENCE, model_save_path=self.model_save_path)
        
        self.model.forward = original_model_forward # Restore
        
        # Expected epochs: 1 (initial best) + PATIENCE (no improvement) = 1 + 2 = 3
        self.assertEqual(len(history['val_loss']), TEST_EARLY_STOPPING_PATIENCE + 1)
        self.assertTrue(os.path.exists(self.model_save_path))
        # Ensure deepcopy was called at least once for the best model state
        mock_deepcopy.assert_called()


    def test_model_saving_on_improvement(self):
        with patch('torch.save') as mock_torch_save:
            # Make val_loss improve then worsen
            val_losses = [0.5, 0.3, 0.6] # epoch 1, 2 (best), 3
            val_loss_iter = iter(val_losses)
            
            original_criterion = self.criterion
            def mock_criterion_eval(output, target):
                if not self.model.training: # eval mode
                    try:
                        return torch.tensor(next(val_loss_iter))
                    except StopIteration: # Should not happen if num_epochs matches val_losses length
                        return original_criterion(output, target) 
                return original_criterion(output, target)

            self.criterion = mock_criterion_eval
            
            train_model(self.model, self.train_loader, self.val_loader, self.criterion,
                        self.optimizer, self.scheduler, num_epochs=3, device=self.device,
                        patience=TEST_EARLY_STOPPING_PATIENCE, model_save_path=self.model_save_path)
            
            self.criterion = original_criterion # Restore

            # torch.save should be called when val_loss improves.
            # Epoch 1 (initial save), Epoch 2 (improvement)
            self.assertGreaterEqual(mock_torch_save.call_count, 2)
            mock_torch_save.assert_any_call(unittest.mock.ANY, self.model_save_path)

    def test_no_validation_loader(self):
        _, history = train_model(self.model, self.train_loader, None, self.criterion, 
                                 self.optimizer, None, num_epochs=TEST_NUM_EPOCHS, device=self.device, 
                                 patience=TEST_EARLY_STOPPING_PATIENCE, model_save_path=self.model_save_path)
        self.assertEqual(len(history['train_loss']), TEST_NUM_EPOCHS)
        # val_loss list might contain None or be shorter depending on implementation. Script appends None.
        self.assertEqual(len(history['val_loss']), TEST_NUM_EPOCHS)
        self.assertTrue(all(vl is None for vl in history['val_loss']))
        self.assertTrue(os.path.exists(self.model_save_path)) # Saves last model

    @patch.object(optim.lr_scheduler.ReduceLROnPlateau, 'step')
    def test_lr_scheduler_step_called(self, mock_scheduler_step):
        train_model(self.model, self.train_loader, self.val_loader, self.criterion,
                    self.optimizer, self.scheduler, num_epochs=1, device=self.device,
                    patience=TEST_EARLY_STOPPING_PATIENCE, model_save_path=self.model_save_path)
        mock_scheduler_step.assert_called_once()


if __name__ == '__main__':
    unittest.main()
