import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from argparse import Namespace

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Functions and main from the script to be tested
from visualization.plot_performance_metrics import (
    load_training_logs,
    load_model_predictions,
    plot_training_curves,
    plot_regression_metric_curves,
    plot_predicted_vs_actual,
    plot_residual_plot,
    main as visualization_main
)

class TestPlottingFunctions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

        # Dummy data for testing plotting functions
        self.dummy_logs_df = pd.DataFrame({
            'epoch': range(1, 6),
            'train_loss': np.random.rand(5) * 0.1 + 0.05,
            'val_loss': np.random.rand(5) * 0.1 + 0.08,
            'val_pearson_r': np.random.rand(5) * 0.3 + 0.5,
            'val_mae': np.random.rand(5) * 0.1 + 0.1
        })
        self.dummy_predictions_df = pd.DataFrame({
            'actual_correlation': np.random.rand(20) * 2 - 1,
            'predicted_correlation': np.random.rand(20) * 2 - 1
        })
        # Ensure some correlation for pearsonr in plot_predicted_vs_actual to not fail
        self.dummy_predictions_df['predicted_correlation'] = self.dummy_predictions_df['actual_correlation'] * 0.5 + (np.random.rand(20) * 0.3 - 0.15)


    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_training_curves(self, mock_close, mock_savefig):
        output_path = os.path.join(self.output_dir, "training_loss.png")
        plot_training_curves(self.dummy_logs_df, output_path=output_path)
        mock_savefig.assert_called_once_with(output_path)
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_regression_metric_curves(self, mock_close, mock_savefig):
        prefix = os.path.join(self.output_dir, "val_metrics")
        plot_regression_metric_curves(self.dummy_logs_df, output_path_prefix=prefix)
        expected_calls = [
            unittest.mock.call(f"{prefix}_pearson_correlation.png"),
            unittest.mock.call(f"{prefix}_mae.png")
        ]
        mock_savefig.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(mock_savefig.call_count, 2)
        self.assertEqual(mock_close.call_count, 2)


    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_predicted_vs_actual(self, mock_close, mock_savefig):
        output_path = os.path.join(self.output_dir, "pred_vs_actual.png")
        plot_predicted_vs_actual(self.dummy_predictions_df, output_path=output_path)
        mock_savefig.assert_called_once_with(output_path)
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_residual_plot(self, mock_close, mock_savefig):
        output_path = os.path.join(self.output_dir, "residuals.png")
        plot_residual_plot(self.dummy_predictions_df, output_path=output_path)
        mock_savefig.assert_called_once_with(output_path)
        mock_close.assert_called_once()

class TestMainFunctionVisualization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir_main = os.path.join(self.temp_dir.name, "main_plots")

        self.dummy_logs_df_main = pd.DataFrame({
            'epoch': range(1, 3), 'train_loss': [0.1, 0.05], 'val_loss': [0.12, 0.07],
            'val_pearson_r': [0.5, 0.6], 'val_mae': [0.2, 0.15]
        })
        self.dummy_predictions_df_main = pd.DataFrame({
            'actual_correlation': [0.5, 0.8], 'predicted_correlation': [0.4, 0.85]
        })
        
    def tearDown(self):
        self.temp_dir.cleanup()

    # Helper to run main with mocked components
    def _run_main_with_mocks(self, mock_args_val, mock_load_logs_return, mock_load_preds_return,
                             expect_train_curves, expect_reg_metrics, expect_pred_actual, expect_residuals):

        with patch('visualization.plot_performance_metrics.argparse.ArgumentParser.parse_args', return_value=mock_args_val) as mock_argparse,\
             patch('visualization.plot_performance_metrics.load_training_logs', return_value=mock_load_logs_return) as mock_load_logs_func,\
             patch('visualization.plot_performance_metrics.load_model_predictions', return_value=mock_load_preds_return) as mock_load_preds_func,\
             patch('visualization.plot_performance_metrics.plot_training_curves') as mock_plot_train_curves_func,\
             patch('visualization.plot_performance_metrics.plot_regression_metric_curves') as mock_plot_reg_metrics_func,\
             patch('visualization.plot_performance_metrics.plot_predicted_vs_actual') as mock_plot_pred_actual_func,\
             patch('visualization.plot_performance_metrics.plot_residual_plot') as mock_plot_residuals_func,\
             patch('os.makedirs') as mock_os_makedirs:
            
            visualization_main()

            mock_os_makedirs.assert_called_with(mock_args_val.output_dir, exist_ok=True)

            if expect_train_curves:
                mock_plot_train_curves_func.assert_called_once()
                self.assertEqual(mock_plot_train_curves_func.call_args[0][0].equals(mock_load_logs_return), True)
                self.assertEqual(mock_plot_train_curves_func.call_args[1]['output_path'], 
                                 os.path.join(mock_args_val.output_dir, "training_loss_curves.png"))
            else:
                mock_plot_train_curves_func.assert_not_called()
            
            if expect_reg_metrics:
                mock_plot_reg_metrics_func.assert_called_once()
                self.assertEqual(mock_plot_reg_metrics_func.call_args[0][0].equals(mock_load_logs_return), True)
                self.assertEqual(mock_plot_reg_metrics_func.call_args[1]['output_path_prefix'], 
                                 os.path.join(mock_args_val.output_dir, "validation_metrics"))
            else:
                mock_plot_reg_metrics_func.assert_not_called()

            if expect_pred_actual:
                mock_plot_pred_actual_func.assert_called_once()
                self.assertEqual(mock_plot_pred_actual_func.call_args[0][0].equals(mock_load_preds_return), True)
                self.assertEqual(mock_plot_pred_actual_func.call_args[1]['output_path'], 
                                 os.path.join(mock_args_val.output_dir, "predicted_vs_actual.png"))
            else:
                mock_plot_pred_actual_func.assert_not_called()

            if expect_residuals:
                mock_plot_residuals_func.assert_called_once()
                self.assertEqual(mock_plot_residuals_func.call_args[0][0].equals(mock_load_preds_return), True)
                self.assertEqual(mock_plot_residuals_func.call_args[1]['output_path'], 
                                 os.path.join(mock_args_val.output_dir, "residual_plot.png"))
            else:
                mock_plot_residuals_func.assert_not_called()

    def test_main_all_files_provided_and_valid(self):
        args = Namespace(training_logs_file="dummy_logs.csv", predictions_file="dummy_preds.csv", output_dir=self.output_dir_main)
        self._run_main_with_mocks(args, self.dummy_logs_df_main, self.dummy_predictions_df_main, 
                                  True, True, True, True)

    def test_main_no_files_provided_uses_placeholders(self):
        args = Namespace(training_logs_file=None, predictions_file=None, output_dir=self.output_dir_main)
        # The script's load functions return placeholder data when path is None or invalid
        # Our mocks for load_training_logs/load_model_predictions will effectively simulate this by returning a valid df
        self._run_main_with_mocks(args, self.dummy_logs_df_main, self.dummy_predictions_df_main, 
                                  True, True, True, True)

    def test_main_logs_only(self):
        args = Namespace(training_logs_file="dummy_logs.csv", predictions_file=None, output_dir=self.output_dir_main)
        self._run_main_with_mocks(args, self.dummy_logs_df_main, pd.DataFrame(), # Empty df for predictions
                                  True, True, False, False)

    def test_main_predictions_only(self):
        args = Namespace(training_logs_file=None, predictions_file="dummy_preds.csv", output_dir=self.output_dir_main)
        self._run_main_with_mocks(args, pd.DataFrame(), self.dummy_predictions_df_main, # Empty df for logs
                                  False, False, True, True)
    
    def test_main_logs_file_not_found_uses_placeholder(self):
        args = Namespace(training_logs_file="non_existent.csv", predictions_file="dummy_preds.csv", output_dir=self.output_dir_main)
        # Mock load_training_logs to simulate FileNotFoundError then placeholder load
        mock_load_logs = MagicMock()
        mock_load_logs.side_effect = [
            FileNotFoundError("File not found"), # First call (actual file) fails
            self.dummy_logs_df_main             # Second call (placeholder) succeeds
        ]
        with patch('visualization.plot_performance_metrics.load_training_logs', mock_load_logs): # Patch specific mock
            self._run_main_with_mocks(args, self.dummy_logs_df_main, self.dummy_predictions_df_main,
                                      True, True, True, True)
            # Ensure load_training_logs was called twice: once for the file, once for placeholder
            self.assertEqual(mock_load_logs.call_count, 2) 
            mock_load_logs.assert_any_call("non_existent.csv")
            mock_load_logs.assert_any_call("placeholder_logs")

    def test_main_logs_missing_metric_columns(self):
        args = Namespace(training_logs_file="logs_missing_cols.csv", predictions_file=None, output_dir=self.output_dir_main)
        logs_missing_cols_df = pd.DataFrame({
            'epoch': [1,2], 'train_loss': [0.1,0.05], 'val_loss': [0.2,0.1]
        })
        self._run_main_with_mocks(args, logs_missing_cols_df, pd.DataFrame(), 
                                  True, False, False, False) # expect_reg_metrics is False


if __name__ == '__main__':
    unittest.main()
