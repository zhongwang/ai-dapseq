import unittest
import pandas as pd
import os
import sys
import tempfile

# Add the project root to the Python path to allow importing from data_processing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.step1_extract_promoter_sequences import extract_promoter_sequences

class TestStep1ExtractPromoterSequences(unittest.TestCase):

    def setUp(self):
        """Set up temporary files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a dummy FASTA file
        self.fasta_file = os.path.join(self.temp_dir.name, "dummy_genome.fasta")
        with open(self.fasta_file, "w") as f:
            f.write(">chr1\\n")
            f.write("AGCTAGCTAGTTAGTTAGTTCCGGCCGGCCAATTAATTAAGGCGCGGCGC\\n") # 50bp
            f.write(">chr2\\n")
            f.write("TCGATCGATCGGAACGGAACAATTCCAATTGGCCAAGGCCTTCGAATTCG\\n") # 50bp

        # Create a dummy BED file
        self.bed_file = os.path.join(self.temp_dir.name, "dummy_promoters.bed")
        bed_data = [
            ("chr1", 0, 10, "gene1", "0", "+"),
            ("chr1", 10, 20, "gene2", "0", "-"),
            ("chr1", 20, 30, "gene3", "0", "+"),
            ("chr1", 30, 40, "gene4", "0", "-"),
            ("chr1", 40, 50, "gene5", "0", "+"),
            ("chr2", 0, 10, "gene6", "0", "+"),
            ("chr2", 10, 20, "gene7", "0", "-"),
            ("chr2", 20, 30, "gene8", "0", "+"),
            ("chr2", 30, 40, "gene9", "0", "-"),
            ("chr2", 40, 50, "gene10", "0", "+"),
        ]
        with open(self.bed_file, "w") as f:
            for row in bed_data:
                f.write("\\t".join(map(str, row)) + "\\n")

        self.output_file = os.path.join(self.temp_dir.name, "output_promoter_sequences.tsv")

        # Define expected output
        self.expected_data = [
            ("gene1", "chr1", 0, 10, "+", "AGCTAGCTAG"),
            ("gene2", "chr1", 10, 20, "-", "AACTAACTAA"), # Original: TTAGTTAGTT
            ("gene3", "chr1", 20, 30, "+", "CCGGCCGGCC"),
            ("gene4", "chr1", 30, 40, "-", "TTAATTAATT"), # Original: AATTAATTAA
            ("gene5", "chr1", 40, 50, "+", "GGCGCGGCGC"),
            ("gene6", "chr2", 0, 10, "+", "TCGATCGATC"),
            ("gene7", "chr2", 10, 20, "-", "GTTCCGTTCC"), # Original: GGAACGGAAC
            ("gene8", "chr2", 20, 30, "+", "AATTCCAATT"),
            ("gene9", "chr2", 30, 40, "-", "GGCCTTGGCC"), # Original: GGCCAAGGCC
            ("gene10", "chr2", 40, 50, "+", "TTCGAATTCG"),
        ]
        self.expected_df = pd.DataFrame(
            self.expected_data,
            columns=['gene_id', 'chromosome', 'promoter_start', 'promoter_end', 'strand', 'promoter_dna_sequence']
        )
        # Ensure correct types for comparison, especially for start/end which are read as int by pandas
        self.expected_df['promoter_start'] = self.expected_df['promoter_start'].astype(int)
        self.expected_df['promoter_end'] = self.expected_df['promoter_end'].astype(int)


    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_extract_promoter_sequences_normal_case(self):
        """Test the normal operation of extract_promoter_sequences."""
        extract_promoter_sequences(self.bed_file, self.fasta_file, self.output_file)

        self.assertTrue(os.path.exists(self.output_file), "Output file was not created.")

        output_df = pd.read_csv(self.output_file, sep='\\t')

        # Sort dataframes by gene_id to ensure consistent comparison
        output_df = output_df.sort_values(by='gene_id').reset_index(drop=True)
        expected_df_sorted = self.expected_df.sort_values(by='gene_id').reset_index(drop=True)
        
        pd.testing.assert_frame_equal(output_df, expected_df_sorted)

if __name__ == "__main__":
    unittest.main()

# Imports for TestStep2ExtractTfBindingSignals
import numpy as np
from unittest.mock import patch, MagicMock
from data_processing.step2_extract_tf_binding_signals import extract_tf_binding_signals

class TestStep2ExtractTfBindingSignals(unittest.TestCase):

    def setUp(self):
        """Set up temporary files and mock data for Step 2 tests."""
        self.temp_dir = tempfile.TemporaryDirectory()

        self.num_genes = 10
        self.promoter_length = 10
        self.tf_names = ["TF1", "TF2", "TF3"]
        self.gene_ids = [f"gene{i+1}" for i in range(self.num_genes)]

        # Create dummy promoter TSV file for Step 2 input
        self.promoter_file_step2 = os.path.join(self.temp_dir.name, "promoters_for_step2.tsv")
        promoter_data_step2 = []
        for i, gene_id in enumerate(self.gene_ids):
            # All on chr1, 10bp long, non-overlapping
            # Start coordinates: 0, 10, 20, ..., 90
            # End coordinates: 10, 20, 30, ..., 100
            promoter_data_step2.append((gene_id, "chr1", i * self.promoter_length, (i + 1) * self.promoter_length, "+", "N" * self.promoter_length))
        
        promoter_df_step2 = pd.DataFrame(
            promoter_data_step2,
            columns=['gene_id', 'chromosome', 'promoter_start', 'promoter_end', 'strand', 'promoter_dna_sequence']
        )
        promoter_df_step2.to_csv(self.promoter_file_step2, sep='\t', index=False)

        # Create dummy bigwig directory and TF files (empty, names are important for glob)
        self.bigwig_dir_step2 = os.path.join(self.temp_dir.name, "bigwigs_step2")
        os.makedirs(self.bigwig_dir_step2, exist_ok=True)
        self.bigwig_files_paths = []
        for tf_name in self.tf_names:
            bw_path = os.path.join(self.bigwig_dir_step2, f"{tf_name}.bigWig")
            with open(bw_path, "w") as f: # Create empty file
                f.write("")
            self.bigwig_files_paths.append(bw_path)

        # Output directory for step 2 (the script should create this)
        self.output_dir_step2 = os.path.join(self.temp_dir.name, "step2_output_npy")

        # Mock signal data for pyBigWig.values()
        # Structure: self.raw_signals_matrix_per_gene[gene_id] = np.array (num_tfs, promoter_length)
        self.raw_signals_matrix_per_gene = {}
        for i, gene_id in enumerate(self.gene_ids):
            gene_matrix_rows = []
            # TF1 signals: (i+1) repeated promoter_length times
            # TF2 signals: (i+1)*10 repeated promoter_length times
            # TF3 signals: (i+1)*100 repeated promoter_length times
            tf1_signal = np.full(self.promoter_length, float(i + 1))
            tf2_signal = np.full(self.promoter_length, float((i + 1) * 10))
            tf3_signal = np.full(self.promoter_length, float((i + 1) * 100))
            
            gene_matrix_rows.append(tf1_signal)
            gene_matrix_rows.append(tf2_signal)
            gene_matrix_rows.append(tf3_signal)
            self.raw_signals_matrix_per_gene[gene_id] = np.array(gene_matrix_rows)

    def tearDown(self):
        """Clean up temporary files and directories."""
        self.temp_dir.cleanup()

    def _dynamic_mock_bw_values(self, tf_name_opened_by_pybw, chrom, start, end):
        """Called by the mocked pyBigWig file object's .values() method."""
        # Determine gene_id based on start coordinate (promoters are contiguous in test setup)
        gene_idx = start // self.promoter_length
        if gene_idx < 0 or gene_idx >= self.num_genes: # Should not happen with valid coords
             return np.full(end - start, np.nan)
        gene_id = self.gene_ids[gene_idx]
        
        # Find the index of the TF based on its name
        try:
            tf_idx = self.tf_names.index(tf_name_opened_by_pybw)
        except ValueError: # TF name not in self.tf_names
            return np.full(end - start, np.nan)

        if gene_id in self.raw_signals_matrix_per_gene:
            # The raw_signals_matrix_per_gene stores the matrix for ALL TFs for that gene.
            # We need to return the row corresponding to tf_idx.
            signal_array_for_tf = self.raw_signals_matrix_per_gene[gene_id][tf_idx, :]
            
            expected_length = end - start
            if len(signal_array_for_tf) == expected_length:
                return signal_array_for_tf.copy() # Return a copy
            else:
                # This case indicates a mismatch between requested length and stored mock data length
                return np.full(expected_length, np.nan) 
        return np.full(end - start, np.nan) # Default to NaNs if gene_id not found

    @patch('data_processing.step2_extract_tf_binding_signals.pyBigWig')
    def _run_extraction_test(self, mock_pyBigWig_module, log_transform, zscore_normalize, pseudocount=1.0):
        """Helper function to run signal extraction and test outputs."""
        
        # Configure the mock for pyBigWig.open()
        def side_effect_pyBigWig_open(file_path_arg):
            # This function is called when pyBigWig.open(file_path_arg) is invoked in the SUT.
            # It should return a mock BigWig file object.
            tf_name_from_path = os.path.basename(file_path_arg).replace(".bigWig", "").replace(".bw", "")
            
            mock_bw_file_object = MagicMock()
            mock_bw_file_object.chroms.return_value = {'chr1': self.num_genes * self.promoter_length * 2} # Dummy chromosome size
            
            # The .values() method of the mock_bw_file_object needs to use tf_name_from_path
            mock_bw_file_object.values = lambda chrom, start, end, numpy=True: \
                self._dynamic_mock_bw_values(tf_name_from_path, chrom, start, end)
            
            mock_bw_file_object.close = MagicMock()
            return mock_bw_file_object

        mock_pyBigWig_module.open.side_effect = side_effect_pyBigWig_open

        # Run the actual function from the SUT
        extract_tf_binding_signals(
            promoter_file_path=self.promoter_file_step2,
            bigwig_dir_path=self.bigwig_dir_step2, # Script will glob this for *.bigWig files
            output_dir_path=self.output_dir_step2,
            num_cores=1, # Crucial for deterministic testing
            log_transform_flag=log_transform,
            zscore_normalize_flag=zscore_normalize,
            pseudocount_val=pseudocount
        )

        self.assertTrue(os.path.exists(self.output_dir_step2), "Output directory was not created.")

        # Calculate expected matrices based on transformations
        expected_final_matrices = {} 

        # 1. Start with raw or log-transformed signals
        intermediate_matrices = {}
        for gene_id in self.gene_ids:
            raw_matrix_for_gene = self.raw_signals_matrix_per_gene[gene_id].copy()
            if log_transform:
                intermediate_matrices[gene_id] = np.log2(raw_matrix_for_gene + pseudocount)
            else:
                intermediate_matrices[gene_id] = raw_matrix_for_gene
        
        # 2. Apply Z-score normalization if enabled (operates on intermediate_matrices)
        if zscore_normalize:
            num_tfs_in_test = len(self.tf_names)
            # Collect all signals for each TF across all genes and positions
            tf_all_signals_list = [[] for _ in range(num_tfs_in_test)]

            for gene_id in self.gene_ids: # Iterate in defined order of genes
                signal_matrix = intermediate_matrices[gene_id] 
                for tf_idx in range(num_tfs_in_test):
                    tf_all_signals_list[tf_idx].extend(signal_matrix[tf_idx, :].flatten().tolist())
            
            tf_means = np.zeros(num_tfs_in_test)
            tf_stds = np.ones(num_tfs_in_test) # Default std to 1 to avoid division by zero

            for tf_idx in range(num_tfs_in_test):
                flat_signals = np.array(tf_all_signals_list[tf_idx])
                if flat_signals.size > 0: # Ensure there's data to process
                    tf_means[tf_idx] = np.mean(flat_signals)
                    std_val = np.std(flat_signals)
                    tf_stds[tf_idx] = std_val if std_val > 1e-9 else 1.0 # Avoid division by zero or very small std
            
            # Apply Z-score to each gene's matrix
            for gene_id in self.gene_ids:
                matrix_to_normalize = intermediate_matrices[gene_id]
                # tf_means and tf_stds are (num_tfs,). Need to broadcast for (num_tfs, promoter_length) matrix.
                expected_final_matrices[gene_id] = (matrix_to_normalize - tf_means[:, np.newaxis]) / tf_stds[:, np.newaxis]
        else: # No Z-score, so final is just the intermediate (raw or log-transformed)
            expected_final_matrices = intermediate_matrices

        # 3. Verify output .npy files
        for gene_id in self.gene_ids:
            npy_file_path = os.path.join(self.output_dir_step2, f"{gene_id}.npy")
            self.assertTrue(os.path.exists(npy_file_path), f"Output .npy file not found: {npy_file_path}")
            
            loaded_matrix = np.load(npy_file_path)
            
            self.assertEqual(loaded_matrix.shape, (len(self.tf_names), self.promoter_length),
                             f"Matrix shape mismatch for {gene_id}")
            
            np.testing.assert_array_almost_equal(
                loaded_matrix,
                expected_final_matrices[gene_id],
                decimal=6, # Using a common precision for float comparisons
                err_msg=f"Matrix content mismatch for {gene_id} (log: {log_transform}, zscore: {zscore_normalize})"
            )

    # Test cases calling the helper
    def test_extract_signals_no_normalization(self):
        self._run_extraction_test(mock_pyBigWig_module=None, log_transform=False, zscore_normalize=False)

    def test_extract_signals_log_transform_only(self):
        self._run_extraction_test(mock_pyBigWig_module=None, log_transform=True, zscore_normalize=False, pseudocount=1.0)

    def test_extract_signals_zscore_normalize_only(self):
        self._run_extraction_test(mock_pyBigWig_module=None, log_transform=False, zscore_normalize=True)

    def test_extract_signals_log_then_zscore(self):
        self._run_extraction_test(mock_pyBigWig_module=None, log_transform=True, zscore_normalize=True, pseudocount=1.0)

from data_processing.step3_prepare_coexpression_data import prepare_coexpression_data

class TestStep3PrepareCoexpressionData(unittest.TestCase):

    def setUp(self):
        """Set up temporary files for Step 3 tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_coexp_file = os.path.join(self.temp_dir.name, "dummy_coexpression.tsv")
        self.output_coexp_file = os.path.join(self.temp_dir.name, "output_coexpression.tsv")
        self.processed_signals_dir = os.path.join(self.temp_dir.name, "mock_processed_signals")
        os.makedirs(self.processed_signals_dir, exist_ok=True)

        # Genes that will have mock .npy files
        self.existing_genes = ["gene1", "gene2", "gene3", "gene4"]
        for gene_id in self.existing_genes:
            with open(os.path.join(self.processed_signals_dir, f"{gene_id}.npy"), "w") as f:
                f.write("dummy npy content") # Content doesn't matter, only existence

        # Co-expression data: gene1_id, gene2_id, correlation_coefficient
        self.coexp_data = [
            ("gene1", "gene2", 0.8), # Both exist
            ("gene3", "gene4", 0.7), # Both exist
            ("gene1", "gene5", 0.6), # gene5 missing
            ("gene6", "gene2", 0.5), # gene6 missing
            ("gene7", "gene8", 0.9), # Both gene7, gene8 missing
        ]
        self.coexp_df = pd.DataFrame(self.coexp_data, columns=['gene1_id', 'gene2_id', 'correlation_coefficient'])
        self.coexp_df.to_csv(self.input_coexp_file, sep='\t', index=False)

    def tearDown(self):
        """Clean up temporary files and directories."""
        self.temp_dir.cleanup()

    def test_no_validation(self):
        """Test co-expression preparation without gene validation."""
        prepare_coexpression_data(
            coexpression_file_path=self.input_coexp_file,
            output_file_path=self.output_coexp_file,
            validate_genes_flag=False
        )
        self.assertTrue(os.path.exists(self.output_coexp_file))
        output_df = pd.read_csv(self.output_coexp_file, sep='\t')
        self.assertEqual(len(output_df), len(self.coexp_data))
        pd.testing.assert_frame_equal(output_df.sort_values(by=["gene1_id", "gene2_id"]).reset_index(drop=True),
                                      self.coexp_df.sort_values(by=["gene1_id", "gene2_id"]).reset_index(drop=True))

    def test_with_validation_filters_missing_genes(self):
        """Test co-expression preparation with validation, filtering out pairs with missing genes."""
        prepare_coexpression_data(
            coexpression_file_path=self.input_coexp_file,
            output_file_path=self.output_coexp_file,
            processed_signals_dir_path=self.processed_signals_dir,
            validate_genes_flag=True
        )
        self.assertTrue(os.path.exists(self.output_coexp_file))
        output_df = pd.read_csv(self.output_coexp_file, sep='\t')
        self.assertEqual(len(output_df), 2) # gene1-gene2 and gene3-gene4 should remain
        
        expected_data_after_filtering = [
            ("gene1", "gene2", 0.8),
            ("gene3", "gene4", 0.7),
        ]
        expected_df = pd.DataFrame(expected_data_after_filtering, columns=['gene1_id', 'gene2_id', 'correlation_coefficient'])
        pd.testing.assert_frame_equal(output_df.sort_values(by=["gene1_id", "gene2_id"]).reset_index(drop=True),
                                      expected_df.sort_values(by=["gene1_id", "gene2_id"]).reset_index(drop=True))

    def test_with_validation_all_genes_present(self):
        """Test validation when all genes in coexp file have corresponding .npy files."""
        # Create a specific coexp file for this test where all genes are in self.existing_genes
        small_coexp_data = [
            ("gene1", "gene2", 0.88),
            ("gene3", "gene1", 0.77),
        ]
        small_coexp_df = pd.DataFrame(small_coexp_data, columns=['gene1_id', 'gene2_id', 'correlation_coefficient'])
        small_input_file = os.path.join(self.temp_dir.name, "small_coexp.tsv")
        small_coexp_df.to_csv(small_input_file, sep='\t', index=False)

        prepare_coexpression_data(
            coexpression_file_path=small_input_file,
            output_file_path=self.output_coexp_file,
            processed_signals_dir_path=self.processed_signals_dir,
            validate_genes_flag=True
        )
        self.assertTrue(os.path.exists(self.output_coexp_file))
        output_df = pd.read_csv(self.output_coexp_file, sep='\t')
        self.assertEqual(len(output_df), len(small_coexp_data))
        pd.testing.assert_frame_equal(output_df.sort_values(by=["gene1_id", "gene2_id"]).reset_index(drop=True),
                                      small_coexp_df.sort_values(by=["gene1_id", "gene2_id"]).reset_index(drop=True))

    def test_validation_error_signals_dir_not_provided(self):
        """Test validation error if signals directory is not provided when flag is true."""
        # Ensure output file does not exist before test or is cleaned up
        if os.path.exists(self.output_coexp_file):
            os.remove(self.output_coexp_file)
            
        prepare_coexpression_data(
            coexpression_file_path=self.input_coexp_file,
            output_file_path=self.output_coexp_file,
            processed_signals_dir_path=None, # Explicitly None
            validate_genes_flag=True
        )
        # The script prints an error and returns, so output file might not be created or could be empty if created before error.
        # Depending on script's error handling, we might check that it's not created or is empty.
        # Current script logic: prints error and returns. If output dir was created by prev step, file might be empty.
        # For this test, let's assume if it returns early, it won't write a valid/full output.
        if os.path.exists(self.output_coexp_file):
            output_df = pd.read_csv(self.output_coexp_file, sep='\t')
            self.assertTrue(output_df.empty, "Output file should be empty or not created on error.")
        # else: self.assertFalse(os.path.exists(self.output_coexp_file)) # also a valid check

    def test_validation_error_invalid_signals_dir(self):
        """Test validation error if signals directory is invalid/doesn't exist."""
        if os.path.exists(self.output_coexp_file):
            os.remove(self.output_coexp_file)

        invalid_signals_dir = os.path.join(self.temp_dir.name, "non_existent_signals_dir")
        prepare_coexpression_data(
            coexpression_file_path=self.input_coexp_file,
            output_file_path=self.output_coexp_file,
            processed_signals_dir_path=invalid_signals_dir,
            validate_genes_flag=True
        )
        if os.path.exists(self.output_coexp_file):
            output_df = pd.read_csv(self.output_coexp_file, sep='\t')
            self.assertTrue(output_df.empty, "Output file should be empty or not created on error.")

# Imports for TestStep4CreateFeatureVectors
from unittest.mock import patch, MagicMock, mock_open
from argparse import Namespace
# Assuming step4_create_feature_vectors uses numpy, pandas, os, etc.
# Need to import the specific functions/classes from step4 if testing them directly or if needed for setup
from data_processing.step4_create_feature_vectors import main as step4_main
from data_processing.step4_create_feature_vectors import one_hot_encode_sequence, DNA_ONE_HOT_MAP, PAD_DNA_VECTOR

class TestStep4CreateFeatureVectors(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.max_len = 10
        self.num_tfs = 3
        self.feature_dim = 4 + self.num_tfs # 4 for DNA OHE, num_tfs for signals

        self.dna_input_path = os.path.join(self.temp_dir.name, "dna_input.tsv")
        self.tf_signals_input_dir = os.path.join(self.temp_dir.name, "tf_signals_input")
        self.feature_output_dir = os.path.join(self.temp_dir.name, "feature_output")

        os.makedirs(self.tf_signals_input_dir, exist_ok=True)
        # os.makedirs(self.feature_output_dir, exist_ok=True) # Script's main creates this

        # Gene data for testing various scenarios
        self.gene_data_specs = {
            "geneA": {"dna": "AGCTAGCTAG", "tf_len": 10, "tf_vals_base": [1.1, 1.2, 1.3]},
            "geneB": {"dna": "CGTA",       "tf_len": 4,  "tf_vals_base": [2.1, 2.2, 2.3]},
            "geneC": {"dna": "GATTACAGATTACA", "tf_len": 14, "tf_vals_base": [3.1, 3.2, 3.3]},
            "geneD": {"dna": "AGCTXN",     "tf_len": 6,  "tf_vals_base": [4.1, 4.2, 4.3]}, # X becomes N
            "geneE": {"dna": "ACTGACTG",   "tf_len": 8,  "tf_vals_base": [5.1, 5.2, 5.3], "skip_tf_file": True},
            "geneF": {"dna": "GTCAGTCA",   "tf_len": 8,  "tf_vals_base": [6.1, 6.2], "wrong_num_tfs": True},
            "geneG": {"dna": np.nan,       "tf_len": 5,  "tf_vals_base": [7.1, 7.2, 7.3]} # Invalid DNA sequence type
        }

        dna_tsv_rows = ["gene_id\tpromoter_dna_sequence"]
        for gene_id, spec in self.gene_data_specs.items():
            dna_seq_val = spec["dna"]
            if pd.isna(dna_seq_val): # Handle NaN for geneG to make it a blank string in TSV
                dna_tsv_rows.append(f"{gene_id}\t")
            else:
                dna_tsv_rows.append(f"{gene_id}\t{dna_seq_val}")
            
            if spec.get("skip_tf_file"):
                continue

            tf_signal_file_path = os.path.join(self.tf_signals_input_dir, f"{gene_id}.npy")
            current_num_tfs = self.num_tfs
            if spec.get("wrong_num_tfs"):
                current_num_tfs = self.num_tfs -1 # e.g. 2 instead of 3
            
            # Create TF signal matrix: (num_tfs, tf_len)
            # Each TF has a constant value across its length, derived from tf_vals_base
            tf_matrix_rows = [] 
            for i in range(current_num_tfs):
                tf_matrix_rows.append(np.full(spec["tf_len"], spec["tf_vals_base"][i], dtype=np.float32))
            if not tf_matrix_rows: # If current_num_tfs was 0 or less somehow
                 tf_signal_matrix = np.empty((0, spec["tf_len"]), dtype=np.float32)
            else:
                tf_signal_matrix = np.array(tf_matrix_rows, dtype=np.float32)
            np.save(tf_signal_file_path, tf_signal_matrix)

        with open(self.dna_input_path, "w") as f:
            f.write("\n".join(dna_tsv_rows) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_one_hot_encode_sequence_logic(self):
        # Exact length
        seq1 = "AGCT"
        expected1 = np.array([DNA_ONE_HOT_MAP['A'], DNA_ONE_HOT_MAP['C'], DNA_ONE_HOT_MAP['G'], DNA_ONE_HOT_MAP['T']], dtype=np.float32)
        np.testing.assert_array_equal(one_hot_encode_sequence(seq1, 4), expected1)

        # Shorter (padding)
        seq2 = "AC"
        expected2 = np.array([DNA_ONE_HOT_MAP['A'], DNA_ONE_HOT_MAP['C'], PAD_DNA_VECTOR, PAD_DNA_VECTOR], dtype=np.float32)
        np.testing.assert_array_equal(one_hot_encode_sequence(seq2, 4), expected2)

        # Longer (truncation)
        seq3 = "GATTACA"
        expected3 = np.array([DNA_ONE_HOT_MAP['G'], DNA_ONE_HOT_MAP['A'], DNA_ONE_HOT_MAP['T'], DNA_ONE_HOT_MAP['T']], dtype=np.float32)
        np.testing.assert_array_equal(one_hot_encode_sequence(seq3, 4), expected3)

        # With N and other char (becomes N)
        seq4 = "ANXG"
        expected4 = np.array([DNA_ONE_HOT_MAP['A'], DNA_ONE_HOT_MAP['N'], DNA_ONE_HOT_MAP['N'], DNA_ONE_HOT_MAP['G']], dtype=np.float32)
        np.testing.assert_array_equal(one_hot_encode_sequence(seq4, 4), expected4)
        
        # Empty sequence
        seq5 = ""
        expected5 = np.array([PAD_DNA_VECTOR]*4, dtype=np.float32)
        np.testing.assert_array_equal(one_hot_encode_sequence(seq5,4), expected5)

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_feature_creation_and_error_handling(self, mock_parse_args):
        mock_args = Namespace(
            dna_input_file=self.dna_input_path,
            tf_signals_dir=self.tf_signals_input_dir,
            output_dir=self.feature_output_dir,
            max_promoter_length=self.max_len,
            expected_num_tfs=self.num_tfs,
            num_workers=1
        )
        mock_parse_args.return_value = mock_args

        step4_main() # Call the main function from the script

        self.assertTrue(os.path.exists(self.feature_output_dir))

        # Genes expected to be processed successfully
        successful_genes = ["geneA", "geneB", "geneC", "geneD"]
        for gene_id in successful_genes:
            spec = self.gene_data_specs[gene_id]
            output_file = os.path.join(self.feature_output_dir, f"{gene_id}.npy")
            self.assertTrue(os.path.exists(output_file), f"{gene_id}.npy not created")
            
            feature_matrix = np.load(output_file)
            self.assertEqual(feature_matrix.shape, (self.max_len, self.feature_dim))

            # Verify DNA part
            dna_seq_original = spec["dna"]
            expected_ohe_dna = one_hot_encode_sequence(dna_seq_original, self.max_len)
            np.testing.assert_array_equal(feature_matrix[:, :4], expected_ohe_dna,
                                          err_msg=f"DNA OHE mismatch for {gene_id}")

            # Verify TF signals part
            expected_tf_signals = np.zeros((self.max_len, self.num_tfs), dtype=np.float32)
            original_tf_len = spec["tf_len"]
            len_to_copy = min(original_tf_len, self.max_len)
            base_vals = spec["tf_vals_base"]
            for i in range(self.num_tfs):
                expected_tf_signals[:len_to_copy, i] = base_vals[i]
            
            np.testing.assert_array_almost_equal(feature_matrix[:, 4:], expected_tf_signals, decimal=6,
                                                 err_msg=f"TF signals mismatch for {gene_id}")

        # Genes expected to be skipped or fail
        skipped_genes = ["geneE", "geneF", "geneG"]
        for gene_id in skipped_genes:
            output_file = os.path.join(self.feature_output_dir, f"{gene_id}.npy")
            self.assertFalse(os.path.exists(output_file), f"{gene_id}.npy created but should have been skipped")

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_empty_dna_input_file(self, mock_parse_args):
        # Create empty DNA input file
        with open(self.dna_input_path, "w") as f:
            f.write("") # Empty file

        mock_args = Namespace(
            dna_input_file=self.dna_input_path,
            tf_signals_dir=self.tf_signals_input_dir,
            output_dir=self.feature_output_dir,
            max_promoter_length=self.max_len,
            expected_num_tfs=self.num_tfs,
            num_workers=1
        )
        mock_parse_args.return_value = mock_args
        step4_main()
        # Output dir might be created, but no .npy files should be present
        if os.path.exists(self.feature_output_dir):
             self.assertEqual(len(os.listdir(self.feature_output_dir)), 0, "No .npy files should be created for empty DNA input")

    @patch('argparse.ArgumentParser.parse_args')
    def test_main_dna_input_file_missing_columns(self, mock_parse_args):
        # Create DNA input file with wrong column names
        with open(self.dna_input_path, "w") as f:
            f.write("gene\tsequence\n")
            f.write("some_gene\tAGCT\n")
            
        mock_args = Namespace(
            dna_input_file=self.dna_input_path,
            tf_signals_dir=self.tf_signals_input_dir,
            output_dir=self.feature_output_dir,
            max_promoter_length=self.max_len,
            expected_num_tfs=self.num_tfs,
            num_workers=1
        )
        mock_parse_args.return_value = mock_args
        step4_main()
        if os.path.exists(self.feature_output_dir):
            self.assertEqual(len(os.listdir(self.feature_output_dir)), 0, "No .npy files should be created for DNA input with missing columns")
