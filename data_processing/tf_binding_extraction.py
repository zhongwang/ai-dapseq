import argparse
import pandas as pd
import pyBigWig
import numpy as np
import os
import logging
from multiprocessing import Pool, cpu_count

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_signal_for_gene(args):
    """
    Worker function to extract TF binding signals for a single gene promoter.
    
    Args:
        args (tuple): A tuple containing (gene_info, bigwig_files, output_dir).
                      gene_info is a Series/dict with promoter coordinates.
                      bigwig_files is a list of paths to bigWig files.
                      output_dir is the directory to save the output .npy file.
    
    Returns:
        str: The path to the saved .npy file or None if an error occurred.
    """
    gene_info, bigwig_files, output_dir = args
    gene_id = gene_info['gene_id']
    chrom = gene_info['chromosome']
    start = int(gene_info['promoter_start'])
    end = int(gene_info['promoter_end'])
    
    # Ensure start is not negative
    if start < 0:
        logging.warning(f"Skipping gene {gene_id} due to negative start coordinate: {start}")
        return None

    gene_signals = []
    for bw_file in bigwig_files:
        try:
            bw = pyBigWig.open(bw_file)
            # Get values, fill missing data with 0.0
            values = bw.values(chrom, start, end)
            values = np.nan_to_num(values, nan=0.0)
            
            # Ensure consistent length (2501 bp)
            if len(values) != (end - start):
                 # This can happen if the promoter region is off the end of the chromosome
                padded_values = np.zeros(end - start)
                padded_values[:len(values)] = values
                values = padded_values

            gene_signals.append(values)
            bw.close()
        except Exception as e:
            logging.error(f"Error processing {bw_file} for gene {gene_id}: {e}")
            # Append a vector of zeros if a file fails
            gene_signals.append(np.zeros(end - start))

    signal_matrix = np.array(gene_signals, dtype=np.float32)
    output_path = os.path.join(output_dir, f"{gene_id}.npy")
    np.save(output_path, signal_matrix)
    return output_path

def main():
    """
    Main function to run the TF binding signal extraction script.
    """
    parser = argparse.ArgumentParser(description="Extract TF binding signals for promoter regions.")
    parser.add_argument("-p", "--promoters", required=True, help="Path to the promoter coordinates TSV file.")
    parser.add_argument("-b", "--bigwigs", required=True, help="Path to a file containing a list of bigWig file paths.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the output .npy files.")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="Number of worker processes to use.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Loading promoter data from {args.promoters}")
    promoters_df = pd.read_csv(args.promoters, sep='\t')

    logging.info(f"Reading bigWig file list from {args.bigwigs}")
    with open(args.bigwigs, 'r') as f:
        bigwig_files = [line.strip() for line in f.readlines()]
    
    logging.info(f"Found {len(bigwig_files)} bigWig files.")

    # Prepare arguments for multiprocessing
    tasks = [(row, bigwig_files, args.output_dir) for index, row in promoters_df.iterrows()]

    logging.info(f"Starting signal extraction with {args.processes} processes.")
    with Pool(args.processes) as pool:
        results = pool.map(extract_signal_for_gene, tasks)
    
    successful_files = [res for res in results if res is not None]
    logging.info(f"Successfully processed {len(successful_files)} out of {len(tasks)} genes.")
    logging.info("TF binding signal extraction complete.")

if __name__ == "__main__":
    main()