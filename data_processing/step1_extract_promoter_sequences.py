\
import argparse
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
import os

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract promoter DNA sequences based on a BED file and genome FASTA file.")
    parser.add_argument("--bed_file", required=True, help="Path to the promoter BED file (chromosome, start, end, gene_id, strand).")
    parser.add_argument("--fasta_file", required=True, help="Path to the A. thaliana genome FASTA file.")
    parser.add_argument("--output_file", required=True, help="Path to the output TSV file (gene_id, chromosome, promoter_start, promoter_end, strand, promoter_dna_sequence).")
    return parser.parse_args()

def extract_promoter_sequences(bed_file_path, fasta_file_path, output_file_path):
    """
    Extracts promoter sequences and writes them to a file.

    Args:
        bed_file_path (str): Path to the BED file.
        fasta_file_path (str): Path to the genome FASTA file.
        output_file_path (str): Path to the output TSV file.
    """
    try:
        # 1. Parse BED file
        bed_df = pd.read_csv(
            bed_file_path,
            sep='\\t',
            header=None,
            names=['chromosome', 'start', 'end', 'gene_id', 'score', 'strand'], # Assuming standard BED6 + gene_id, strand. Adjust if format differs.
            comment='#' # Skip header lines if any
        )
        # Ensure correct types, especially for start/end
        bed_df['start'] = bed_df['start'].astype(int)
        bed_df['end'] = bed_df['end'].astype(int)

        print(f"Successfully parsed BED file: {bed_file_path}. Found {len(bed_df)} promoter regions.")

    except FileNotFoundError:
        print(f"Error: BED file not found at {bed_file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: BED file {bed_file_path} is empty.")
        return
    except Exception as e:
        print(f"Error parsing BED file {bed_file_path}: {e}")
        return

    try:
        # 2. Index FASTA file for efficient access
        print(f"Indexing FASTA file: {fasta_file_path}...")
        genome_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file_path, "fasta"))
        print("FASTA file indexed successfully.")
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_file_path}")
        return
    except Exception as e:
        print(f"Error reading or indexing FASTA file {fasta_file_path}: {e}")
        return

    results = []
    processed_genes = 0
    failed_genes = 0

    # 3. Extract DNA Sequences
    for index, row in bed_df.iterrows():
        chrom = str(row['chromosome']) # Ensure chromosome is a string
        start = int(row['start'])
        end = int(row['end'])
        gene_id = row['gene_id']
        strand = row['strand']

        try:
            if chrom not in genome_sequences:
                print(f"Warning: Chromosome '{chrom}' for gene '{gene_id}' not found in FASTA file. Skipping.")
                failed_genes += 1
                continue

            # Biopython SeqRecord slicing is 0-based, BED is 0-based start, 1-based end
            # So, if BED is [start, end), then sequence is genome[chrom].seq[start:end]
            promoter_seq_record = genome_sequences[chrom].seq[start:end]

            if strand == '-':
                promoter_dna_sequence = str(promoter_seq_record.reverse_complement())
            else:
                promoter_dna_sequence = str(promoter_seq_record)

            results.append({
                'gene_id': gene_id,
                'chromosome': chrom,
                'promoter_start': start,
                'promoter_end': end,
                'strand': strand,
                'promoter_dna_sequence': promoter_dna_sequence
            })
            processed_genes +=1
        except Exception as e:
            print(f"Error processing gene {gene_id} (chr: {chrom}, start: {start}, end: {end}): {e}")
            failed_genes += 1

    if not results:
        print("No sequences were successfully extracted. Please check your input files and formats.")
        return

    # 4. Output to structured file
    output_df = pd.DataFrame(results)
    try:
        output_df.to_csv(output_file_path, sep='\\t', index=False,
                         columns=['gene_id', 'chromosome', 'promoter_start', 'promoter_end', 'strand', 'promoter_dna_sequence'])
        print(f"Successfully wrote {len(output_df)} promoter sequences to {output_file_path}")
        print(f"Total genes processed: {processed_genes}, failed: {failed_genes}")
    except Exception as e:
        print(f"Error writing output file {output_file_path}: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    extract_promoter_sequences(args.bed_file, args.fasta_file, args.output_file)
    print("Step 1: Promoter DNA sequence extraction finished.")

# Example usage (comment out or remove before running as a script):
# python step1_extract_promoter_sequences.py \\
#   --bed_file "/path/to/your/promoter.bed" \\
#   --fasta_file "/path/to/your/genome.fasta" \\
#   --output_file "/path/to/your/output_promoter_sequences.tsv"
