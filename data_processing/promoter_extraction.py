import argparse
import pandas as pd
from BCBio import GFF
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_gff(gff_file):
    """
    Parses a GFF3 file to extract information for protein-coding genes.

    Args:
        gff_file (str): Path to the GFF3 file.

    Returns:
        list: A list of dictionaries, where each dictionary contains information
              for a single protein-coding gene.
    """
    genes = []
    with open(gff_file) as in_handle:
        for rec in GFF.parse(in_handle):
            for feature in rec.features:
                if feature.type == 'gene':
                    # Assuming the first mRNA and first CDS are the primary ones
                    mrna = feature.sub_features[0]
                    cds = next((sf for sf in mrna.sub_features if sf.type == 'CDS'), None)
                    
                    if cds:
                        gene_id = feature.id
                        chrom = rec.id
                        strand = cds.strand
                        
                        # Get the start codon position
                        if strand == 1: # Forward strand
                            start_codon = cds.location.start.position
                        else: # Reverse strand
                            start_codon = cds.location.end.position
                            
                        genes.append({
                            'gene_id': gene_id,
                            'chromosome': chrom,
                            'strand': strand,
                            'start_codon': start_codon
                        })
    return genes

def define_promoters(genes, upstream=2000, downstream=500):
    """
    Defines promoter regions based on gene information.

    Args:
        genes (list): A list of gene dictionaries from parse_gff.
        upstream (int): The number of base pairs upstream of the start codon.
        downstream (int): The number of base pairs downstream of the start codon.

    Returns:
        pd.DataFrame: A DataFrame with promoter coordinates for each gene.
    """
    promoter_data = []
    for gene in genes:
        if gene['strand'] == 1: # Forward strand
            promoter_start = gene['start_codon'] - upstream
            promoter_end = gene['start_codon'] + downstream
        else: # Reverse strand
            promoter_start = gene['start_codon'] - downstream
            promoter_end = gene['start_codon'] + upstream
            
        promoter_data.append({
            'gene_id': gene['gene_id'],
            'chromosome': gene['chromosome'],
            'strand': gene['strand'],
            'promoter_start': promoter_start,
            'promoter_end': promoter_end
        })
        
    return pd.DataFrame(promoter_data)

def main():
    """
    Main function to run the promoter extraction script.
    """
    parser = argparse.ArgumentParser(description="Extract promoter regions from a GFF3 file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input GFF3 file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output TSV file.")
    parser.add_argument("--upstream", type=int, default=2000, help="Upstream distance from start codon.")
    parser.add_argument("--downstream", type=int, default=500, help="Downstream distance from start codon.")
    args = parser.parse_args()

    logging.info(f"Parsing GFF file: {args.input}")
    genes = parse_gff(args.input)
    logging.info(f"Found {len(genes)} protein-coding genes.")

    logging.info(f"Defining promoter regions: {args.upstream}bp upstream, {args.downstream}bp downstream.")
    promoters_df = define_promoters(genes, args.upstream, args.downstream)

    logging.info(f"Saving promoter data to: {args.output}")
    promoters_df.to_csv(args.output, sep='\t', index=False)
    logging.info("Promoter extraction complete.")

if __name__ == "__main__":
    main()