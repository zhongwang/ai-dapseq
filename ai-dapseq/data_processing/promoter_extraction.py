import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
# Biopython's GFF parsing is a bit limited, might need gffpandas or custom logic
# from Bio.SeqFeature import FeatureLocation, CompoundLocation

# Placeholder paths - replace with actual file paths
GFF3_FILE = "../path/to/arabidopsis.gff3"
FASTA_FILE = "../path/to/arabidopsis.fasta"
OUTPUT_FILE = "./promoter_regions.tsv"

PROMOTER_UPSTREAM = 2000
PROMOTER_DOWNSTREAM = 500

def parse_gff3(gff3_path):
    """
    Parses a GFF3 file to extract gene information.
    Returns a dictionary mapping gene ID to (chromosome, start_codon_pos, strand).
    Note: Biopython's GFF parsing can be tricky. This is a basic approach.
    More robust parsing might require gffpandas or custom logic based on GFF3 structure.
    """
    gene_info = {}
    print(f"Parsing GFF3 file: {gff3_path}")
    # This is a simplified parser. A real GFF3 parser needs to handle features carefully.
    # Using a simple line-by-line approach assuming standard gene/mRNA/CDS structure.
    # A more robust approach would use a dedicated GFF3 parsing library or handle parent/child relationships.
    try:
        with open(gff3_path, "r") as infile:
            for line in infile:
                if line.startswith("#"):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue

                seqname, source, feature_type, start, end, score, strand, frame, attributes = parts

                # We are looking for the start codon position, which is typically part of a CDS feature
                # associated with a gene or mRNA. This simplified parser looks for 'gene' features
                # and assumes the 'start' coordinate is the TSS for '+' strand and 'end' for '-' strand
                # in a simplified GFF3, or looks for CDS and infers from there.
                # A proper implementation needs to find the *translation start site* (start codon)
                # which is the beginning of the CDS feature on the correct strand.

                # Simplified approach: Look for 'gene' features and use start/end based on strand
                # This is NOT strictly correct for TSS, which is the start codon of the CDS.
                # A better approach would find the CDS feature linked to a gene/mRNA.
                if feature_type == "gene":
                     # Need to extract gene ID from attributes
                    gene_id = None
                    attrs = attributes.split(';')
                    for attr in attrs:
                        if attr.startswith("ID=gene:"):
                            gene_id = attr.split(':')[1]
                            break
                        elif attr.startswith("ID="): # More general ID
                             gene_id = attr.split('=')[1]
                             # Check if it looks like a gene ID, e.g., starts with AT
                             if not gene_id.startswith("AT"):
                                 gene_id = None # Not a gene ID we want
                                 continue
                             # Further check if it's a primary gene ID, not a component
                             # This requires more specific knowledge of the GFF3 format used (e.g., TAIR)
                             # For simplicity, let's assume ID=gene:XXXX or ID=ATXGXXXXX is the gene ID
                             if gene_id and not gene_id.startswith("gene:"): # Avoid internal IDs if possible
                                 pass # Keep this ID
                             else:
                                 gene_id = None


                    if gene_id:
                        # For a gene feature, start/end are gene boundaries, not TSS.
                        # We need the CDS feature's start codon.
                        # This simplified parser will just store gene boundaries for now.
                        # A proper implementation would store gene_id, chrom, and look up CDS later.
                        # Let's refine this to look for CDS features and link them to genes.
                        pass # Skip gene feature for now, look for CDS

                # More accurate approach: Find CDS features and their start codons
                if feature_type == "CDS":
                    parent_gene_id = None
                    attrs = attributes.split(';')
                    for attr in attrs:
                         if attr.startswith("Parent=mRNA:"):
                             # Link mRNA to gene if necessary, or directly link CDS to gene if Parent is gene
                             # Assuming Parent=mRNA:XXXX, and mRNA ID is gene:YYYY.mRNA.Z
                             # This is highly dependent on the GFF3 format.
                             # Let's assume Parent=gene:XXXX or Parent=ATXGXXXXX for simplicity for now.
                             # A robust parser would build a tree/graph of features.
                             parent_id_str = attr.split('=')[1]
                             if parent_id_str.startswith("mRNA:"):
                                 # Need to map mRNA ID back to Gene ID. This is complex with simple parsing.
                                 # Let's assume a simpler GFF3 where CDS Parent is the gene ID directly or indirectly linkable.
                                 # For now, let's just store CDS info and figure out linking later if needed.
                                 pass # Skip complex linking for now

                         # Let's try to find a Parent attribute that looks like a gene ID
                         if attr.startswith("Parent="):
                             p_id = attr.split('=')[1]
                             # Simple heuristic: if Parent ID starts with AT or gene:
                             if p_id.startswith("AT") or p_id.startswith("gene:"):
                                 # Clean up gene: prefix if present
                                 parent_gene_id = p_id.replace("gene:", "")
                                 break # Found a potential parent gene ID

                    if parent_gene_id:
                        # The start codon is the 'start' coordinate for '+' strand CDS
                        # and the 'end' coordinate for '-' strand CDS.
                        start_pos = int(start)
                        end_pos = int(end)
                        # The start codon is the coordinate where translation begins.
                        # For '+', it's the 'start' of the CDS. For '-', it's the 'end' of the CDS.
                        start_codon_pos = start_pos if strand == '+' else end_pos

                        # Store the first CDS found for a gene as its primary TSS location.
                        # In reality, genes can have multiple transcripts/CDSs.
                        # This simplified approach takes the first one encountered.
                        if parent_gene_id not in gene_info:
                             gene_info[parent_gene_id] = (seqname, start_codon_pos, strand)


    except Exception as e:
        print(f"Error parsing GFF3: {e}")
        # In a real scenario, log the error and potentially stop or skip the file.
        # For this script, we'll just print and continue with potentially incomplete data.
        pass # Continue with potentially incomplete gene_info

    print(f"Found info for {len(gene_info)} genes.")
    return gene_info

def extract_promoter_sequences(fasta_path, gene_info, upstream, downstream):
    """
    Extracts promoter sequences based on gene information and FASTA genome.
    """
    promoter_data = []
    print(f"Loading genome FASTA: {fasta_path}")
    try:
        genome = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        print(f"Genome loaded with {len(genome)} chromosomes/contigs.")
    except Exception as e:
        print(f"Error loading FASTA: {e}")
        return [] # Return empty list if FASTA loading fails

    print("Extracting promoter sequences and coordinates...")
    for gene_id, (chrom, start_codon_pos, strand) in gene_info.items():
        if chrom not in genome:
            print(f"Warning: Chromosome {chrom} for gene {gene_id} not found in FASTA. Skipping.")
            continue

        # Calculate promoter coordinates based on strand (1-based)
        if strand == '+':
            promoter_start_1based = start_codon_pos - upstream
            promoter_end_1based = start_codon_pos + downstream
        elif strand == '-':
            promoter_start_1based = start_codon_pos - downstream # Downstream is numerically smaller on '-' strand
            promoter_end_1based = start_codon_pos + upstream   # Upstream is numerically larger on '-' strand
        else:
            print(f"Warning: Unknown strand '{strand}' for gene {gene_id}. Skipping.")
            continue

        # Ensure coordinates are within chromosome bounds (1-based)
        chrom_length = len(genome[chrom].seq)
        if promoter_start_1based < 1:
            print(f"Warning: Promoter start for gene {gene_id} ({promoter_start_1based}) is before chromosome start. Clamping to 1.")
            promoter_start_1based = 1
        if promoter_end_1based > chrom_length:
             print(f"Warning: Promoter end for gene {gene_id} ({promoter_end_1based}) is after chromosome end ({chrom_length}). Clamping.")
             promoter_end_1based = chrom_length


        # Extract sequence
        try:
            # Convert 1-based genomic coordinates to 0-based for slicing (end-exclusive)
            promoter_start_0based = promoter_start_1based - 1
            promoter_end_0based_exclusive = promoter_end_1based

            seq = genome[chrom].seq[promoter_start_0based : promoter_end_0based_exclusive]

            # If gene is on '-' strand, get the reverse complement sequence
            if strand == '-':
                seq = seq.reverse_complement()

            promoter_data.append({
                'gene_id': gene_id,
                'chromosome': chrom,
                'start': promoter_start_1based, # Store 1-based coordinates
                'end': promoter_end_1based,     # Store 1-based coordinates
                'strand': strand,
                'sequence': str(seq)
            })

        except Exception as e:
            print(f"Error extracting sequence for gene {gene_id} on {chrom}:{promoter_start_1based}-{promoter_end_1based} ({strand}): {e}. Skipping.")
            continue

    print(f"Extracted data for {len(promoter_data)} promoters.")
    return promoter_data

def write_promoter_data(promoter_data, output_path):
    """
    Writes extracted promoter data (including coordinates and sequence) to a TSV file.
    """
    print(f"Writing promoter data to {output_path}")
    try:
        # Use pandas to write the list of dictionaries to a TSV
        promoter_df = pd.DataFrame(promoter_data)
        # Ensure columns are in a consistent order
        column_order = ['gene_id', 'chromosome', 'start', 'end', 'strand', 'sequence']
        promoter_df = promoter_df[column_order]
        promoter_df.to_csv(output_path, sep='\t', index=False)
        print("Writing complete.")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Step 1: Parse GFF3 and get gene info
    gene_info = parse_gff3(GFF3_FILE)

    # Step 2: Extract promoter sequences and coordinates
    promoter_data = extract_promoter_sequences(FASTA_FILE, gene_info, PROMOTER_UPSTREAM, PROMOTER_DOWNSTREAM)

    # Step 3: Write results
    if promoter_data:
        write_promoter_data(promoter_data, OUTPUT_FILE)
    else:
        print("No promoter data extracted. Output file not created.")