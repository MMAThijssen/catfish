#!/usr/bin/env python3

"""
Retrieve information on read quality.

Adapted from Carlos.
"""

import mappy as mp
import yaml
from sys import argv

def read_quality(reads, ref_fasta, output):
    """
    Saves read quality of FAST5 reads to file.
    
    Args:
        reads -- str, path to folder
        ref_fasta -- str, path to FASTA file containing reference
        output -- str, name of output file
        
    Returns: None
    """
    result_dict = {
        'nb_mappings': [],
        'matches': 0,
        'mismatches': 0,
        'deletions': 0,
        'insertions': 0,
        'mapping_quality': []}
        
    aligner = mp.Aligner(ref_fasta)

    for name, seq, qual in mp.fastx_read(reads):
        nb_hits = 0
        for hit in aligner.map(seq):
            if hit.is_primary:
                matches_mismatches = sum([c[0] for c in hit.cigar if c[1] == 0])
                result_dict['matches'] += hit.mlen
                result_dict['mismatches'] += matches_mismatches - hit.mlen
                result_dict['insertions'] += sum([c[0] for c in hit.cigar if c[1] == 1])
                result_dict['deletions'] += sum([c[0] for c in hit.cigar if c[1] == 2])
                result_dict['mapping_quality'].append(hit.mapq)
            nb_hits += 1
        result_dict['nb_mappings'].append(nb_hits)
        
    with open(output, 'w') as outfile:
        yaml.dump(result_dict, outfile)


if __name__ == "__main__":
    reads = argv[1]
    ref_fasta = argv[2]
    output = argv[3]
    read_quality(reads, ref_fasta, output)
