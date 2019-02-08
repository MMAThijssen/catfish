#!/usr/bin/env python3

"""
Retrieve information on read quality.

Adapted from Carlos.
"""

import mappy as mp
import os
import yaml
from sys import argv

def read_quality(read, ref_fasta, result_dict={'nb_mappings': [],
                                                'matches': 0,
                                                'mismatches': 0,
                                                'deletions': 0,
                                                'insertions': 0,
                                                'mapping_quality': []}):
    """
    Saves read quality of FASTQ/A reads to file.
    
    Args:
        reads -- str, path to read in FASTA or FAST5 format
        ref_fasta -- str, path to FASTA file containing reference
        output -- str, name of output file
        
    Returns: None
    """        
    aligner = mp.Aligner(ref_fasta)                                             # constructor that indexes reference

    for name, seq, qual in mp.fastx_read(read):                                # generator that open FASTA/Q and yiels name, seq, qual
        # ~ print(name)
        
        nb_hits = 0
        for hit in aligner.map(seq):                                            # aligns seq against index (generates Alignment object that describe alignment)
            if hit.is_primary:                                                  # usually best and first              
                matches_mismatches = sum([c[0] for c in hit.cigar if c[1] == 0])    # from CIGAR
                result_dict['matches'] += hit.mlen
                result_dict['mismatches'] += matches_mismatches - hit.mlen
                result_dict['insertions'] += sum([c[0] for c in hit.cigar if c[1] == 1])
                result_dict['deletions'] += sum([c[0] for c in hit.cigar if c[1] == 2])
                result_dict['mapping_quality'].append(hit.mapq)
            nb_hits += 1
        result_dict['nb_mappings'].append(nb_hits)
    
    return result_dict


def save_RQ(result_dict, output):
   
    if os.path.isfile(os.path.abspath(output)):
        raise IOError("ATTENTION - read quality file exists already.")
    
    with open(output, 'w') as outfile:
        yaml.dump(result_dict, outfile)


if __name__ == "__main__":
    reads = argv[1]
    ref_fasta = argv[2]
    output = argv[3]

    result_dict={'nb_mappings': [],
                 'matches': 0,
                 'mismatches': 0,
                 'deletions': 0,
                 'insertions': 0,
                 'mapping_quality': []}

    read_list = os.listdir(reads)
    for r in read_list:
        result_dict = read_quality("{}/{}".format(reads, r), ref_fasta, result_dict)
    save_RQ(result_dict, output)
    print("Finished writing results on read quality to {}".format(output))
