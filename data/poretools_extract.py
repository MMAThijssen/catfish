#!/usr/bin/env python3

import os
import subprocess
from sys import argv


def poretools_fastq():
    """
    Extracts FASTQ from FAST5 in subfolders using poretools.
    
    Args:
        my_dir -- str, path to main directory
        
    Returns: None
    """
    dirs = os.listdir(my_dir)
    for folder in dirs:
        path_to_folder = os.path.join(my_dir, folder)
        subprocess.check_output("poretools fastq --type fwd {}//*.fast5 > {}_poretools.fq"
                                .format(path_to_folder, path_to_folder), shell=True)
        print("Finished folder {}".format(folder))
    print("Finished extractions of FASTQs.")
    

if __name__ == "__main__":
    #~ my_dir = "/mnt/nexenta/thijs030/data/Ecoli_MinKNOW_1.4_RAD002_Sambrook/" 
    my_dir = argv[1]
    poretools_fastq(my_dir)
