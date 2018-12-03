#!/usr/bin/env python3
import os
import subprocess
from sys import argv

main(my_dir):
    """
    Performs tombo resquiggle on all subfolders in specified folder.
    
    Args:
        my_dir -- str, path to main directory
        
    Returns: None    
    """
    #~ my_dir = "/mnt/nexenta/thijs030/data/Ecoli_MinKNOW_1.4_RAD002_Sambrook/" 
    dirs = os.listdir(my_dir)
    for folder in dirs:
        subprocess.check_output("tombo resquiggle {}/{} ./reference/ecoli_ref.fasta --dna --processes 4 --failed-reads-filename failed_reads_{}"
                                .format(my_dir, folder, folder), shell=True)
        print("Finished folder {}".format(folder))  
    print("Finished tombo resquiggle execution.")
    

if __name__ == "__main__":
    my_dir = argv[1]
    main(my_dir)
