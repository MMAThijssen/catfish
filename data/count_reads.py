#!usr/bin/env python3

"""
Module to check FAST5s on sequencing and MUX reads.

@author: Marijke Thijssen
"""

import os
import subprocess
from sys import argv


def count_reads(my_dir):
    """
    Counts number of total, sequencing, mux reads in a given directory.
    
    Args:
        my_dir -- str, absolute path to directory
        
    Returns: number of total files (int), FAST5 files (int), MUX reads (int), sequencing reads (int)
    """
    count_all = subprocess.check_output("ls | wc -l" , shell=True)            
    count_f5 = subprocess.check_output("ls *.fast5 | wc -l", shell=True)
    count_mux = subprocess.check_output("ls -I \"*sequencing*.fast5\" | wc -l", shell=True)
    count_seq = subprocess.check_output("ls -I \"*mux*.fast5\" | wc -l", shell=True)        
    return count_all.decode("utf-8").strip(), count_f5.decode("utf-8").strip(),
           count_mux.decode("utf-8").strip(), count_seq.decode("utf-8").strip()
           

def main(my_dir):
    """
    Count number of reads in all subdirectories.
    
    Args:
        my_dir -- str, path to main directory
        
    Returns: None
    """
    total_count = 0
    total_f5 = 0
    total_mux = 0
    total_seq = 0
    dirs = os.listdir(my_dir)
    print("DIR\tALL\tFAST5\tMUX\tsequencing")    
    for folder in dirs:
        os.chdir("{}//{}".format(my_dir, folder))
        count_all, count_f5, count_mux, count_seq = count_reads(folder)
        total_count += int(count_all)
        total_f5 += int(count_f5)
        total_mux += int(count_mux)
        total_seq += int(count_seq)
        print("{}\t{}\t{}\t{}\t{}".format(folder.strip(), count_all, count_f5,
              count_mux, count_seq))
    print("total files: {}\ntotal reads: {}\ntotal mux: {}\ntotal seq: {}"
            .format(total_count, total_f5, total_mux, total_seq))     
        
    
if __name__ == "__main__":
    my_dir = argv[1]
    main(my_dir)
   
