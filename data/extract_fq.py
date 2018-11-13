# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:46:33 2018

@author: thijs030
"""
import os
import subprocess
from sys import argv

if __name__ == "__main__":
#    my_dir = "/mnt/nexenta/thijs030/data/Ecoli_MinKNOW_1.4_RAD002_Sambrook/1/" 
#    subprocess.check_output("poretools fastq --type fwd {}//*.fast5 > {}_poretools.fa"
#                            .format(my_dir, my_dir), shell=True)
#    print(path_to_folder)
#    my_dir = "/mnt/nexenta/thijs030/data/Ecoli_MinKNOW_1.4_RAD002_Sambrook/" 
    my_dir = argv[1]
    dirs = os.listdir(my_dir)
    for folder in dirs:
        path_to_folder = os.path.join(my_dir, folder)
        subprocess.check_output("poretools fastq --type fwd {}//*.fast5 > {}_poretools.fq"
                                .format(path_to_folder, path_to_folder), shell=True)
        print("Finished folder {}".format(folder))