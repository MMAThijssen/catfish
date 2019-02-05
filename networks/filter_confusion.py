#!/usr/bin/env python3

import os
from os.path import basename
import re
from sys import argv

def write_confusion_to_file(input_file, dest_file):
    with open(dest_file, "a+") as dest:
        with open(input_file, "r") as source:
            file_name = basename(input_file).split(".")[0]
            for line in source:
                if line.strip().startswith("Detected"):
                    line = line.split()
                    # int part in unnecessary
                    tp = int(line[1])
                    fp = int(line[4])
                    tn = int(line[7])
                    fn = int(line[10])
        dest.write("{}\t{}\t{}\t{}\t{}\n".format(file_name, tp, fp, tn, fn))

    print("Wrote TP, FP, TN and FN to file for {}.".format(file_name))


if __name__ == "__main__":
    input_dir = argv[1]
    output_name = argv[2]       # with .txt!
    
    input_files = os.listdir(input_dir)
    input_files = [f for f in input_files if "_validateall.txt" in f]
    print(input_files)
    for f in input_files:
        write_confusion_to_file("{}/{}".format(input_dir, f), output_name)
