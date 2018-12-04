#!/usr/bin/env python3 

"""
Divide reads into training, validation and test set.

@author: thijs030
"""

import os
from sys import argv
import random

def retrieve_fast5(folder, out_file="fast5_all.txt", simulated=False):
    """
    Save all FAST5 file names to a single file.
    
    Args:
        folder -- str, absolute path to main folder
        out_file -- str, name of output file [default: "fast5_all.txt"]
        simulated -- bool, simulated reads or real reads [default: False]
    
    Returns: file (str)
    """
    folder_list = os.listdir(os.path.abspath(folder))
    with open(out_file, "w") as dest:
        if simulated:
            for fld in folder_list:
                    dest.write(os.path.abspath(fld))
                    dest.write("\n")
        else:
            for fld in folder_list:
                path_to_folder = "{}/{}".format(folder, fld)
                folder_files = os.listdir(path_to_folder)
                fast5_files = list(filter(lambda fle: fle.endswith(".fast5"), folder_files))
                for fast5 in fast5_files:
                    dest.write("{}/{}".format(path_to_folder, fast5))
                    dest.write("\n")
                    
    return out_file
 
    
def get_sets(fast5_file, n_total, p_train=0.7, p_val=0.15, p_test=0.15):
    """
    Divides reads from file in sets of given proportions.
    
    Args:
        fast5_file -- str, name of file containing all FAST5 file names
        n_total -- int, total number of reads to divide
        p_train -- float, proportion for training set   [default: 0.7]
        p_val -- float, proportion for validation set   [default: 0.15]
        p_test -- float, proportion for test set        [default: 0.15]
    
    Returns: train_set, val_set, test_set (list of str (absolute paths to FAST5 files))
    """
    random.seed(83)                                                             # to get same division every time
    if (p_train + p_val + p_test) != 1:
        raise ValueError("Proportions should add up to 1.")
    sample_list = []
    with open(fast5_file, "r") as source:
        for line in source:
            if not line.strip():
                continue
            else:
                sample_list.append(line.strip())
    random.shuffle(sample_list)
    # divide reads over main groups, so if you use same proportions more or less reads can be added and are still in the same group
    total_samples = len(sample_list)
    n_total_train = round(total_samples * p_train)
    n_total_val = round(total_samples * p_val)
    # create subsets:
    subset_train = sample_list[: n_total_train]
    subset_val = sample_list[n_total_train : n_total_train + n_total_val]
    subset_test = sample_list[n_total_train + n_total_val :]
    n_train = round(n_total * p_train)
    n_val = round(n_total * p_val)
    n_test = n_total - n_train - n_val
    train_set = subset_train[: n_train]
    val_set = subset_val[ : n_val]
    test_set = subset_test[ : n_test]
    
    return train_set, val_set, test_set
    
    
def training_dir(dest_dir, train_set):
    """
    Copies reads to specified folder.
    
    Args:
        dest_dir -- str, path to new or existing folder
        train_set -- list of str, containing paths to files
        
    Returns: None
    """
    # Create new directory if does not exist already
    if not os.path.exists(dest_dir):    
        os.mkdir(dest_dir)
        print("Created directory {}.".format(dest_dir))
    else:
        print("Directory {} already exists.".format(dest_dir))

    # Copy reads to directory:
    n_copied = 0
    for train in train_set:
        os.popen("cp {} {}/{}".format(train, dest_dir, os.path.basename(train)))
        n_copied += 1
        if n_copied % 100 == 0:
            print("Copied {}".format(n_copied))
    print("Finished copying files to directory.")
        
   
    
if __name__ == "__main__":
    #1. Get a file with the names as in absolute path of the files to sample from
    my_dir = argv[1]
    output_file = argv[2]
    #~ if len(argv) == 4:
        #~ if argv[3] == "True":
            #~ simulated = True
    all_samples = retrieve_fast5(my_dir, output_file, simulated)

    # 2. Divide reads over sets
    total_number = int(argv[3])
    name_traindb = argv[4]
    name_valdb = argv[5]
    name_testdb = argv[6]

    tr_set, val_set, ts_set = get_sets(all_samples, total_number) 
    
    print(len(tr_set))
    print(len(val_set))
    print(len(ts_set))
    
    # 3. Make directory for trainingDB with training samples
    training_dir(name_traindb, tr_set)    
    training_dir(name_valdb, val_set)
    training_dir(name_testdb, ts_set)    
