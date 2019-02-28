#!/usr/bin/env python3
"""
Created on Mon Oct 22 13:37:40 2018

@author: thijs030
"""
import numpy as np
import os
from sys import argv

def load_npz(npz_file):
    """
    Retrieves information saved in NPZ file.
    
    Args:
        npz_file -- str, path to .npz file
        
    Returs: tuple (np.array of raw signal, np.array of labels)
    """
    with np.load(npz_file) as npz:
        x = npz["raw"]
        y = npz["base_labels"]                                                  # labels are 0 (non-HP) or 1 (HP)
    return x, y


def load_npz_labels(npz_file):
    """
    Retrieves information saved in NPZ file.
    
    Args:
        npz_file -- str, path to .npz file
        
    Returs: tuple (np.array of raw signal, np.array of labels)
    """
    with np.load(npz_file) as npz:
        labels = npz["base_labels"]                                                  # labels are 0 (non-HP) or 1 (HP)
    return labels
    

def load_npz_raw(npz_file):
    """
    Retrieves information saved in NPZ file.
    
    Args:
        npz_file -- str, path to .npz file
        
    Returs: tuple (np.array of raw signal, np.array of labels)
    """
    with np.load(npz_file) as npz:
        raw = npz["raw"]
    return raw


if __name__ == "__main__":
    input_dir = argv[1]
    files = os.listdir(input_dir)
    total_length = 0
    for f in files:
        c_len = len(load_npz_labels("{}/{}".format(input_dir, f)))
        total_length += c_len
    print("total length: ", total_length)
    print("average length: ", total_length / len(files))
