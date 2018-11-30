#!/usr/bin/env python3
"""
Created on Mon Oct 22 13:37:40 2018

@author: thijs030
"""
import numpy as np

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
