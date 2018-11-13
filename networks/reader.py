#!/usr/bin/env python3
"""
Created on Mon Oct 22 13:37:40 2018

@author: thijs030
"""
import numpy as np
#### script to convert input of trainingDB etc. to TensorFlow objects

# read in small chunks


# read in npz files
def load_npz(npz_file):
    with np.load(npz_file) as npz:
        x = npz["raw"]
        y = npz["base_labels"]
    return x, y     # Removed parentheses
