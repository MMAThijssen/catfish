#!/usr/bin/env
import numpy as np
import reader
from sys import argv
import tensorflow as tf

def main(db_dir, max_seq_length):
    """
    Validate model on full reads
    
    Args:
        db_dir -- str, path to validation database
    """
    # 0. Restore model
    
    
    # 1. Load validation data
    print("Loading database..")
    db, squiggles = helper_functions.load_db(db_dir)
    
    # 2. Validate model 
    valid_reads = 0
    # per squiggle:
    for squig in squiggles:
        data_sq, labels_sq = reader.load_npz(squig)
            
        if len(data_sq) >= max_seq_length:
            data.append(data_sq[: max_seq_length])
            labels.append(labels_sq[: max_seq_length]) 
            valid_reads += 1
    
        set_x = reshape_input(data, window, n_inputs)
        set_y = reshape_input(labels, window, n_outputs)
        
        network.test_network(set_x, set_y)
        
    
    print("Validated model {} on {} raw signals.".format(network.model_type, valid_reads))

if __name__ ==  "__main__":
    db_dir = argv[1]
