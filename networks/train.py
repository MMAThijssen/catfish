#!/usr/bin/env python3
import helper_functions
import numpy as np
import reader
from sys import argv
import tensorflow as tf

tf.set_random_seed(16)

def reshape_input(data, window, n_inputs):
    #~ data = np.concatenate(data)     
    #~ data = data.reshape(-1, window, n_inputs)
    data = np.reshape(data, (-1, window, n_inputs))  
    return data      
    
    
def retrieve_set(db_dir, training_nr, training_type, balanced="train"):
    window = 35
    n_inputs = 1
    n_outputs = 1

    #~ db_dir = argv[1]
    #~ training_nr = argv[2]           # is either number of trainingreads or max_seq_length      
    
    db, squiggles = helper_functions.load_db(db_dir)
    
    if training_type == "trainingreads":
        # 1. Train on training reads:
        balanced = "train"          # balanced set
        if balanced != "train":
            balanced = "test"       # unbalanced set
            print("\nUnbalanced training set\n")

        data, labels = db.get_training_set(training_nr , balanced)
    
    if training_type == "squiggles":
        # 2. Train on squiggles:
        data = []
        labels = []
        for squig in squiggles:
            data_sq, labels_sq = reader.load_npz(squig)

            data.append(data_sq[: training_nr])
            labels.append(labels_sq[: training_nr]) 
    
    set_x = reshape_input(data, window, n_inputs)
    set_y = reshape_input(labels, window, n_outputs)
    
    return set_x, set_y

    
if __name__ == "__main__":
    category = "trainingreads"
    #~ category = "squiggles"
    db_dir = argv[1]                # "/mnt/nexenta/thijs030/data/trainingDB/test3/"
    training_nr = int(argv[2])           # is either number of trainingreads or max_seq_length  
    train_x, train_y = retrieve_set(db_dir, training_nr, category)
    print("Shape data : ", train_x.shape)


