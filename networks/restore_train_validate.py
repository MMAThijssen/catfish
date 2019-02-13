#!/usr/bin/env python3
import datetime
import trainingDB.helper_functions
import numpy as np
import os
import psutil
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf
from train_validate import build_model, generate_random_hyperparameters, train_and_validate, validate


def retrieve_hyperparams(model_file, split_on=": "):
    """
    Retrieve hyperparameters from model file.
    
    Args:
        model_file -- str, path to file on model created by train_validate.build_model
        split_on -- str, combination of characters to split on [default: ": "]
    
    Returns: dict of hyperparameters
    """
    hpm_dict = {}
    with open(model_file, "r") as source:
        for line in source:
            if line.startswith("batch_size"):
                hpm_dict["batch_size"] = int(line.strip().split(split_on)[1])
            elif line.startswith("optimizer_choice"):
                hpm_dict["optimizer_choice"] = line.strip().split(split_on)[1]
            elif line.startswith("learning_rate"):
                hpm_dict["learning_rate"] = float(line.strip().split(split_on)[1])
            elif line.startswith("layer_size:"):
                hpm_dict["layer_size"] = int(line.strip().split(split_on)[1])
            elif line.startswith("n_layers:"):
                hpm_dict["n_layers"] = int(line.strip().split(split_on)[1])
            elif line.startswith("keep_prob"):
                hpm_dict["keep_prob"] = float(line.strip().split(split_on)[1])
            elif line.startswith("layer_size_res"):
                hpm_dict["layer_size_res"] = int(line.strip().split(split_on)[1])
            elif line.startswith("n_layers_res"):   
                hpm_dict["n_layers_res"] = int(line.strip().split(split_on)[1])
                
    return hpm_dict
    

if __name__ == "__main__":
    #0. Get input
    if not len(argv) >= 8:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-path to saved network\n\t-checkpoint to restore" +
                         "\n\t-path to training db\n\t-number of training reads" + 
                         "\n\t-path to validation db\n\t-max length of validation reads" + 
                         "\nOPTIONAL:\n\t-only validation, no training")
    print("THRESHOLD ON 0.8!")
    
    network_type = argv[1]
    network_path = argv[2]
    checkpoint = argv[3]
    db_dir_train = argv[4]
    training_nr = int(argv[5])
    db_dir_val = argv[6]
    max_seq_length = int(argv[7])
    only_validation = False
    saving = True
    if len(argv) == 9:
        only_validation = True
        saving = False
        print("Only validating now, NO training")
    validation_start = "complete" #"random"  # #0 #30000
    max_number = 12256  #856          
    
    # Keep track of memory and time
    p = psutil.Process(os.getpid())
    t1 = datetime.datetime.now() 
    m1 = p.memory_full_info().pss
    print("\nMemory use at start is", m1)  
    print("Started script at {}\n".format(t1))
    

    #~ # 1a. Restore model
    hpm_dict = retrieve_hyperparams("{}.txt".format(network_path))
    model = build_model(network_type, save=saving, **hpm_dict)
    model.restore_network("{}/checkpoints".format(network_path), ckpnt="ckpnt-{}".format(checkpoint))
    
    # 1b. Extend RNN model
    #~ hpm_dict = retrieve_hyperparams("{}.txt".format(network_path))
    #~ resnet_dict = generate_random_hyperparameters(network_type)
    #~ hpm_dict["layer_size_res"] = resnet_dict["layer_size_res"]
    #~ hpm_dict["n_layers_res"] = resnet_dict["n_layers_res"]
    #~ model = build_model(network_type, save=True, **hpm_dict)
    # ~ model.initialize_network() 

 
    # 2. Train model
    if not only_validation:
        file_path = model.model_path
        print("Saving to information to {} extended".format(file_path))
        print("Loading training database..")
        db_train = trainingDB.helper_functions.load_db(db_dir_train)
        print("Loading validation database..")
        squiggles = trainingDB.helper_functions.load_squiggles(db_dir_val)
        t2 = datetime.datetime.now()
        train_and_validate(model, db_train, training_nr, squiggles, max_seq_length, file_path, validation_start, max_number)
        t3 = datetime.datetime.now()  
        m3 = p.memory_full_info().pss
        print("\nMemory after training is ", m3)
        print("Trained and validated model in {}\n".format(t3 - t2))
        
    if only_validation:
        file_path = network_path + "_"
        print("Saving to information to {} extended".format(file_path))
        print("Loading validation database..")
        squiggles = trainingDB.helper_functions.load_squiggles(db_dir_val)
        validate(model, squiggles, max_seq_length, file_path, validation_start, max_number)  
        
    tf.reset_default_graph()



