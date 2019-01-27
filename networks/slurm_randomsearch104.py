#!/usr/bin/env python3
import datetime
import helper_functions
import numpy as np
import os
import psutil
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf
from train_validate import train_and_validate, validate, build_model

def generate_random_hyperparameters(network_type,
                                    learning_rate_min=-4,      
                                    learning_rate_max=0,
                                    optimizer_list=["Adam", "RMSProp"],
                                    layer_size_list=[16, 32, 64, 128, 256],
                                    n_layers_min=1,
                                    n_layers_max=6,   
                                    batch_size_list=[128, 256, 512],
                                    dropout_min=0.2,
                                    dropout_max=0.8,
                                    n_layers_res_min=1,
                                    n_layers_res_max=12,
                                    size_layers_res_list=[16, 32, 64, 128, 256]):
    """
    Generates random hyperparameters.
    
    Args:
        network_type -- str, type of network (empty (RNN) or "ResNetRNN")
        learning_rate_min -- int, minimal negative number in exponential for learning rate [default: -4]
        learning_rate_max -- int, maximal positive number in exponential for learning rate [default: 0]
    
    Returns: dict {str: value for hyperparameter}
    """   
    # pick random hyperparameter:
    learning_rate = 10 ** np.random.randint(learning_rate_min, learning_rate_max)
    optimizer = np.random.choice(optimizer_list)
    layer_size = np.random.choice(layer_size_list)
    n_layers = np.random.randint(n_layers_min, n_layers_max)
    batch_size = np.random.choice(batch_size_list)
    dropout = round(np.random.uniform(dropout_min, dropout_max), 1)
    
    # create dict:
    hpm_dict = {"batch_size": batch_size, "optimizer_choice": optimizer, 
                "learning_rate": learning_rate, "layer_size": layer_size, 
                "n_layers": n_layers, "keep_prob": dropout}
    
    # extend dict if network is ResNetRNN:            
    if network_type == "ResNetRNN":
        n_layers_res = np.random.randint(n_layers_res_min, n_layers_res_max)
        size_layers_res = np.random.choice(size_layers_res_list)
        
        hpm_dict["layer_size_res"] = size_layers_res
        hpm_dict["n_layers_res"] = n_layers

    return hpm_dict



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
    #~ if not len(argv) == 6:
        #~ raise ValueError("The following arguments should be provided in this order:\n" + 
                         #~ "\t-network type\n\t-model id\n\t-path to training db" +
                         #~ "\n\t-number of training reads\n\t-number of epochs" + 
                         #~ "\n\t-path to validation db\n\t-max length of validation reads")
    
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
    validation_start = "random"
    max_number = 856
    #~ db_dir_train = argv[2]
    #~ training_nr = int(argv[3])
    #~ n_epochs = int(argv[4])
    #~ db_dir_val = argv[5]
    #~ max_seq_length = int(argv[6])                  
    
    # Keep track of memory and time
    p = psutil.Process(os.getpid())
    t1 = datetime.datetime.now() 
    m1 = p.memory_full_info().pss
    print("\nMemory use at start is", m1)  
    print("Started script at {}\n".format(t1))
        
    #~ # 1. Create model
    #~ hpm_dict = generate_random_hyperparameters(network_type)
    #~ model = build_model(network_type, **hpm_dict)
    #~ model.initialize_network()
    #~ t2 = datetime.datetime.now()  
    #~ m2 = p.memory_full_info().pss
    #~ print("\nMemory after building model is ", m2)
    #~ print("Built and initialized model in {}\n".format(t2 - t1))

    # 1b. Restore model
    hpm_dict = retrieve_hyperparams("{}.txt".format(network_path))
    model = build_model(network_type, save=saving, **hpm_dict)
    model.restore_network("{}/checkpoints".format(network_path, ckpnt="ckpnt-{}".format(checkpoint)))
    
    # 1c. Extend RNN model
    #~ hpm_dict = retrieve_hyperparams("/mnt/nexenta/thijs030/networks/biGRU-RNN_104.txt")
    #~ resnet_dict = generate_random_hyperparameters(network_type)
    #~ hpm_dict["layer_size_res"] = resnet_dict["layer_size_res"]
    #~ hpm_dict["n_layers_res"] = resnet_dict["n_layers_res"]
    #~ model = build_model(network_type, **hpm_dict)
    #~ model.initialize_network()    
 
    # 2. Train model
    if not only_validation:
        file_path = model.model_path
        print("Saving to information to {} extended".format(file_path))
        print("Loading training database..")
        db_train = helper_functions.load_db(db_dir_train)
        print("Loading validation database..")
        squiggles = helper_functions.load_squiggles(db_dir_val)
        t2 = datetime.datetime.now()
        train_and_validate(model, db_train, training_nr, squiggles, max_seq_length, file_path, validation_start, max_number)
        t3 = datetime.datetime.now()  
        m3 = p.memory_full_info().pss
        print("\nMemory after training is ", m3)
        print("Trained and validated model in {}\n".format(t3 - t2))
        
    if only_validation:
        file_path = network_path + "_val_"
        print("Saving to information to {} extended".format(file_path))
        print("Loading validation database..")
        squiggles = helper_functions.load_squiggles(db_dir_val)
        validate(model, squiggles, max_seq_length, file_path, validation_start, max_number)     




