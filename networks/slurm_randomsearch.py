#!/usr/bin/env python3
import helper_functions
import numpy as np
import os
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf
from train_validate import train, validate, build_model

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
    Generates random hyperparameters
    
    Args:
        network_type -- str, type of network (empty (RNN) or "ResNetRNN")
        learning_rate_min -- int, minimal negative number in exponential for learning rate [default: -4]
        learning_rate_max -- int, maximal positive number in exponential for learning rate [default: 0]
    Returns: dict of hyperparameters
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



def retrieve_hyperparams(model_file):
    """
    Retrieve hyperparameters from model file.
    
    Args:
        model_file -- str, path to file on model created by train_validate.build_model
    
    Returns: dict of hyperparameters
    """
    hpm_dict = {}
    with open(model_file, "r") as source:
        for line in source:
            if line.startswith("batch_size"):
                hpm_dict["batch_size"] = int(line.strip().split(": ")[1])
            if line.startswith("optimizer_choice"):
                hpm_dict["optimizer_choice"] = line.strip().split(": ")[1]
            if line.startswith("learning_rate"):
                hpm_dict["learning_rate"] = float(line.strip().split(": ")[1])
            if line.startswith("layer_size"):
                hpm_dict["layer_size"] = int(line.strip().split(": ")[1])
            if line.startswith("n_layers"):
                hpm_dict["n_layers"] = int(line.strip().split(": ")[1])
            if line.startswith("keep_prob"):
                hpm_dict["keep_prob"] = float(line.strip().split(": ")[1])
            if line.startswith("layer_size_res"):
                hpm_dict["layer_size_res"] = int(line.strip().split(": ")[1])
            if line.startswith("n_layers_res"):   
                hpm_dict["n_layers_res"] = int(line.strip().split(": ")[1])
                
    return hpm_dict



if __name__ == "__main__":
    #0. Get input
    if not len(argv) == 7:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-model id\n\t-path to training db" +
                         "\n\t-number of training reads\n\t-number of epochs" + 
                         "\n\t-path to validation db\n\t-max length of validation reads")
    
    network_type = argv[1]
    db_dir_train = argv[2]
    training_nr = int(argv[3])
    n_epochs = int(argv[4])
    db_dir_val = argv[5]
    max_seq_length = int(argv[6])                  
        
        
    # 1. Create model or restore model:
    #~ hpm_dict = generate_random_hyperparameters(network_type)
    hpm_dict = retrieve_hyperparams("/mnt/nexenta/thijs030/networks/biGRU-RNN_165.txt")
    model = build_model(network_type, **hpm_dict)
    #~ model.initialize_network()
    model.restore_network("/mnt/nexenta/thijs030/networks/biGRU-RNN_165/checkpoints")

 
    # 2. Train models
    print("Loading training database..")
    db_train = helper_functions.load_db(db_dir_train)
    train(model, db_train, training_nr, n_epochs)

    
    #3. Assess performance on validation set
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    validate(model, squiggles, max_seq_length)

