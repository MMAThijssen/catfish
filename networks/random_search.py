#!/usr/bin/env python3
import numpy as np
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf
from train_validate import train, validate


def generate_random_hyperparameters(network_type):
    """
    Returns: learning_rate, optimizer, layer_size, n_layers, batch_size, dropout
    """
    # HYPERPARAMETERS #
    learning_rate_min = -4      # 10 ^ -4
    learning_rate_max = 0

    optimizer_list = ["Adam", "RMSProp"]

    layer_size_list = [16, 32, 64, 128, 256]

    n_layers_min = 1
    n_layers_max = 6        # randint is exclusive

    batch_size_list = [64, 128, 256]

    dropout_min = 0.2
    dropout_max = 0.8
    
    n_layers_res_min = 1             # 6, 12, 18, 34 / 3
    n_layers_res_max = 12
    
    size_layers_res_list = [16, 32, 64, 128, 256]
    
    # pick random hyperparameter:
    learning_rate = 10 ** np.random.randint(learning_rate_min, learning_rate_max)
    optimizer = np.random.choice(optimizer_list)
    layer_size = np.random.choice(layer_size_list)
    n_layers = np.random.randint(n_layers_min, n_layers_max)
    batch_size = np.random.choice(batch_size_list)
    dropout = round(np.random.uniform(dropout_min, dropout_max), 1)
    
    if network_type == "ResNetRNN":
        n_layers_res = np.random.randint(n_layers_res_min, n_layers_res_max)
        size_layers_res = np.random.choice(size_layers_res_list)
        
        return learning_rate, optimizer, layer_size, n_layers, batch_size, dropout, n_layers_res, size_layers_res
    
    return learning_rate, optimizer, layer_size, n_layers, batch_size, dropout


def create_model(model_id, network_type):                        # werkt
    with tf.variable_scope(None, 'model'):
        if network_type == "RNN":
            lr, opt, l_size, n_layers, batch_size, dropout = generate_random_hyperparameters(network_type)
            return RNN(model_id, learning_rate=lr, optimizer_choice=opt, n_layers=n_layers,
                    layer_size=l_size, batch_size=batch_size, keep_prob=dropout) 
        elif network_type == "ResNetRNN":
            lr, opt, l_size, n_layers, batch_size, dropout, 
            n_lay_res, size_lay_res = generate_random_hyperparameters(network_type)
            return ResNetRNN(model_id, learning_rate=lr, optimizer_choice=opt, n_layers=n_layers,
                    layer_size=l_size, batch_size=batch_size, keep_prob=dropout, 
                    n_layers_res=n_lay_res, layer_size_res=size_lay_res)             
       
                    

if __name__ == "__main__":
    #0. Get input
    if not len(argv) == 8:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-number of models\n\t-path to training db" +
                         "\n\t-number of training reads\n\t-number of epochs" + 
                         "\n\t-path to validation db\n\t-max length of validation reads")
    
    network_type = argv[1]
    POPULATION_SIZE = int(argv[2])
    db_dir_train = argv[3]
    training_nr = int(argv[4])
    n_epochs = int(argv[5])
    db_dir_val = argv[6]
    max_seq_length = int(argv[7])                   
    
    # 1. Create models
    models = [create_model(i, network_type) for i in range(POPULATION_SIZE)]
    
    # 2. Train models
    for m in models:
        print("------------------------------MODEL {}------------------------------".format(m.model_id))
        train(m, db_dir_train, training_nr, n_epochs)
    
    #3. Assess performance on validation set
    for m in models:
        print("------------------------------MODEL {}------------------------------".format(m.model_id))
        validate(m, db_dir_val, max_seq_length)
