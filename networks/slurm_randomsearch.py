#!/usr/bin/env python3
import helper_functions
import numpy as np
import os
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf
from train_validate import train, validate, build_model

tf.set_random_seed(33)

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
    with tf.variable_scope(None, "Model_{}".format(model_id)):
        if network_type == "RNN":
            lr, opt, l_size, n_layers, batch_size, drpt = generate_random_hyperparameters(network_type)
            return RNN(model_id, learning_rate=lr, optimizer_choice=opt, n_layers=n_layers,
                    layer_size=l_size, batch_size=batch_size, keep_prob=drpt) 
        elif network_type == "ResNetRNN":
            lr, opt, l_size, n_layers, batch_size, drpt, n_lay_res, size_lay_res = generate_random_hyperparameters(network_type)
            return ResNetRNN(model_id, learning_rate=lr, optimizer_choice=opt, n_layers=n_layers,
                    layer_size=l_size, batch_size=batch_size, keep_prob=drpt, 
                    n_layers_res=n_lay_res, layer_size_res=size_lay_res)             
       


if __name__ == "__main__":
    #0. Get input
    if not len(argv) == 8:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-model id\n\t-path to training db" +
                         "\n\t-number of training reads\n\t-number of epochs" + 
                         "\n\t-path to validation db\n\t-max length of validation reads")
    
    network_type = argv[1]
    model_id = int(argv[2])
    db_dir_train = argv[3]
    training_nr = int(argv[4])
    n_epochs = int(argv[5])
    db_dir_val = argv[6]
    max_seq_length = int(argv[7])                  
    
    #~ print("CPU stats: ", psutil.cpu_stats())
    #~ print("Virtual memory stats: ", psutil.virtual_memory())
    #~ p = psutil.Process(os.getpid())
    #~ print("CPU percent p: ", p.cpu_percent())
    
    # 1. Create model
    #~ model = create_model(model_id, network_type)
    #~ model.initialize_network()
    #~ print(model.sess)
    
    # or restore model:
    model = build_model("RNN", from_kwargs=True,batch_size=256, optimizer_choice="Adam", learning_rate=0.001, layer_size=64, n_layers=1, keep_prob=0.3, layer_size_res=128, n_layers_res=11)
    #~ model.restore_network("/mnt/nexenta/thijs030/networks/biGRU-RNN_123/checkpoints")
    model.initialize_network()

 
    #~ # 2. Train models
    #~ print("Loading training database..")
    db_train = helper_functions.load_db(db_dir_train)
    #~ print("------------------------------MODEL {}------------------------------".format(model_id))
    train(model, db_train, training_nr, n_epochs)
    
    #~ training = [print("------------------------------MODEL {}------------------------------".format(m.model_id)), 
                #~ train(m, db_train, training_nr, n_epochs) for m in models]
    
    #3. Assess performance on validation set
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    #~ print("------------------------------MODEL {}------------------------------".format(model_id))
    validate(model, squiggles, max_seq_length)

