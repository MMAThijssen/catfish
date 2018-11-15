#!/usr/bin/env python3

import numpy as np

def generate_random_hyperparameters():
    """
    Returns: learning_rate, optimizer, layer_size, n_layers, batch_size, dropout
    """
    # HYPERPARAMETERS #
    learning_rate_min = -4      # 10 ^ -4
    learning_rate_max = 0

    optimizer_list = ["Adam", "RMSProp"]

    layer_size_list = [16, 32, 64, 128, 256]

    n_layers_min = 1
    n_layers_max = 5

    batch_size_list = [32, 64]

    dropout_min = 0.0
    dropout_max = 0.8
    
    # pick random hyperparameter:
    learning_rate = 10 ** np.random.randint(learning_rate_min, learning_rate_max)
    optimizer = np.random.choice(optimizer_list)
    layer_size = np.random.choice(layer_size_list)
    n_layers = np.random.randint(n_layers_min, n_layers_max)
    batch_size = np.random.choice(batch_size_list)
    dropout = round(np.random.uniform(dropout_min, dropout_max), 1)
    
    return learning_rate, optimizer, layer_size, n_layers, batch_size, dropout


print(len(list(set([(0.1, 'RMSProp', 64, 4, 32, 0.6), (0.001, 'RMSProp', 32, 4, 32, 0.7), (0.1, 'RMSProp', 64, 4, 32, 0.2), (0.001, 'RMSProp', 16, 2, 32, 0.7), (0.001, 'RMSProp', 128, 4, 64, 0.1), (0.001, 'Adam', 128, 4, 32, 0.2), (0.0001, 'RMSProp', 128, 4, 64, 0.0), (0.01, 'RMSProp', 128, 3, 64, 0.7), (0.001, 'Adam', 32, 1, 64, 0.3), (0.1, 'Adam', 64, 2, 64, 0.8), (0.001, 'RMSProp', 64, 3, 32, 0.6), (0.01, 'Adam', 128, 1, 32, 0.7), (0.1, 'RMSProp', 32, 1, 64, 0.3), (0.1, 'RMSProp', 128, 3, 64, 0.1), (0.0001, 'Adam', 128, 3, 32, 0.6), (0.1, 'RMSProp', 16, 1, 64, 0.6), (0.001, 'RMSProp', 256, 4, 64, 0.7), (0.001, 'Adam', 128, 1, 64, 0.6), (0.0001, 'RMSProp', 16, 2, 32, 0.1), (0.1, 'RMSProp', 64, 1, 32, 0.4), (0.01, 'Adam', 256, 4, 32, 0.5), (0.1, 'RMSProp', 16, 3, 64, 0.2), (0.0001, 'Adam', 64, 1, 32, 0.6), (0.1, 'Adam', 256, 3, 32, 0.7), (0.1, 'RMSProp', 16, 4, 32, 0.6), (0.001, 'RMSProp', 32, 2, 32, 0.2), (0.001, 'Adam', 256, 4, 64, 0.1), (0.01, 'Adam', 64, 2, 64, 0.7), (0.1, 'Adam', 256, 3, 32, 0.5), (0.1, 'Adam', 256, 3, 32, 0.5)]))))


#hp_list = [] 
#for i in range(30):
#    hp_list.append(generate_random_hyperparameters())
#print(list(set(hp_list)))
#print(len(list(set(hp_list))))