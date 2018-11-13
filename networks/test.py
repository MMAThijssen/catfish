#!/usr/bin/env python3
import random
import numpy as np

def explore():
    hps = ["learning_rate", "optimizer_choice", "layer_size", "n_layers", "batch_size", "keep_prob"]
    random_hps = generate_random_hyperparameters()
#    hp_int = random.choice(range(len(random_hps)))
    return list(zip(hps, random_hps))

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
  
exp = explore()
print(type(exp[1]))
print(exp)