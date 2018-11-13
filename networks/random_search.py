#!/usr/bin/env python3
import numpy as np
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf
import train


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

    batch_size_list = [64, 128, 256]

    dropout_min = 0.2
    dropout_max = 0.8
    
    # pick random hyperparameter:
    learning_rate = 10 ** np.random.randint(learning_rate_min, learning_rate_max)
    optimizer = np.random.choice(optimizer_list)
    layer_size = np.random.choice(layer_size_list)
    n_layers = np.random.randint(n_layers_min, n_layers_max)
    batch_size = np.random.choice(batch_size_list)
    dropout = round(np.random.uniform(dropout_min, dropout_max), 1)
    
    return learning_rate, optimizer, layer_size, n_layers, batch_size, dropout


def create_model(model_id):                        # werkt
    with tf.variable_scope(None, 'model'):
        lr, opt, l_size, n_layers, batch_size, dropout = generate_random_hyperparameters()
            #~ return ResNetRNN(**kwargs)       # change model to desired model
        return RNN(model_id, learning_rate=lr, optimizer_choice=opt, n_layers=n_layers,
                    layer_size=l_size, batch_size=batch_size, keep_prob=dropout)        
                    

if __name__ == "__main__":
    #1. Create multiple models
    window = 35 
    if not len(argv) == 7:
        raise ValueError("The following arguments should be provided:\n\t-number of models\n" +
                         "\t-trainingdb\n\t-nr trainingreads\n\t-nr epochs\n\t-valdb\n\t-nr validationreads")
    
    POPULATION_SIZE = int(argv[1])
    models = [create_model(i) for i in range(POPULATION_SIZE)]
    
    #2. Train models
    db_dir = argv[2]
    training_nr = int(argv[3])
    print("Training on : {} windows".format(training_nr))
    n_epochs = int(argv[4])
    train_x, train_y = train.retrieve_set(db_dir, training_nr, "trainingreads")
    for m in models:
        print("------------------------------MODEL {}------------------------------".format(m.model_id))
        m.train_network(train_x, train_y, n_epochs)
    
    #3. Assess performance on validation set (squiggles)
    db_dir_val = argv[5]
    val_nr = int(argv[6])                   # // window * window
    print("Validating on: {} squiggles".format(val_nr))
    val_x, val_y = train.retrieve_set(db_dir_val, val_nr, "squiggles")
    for m in models:
        print("------------------------------MODEL {}------------------------------".format(m.model_id))
        m.test_network(val_x, val_y)
