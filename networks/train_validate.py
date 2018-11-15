#!/usr/bin/env python3
import helper_functions
import numpy as np
import reader
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf


def reshape_input(data, window, n_inputs):
    data = np.reshape(data, (-1, window, n_inputs))  
    return data 


def build_model(network_type):
    model_id = 0
    batch_size = 64
    learning_rate = 0.01
    n_epochs = 10
    n_layers = 2
    layer_size = 128
    keep_prob = 0.8     
    optimizer_choice = "Adam"
    layer_size_res = 16
    depth = 1

    if network_type == "RNN":
        network = RNN(model_id, batch_size=batch_size, layer_size=layer_size, 
                        n_layers=n_layers, optimizer_choice=optimizer_choice,  
                        learning_rate=learning_rate, keep_prob=keep_prob)
    
    if network_type == "ResNetRNN":                   
        network = ResNetRNN(model_id, batch_size=batch_size, layer_size=layer_size, 
                        n_layers=n_layers, optimizer_choice=optimizer_choice, 
                        learning_rate=learning_rate, keep_prob=keep_prob, 
                        n_layers_res=depth, layer_size_res=layer_size_res)
                        
    return network


def train(network, db_dir, training_nr, n_epochs):
    """
    Training on windows
    
    Args:
        db_dir -- str, path to database
        training_nr -- int, number of training examples
    """
    saving = True
    
    # 1. load db
    print("Loading database..")
    db, squiggles = helper_functions.load_db(db_dir)
    
    # 2. train network
    n_examples = training_nr // network.batch_size * network.batch_size
    n_batches = n_examples // network.batch_size
    print("Training on {} examples in {} batches\n".format(n_examples, n_batches))
    step = 0
    positives = 0       # positives here are labelled as 1
    negatives = 0
    for n in range(n_epochs):
        for b in range(n_batches):
            # load batch sized training examples:
            data, labels, pos, neg = db.get_training_set(network.batch_size)      # TODO: take care that batches contain different examples
            set_x = reshape_input(data, network.window, network.n_inputs)
            set_y = reshape_input(labels, network.window, network.n_outputs)
            
            positives += pos
            negatives += neg
            
            # train on batch:
            step += 1
            network.train_network(set_x, set_y, step)
            
            if step % network.saving_step == 0:   
                network.saver.save(network.sess, network.model_path + "/checkpoints", global_step=step, write_meta_graph=False)            
 
        print("Finished epoch: ", n)      
        
    network.saver.save(network.sess, network.model_path + "/checkpoints", global_step=step)
    print("\nSaved final checkpoint\n")
    
    total = positives + negatives
    print("True percentage HPs: {:.2%}".format(positives / total))  
    print("Finished training :)")
  
    
def validate(network, db_dir, max_seq_length):
    """
    Validate model on full reads
    
    Args:
        db_dir -- str, path to validation database
    """  
    # 1. Load validation data
    print("Loading database..")
    db, squiggles = helper_functions.load_db(db_dir)
    
    # 2. Validate model 
    valid_reads = 0
    max_seq_length = max_seq_length // network.window * network.window
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    roc_auc = 0
    # per squiggle:
    for squig in squiggles:
        data_sq, labels_sq = reader.load_npz(squig)
            
        if len(data_sq) >= max_seq_length:
            data = data_sq[: max_seq_length]
            labels = labels_sq[: max_seq_length] 
            valid_reads += 1
    
            set_x = reshape_input(data, network.window, network.n_inputs)
            set_y = reshape_input(labels, network.window, network.n_outputs)
            
            sgl_acc, sgl_precision, sgl_recall, sgl_f1, sgl_auc = network.test_network(set_x, set_y)
            accuracy += sgl_acc
            precision += sgl_precision
            recall += sgl_recall
            f1 += sgl_f1
            roc_auc += sgl_auc
        
        else:
            continue
        
    # averaged performance:
    print("\nPerformance of validation set:")
    print("\tAccuracy: {:.2%}".format(accuracy / valid_reads))
    print("\tPecision: {:.2%}\nRecall: {:.2%}".format(precision / valid_reads, recall / valid_reads))
    print("\tWeighted F1 measure: {0:.4f}".format(float(f1 / valid_reads)))
    print("\tAUC: {0:.4f}".format(roc_auc / valid_reads))
        
    print("Finished validation of model {} on {} raw signals of length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   max_seq_length))

if __name__ == "__main__":
    # get input
    if not len(argv) == 7:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-path to training db" +
                         "\n\t-number of training reads\n\t-number of epochs" + 
                         "\n\t-path to validation db\n\t-max length of validation reads")
    network_type = argv[1]
    db_dir_train = argv[2]
    training_nr = int(argv[3])
    n_epochs = int(argv[4])
    
    db_dir_val = argv[5]
    max_seq_length = int(argv[6])
    
    # train and validate network
    network = build_model(network_type)
    train(network, db_dir_train, training_nr, n_epochs)
    validate(network, db_dir_val, max_seq_length)
