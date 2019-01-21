#!/usr/bin/env python3
import datetime
import helper_functions
import metrics
import numpy as np
import os
import psutil
import random
import reader
from resnet_class import ResNetRNN
from rnn_class import RNN
from sys import argv
import tensorflow as tf


def reshape_input(data, window, n_inputs):
    """
    Reshapes input to fit input tensor
    
    Args:
        data -- np.array, data to input
        window -- int, number of input points
        n_input -- int, number of features
        
    Returns: np.array
    """
    try:
        data = np.reshape(data, (-1, window, n_inputs)) 
    except ValueError:
        print(len(data))
        print(len(data[0]))
    return data 


def build_model(network_type, **kwargs):
    """
    Constructs neural network
    
    Args:
        network_type -- str, type of network ["RNN" or "ResNetRNN"]
        
    Returns: network object
    """
    if network_type == "RNN":
        network = RNN(**kwargs)
    
    elif network_type == "ResNetRNN":                   
        network = ResNetRNN(**kwargs)
                        
    return network   

def train(network, db, training_nr, squiggles, max_seq_length):
    """
    Training on windows
    
    Args:
        db -- str, ZODB database
        training_nr -- int, number of training examples
        n_epochs -- str, number of epochs
    
    Returns: training accuracy (float)
    """
    saving = True
    print(datetime.datetime.now())
        
    # 2. train network
    n_examples = training_nr // network.batch_size * network.batch_size
    n_batches = n_examples // network.batch_size
    print("\nTraining on {} examples in {} batches\n".format(n_examples, n_batches))
    with open(network.model_path + ".txt", "a") as dest:
        dest.write("Training on {} examples in {} batches\n".format(n_examples, n_batches))
    
    step = 0
    positives = 0

    for b in range(n_batches):
        # load batch sized training examples:
        data, labels, pos = db.get_training_set(network.batch_size)  
        positives += pos
        
        set_x = reshape_input(data, network.window, network.n_inputs)
        set_y = reshape_input(labels, network.window, network.n_outputs)
        
        # train on batch:
        step += 1                                                           # step is per batch
        network.train_network(set_x, set_y, step)
        
        if step % 1 == 0:   
            network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", global_step=step, write_meta_graph=True)            
            print("Saved checkpoint at step ", step)
            train_acc, train_loss = network.sess.run([network.accuracy, network.loss], feed_dict={network.x:set_x, network.y:set_y, network.p_dropout: network.keep_prob})
            print("Training accuracy: ", train_acc)
            print("Training loss: ", train_loss)
        
            print(step * network.batch_size)
            print(datetime.datetime.now())
            
            val_acc, whole_precision, whole_recall = validate(network, squiggles, max_seq_length)
            print("Validation precision: ", whole_precision)
            print("Validation recall: ", whole_recall)
            network.tp = 0
            network.fp = 0
            network.tn = 0
            network.fn = 0
    
    try:
        train_hp = positives / (network.window * n_examples)  
    except ZeroDivisionError:
        train_hp = 0
    print("Training set had {:.2%} HPs".format(train_hp))
        
    network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", global_step=step)
    print("\nSaved final checkpoint at step ", step, "\n")
    train_acc, train_loss = network.sess.run([network.accuracy, network.loss], feed_dict={network.x:set_x, network.y:set_y, network.p_dropout: network.keep_prob})
    print("Training accuracy: ", train_acc)
    print("Training loss: ", train_loss)
    
    print("\nFinished training!")
  
    return train_acc        # also return step?
    
    
def validate(network, squiggles, max_seq_length, file_path):
    """
    Validate model on full reads
    
    Args:
        network -- RNN object, network model
        squiggles -- npz files, raw signals and labels of MinION sequencing data
        max_seq_length -- int, maximum length of read to take in
    """  
    # 2. Validate model 
    max_seq_length = max_seq_length // network.window * network.window
    accuracy = 0
    loss = 0
    valid_reads = 0

    # per squiggle:
    max_number = 856
    #~ random.shuffle(squiggles)
    with open(file_path + "_checkpoint20000.txt", "a+") as dest: 
        for squig in squiggles:
            t1 = datetime.datetime.now()
            data_sq, labels_sq = reader.load_npz(squig)
            t2 = datetime.datetime.now()
             
            # OR specify start val:          
            start_val = 30000
            if len(data_sq) >= start_val + max_seq_length:           

            # get random start val:
            #~ if len(data_sq) >= max_seq_length:
                #~ start_val = random.randint(0, len(data_sq) - max_seq_length)
                #~ dest.write("\nStart validation at point: {}".format(start_val)) 
                labels = labels_sq[start_val: start_val + max_seq_length] 
                data = data_sq[start_val: start_val + max_seq_length]
                    
                read_name = os.path.basename(squig).split(".npz")[0]
                valid_reads += 1
                
                set_x = reshape_input(data, network.window, network.n_inputs)
                set_y = reshape_input(labels, network.window, network.n_outputs)
                
                t3 = datetime.datetime.now()
                sgl_acc, sgl_loss  = network.test_network(set_x, set_y, valid_reads, read_name, file_path)
                t4 = datetime.datetime.now()
                
                accuracy += sgl_acc
                loss += sgl_loss
            
            #~ # for 104:    
            if valid_reads >= max_number:
                break
            
            else:
                continue
        
        whole_accuracy = metrics.calculate_accuracy(network.tp, network.fp, network.tn, network.fn)
        whole_precision, whole_recall = metrics.precision_recall(network.tp, network.fp, network.fn)
        whole_f1 = metrics.f1(whole_precision, whole_recall)
    
    # averaged performance:         
        dest.write("\nNEXT EPOCH")
        dest.write("\nAverage performance of validation set:\n")
        dest.write("\tAccuracy: {:.2%}\n".format(accuracy / valid_reads))
        dest.write("\tLoss: {:.2%}\n".format(loss / valid_reads))
        
    # over whole set:
        dest.write("\nPerformance over whole set: \n")
        dest.write("\tDetected {} true positives, {} false positives, {} true negatives, {} false negatives in total.\n".
                format(network.tp, network.fp, network.tn, network.fn))
        dest.write("\tTrue number of HPs: {} \tTrue percentage: {:.2%}\t Predicted percentage HPs: {:.2%}\n".
                format(network.tp + network.fn, (network.tp + network.fn) / (valid_reads * max_seq_length), (network.tp + network.fp) / (valid_reads * max_seq_length)))
        dest.write("\tAccuracy: {:.2%}".format(whole_accuracy))
        dest.write("\n\tPrecision: {:.2%}\n\tRecall: {:.2%}".format(whole_precision, whole_recall))
        dest.write("\t\nF1 score: {0:.4f}".format(whole_f1))
        dest.write("\nFinished validation of model {} on {} raw signals of length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   max_seq_length))    
    
    network.tp = 0
    network.fn = 0
    network.tn = 0
    network.fp = 0
    print("\nFinished validation of model {} on {} raw signals of length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   max_seq_length))



if __name__ == "__main__":
    # get input
    db_dir_val = argv[1]
      
    # validate network
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    print(len(squiggles))
    print(squiggles[:2])
    random.shuffle(squiggles)
    print(squiggles[:2])

