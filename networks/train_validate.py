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
        network_type -- str, type of network: "RNN" or "ResNetRNN"
        
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
        
        if step % 2 == 0:   
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
    
    
def validate(network, squiggles, max_seq_length):
    """
    Validate model on full reads
    
    Args:
        network -- RNN object, network model
        squiggles -- npz files, raw signals and labels of MinION sequencing data
        max_seq_length -- int, maximum length of read to take in
        
    Returns: validation accuracy
    """  
    max_seq_length = max_seq_length // network.window * network.window
    accuracy = 0
    loss = 0
    valid_reads = 0

    # per squiggle:
    for squig in squiggles:
        t1 = datetime.datetime.now()
        data_sq, labels_sq = reader.load_npz(squig)
        t2 = datetime.datetime.now()

        if len(data_sq) >= max_seq_length:
            labels = labels_sq[: max_seq_length] 
            data = data_sq[: max_seq_length]
            
            read_name = os.path.basename(squig).split(".npz")[0]
            valid_reads += 1
    
            set_x = reshape_input(data, network.window, network.n_inputs)
            set_y = reshape_input(labels, network.window, network.n_outputs)
            
            t3 = datetime.datetime.now()
            sgl_acc, sgl_loss  = network.test_network(set_x, set_y, valid_reads, read_name)
            t4 = datetime.datetime.now()
            
            #~ if valid_reads % network.saving_step == 0:
                #~ metrics.plot_squiggle(data, "Squiggle_{}_{}".format(os.path.basename(network.model_path), valid_reads))
                #~ metrics.plot_squiggle(data[2000 : 3001], "Squiggle_{}_{}_middle".format(os.path.basename(network.model_path), valid_reads))
                
            accuracy += sgl_acc
            loss += sgl_loss
        
        else:
            continue
    
    whole_accuracy = metrics.calculate_accuracy(network.tp, network.fp, network.tn, network.fn)
    whole_precision, whole_recall = metrics.precision_recall(network.tp, network.fp, network.fn)
    whole_f1 = metrics.weighted_f1(whole_precision, whole_recall, (network.tp + network.fn), valid_reads * max_seq_length)
    
    # averaged performance:  
    with open(network.model_path + ".txt", "a+") as dest: 
        dest.write("\nNEXT EPOCH")
        dest.write("\nAverage performance of validation set:\n")
        dest.write("\tAccuracy: {:.2%}\n".format(accuracy / valid_reads))
        dest.write("\tLoss: {0:.4f}".format(loss / valid_reads))

    # over whole set:
        dest.write("\nPerformance over whole set: \n")
        dest.write("\tDetected {} true positives, {} false positives, {} true negatives, {} false negatives in total.\n".
                format(network.tp, network.fp, network.tn, network.fn))
        dest.write("\tTrue number of HPs: {} \tTrue percentage: {:.2%}\t Predicted percentage HPs: {:.2%}\n".
                format(network.tp + network.fn, (network.tp + network.fn) / (valid_reads * max_seq_length), (network.tp + network.fp) / (valid_reads * max_seq_length)))
        dest.write("\tAccuracy: {:.2%}".format(whole_accuracy))
        dest.write("\n\tPrecision: {:.2%}\n\tRecall: {:.2%}".format(whole_precision, whole_recall))
        dest.write("\t\nWeighed F1: {0:.4f}".format(whole_f1))
        dest.write("\nFinished validation of model {} on {} raw signals of length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   max_seq_length))    
    print("\nFinished validation of model {} on {} raw signals of length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   max_seq_length))
    print("Validation accuracy: ", whole_accuracy)
    print("Validation loss: ", loss / valid_reads)
    return whole_accuracy, whole_precision, whole_recall



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
    
    #~ hpm_dict = {"batch_size": 16, "learning_rate": 0.01, "n_layers": 2,
                #~ "layer_size": 16, "keep_prob": 0.6, "optimizer_choice": "Adam"}
    hpm_dict = {"batch_size":256, "optimizer_choice": "Adam", "learning_rate":0.001, "layer_size":64, "n_layers":1, "keep_prob":0.3, "layer_size_res":128, "n_layers_res":11}
    #~ hpm_dict = {"batch_size": 256, "optimizer_choice": "Adam", "learning_rate":0.001, "layer_size":64, "n_layers":1, "keep_prob":0.3, "layer_size_res":128, "n_layers_res": 10}

    p = psutil.Process(os.getpid())
    m1 = p.memory_full_info().pss
    print("Memory use at start is", m1)
    
    # build model
    print("Started script at ", datetime.datetime.now())
    t1 = datetime.datetime.now()
    network = build_model(network_type, **hpm_dict)
    t2 = datetime.datetime.now()
    m2 = p.memory_full_info().pss
    print("Extra memory use after building network is", m2 - m1)
    #~ network.restore_network("/mnt/nexenta/thijs030/networks/ResNet-RNN_9/checkpoints")
    network.initialize_network()
    t22 = datetime.datetime.now()
    m3 = p.memory_full_info().pss
    print("Building model took {}".format(t2 - t1))
    print("Initialized model in {}".format(t22 - t2))
    print("Extra memory use after initialization ", m3 - m2)
    
    # train and validate network
    print("Loading training database..")
    print(datetime.datetime.now())
    t3 = datetime.datetime.now()
    db_train = helper_functions.load_db(db_dir_train)
    t4 = datetime.datetime.now()
    m4 = p.memory_full_info().pss
    print("Extra memory use after loading db is", m4 - m3)
    print("Loaded db in {}".format(t4 - t3))
    
    seed = random.randint(0, 1000000000)
    
    t0 = datetime.datetime.now()
    pos_range, neg_range = db_train.set_ranges(seed)
    print("Set ranges in {}".format(datetime.datetime.now() - t0))
    for n in range(n_epochs):
        db_train.range_ps = pos_range
        db_train.range_ns = neg_range
        network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", write_meta_graph=True)
        print("Saved checkpoint at start of epoch {}".format(n))
        t5 = datetime.datetime.now()
        train_acc = train(network, db_train, training_nr)
        t6 = datetime.datetime.now()
        m5 = p.memory_full_info().pss
        print("Extra memory use after training is", m5 - m4)
        print("Trained network in {}".format(t6 - t5))
        print("Finished epoch: {}".format(n))
    
    #~ # validate network
    #~ print("Loading validation database..")
    #~ squiggles = helper_functions.load_squiggles(db_dir_val)
    #~ t7 = datetime.datetime.now()
    #~ m6 = p.memory_full_info().pss
    #~ print("Extra memory use after loading squiggles is ", m6 - m5)
    #~ print("Loaded squiggles in {}".format(t7 - t6))
    #~ val_acc = validate(network, squiggles, max_seq_length)
    #~ t8 = datetime.datetime.now()
    #~ m7 = p.memory_full_info().pss
    #~ print("Extra memory use after validation is ", m7 - m6)
    #~ print("Validated network in {}".format(t8 - t7))
    #~ print("Finished script at ", t8)
    
    #~ print("Memory use at end is ", p.memory_full_info().pss)
