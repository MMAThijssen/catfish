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
    try:
        data = np.reshape(data, (-1, window, n_inputs)) 
    except ValueError:
        print(len(data))
        print(len(data[0]))
    return data 


def build_model(network_type, **kwargs):
    batch_size = kwargs["batch_size"]
    optimizer_choice = kwargs["optimizer_choice"]
    learning_rate = kwargs["learning_rate"]
    layer_size = kwargs["layer_size"]        
    n_layers = kwargs["n_layers"]
    keep_prob = kwargs["keep_prob"]

    if network_type == "RNN":
        network = RNN(batch_size=batch_size, layer_size=layer_size, 
                        n_layers=n_layers, optimizer_choice=optimizer_choice,  
                        learning_rate=learning_rate, keep_prob=keep_prob)
    
    elif network_type == "ResNetRNN":        
        n_layers_res = kwargs["n_layers_res"]
        layer_size_res = kwargs["layer_size_res"]
           
        network = ResNetRNN(batch_size=batch_size, layer_size=layer_size, 
                        n_layers=n_layers, optimizer_choice=optimizer_choice, 
                        learning_rate=learning_rate, keep_prob=keep_prob, 
                        n_layers_res=n_layers_res, layer_size_res=layer_size_res)
                        
    return network   


def train(network, db, training_nr, n_epochs):
    """
    Training on windows
    
    Args:
        db -- str, ZODB database
        training_nr -- int, number of training examples
        n_epochs -- str, number of epochs
    """
    saving = True
    p = psutil.Process(os.getpid())
    m1 = p.memory_full_info().pss
    print("Memory use at start of training (in def): ", m1)
        
    # 2. train network
    n_examples = training_nr // network.batch_size * network.batch_size
    n_batches = n_examples // network.batch_size
    print("Training on {} examples in {} batches\n".format(n_examples, n_batches))
    with open(network.model_path + ".txt", "a") as dest:
        dest.write("Training on {} examples in {} batches\n".format(n_examples, n_batches))
    
    step = 0
    positives = 0
    seed = random.randint(0, 1000000000)
    for n in range(n_epochs):
        network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", write_meta_graph=True)
        print("\nSaved checkpoint at start op epoch {} at step {}\n".format(n, step))
        m2 = p.memory_full_info().pss
        db.set_ranges(seed)
        m3 = p.memory_full_info().pss
        print("Memory used to set ranges: ", m3 - m2)
        m8 = p.memory_full_info().pss
        for b in range(n_batches):
            # load batch sized training examples:
            t1 = datetime.datetime.now()
            m4 = m1 = p.memory_full_info().pss
            data, labels, pos = db.get_training_set(network.batch_size)  
            t2 = datetime.datetime.now()
            m5 = p.memory_full_info().pss
    
            positives += pos
            
            set_x = reshape_input(data, network.window, network.n_inputs)
            set_y = reshape_input(labels, network.window, network.n_outputs)
            
            # train on batch:
            step += 1
            t4 = datetime.datetime.now()
            m6 = p.memory_full_info().pss
            network.train_network(set_x, set_y, step)
            t5 = datetime.datetime.now()
            m7 = p.memory_full_info().pss
            
            if step % network.saving_step == 0:   
                network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", global_step=step, write_meta_graph=True)            
                print("Saved checkpoint at step ", step)
                train_acc = network.sess.run(network.accuracy, feed_dict={network.x:set_x, network.y:set_y, network.p_dropout: network.keep_prob})
                print("Training accuracy: ", train_acc)
                print("Time to get training set {}".format(t2 - t1))
                print("Time to train network {}".format(t5 - t4))
                print("Memory taken by getting training set on one batch: ", m5 -m4)
                print("Memory taken by training on one batch: ", m7 - m6)
                
        print("Finished epoch: ", n)   
        m9 = p.memory_full_info().pss   
        print("Memory taken after one epoch: ", m9 - m8)
    
   
    network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", global_step=step)
    print("\nSaved final checkpoint at step ", step, "\n")
    t6 = datetime.datetime.now()
    print("Training set had {:.2%} HPs".format(positives / (network.window * n_examples)))
    print("Time to save {}".format(t6 - t5))
    
    print("Finished training!\n")
  
    
def validate(network, squiggles, max_seq_length):
    """
    Validate model on full reads
    
    Args:
        network -- RNN object, network model
        squiggles -- npz files, raw signals and labels of MinION sequencing data
        max_seq_length -- int, maximum length of read to take in
    """  
   
    # 2. Validate model 
    valid_reads = 0
    max_seq_length = max_seq_length // network.window * network.window
    accuracy = 0
    #~ precision = 0
    #~ recall = 0
    #~ f1 = 0
    #~ roc_auc = 0

    # per squiggle:
    for squig in squiggles:
        t1 = datetime.datetime.now()
        data_sq, labels_sq = reader.load_npz(squig)
        t2 = datetime.datetime.now()

        if len(data_sq) >= max_seq_length:
            labels = labels_sq[: max_seq_length] 
            #~ if np.count_nonzero(labels == 1) > 0:          # 0 because is neg label
            data = data_sq[: max_seq_length]
            
            valid_reads += 1
    
            set_x = reshape_input(data, network.window, network.n_inputs)
            set_y = reshape_input(labels, network.window, network.n_outputs)
            
            t3 = datetime.datetime.now()
            #~ sgl_acc, sgl_auc, sgl_precision, sgl_recall, sgl_f1  = network.test_network(set_x, set_y, valid_reads)
            sgl_acc  = network.test_network(set_x, set_y, valid_reads)
            t4 = datetime.datetime.now()
            if valid_reads % 100 == 0:
                metrics.plot_squiggle(data, "Squiggle_{}_{}".format(os.path.basename(network.model_path), valid_reads))
                metrics.plot_squiggle(data[2000 : 3001], "Squiggle_{}_{}_middle".format(os.path.basename(network.model_path), valid_reads))
                print("Time to load one squiggle {}".format(t2 - t1))
                print("Time to validate network on one batch {}".format(t4 - t3))
                
            accuracy += sgl_acc
            #~ precision += sgl_precision
            #~ recall += sgl_recall
            #~ f1 += sgl_f1
            #~ roc_auc += sgl_auc
        
        else:
            continue
    
    whole_accuracy = metrics.calculate_accuracy(network.tp, network.fp, network.tn, network.fn)
    whole_precision, whole_recall = metrics.precision_recall(network.tp, network.fp, network.fn)
    whole_f1 = metrics.weighted_f1(whole_precision, whole_recall, (network.tp + network.fn), valid_reads * max_seq_length)
    # averaged performance:  
    with open(network.model_path + ".txt", "a") as dest: 
        dest.write("\nAverage performance of validation set:\n")
        dest.write("\tAccuracy: {:.2%}\n".format(accuracy / valid_reads))
        #~ dest.write("\tPrecision: {:.2%}\n\tRecall: {:.2%}\n".format(precision / valid_reads, recall / valid_reads))
        #~ dest.write("\tWeighted F1 measure: {0:.4f}\n".format(float(f1 / valid_reads)))
        #~ dest.write("\tAUC: {0:.4f}\n".format(roc_auc / valid_reads))

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
                #~ "layer_size": 16, "keep_prob": 0.6, "optimizer_choice": "Adam", 
                #~ "layer_size_res": 16, "n_layers_res" = 1}
    hpm_dict = {"batch_size":256, "optimizer_choice": "Adam", "learning_rate":0.001, "layer_size":64, "n_layers":1, "keep_prob":0.3, "layer_size_res":128, "n_layers_res":11}
    
    p = psutil.Process(os.getpid())
    m1 = p.memory_full_info().pss
    print("Memory use at start is", m1)
    
    # buid model
    print("Started script at ", datetime.datetime.now())
    t1 = datetime.datetime.now()
    network = build_model(network_type, **hpm_dict)
    t2 = datetime.datetime.now()
    m2 = p.memory_full_info().pss
    print("Extra memory use after building network is", m2 - m1)
    #~ network.restore_network("/mnt/nexenta/thijs030/networks/biGRU-RNN_132/checkpoints")
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
    
    t5 = datetime.datetime.now()
    train(network, db_train, training_nr, n_epochs)
    t6 = datetime.datetime.now()
    m5 = p.memory_full_info().pss
    print("Extra memory use after training is", m5 - m4)
    print("Trained network in {}".format(t6 - t5))
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    t7 = datetime.datetime.now()
    m6 = p.memory_full_info().pss
    print("Extra memory use after loading squiggles is ", m6 - m5)
    print("Loaded squiggles in {}".format(t7 - t6))
    validate(network, squiggles, max_seq_length)
    t8 = datetime.datetime.now()
    m7 = p.memory_full_info().pss
    print("Extra memory use after validation is ", m7 - m6)
    print("Validated network in {}".format(t8 - t7))
    print("Finished script at ", t8)
    
    print("Memory use at end is ", p.memory_full_info().pss)
