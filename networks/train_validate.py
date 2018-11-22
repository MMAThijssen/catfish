#!/usr/bin/env python3
import datetime
import helper_functions
import metrics
import numpy as np
import os
import psutil
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


def build_model(network_type, from_kwargs=True, **kwargs):
    model_id = 0
    batch_size = 16
    learning_rate = 0.01
    n_layers = 2
    layer_size = 16
    keep_prob = 0.6     
    optimizer_choice = "Adam"
    layer_size_res = 16
    n_layers_res = 1
    
        #~ model_id = 0
    #~ batch_size = 128
    #~ learning_rate = 0.01
    #~ n_layers = 3
    #~ layer_size = 64
    #~ keep_prob = 0.8     
    #~ optimizer_choice = "Adam"
    #~ layer_size_res = 32
    #~ n_layers_res = 2
    
    if from_kwargs:
        batch_size = kwargs["batch_size"]
        optimizer_choice = kwargs["optimizer_choice"]
        learning_rate = kwargs["learning_rate"]
        layer_size = kwargs["layer_size"]        
        n_layers = kwargs["n_layers"]
        keep_prob = kwargs["keep_prob"]
        if network_type == "ResNetRNN":
            n_layers_res = kwargs["n_layers_res"]
            layer_size_res = kwargs["layer_size_res"]

    if network_type == "RNN":
        network = RNN(model_id, batch_size=batch_size, layer_size=layer_size, 
                        n_layers=n_layers, optimizer_choice=optimizer_choice,  
                        learning_rate=learning_rate, keep_prob=keep_prob)
    
    elif network_type == "ResNetRNN":                   
        network = ResNetRNN(model_id, batch_size=batch_size, layer_size=layer_size, 
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
        
    # 2. train network
    n_examples = training_nr // network.batch_size * network.batch_size
    n_batches = n_examples // network.batch_size
    print("Training on {} examples in {} batches\n".format(n_examples, n_batches))
    with open(network.model_path + ".txt", "a") as dest:
        dest.write("Training on {} examples in {} batches\n".format(n_examples, n_batches))
    step = 0
    
    #~ time_reshape = 0
    #~ time_gettrainingset = 0
    #~ time_train = 0
    for n in range(n_epochs):
        network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", write_meta_graph=True)
        print("\nSaved checkpoint at start op epoch {} at step {}\n".format(n, step))
        db.set_ranges()
        for b in range(n_batches):
            # load batch sized training examples:
            t1 = datetime.datetime.now()
            data, labels = db.get_training_set(network.batch_size)      # TODO: take care that batches contain different examples
            t2 = datetime.datetime.now()
            print("Time to get training set {}".format(t2 - t1))

            set_x = reshape_input(data, network.window, network.n_inputs)
            set_y = reshape_input(labels, network.window, network.n_outputs)
            
            # train on batch:
            step += 1
            t4 = datetime.datetime.now()
            network.train_network(set_x, set_y, step)
            t5 = datetime.datetime.now()
            #~ time_train += (t5 - t4)
            print("Time to train network {}".format(t5 - t4))
            
            if step % network.saving_step == 0:   
                network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", global_step=step, write_meta_graph=True)            
                print("Saved checkpoint at step ", step)
                #~ train_acc = network.sess.run(network.accuracy, feed_dict={network.x:set_x, network.y:set_y, network.p_dropout: network.keep_prob})
                #~ print("Training accuracy: ", train_acc)
        print("Finished epoch: ", n)      
    
    #~ print("Time to get training set {}".format(time_gettrainingset))
    #~ print("Time to reshape {}".format(time_reshape))
    #~ print("Time to train network {}".format(time_train))    
    network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", global_step=step)
    print("\nSaved final checkpoint at step ", step, "\n")
    t6 = datetime.datetime.now()
    print("Time to save {}".format(t6 - t5))
    
    print("Finished training :)")
  
    
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
        print("Time to load one squiggle {}".format(t2 - t1))
        if len(data_sq) >= max_seq_length:
            labels = labels_sq[: max_seq_length] 
            if np.count_nonzero(labels == 1) > 0:          # 0 because is neg label
                data = data_sq[: max_seq_length]
                
                valid_reads += 1
        
                set_x = reshape_input(data, network.window, network.n_inputs)
                set_y = reshape_input(labels, network.window, network.n_outputs)
                
                t3 = datetime.datetime.now()
                #~ sgl_acc, sgl_auc, sgl_precision, sgl_recall, sgl_f1  = network.test_network(set_x, set_y, valid_reads)
                sgl_acc  = network.test_network(set_x, set_y, valid_reads)
                t4 = datetime.datetime.now()
                print("Time to validate network on one batch {}".format(t4 - t3))
                if valid_reads % 100 == 0:
                    metrics.plot_squiggle(data, "Squiggle_{}_{}".format(os.path.basename(network.model_path), valid_reads))
                    metrics.plot_squiggle(data[2000 : 3001], "Squiggle_{}_{}_middle".format(os.path.basename(network.model_path), valid_reads))
                    
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
    # moeten arrays zijn:
    #~ tpr = network.tp / (network.tp + network.fn)
    #~ fpr = network.fp / (network.fp + network.tn)
    #~ auc = metrics.compute_auc(tpr, fpr)
    #~ metrics.draw_roc(tpr, fpr, auc, "ROC_{}".format(network.model_id))
        dest.write("\nFinished validation of model {} on {} raw signals of length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   max_seq_length))    
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
    
    p = psutil.Process(os.getpid())
    print(p.memory_full_info().pss)
    
    # buid model
    print("Start building model at ", datetime.datetime.now())
    t1 = datetime.datetime.now()
    network = build_model(network_type, from_kwargs=False)
    t2 = datetime.datetime.now()
    network.initialize_network()
    t22 = datetime.datetime.now()
    print("Building model took {}".format(t2 - t1))
    print("Initialized model in {}".format(t22 - t2))
    print("Finished building model at ", datetime.datetime.now())
    
    # train and validate network
    print("Loading training database..")
    print(datetime.datetime.now())
    t3 = datetime.datetime.now()
    db_train = helper_functions.load_db(db_dir_train)
    t4 = datetime.datetime.now()
    print("Loaded db in {}".format(t4 - t3))
    print("Finished load_db at ", datetime.datetime.now())
    
    t5 = datetime.datetime.now()
    train(network, db_train, training_nr, n_epochs)
    t6 = datetime.datetime.now()
    print("Trained network in {}".format(t6 - t5))
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    t7 = datetime.datetime.now()
    print("Loaded squiggles in {}".format(t7 - t6))
    validate(network, squiggles, max_seq_length)
    t8 = datetime.datetime.now()
    print("Validated network in {}".format(t8 - t7))
    print(t8)
    
    print(p.memory_full_info().pss)
