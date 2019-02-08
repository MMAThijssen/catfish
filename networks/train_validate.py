#!/usr/bin/env python3
import datetime
import trainingDB.helper_functions
import trainingDB.metrics
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
    
def padding(data, window=35, n_input=1):
    # pad if needed
    if not (len(data) / window).is_integer():
        # pad
        padding_size = window - (len(data) - (len(data) // window * window))
        padding = np.array(padding_size * [0])
        data = np.hstack((data, padding))
    else:
        padding_size = 0
    #~ # take per 35 from raw and predict scores       - IDEA: really take per 35 and immediately put through right basecaller
    n_batches = len(data) // window
    data = reshape_input(data, window, n_input)
    
    return data, padding_size

    
def generate_random_hyperparameters(network_type,
                                    learning_rate_min=-4,      
                                    learning_rate_max=0,
                                    optimizer_list=["Adam", "RMSProp"],
                                    layer_size_list=[16, 32, 64, 128, 256],
                                    n_layers_min=1,
                                    n_layers_max=6,   
                                    batch_size_list=[128, 256, 512],
                                    dropout_min=0.2,
                                    dropout_max=0.8,
                                    n_layers_res_min=1,
                                    n_layers_res_max=12,
                                    size_layers_res_list=[16, 32, 64, 128, 256]):
    """
    Generates random hyperparameters.
    
    Args:
        network_type -- str, type of network (empty (RNN) or "ResNetRNN")
        learning_rate_min -- int, minimal negative number in exponential for learning rate [default: -4]
        learning_rate_max -- int, maximal positive number in exponential for learning rate [default: 0]
    
    Returns: dict {str: value for hyperparameter}
    """   
    # pick random hyperparameter:
    learning_rate = 10 ** np.random.randint(learning_rate_min, learning_rate_max)
    optimizer = np.random.choice(optimizer_list)
    layer_size = np.random.choice(layer_size_list)
    n_layers = np.random.randint(n_layers_min, n_layers_max)
    batch_size = np.random.choice(batch_size_list)
    dropout = round(np.random.uniform(dropout_min, dropout_max), 1)
    
    # create dict:
    hpm_dict = {"batch_size": batch_size, "optimizer_choice": optimizer, 
                "learning_rate": learning_rate, "layer_size": layer_size, 
                "n_layers": n_layers, "keep_prob": dropout}
    
    # extend dict if network is ResNetRNN:            
    if network_type == "ResNetRNN":
        n_layers_res = np.random.randint(n_layers_res_min, n_layers_res_max)
        size_layers_res = np.random.choice(size_layers_res_list)
        
        hpm_dict["layer_size_res"] = size_layers_res
        hpm_dict["n_layers_res"] = n_layers

    return hpm_dict


def train_and_validate(network, db, training_nr, squiggles, max_seq_length, file_path, validation_start, max_number):
    """
    Train and validate on examples from trainingDB.
    
    Args:
        db -- str, ZODB database
        training_nr -- int, number of training examples
        squiggles -- list, raw signal with labels to validate on
        max_seq_length -- int, maximal length of read to take in
        file_path -- str, path to save validation information file to 
        validation_start -- str / int, if int: specifies start position for validation, else: random start position 
        max_number -- int, maximum number of squiggles to use in validation 
    
    Returns: training accuracy (float)
    """
    print("Start training at {}".format(datetime.datetime.now()))
        
    # Train network
    n_examples = training_nr // network.batch_size * network.batch_size
    n_batches = n_examples // network.batch_size
    print("\nTraining on {} examples in {} batches\n".format(n_examples, n_batches))
    with open(file_path + ".txt", "a+") as dest:
        dest.write("\nTraining on {} examples in {} batches\n".format(n_examples, n_batches))
    
        step = 0
        positives = 0

        for b in range(n_batches):
            # load batch sized training examples:
            data, labels, pos = db.get_training_set(network.batch_size, ratio=2)
            positives += pos
            
            set_x = reshape_input(data, network.window, network.n_inputs)
            set_y = reshape_input(labels, network.window, network.n_outputs)
            
            # train on batch:
            step += 1                                                           # step is per batch
            network.train_network(set_x, set_y, step)
            
            # validate network:
            if step == n_batches or step % network.saving_step == 0:                           # network.saving_step                                            
                network.saver.save(network.sess, network.model_path + "/checkpoints/ckpnt", global_step=step, write_meta_graph=True)            
                print("Saved checkpoint at step {}\n".format(step))
                dest.write("\nSaved checkpoint at step {}\n".format(step))
                if step == n_batches - 1:
                    print("This was the final checkpoint\n")
                
                # compute training performance:
                t1 = datetime.datetime.now()
                train_acc, train_loss = network.sess.run([network.accuracy, network.loss], feed_dict={network.x:set_x, network.y:set_y, network.p_dropout: network.keep_prob})
                t2 = datetime.datetime.now()
                print("Validated in {}".format(t2 - t1))
                dest.write("\nTraining accuracy: {}\n".format(train_acc))
                dest.write("Training loss: {}\n".format(train_loss))
                
                # compute validation performance:
                t1 = datetime.datetime.now()
                val_acc, whole_precision, whole_recall = validate(network, squiggles, max_seq_length, file_path, validation_start, max_number)
                t2 = datetime.datetime.now()
                print("Validated in {}".format(t2 - t1))
                dest.write("Validation precision: {}\n".format(whole_precision))
                dest.write("Validation recall: {}\n".format(whole_recall))
    
        try:
            train_hp = positives / (network.window * n_examples)  
        except ZeroDivisionError:
            train_hp = 0
        dest.write("\nTraining set had {:.2%} HPs\n".format(train_hp))

        dest.write("\nFinished training!\n\n")

    return train_acc        
    
    
def validate(network, squiggles, max_seq_length, file_path, validation_start="random", max_number=856):
    """
    Validate model on specified number of reads. 
    Information is saved to 'file_path_validation.txt".
    
    Args:
        network -- RNN object, network model
        squiggles -- list of .npz files, raw signals and labels of MinION sequencing data
        max_seq_length -- int, maximum length of read to take in
        file_path -- str, path to save validation information file to 
        validation_start -- str / int, if int: specifies start position for validation, else: random start position [default: "random"]
        max_number -- int, maximum number of squiggles to use in validation [default: 856]
        
    Returns: validation accuracy
    """  
    total_length = 0
    accuracy = 0
    loss = 0
    valid_reads = 0
    
    print("Max length is {}".format(max_seq_length))
    print("Validation start is {}".format(validation_start))
    # per squiggle:
    #~ random.shuffle(squiggles)                                                   # to assure reads are selected randomly
    file_path = file_path.split("/")[-1]
    #~ with open(file_path + "_training.txt", "a+") as dest:                    # for saving start pos
    for squig in squiggles:
        data_sq, labels_sq = reader.load_npz(squig)

        if validation_start == "complete":
            total_length += len(data_sq)
        else:
            max_seq_length = max_seq_length // network.window * network.window
            if type(validation_start) == int: 
                if len(data_sq) >= validation_start + max_seq_length: 
                    start_val = validation_start
                else: 
                    continue
            elif validation_start == "random":
                if len(data_sq) >= max_seq_length:
                    # pick random point to start validation from:
                    start_val = random.randint(0, len(data_sq) - max_seq_length)
                else:
                    continue
            labels_sq = labels_sq[start_val: start_val + max_seq_length] 
            data_sq = data_sq[start_val: start_val + max_seq_length]
            #~ dest.write("\nStart validation at point: {}".format(start_val))
            total_length += max_seq_length
        
        read_name = os.path.basename(squig).split(".npz")[0]
        valid_reads += 1
        
        set_x, padding_size = padding(data_sq, network.window, network.n_inputs)
        set_y, _ = padding(labels_sq, network.window, network.n_inputs)

        #~ set_x = reshape_input(data, network.window, network.n_inputs)
        #~ set_y = reshape_input(labels, network.window, network.n_outputs)

        sgl_acc, sgl_loss  = network.test_network(set_x, set_y, read_name, file_path, padding_size, threshold=0.8)
        
        if valid_reads >= max_number:
            break
        
        #~ if valid_reads % network.saving_step == 0:
            #~ trainingDB.metrics.plot_squiggle(data, "Squiggle_{}_{}".format(os.path.basename(network.model_path), valid_reads))
            #~ trainingDB.metrics.plot_squiggle(data[2000 : 3001], "Squiggle_{}_{}_middle".format(os.path.basename(network.model_path), valid_reads))
            
        accuracy += sgl_acc
        loss += sgl_loss


    whole_accuracy = trainingDB.metrics.calculate_accuracy(network.tp, network.fp, network.tn, network.fn)
    whole_precision, whole_recall = trainingDB.metrics.precision_recall(network.tp, network.fp, network.fn)
    whole_f1 = trainingDB.metrics.f1(whole_precision, whole_recall)
    
     
    with open(file_path + ".txt", "a+") as dest: 
        # averaged performance: 
        dest.write("\n---NEXT ROUND OF VALIDATION---")
        dest.write("\nAverage performance of validation set:\n")
        dest.write("\tAccuracy: {:.2%}\n".format(accuracy / valid_reads))
        dest.write("\tLoss: {0:.4f}".format(loss / valid_reads))

        # over whole set:
        dest.write("\nPerformance over whole set: \n")
        dest.write("\tDetected {} true positives, {} false positives, {} true negatives, {} false negatives in total.\n".
                format(network.tp, network.fp, network.tn, network.fn))
        dest.write("\tTrue number of HPs: {} \tTrue percentage: {:.2%}\t Predicted percentage HPs: {:.2%}\n".
                format(network.tp + network.fn, (network.tp + network.fn) / (total_length), (network.tp + network.fp) / (total_length)))
        dest.write("\tAccuracy: {:.2%}".format(whole_accuracy))
        dest.write("\n\tPrecision: {:.2%}\n\tRecall: {:.2%}".format(whole_precision, whole_recall))
        dest.write("\t\nF1 score: {0:.4f}".format(whole_f1))
        dest.write("\nFinished validation of model {} on {} raw signals of average length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   total_length / valid_reads))    
    print("\nFinished validation of model {} on {} raw signals of average length {}.".format(network.model_type, 
                                                                                   valid_reads,
                                                                                   total_length / valid_reads))
    # clean for new round of validation:
    network.tp = 0
    network.fn = 0
    network.tn = 0
    network.fp = 0
    # print final information
    print("Validation accuracy: ", whole_accuracy)
    print("Validation loss: ", loss / valid_reads)
    
    return whole_accuracy, whole_precision, whole_recall



if __name__ == "__main__":
    # get input
    if len(argv) < 6:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-path to training db" +
                         "\n\t-number of training reads\n\t-path to validation db" + 
                         "\n\t-max length of validation reads\n\nOptional:" +
                         "\n\t-path to file to save information on validation to" + 
                         "\n\t-start position for validation\n\t-maximum number of reads for validation")
    network_type = argv[1]
    db_dir_train = argv[2]
    training_nr = int(argv[3])
    db_dir_val = argv[4]
    max_seq_length = int(argv[5])
    validation_start = "random"
    max_number = 856
    only_validation = False
    
    if len(argv) >= 7:
        validation_start = int(argv[6])
    if len(argv) >= 8:
        max_number = int(argv[7])
    if len(argv) >= 9:
        validation_path = argv[8]
    if len(argv) == 10:
        only_validation = True
    
    p = psutil.Process(os.getpid())
    m1 = p.memory_full_info().pss
    print("Memory use at start is", m1)
    
    if not only_validation:
        # build model
        hpm_dict = generate_random_hyperparameters(network_type)
        #~ hpm_dict = {"batch_size":128, "optimizer_choice": "Adam", "learning_rate":0.001, "layer_size":64, "n_layers":1, "keep_prob":0.3, "layer_size_res":128, "n_layers_res":11}
        network = build_model(network_type, save=True, **hpm_dict)
        network.initialize_network()
        
        validation_path = network.model_path
        
        # train and validate network
        print("Loading training database..")
        db_train = trainingDB.helper_functions.load_db(db_dir_train)
        print("Loading validation database..")
        squiggles = trainingDB.helper_functions.load_squiggles(db_dir_val)
        t5 = datetime.datetime.now()
        train_and_validate(network, db_train, training_nr, squiggles, max_seq_length, 
                            validation_path, validation_start, max_number)
        t6 = datetime.datetime.now()
        print("Trained and validated network in {}".format(t6 - t5))

    # ONLY validate network
    if only_validation:
        
        print("Loading validation database..")
        squiggles = trainingDB.helper_functions.load_squiggles(db_dir_val)
        t7 = datetime.datetime.now()
        validate(network, squiggles, max_seq_length, file_path, validation_start, max_number)
        t8 = datetime.datetime.now()
        print("Validated network in {}".format(t8 - t7))
    
    t8 = datetime.datetime.now()
    print("Finished script at ", t8)
    print("Memory use at end is ", p.memory_full_info().pss)
