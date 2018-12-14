#!/usr/bin/env python3
import datetime
import helper_functions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import psutil
import random
from sys import argv
from slurm_randomsearch import retrieve_hyperparams
from train_validate import train, validate, build_model


def t_and_v(network_type, db_dir_train, training_nr, n_epochs, db_dir_val, max_seq_length, 
            max_nr=100000000):
    """
    Train and validation after each epoch for an increasing number of training reads.
    
    Args:
        network_type -- str, type of neural network: either "RNN" or "ResNetRNN"
        db_dir_train -- str, path to training database
        training_nr -- int, number of training reads
        n_epochs -- int, number of epochs
        db_dir_val -- str, path to validation database
        max_seq_length -- int, maximum length of validation reads
        max_nr -- int, maximum number of training reads
        step_factor -- int, number of steps after which to print accuracies
        
    Returns: training accuracy, validation accuracy, list of training_nr
    """
    # 1. Build model from file
    hpm_dict = retrieve_hyperparams("/mnt/nexenta/thijs030/networks/learning_curve_{}.txt".format(network_type))
    model = build_model(network_type, **hpm_dict)
    model.initialize_network()
    #~ model.restore_network("/mnt/nexenta/thijs030/networks/biGRU-RNN_70/checkpoints")
    
    print("Loading training database..")
    db_train = helper_functions.load_db(db_dir_train)
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    
    seed = random.randint(0, 1000000000)
    
    #2. Train and validate
    #~ while training_nr <= max_nr:
    t0 = datetime.datetime.now()
    pos_range, neg_range = db_train.set_ranges(seed)
    print("Set ranges in {}".format(datetime.datetime.now() - t0))
    for n in range(n_epochs):
        # train model:
        db_train.range_ps = pos_range
        db_train.range_ns = neg_range
        t2 = datetime.datetime.now()
        train_accuracy = train(model, db_train, training_nr, squiggles, max_seq_length)
        t3 = datetime.datetime.now()  
        print("Trained model in {}\n".format(t3 - t2))
        print("Finished epoch: {}".format(n))
        
        # validate model at end:
        val_accuracy = validate(model, squiggles, max_seq_length)
        t4 = datetime.datetime.now()  
        print("Validated model in {}".format(t4 - t3))
        
    #~ training_nr *= step_factor
    
    return train_accuracy, val_accuracy


def compute_lines(samples):
    """
    Takes matrix of lists of errors per sample to compute mean, min and max 
    values per size per sample.
    
    Args:
        samples -- list of lists of (floats), scores of different samples
    
    Returns: list of means, list of min, list of max 
    """    
    min_list = []
    max_list = []   
    mean_list = []

    for s in range(len(samples[0])):   
        total = 0
        temp_list = []                                                          # save list per size
        for i in range(len(samples)):
            total += samples[i][s]
            temp_list.append(samples[i][s])
        mean = total / len(samples)
        mean_list.append(mean)
        
        min_list.append(min(temp_list))
        max_list.append(max(temp_list))
        
        
    return mean_list, min_list, max_list


def parse_txt(infile, measure="accuracy"):
    """
    """
    training = []
    validation = []
    sizes = []
    
    new_round = True
    get_size = False
    with open(infile, "r") as source:
        for line in source:
            if new_round:
                if get_size:
                    size = int(line.strip())
                    get_size = False
                if line.startswith("Training {}".format(measure)):
                    train_acc = float(line.strip().split(": ")[1])
                elif line.startswith("Validation {}".format(measure)):
                    val_acc = float(line.strip().split(": ")[1])
                    new_round = False
                if line.startswith("Training loss"):
                    get_size = True
            else:
                training.append(train_acc)
                validation.append(val_acc)
                sizes.append(size)
                new_round = True
    return training, validation, sizes           
            

def draw_learning_curves(training_scores, validation_scores, train_sizes, img_title, 
                         network_type, measure="accuracy"):
    """
    Plots learning curve. 
    
    Args:
        training_score -- list of lists of float/ints
        validation_score -- list of lists of float/ints
        train_sizes -- list of ints
        img_title -- str, name and title of figure
        network_type -- str, type of network: RNN or ResNetRNN
        measure -- str, 'accuracy' or 'loss' [default: accuracy]
        
    """    
    plt.style.use("seaborn")
    
    if network_type == "RNN":
        c = "darkcyan"
        c2 = "paleturquoise"
    if network_type == "ResNetRNN":
        c = "forestgreen"
        c2 = "lightgreen"
    
    train_means, train_mins, train_maxs = compute_lines(training_scores)
    val_means, val_mins, val_maxs = compute_lines(validation_scores)
    
    plt.plot(train_sizes, train_means, label = 'Training', color=c)
    plt.plot(train_sizes, val_means, label = 'Validation', color=c2)
    plt.fill_between(train_sizes, train_mins, train_maxs, alpha=0.3)
    plt.fill_between(train_sizes, val_mins, val_maxs, alpha=0.3)

    plt.ylabel(measure, fontsize = 14)
    plt.xlabel('number of training examples', fontsize = 14)
    title = "Learning curve"
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.savefig("{}-{}.png".format(img_title, measure[:2]), bbox_inches="tight")
    plt.close()
    #~ plt.show()


if __name__ == "__main__":
    #~ #0. Get input
    #~ if not len(argv) == 7:
        #~ raise ValueError("The following arguments should be provided in this order:\n" + 
                         #~ "\t-network type\n\t-model id\n\t-path to training db" +
                         #~ "\n\t-number of training reads\n\t-number of epochs" + 
                         #~ "\n\t-path to validation db\n\t-max length of validation reads")
    
    network_type = argv[1]
    db_dir_train = argv[2]
    training_nr = int(argv[3])      # at start
    n_epochs = int(argv[4])         
    db_dir_val = argv[5]
    max_seq_length = int(argv[6])  
    
    # Keep track of memory and time
    p = psutil.Process(os.getpid())
    t1 = datetime.datetime.now() 
    print("Started script at {}\n".format(t1))
    
    #1. Train and validate network
    train_error, val_error = t_and_v(network_type, db_dir_train, training_nr, 
                                         n_epochs, db_dir_val, max_seq_length)

    
    #~ # 2. Get training and validation curves
    #~ input_file = argv[2]
    #~ measure = argv[3]
    #~ training, validation, sizes = parse_txt(input_file, measure)
    
    #~ #5. Plot learning curve
    #~ img_title = "Learning_curve_{}".format(network_type)
    #~ draw_learning_curves([training], [validation], sizes, img_title, network_type, measure)
