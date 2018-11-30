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


def main(network_type, db_dir_train, training_nr, n_epochs, db_dir_val, max_seq_length):
    max_nr = 100000000
    #~ max_nr = 10000
    step_factor = 10
    samples = 2

    # 1. Build model from file
    hpm_dict = retrieve_hyperparams("/mnt/nexenta/thijs030/networks/learning_curve_{}.txt".format(network_type))
    model = build_model(network_type, **hpm_dict)
    model.initialize_network()
 
    size_list = [0]
    train_error = [0]
    val_error = [0]
    
    print("Loading training database..")
    db_train = helper_functions.load_db(db_dir_train)
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    
    seed = random.randint(0, 1000000000)
    
    #2. Train and validate
    while training_nr <= max_nr:
        for n in range(n_epochs):
            db_train.set_ranges(seed)
            size_list.append(training_nr)
            # 2. Train model
            t2 = datetime.datetime.now()
            train_accuracy = train(model, db_train, training_nr)
            train_error.append(train_accuracy)
            t3 = datetime.datetime.now()  
            print("Trained model in {}\n".format(t3 - t2))
            print("Finished epoch: {}".format(n))
            
            #3. Assess performance on validation set
            val_accuracy = validate(model, squiggles, max_seq_length)
            val_error.append(val_accuracy)
            t4 = datetime.datetime.now()  
            print("Validated model in {}".format(t4 - t3))
            
        training_nr *= step_factor
    
    return(train_error, val_error, size_list)


def draw_learning_curves(training_score, validation_score, train_sizes, img_title):
    """
    Plots learning curve. 
    
    Args:
        training_score -- list of float/ints
        validation_score -- list of float/ints
        train_sizes -- list of ints
        
    """
    #~ training_scores_mean = training_scores.mean(axis = 1)
    #~ validation_scores_mean = validation_scores.mean(axis = 1)
    
    plt.style.use("seaborn")
    
    plt.plot(train_sizes, training_score, label = 'Training error')
    plt.plot(train_sizes, validation_score, label = 'Validation error')

    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = "Learning curves"
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.savefig("{}.png".format(img_title), bbox_inches="tight")



if __name__ == "__main__":
    #0. Get input
    if not len(argv) == 7:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-model id\n\t-path to training db" +
                         "\n\t-number of training reads\n\t-number of epochs" + 
                         "\n\t-path to validation db\n\t-max length of validation reads")
    
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
    train_error, val_error, sizes = main(network_type, db_dir_train, training_nr, 
                                         n_epochs, db_dir_val, max_seq_length)
    print("Training: ", train_error)
    print("Validation: ", val_error)
    print("Sizes: ", sizes)

    #~ #2. Calculate mean, min and max training and validation scores
    #~ # suppose you have multiple networks, eg. 2:
    #~ for s in range(len(sizes)):
        #~ mean_training_score = train_error[s].mean()
        #~ mean_validation_score = val_error[s].mean()    
    
    #~ #5. Plot learning curve
    #~ img_title = "Learning_curve_{}".format(network_type)
    #~ draw_learning_curves(mean_training_score, mean_validation_score, size_list, img_title)
