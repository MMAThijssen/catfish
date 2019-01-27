#!/usr/bin/env python3

import h5py
import numpy as np
import os
from models.resnet_class import ResNetRNN
from models.rnn_class import RNN
from sys import argv

### FOR INFERING:       # start with input as a single file > work towards directory

def infer_class_from_signal(fast5_file, model, hdf_path="Analyses/Basecall_1D_000", window_size=35):
    """
    Infers classes from raw MinION signal in FAST5.
    
    Args:
        fast5_file -- str, path to FAST5 file
        model -- RNN object, network model
        hdf_path -- str, path in FAST5 leading to signal
        network_type -- str, type of network [default: ResNetRNN]
        window_size -- int, size of window used to train network
    
    Returns: list of classes
    """           
    # open FAST5
    if not os.path.exists(fast5_file):
        raise ValueError("path to FAST5 is not correct.")
    with h5py.File(fast5_file, "r") as fast5:
        # process signal
        raw = process_signal(fast5)
        #~ print("Length of raw signal: ", len(raw))
        #~ raw = raw[: 36]                                                         # VERANDEREN!                                              
        #~ print(len(raw))
        
    # pad if needed
    if not (len(raw) / window_size).is_integer():
        # pad
        padding_size = window_size - (len(raw) - (len(raw) // window_size * window_size))
        padding = np.array(padding_size * [0])
        raw = np.hstack((raw, padding))
    else:
        np.concatenate((raw, [0]), axis=0)
        padding_size = 1
    
    # take per 35 from raw and predict scores       - IDEA: really take per 35 and immediately put through right basecaller
    N_INPUT = 1
    n_batches = len(raw) // window_size
    raw_in = reshape_input(raw, window_size, N_INPUT)
    scores = model.infer(raw_in)
    
    # cut padding
    scores = scores[: -padding_size]
    labels = class_from_threshold(scores)
    predicted_hps = hp_in_pred(labels)
    
    return predicted_hps

# TODO: adjust this to correct model! -- also network_type in second line  -- make default hpm_dict -- change path to point to be dependent on user
# 1. Load network
def load_network(network_type, path_to_network):
    hpm_dict = retrieve_hyperparams(path_to_network + ".txt")                # CHANGE later!
    #~ hpm_dict = {"batch_size": 128, "optimizer_choice": "RMSProp", "learning_rate": 0.001, 
                #~ "layer_size": 256, "n_layers": 4, "keep_prob": 0.2, "layer_size_res": 32, "n_layers_res": 4}
    model = build_model(network_type, **hpm_dict)
    model.restore_network(path_to_network + "/checkpoints")

    return model
    
def retrieve_hyperparams(model_file, split_on=": "):
    """
    Retrieve hyperparameters from model file.
    
    Args:
        model_file -- str, path to file on model created by train_validate.build_model
        split_on -- str, combination of characters to split on [default: ": "]
    
    Returns: dict of hyperparameters
    """
    hpm_dict = {}
    with open(model_file, "r") as source:
        for line in source:
            if line.startswith("batch_size"):
                hpm_dict["batch_size"] = int(line.strip().split(split_on)[1])
            elif line.startswith("optimizer_choice"):
                hpm_dict["optimizer_choice"] = line.strip().split(split_on)[1]
            elif line.startswith("learning_rate"):
                hpm_dict["learning_rate"] = float(line.strip().split(split_on)[1])
            elif line.startswith("layer_size:"):
                hpm_dict["layer_size"] = int(line.strip().split(split_on)[1])
            elif line.startswith("n_layers:"):
                hpm_dict["n_layers"] = int(line.strip().split(split_on)[1])
            elif line.startswith("keep_prob"):
                hpm_dict["keep_prob"] = float(line.strip().split(split_on)[1])
            elif line.startswith("layer_size_res"):
                hpm_dict["layer_size_res"] = int(line.strip().split(split_on)[1])
            elif line.startswith("n_layers_res"):   
                hpm_dict["n_layers_res"] = int(line.strip().split(split_on)[1])
                
    return hpm_dict

# 2. Extract raw signal from file
def process_signal(fast5_file, normalization="median"):
    """
    Process raw signal by trimming start and normalizing the signal.
    
    Args: 
        fast5_file -- str, path to FAST5 file
        normalization -- str, method of normalization [default: median]
        
    Returns: raw signal                                                         # np.array of floats
    """
    first_sample = fast5_file["Analyses/Segmentation_000/Summary/segmentation"].attrs["first_sample_template"]
    read_name = fast5_file["Raw/Reads/"].visit(str)
    raw_signal = fast5_file["Raw/Reads/" + read_name + "/Signal"][()]
    raw_signal = raw_signal[first_sample : ]
    raw_signal = normalize_raw_signal(raw_signal, normalization)
    
    return raw_signal

# from Carlos             
def normalize_raw_signal(raw, norm_method):
    """
    Normalize the raw DAC values
    """
    # Median normalization, as done by nanoraw (see nanoraw_helper.py)
    if norm_method == 'median':
        shift = np.median(raw)
        scale = np.median(np.abs(raw - shift))
    else:
        raise ValueError('norm_method not recognized')
    return (raw - shift) / scale


# 3. Restore network
def build_model(network_type, saving=False, **kwargs):
    """
    Constructs neural network
    
    Args:
        network_type -- str, type of network: "RNN" or "ResNetRNN"
        
    Returns: network object
    """
    if network_type == "RNN":
        network = RNN(save=saving, **kwargs)
    
    elif network_type == "ResNetRNN":                   
        network = ResNetRNN(save=saving, **kwargs)
                        
    return network  

# 4. Infer 
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
    except ValueError:                                                          # TODO: change this part
        print(len(data))
        print(len(data[0]))
    return data 


def class_from_threshold(predicted_scores, threshold=0.5):
    """
    Assigns classes on input based on given threshold.
    
    Args:
        predicted_scores -- list of floats, scores outputted by neural network
        threshold -- float, threshold
    
    Returns: list of class labels (ints)
    """
    return [1 if y >= threshold else 0 for y in predicted_scores]
    
                         
def hp_in_pred(predictions, positive=1):
    """
    Get start positions and length of positive label in predicted input.
    
    Args:
        predictions -- list of int, predicted class labels
        threshold -- int, threshold to correct stretch [default: 15]
    
    Returns: list [[length, start position]]
    """
    compressed_predictions = [[predictions[0], 0, 0]]

    for p in range(len(predictions)):
        if predictions[p] == compressed_predictions[-1][0]:
            compressed_predictions[-1][1] += 1
        else:
            compressed_predictions.append([predictions[p], 1, p])
            
    positives = [compressed_predictions[l][1:] for l in range(len(compressed_predictions)) if compressed_predictions[l][0] == 1]
            
    return positives
