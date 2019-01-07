#!/usr/bin/env python3

import h5py
import numpy as np
import os.path
from models.resnet_class import ResNetRNN
from models.rnn_class import RNN
from sys import argv

### FOR INFERING:       # start with input as a single file > work towards directory

def main(fast5_file, hdf_path="Analyses/Basecall_1D_000", network_type="ResNetRNN", window_size=35):
    """
    Infers classes from raw MinION signal in FAST5.
    
    Args:
        fast5_file -- str, path to file
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
        print("Length of raw signal: ", len(raw))
        raw = raw[: 1000]   # VERANDEREN!
        print(len(raw))
        
    # load network
    # TODO: adjust this to correct model! -- also network_type in second line  -- make default hpm_dict -- change path to point to be dependent on user
    #~ #hpm_dict = retrieve_hyperparams("/mnt/scratch/thijs030/validatenetworks/biGRU-RNN_3.txt")
    hpm_dict = {"batch_size": 128, "optimizer_choice": "RMSProp", "learning_rate": 0.001, 
                "layer_size": 256, "n_layers": 4, "keep_prob": 0.2, "layer_size_res": 32, "n_layers_res": 4}
    model = build_model(network_type, **hpm_dict)
    model.restore_network("/mnt/scratch/thijs030/validatenetworks/ResNet-RNN_3/checkpoints")
        
    # pad if needed
    if not (len(raw) / window_size).is_integer():
        # pad
        padding_size = window_size - (len(raw) - (len(raw) // window_size * window_size))
        print(padding_size)
        padding = np.array(padding_size * [0])
        raw = np.hstack((raw, padding))
    
    # take per 35 from raw and predict scores
    n_input = 1
    n_batches = len(raw) // window_size
    raw_in = reshape_input(raw, window_size, n_input)
    scores = model.infer(raw_in)
    
    # cut padding
    scores = scores[: -padding_size]
    
    return scores
    

#2. Extract raw signal from file
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
