#!/usr/bin/env python3

import h5py
import numpy as np
import os
from models.resnet_class import ResNetRNN
from models.rnn_class import RNN
import reader        # only for testing on "perfect" set
from sys import argv

# Inference
def infer_class_from_signal(fast5_file, model, label=1, window_size=35):
    """
    Infers classes from raw MinION signal in FAST5.
    
    Args:
        fast5_file -- str, path to FAST5 file
        model -- RNN object, network model
        network_type -- str, type of network [default: ResNetRNN]
        window_size -- int, size of window used to train network
    
    Returns: dict of homopolymer positions, length of read
    """      
    # open FAST5
    if not os.path.exists(fast5_file):
        raise ValueError("path to FAST5 is not correct.")
    with h5py.File(fast5_file, "r") as fast5:
        # process signal
        raw = process_signal(fast5)
        
    # pad if needed
    if not (len(raw) / window_size).is_integer():
        # pad
        padding_size = window_size - (len(raw) - (len(raw) // window_size * window_size))
    else:
        padding_size = 35
    padding = np.array(padding_size * [0])
    raw = np.hstack((raw, padding))        
    
    # take per 35 from raw and predict scores                                   # IDEA: really take per 35 and immediately put through right basecaller
    N_INPUT = 1
    n_batches = len(raw) // window_size
    raw_in = reshape_input(raw, window_size, N_INPUT)
    scores = model.infer(raw_in)
    
    # cut padding
    scores = scores[: -padding_size]
    labels = correct_short(class_from_threshold(scores))
    predicted_hps = hp_in_pred(labels)
    
    return predicted_hps, len(labels)

# use npzs
def infer_class_from_npz(npz_file, model, label=1, window_size=35):
    """
    Retrieve classes from labels saved in NPZ files.
    
    Args:
        fast5_file -- str, path to FAST5 file
        model -- RNN object, network model
        network_type -- str, type of network [default: ResNetRNN]
        window_size -- int, size of window used to train network
    
    Returns: dict of homopolymer positions, length of read
    """      
    # open FAST5
    if not os.path.exists(npz_file):
        raise ValueError("path to NPZ is not correct.")
    labels = reader.load_npz_labels(npz_file)

    predicted_hps = hp_in_pred(labels, 0, 0)
    
    return predicted_hps, len(labels)
    
    
# Raw signal
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

          
def normalize_raw_signal(raw, norm_method):                                     # from Carlos
    """
    Normalize the raw DAC values. 
    """
    if norm_method == 'median':
        shift = np.median(raw)
        scale = np.median(np.abs(raw - shift))
    else:
        raise ValueError('norm_method not recognized')
    return (raw - shift) / scale


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


# Classified output
def class_from_threshold(predicted_scores, threshold=0.5):
    """
    Assigns classes on input based on given threshold.
    
    Args:
        predicted_scores -- list of floats, scores outputted by neural network
        threshold -- float, threshold
    
    Returns: list of class labels (ints)
    """
    return [1 if y >= threshold else 0 for y in predicted_scores]
    
                         
def hp_in_pred(predictions, extension_left=11, extension_right=16, label=1):    # end_pos is exclusive                                          # , neg_label=0
    """
    Get start positions and length of positive label in predicted input.
    
    Args:
        predictions -- list of int, predicted class labels
        label -- int, labeled positions to extract [default: 1]
    
    Returns: list [(start position, end position)]
    """
    compressed_predictions = [[predictions[0], 0, 0]]

    for p in range(len(predictions)):
        if predictions[p] == compressed_predictions[-1][0]:
            compressed_predictions[-1][1] += 1
        else:
            compressed_predictions.append([predictions[p], 1, p])
               
    positives = [[compressed_predictions[l][2] - extension_left, compressed_predictions[l][2] + compressed_predictions[l][1] + extension_right]        # removed label: , compressed_predictions[l][0]
                    for l in range(len(compressed_predictions)) if compressed_predictions[l][0] == label]

    return positives
    
    
def get_positions(prediction_list):
    """
    Args:
        prediction_list -- list [[label, length, start position]]
    """
    positives = [(compressed_predictions[l][2], compressed_predictions[l][2] + compressed_predictions[l][1])  for l in range(len(compressed_predictions)) if compressed_predictions[l][0] == 1]
    return positives

                    
def correct_short(predictions, threshold=15):                                   # adapted from Carlos
    """
    Corrects class prediction to negative label if positive stretch is shorter than threshold.
    
    Args:
        predictions -- list of int, predicted class labels
        threshold -- int, threshold to correct stretch [default: 15]
    
    Returns: corrected predictions
    """
    compressed_predictions = [[predictions[0],0]]

    for p in predictions:
        if p == compressed_predictions[-1][0]:
            compressed_predictions[-1][1] += 1
        else:
            compressed_predictions.append([p, 1])

    for pred_ci, pred_c, in enumerate(compressed_predictions):
        if pred_c[0] != 0:
            if pred_c[1] < threshold:
                # remove predictions shorter than threshold
                compressed_predictions[pred_ci][0] = 0
            
    return np.concatenate([np.repeat(pred_c[0], pred_c[1]) for pred_c in compressed_predictions])    
