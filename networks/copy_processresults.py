#!/usr/bin/env python3

from collections import Counter
import h5py
import reader
from sys import argv
import numpy as np

# compare all per position

# COMPOSITION
def composition_hp(predicted_labels, bases):
    """
    Get the composition of the predicted homopolymers.
    
    Args: 
        predicted_labels -- list of ints, predicted labels
        
    Returns: dict {base: count}
    """
    comp_dict = {"A": 0, "C": 0, "T": 0, "G": 0}
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1:
            comp_dict[bases[i]] += 1
    
    return comp_dict

#TODO: finish
def compare_real_bases(predicted_labels, bases, new):
    """
    Compares predicted homopolymer with true number of HPs.
    """
    hp_dict = {}        # hp_dict {start_pos: [length in measurements, base, length_in_bases, start]]}
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1:
            measure_len = 1
            base = bases[i]                                                     # first base is from "-" or "n"; if on "n", base is double
            base_len = 0
            start = False                                                       # check if prediction start on real base
            hp_dict[i] = [measure_len, base, base_len]
            if new[i] == "n":
                hp_dict[i][1] += bases[i] 
                hp_dict[i][2] += 1
                start = True
 
#TODO: test
def bases_in_preds(predicted_labels, new):
    """
    Reports how many separate bases in reality were present in predicted homopolymers.
    
    Returns: dict {hp_start: n_bases}
    """
    # get dict with start pos and length in measurements of predicted hps:
    hp_dict = predicted_hp_dict(predicted_labels)
    basecount_dict = {new[hp : hp_dict[hp]].count("n") for hp in hp_dict}
    return basecount_dict


#~ def bases_in_preds(fast5, predicted_labels):
    #~ """
    #~ Reports how many separate bases in reality were present in predicted homopolymers.
    
    #~ Returns: dict {hp_start: n_bases}
    #~ """
    #~ # get dict with start pos and length in measurements of predicted hps:
    #~ hp_dict = predicted_hp_dict(predicted_labels)
    #~ # get list with start pos of events
    #~ with h5py.File(fast5, "r") as hdf:
        #~ hdf_path = "Analyses/RawGenomeCorrected_000/"
        #~ hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        #~ # get list of event lengths:
        #~ event_lengths = hdf[hdf_events_path]["length"]
        #~ event_lengths = event_lengths[:5005]    # just for checking if it works
        #~ event_starts = [sum(event_lengths[: i + 1]) for i in range(len(event_lengths))]
    #~ # NOT FINISHED
    #~ basecount_dict = {new[hp : hp_dict[hp]].count("n") for hp in hp_dict}
    #~ return basecount_dict
        

# LENGTHS
#TODO: finish
def length_info(predicted_labels):
    """
    Get information on lengths of predicted homopolymers.
    
    Returns: avg length, median length, min length, max length
    """
    start == False
    lengths = []
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1:
            start == True
            hp_len = 0
            while start:
                hp_len += 1
                if predicted_labels:
                    pass
# compare with true avg, median, min and max length

def predicted_hp_dict(predicted_labels):            # works - TESTED
    # get dict with start pos and length in measurements of predicted hps:
    hp_dict = {}
    new_hp = False
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1 and not new_hp:
            new_hp = True
            hp_dict[i] = 1
            start_pos = i
        elif predicted_labels[i] == 1 and new_hp:
            hp_dict[start_pos] += 1
        elif predicted_labels[i] == 0 and new_hp:
            new_hp  = False
    return hp_dict
    
#TODO: finish
def compare_length(predicted_labels, true_labels):
    # get dict with start pos and length in measurements of predicted hps:
    hp_dict = predicted_hp_dict(predicted_labels)
    for hp in hp_dict:
        pass
    
    # check if hp in dict overlaps with real hp
    
        
    
    
# check length of HPs
def check_length():
    pass
    
# compare lengths of predictions with reality

# check if overestimated HPs are surrounding real ones


# COMPOSITION
# 
# check if A, C, G, T HPs are very well or very poorly recognized, or equally well
# check with reality

# What is the composition of misclassified reads?


# EVENTS
# you can check per measurement but also per event
# maybe even collapse predictions to most predicted one and assign that
# as correction to reassign label
# THIS FUNCTION HAS NO USE -- IT DOES: 
# if this tool works great than you can correct the signal with this

def reassign_bases(fast5, predicted_labels, use_tombo=True):        # works so far - TESTED
    
    with h5py.File(fast5, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        # get list of event lengths:
        event_lengths = hdf[hdf_events_path]["length"]
        if use_tombo:
            # get list of base sequence:
            event_bases = hdf[hdf_events_path]["base"].astype(str)

            # check per event if belonging to hp or not according to prediction
            label_list = []
            for i in range(len(event_lengths)):
                label_list.append(most_common(predicted_labels[i : i + event_lengths[i]]))     
                
            # correct ???
        
        else:
            raise Exception("Not yet implemented for uncorrected reads")
            label_list = None
    return label_list
    


def most_common(lst):
    """
    Returns most common element in a list.
    
    Args:
        lst -- list of type
        
    Returns: type
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]
    

if __name__ == "__main__":
    predfile = argv[1]
    with open(predfile, "r") as source:
        for line in source:
            labels = line.strip()[1:-1].split(", ")
            labels = list(map(int, labels))
            #~ print(len(labels))       # 5005
    bases_in_preds(argv[2], labels)
