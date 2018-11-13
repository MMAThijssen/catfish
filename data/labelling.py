#!/usr/bin/env python3
"""
Created on Mon Oct  8 15:30:00 2018

@author: thijs030
"""
from enhance_ref import check_hp
import h5py
from sys import argv

# 1. Assign signal to base
def get_signal(fast5):
    """
    Retrieves raw signal from FAST5 file.
    
    Args:
        fast5 -- string, name of FAST5 file
        
    Returns: numpy ndarray
    """
    with h5py.File(fast5, "r") as hdf:
        raw_signal = list(hdf['/Raw/Reads'].values())[0]['Signal'].value
    return(raw_signal)

# DOES work on Loman data; does NOT work on simulated signal  
# ! simulated signal has FASTQ seperately already
def get_fastq(fast5):
    """
    Retrieves FASTQ.
    """
    with h5py.File(fast5, "r") as hdf:
        fastq = hdf['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'].value
    return(fastq)
  
def get_basecalls(fast5):
    """
    Returns (start, length, base)
    """
    with h5py.File(fast5, "r") as hdf:
        length_list = []
        base_list = []
        event_list = list(hdf["Analyses/RawGenomeCorrected_000/BaseCalled_template/Events"])
        # so far return huge list with all events (..., ..., start, length, base)
        for event in event_list:
            length_list.append(event[3])
            base_list.append(event[4])
        with open("fasta_tombo_nanopore.txt", "w") as dest:
            for base in range(len(base_list)):
                dest.write(chr(base_list[base][0]))
        print(len(base_list))
        basecalls = 0
        
    return(basecalls)

# 2. Assign base to label
def label_bases(seq, threshold=5):
    """
    Labels bases to be part of homopolymer (1) or not (0).
    
    seq -- str, DNA sequence
    
    Returns: list of integers
    """
##########  klopt niet
#    """
#    Saves locations of homopolymers in a dictionary.
#    
#    Args:
#        seq -- str, DNA sequence
#        threshold -- int, minimal number of equal bases to define 
#                        homopolymer stretch [default:5]
#    
#    Returns: dict {hp (str): positions (list of tuples)}
#    """
#    labels = []
#    is_hp = False
#    for i in range(len(seq) - threshold + 1):
#        stretch = seq[i:i + threshold]
#        if not is_hp and check_hp(stretch):
#            is_hp = True            
#            hp_start = i
#            hp_end = i + threshold
#        elif is_hp and check_hp(stretch):
#            hp_end = i + threshold
#        elif is_hp and ((not check_hp(stretch)) or i == len(seq) - threshold):
#            is_hp = False
#            hp = seq[hp_start: hp_end]
#            for base in hp:
#                labels.append("H")
#        else:
#            labels.append("N")
#            
#    return(labels)
    

    
                

# 3. Combine signal with label

# 4. Convert to right format for TensorFlow



if __name__ == "__main__":
# 1. Assign signal to base
# input FAST5 file:
    fast5 = argv[1]         # it does not matter if abs path or rel path
    # raw_signal = get_signal(fast5)
    # fastq = get_fastq(fast5)
    # print(fastq)
    basecalls = get_basecalls(fast5)
    print(basecalls)
#     for base in basecalls:
#         print(base)
           

# get raw signal
# get base belonging to signal - as is signal 0 to 100 belong to base1

# 2. Assign base to label
# check if sequence is homopolymer or not - used check_hp from enhance_ref
# assign label HP or non-HP
#    seq = "AAAAAAGGGGGA"
#    labels = label_bases(seq)    
#    print(len(seq))
#    print(len(labels))
#    print(labels)

# 3. Combine signal with label
# replace bases with labels
# multiply label the length of the signals belong to the base

# 4. Convert to right format for TensorFlow



#https://github.com/jts/nanopolish/blob/master/src/common/nanopolish_fast5_io.cpp#L129
#import h5py
#import re
#from sys import argv
#
#def get_signal(fast5):
#    # for Loman reads
#    if fast5.startswith("nanopore"):
#        read_nr = int(re.search("(?<=read)\d+", fast5).group())
#    # for DeepSimulator reads
#    elif fast5.startswith("signal"):
#    signal = fast5['Raw/Reads/Read_{}/Signal'.format(read_nr)]
#    return(signal)
#
#if __name__ == "__main__":
#    fast5_path = argv[1] # expects full path
#    with h5py.File(fast5_path, 'r') as hdf:
#        # do something
#        pass