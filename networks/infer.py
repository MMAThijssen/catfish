#!/usr/bin/env python3

import h5py
import numpy as np
import os.path
from sys import argv

### FOR INFERING:       # start with input as a single file > work towards directory
# 1. Open fast5
# 2. Extract raw signals
# 3. Process like done in TrainingRead:
#       - cut part till first sample
#       - normalize signal
# 4. Save as npz file       -- necessary or can it be directly given to network?
# 5. Predict (01 for now)
#       - load network
#       - predict homopolymers
# 6. Return output

def main(fast5_file, hdf_path="Analyses/Basecall_1D_000"):
    """
    Infers classes from raw MinION signal in FAST5.
    
    Args:
        fast5_file -- str, path to file
        hdf_path -- str, path in FAST5 leading to signal
    """
    # open FAST5
    if not os.path.exists(fast5_file):
        print("Error: path to FAST5 is not correct.")
        return 1                                                                # 1 indicates error
    with h5py.File(fast5_file, "r") as fast5:
    
        # process signal
        raw = process_signal(fast5)
    
    # load network
    
    # take per 35 from raw
        # pad if needed
    

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

    #~ np.savez(npz_path + splitext(basename(files))[0],
             #~ base_labels=tr.classified, 
             #~ raw=tr.raw[: tr.final_signal])

    

if __name__ == "__main__":
    fast5 = argv[1]
    main(fast5)
