#!/usr/bin/env python3

"""
Assigns the bases that belong to a measurement, while keeping a distinction
between separate bases.
"""

import h5py
import os
from sys import argv


def get_base_signal(fast5, use_tombo=True):
    """
    Takes events to extend base sequence per measurement.
    
    Args:
        fast5 - str, path to FAST5
        use_tombo -- bool, use of corrected or uncorrected reads [default: True]
        
    Returns: list of str
    """
    # maybe should be in TrainingRead and rebuild val and testdb
    with h5py.File(fast5, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        # get list of event lengths:
        event_lengths = hdf[hdf_events_path]["length"]
        if use_tombo:
            # get list of base sequence:
            event_bases = hdf[hdf_events_path]["base"].astype(str)

            # make a base list by copying the base the times of the event length
            bases = []
            for i in range(len(event_lengths)):
                bases.extend(event_lengths[i] * [event_bases[i],])
            
        else:
            raise Exception("Not yet implemented for uncorrected reads")
            bases = None
    return bases
 
    
def get_new_signal(fast5, use_tombo=True, new_base="n", same_base="-"):
    """
    Takes events to create list when new base is detected.
    
    Args:
        fast5 -- str, path to FAST5 file
        use_tombo -- bool, use of corrected or uncorrected reads [default: True]
        new_base -- str, character that points to newly detected base
        same_base -- str, character that points to same base
        
    Returs: list of str
    """
    with h5py.File(fast5, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        # get list of event lengths:
        event_lengths = hdf[hdf_events_path]["length"]
        if use_tombo:
            # make a new list by copying the base the times of the event length
            new = []
            for i in range(len(event_lengths)):
                new.extend([new_base,])
                new.extend((event_lengths[i] - 1) * [same_base,])
            
        else:
            raise Exception("Not yet implemented for uncorrected reads")
            new = None
    return new


def get_base_new_signal(fast5, use_tombo=True, new_base="n", same_base="-"):
    """
    Takes events to extend base sequence per measurement.
    
    Args:
        fast5 - str, path to FAST5
        use_tombo -- bool, use of corrected or uncorrected reads [default: True]
        new_base -- str, character that points to newly detected base
        same_base -- str, character that points to same base
                
    Returns: list of str
    """
    # maybe should be in TrainingRead and rebuild val and testdb
    with h5py.File(fast5, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        # get list of event lengths:
        event_lengths = hdf[hdf_events_path]["length"]
        if use_tombo:
            # get list of base sequence:
            event_bases = hdf[hdf_events_path]["base"].astype(str)
            #~ print("".join(event_bases).find("AAAAAAG")) #1430
            #~ print(len(event_lengths[:1430])) #13055
            #~ print(event_lengths[1430:1437])
            #~ print(event_bases[1430:1437])
            # make a base list by copying the base the times of the event length
            bases = []
            new = []
            for i in range(len(event_lengths)):
                bases.extend(event_lengths[i] * [event_bases[i],])
                half_event = event_lengths[i] // 2
                otherhalf_event = event_lengths[i] - half_event - 1             # -1 for "n"
                new.extend(half_event * [same_base,])
                new.extend([new_base,])
                new.extend(otherhalf_event * [same_base,])
        else:
            raise Exception("Not yet implemented for uncorrected reads")
            bases = None
    #~ start = 10250
    #~ end = 10379
    #~ fake_new = new[start : end]
    #~ fake_bases = bases[start: end]
    #~ seq = [fake_bases[i] for i in range(len(fake_new)) if fake_new[i] == "n"]
    #~ print("".join(seq))
    #~ print(len(bases))
    return bases, new
 

def save_base_signal(fast5, out_file):
    """
    Args:
        fast5 -- str, path to fast5
        out_file -- str, name of output file
    
    Returns: str, .txt file
    """
    #~ labels = reader.load_npz_labels(npz)
    bases, new = get_base_new_signal(fast5)
    with open(out_file, "a+") as dest:
        dest.write(fast5.split(".fast5")[0])
        #~ dest.write("\n")
        #~ dest.write("* {}".format(list(labels)))
        dest.write("\n")
        dest.write("$ {}".format("".join(bases)))
        dest.write("\n")
        dest.write("! {}".format("".join(new)))
        dest.write("\n")
    
if __name__ == "__main__":
    get_base_new_signal(argv[1])
    #~ sgl_folder = argv[1]
    #~ outname = argv[2]
    #~ count = 0
    #~ sgl_files = os.listdir(os.path.abspath(sgl_folder))
    #~ os.chdir(sgl_folder)
    #~ for sgl in sgl_files:
        #~ save_base_signal(sgl, argv[2])
        #~ if count % 100 == 0:
            #~ print("Saved {} sgls to file".format(count))
        #~ count += 1

