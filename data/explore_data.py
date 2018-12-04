#!/usr/bin/env python2
"""
Explore fast5 data using h5py.

Created on Wed Sep 12 09:27:12 2018
@author: thijs030
"""

import h5py
import numpy as np
import os
import statistics
import sys

def Process(fast5_path, runname):
    # Collate the attribute list
    hdf = h5py.File(fast5_path, 'r')
    # Get the names of all groups and subgroups in the file
    list_of_names = []
    hdf.visit(list_of_names.append)
    attribute = []
    for name in list_of_names:
        # Get all the attribute name and value pairs
        itemL = hdf[name].attrs.items()
        for item in itemL:
            attr, val = item
            if type(hdf[name].attrs[attr]) == np.ndarray:
                val = ''.join(hdf[name].attrs[attr])
            val = str(val).replace('\n', '')
            attribute.append([runname, name+'/'+attr, val])
    hdf.close()
    # Print the header
    print('{0}'.format('\t'.join(['runname', 'attribute', 'value'])))
    # Print the attribute list
    print('{0}'.format('\n'.join(['\t'.join([str(x) for x in item]) for item in attribute])))


def compute_on_length(fast5_file, stat):
    """
    Computes median, average or minimal length of a FAST5 read.
    
    Args:
        fast5_file -- str, path to FAST5 file
        stat -- str, statistic to compute: "median", "min"
        
    Returns: median (int) or min (int)
    """
    with h5py.File(f, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{hdf_path}BaseCalled_template/Events'.format(hdf_path=hdf_path)
        event_lengths = hdf[hdf_events_path]["length"]
        if stat == "median":
            return statistics.median(event_lengths)
        elif stat == "min":
            return min(event_lengths)
        else:
            raise ValueError("Could not compute. Choose: 'median' or 'min'.")
            

if __name__ == '__main__':
    #~ if len(sys.argv) < 3:
        #~ print('Usage:   extractattr.py runname fast5_path')
        #~ print('         Extract a table of attributes and values from a fast5 file.')
        #~ print('')
        #~ sys.exit(1)
    
    #~ runname, fast5_path = sys.argv[1:]
    #~ Process(fast5_path, runname)

    main_dir = argv[1]              
    folder_list = os.listdir(main_dir)
    n_folders = 0
    medians = []
    minimals = []
    for folder in folder_list:
        path_to_folder = "{}/{}".format(folder_all, folder)
        print("Started on folder {}".format(folder))
        for f in os.listdir(path_to_folder):
            f = path_to_folder + "/" + f
            n_folders += 1
            median = compute_on_length(f, "median")
            medians.append(median)
            minimal = compute_on_length(f, "min")
            minimals.append(minimal))
            
    print("Minimal length of event: ", min(minimals))
    print("Median of medians: ", statistics.median(medians))
    print("Average of medians: ", sum(medians) / n_folders)
