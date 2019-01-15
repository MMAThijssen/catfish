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
from sys import argv

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
            avg = sum(event_lengths) / len(event_lengths)
            return avg, statistics.median(event_lengths), min(event_lengths), max(event_lengths)
        elif stat == "min":
            return min(event_lengths), max(event_lengths)
        else:
            raise ValueError("Could not compute. Choose: 'median' or 'min'.")
        
def write_to_file(output_name, averages, minimals, maximals, medians):
    with open(output_name, "a+") as dest:
        dest.write("Median of averages: {}\n".format(statistics.median(averages)))
        dest.write("Average of averages: {}\n".format(sum(averages) / len(averages)))
        dest.write("Minimal length of event: {}\n".format(min(minimals)))
        dest.write("Maximal length of event: {}\n".format(max(maximals)))
        dest.write("Median of medians: {}\n".format(statistics.median(medians)))
        dest.write("Average of medians: {}\n".format(sum(medians) / len(medians)))
        dest.write("------------------------------------------------\n")

if __name__ == '__main__':
    #~ if len(sys.argv) < 3:
        #~ print('Usage:   extractattr.py runname fast5_path')
        #~ print('         Extract a table of attributes and values from a fast5 file.')
        #~ print('')
        #~ sys.exit(1)
    
    #~ runname, fast5_path = sys.argv[1:]
    #~ Process(fast5_path, runname)

    main_dir = argv[1]   
    file_out = argv[2]           
    folder_list = os.listdir(main_dir)
    averages = []
    medians = []
    minimals = []
    maximals = []
    for folder in folder_list:
        path_to_folder = "{}/{}".format(main_dir, folder)
        print("Started on folder {}".format(folder))
        for f in os.listdir(path_to_folder):
            f = path_to_folder + "/" + f
            avg, median, minimal, maximal = compute_on_length(f, "median")
            averages.append(avg)
            medians.append(median)
            #~ minimal = compute_on_length(f, "min")
            minimals.append(minimal)
            maximals.append(maximal)
        write_to_file(file_out, averages, minimals, maximals, medians)
        print("Median of averages: ", statistics.median(averages))
        print("Average of averages: ", sum(averages) / len(averages))
        print("Minimal length of event: ", min(minimals))
        print("Maximal length of event: ", max(maximals))
        print("Median of medians: ", statistics.median(medians))
        print("Average of medians: ", sum(medians) / len(medians))
    
    print("TAKEN OVER ALL FOLDERS:")        # will be the same as last print statement before
    print("Median of averages: ", statistics.median(averages))
    print("Average of averages: ", sum(averages) / len(averages))
    print("Minimal length of event: ", min(minimals))
    print("Maximal length of event: ", max(maximals))
    print("Median of medians: ", statistics.median(medians))
    print("Average of medians: ", sum(medians) / len(medians))
