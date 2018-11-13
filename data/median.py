#!/usr/bin/env python3
"""
Calculate median from all data.

@author: thijs030
"""
import h5py
import os
import statistics

medians = []
#minimals = []
           
folder_all = "/mnt/nexenta/thijs030/data/Ecoli_MinKNOW_1.4_RAD002_Sambrook"
folder_list = os.listdir(folder_all)
n_folders = 0
for folder in folder_list:
    path_to_folder = "{}/{}".format(folder_all, folder)
    print("Started on folder {}".format(folder))
    for f in os.listdir(path_to_folder):
        f = path_to_folder + "/" + f
        n_folders += 1
        with h5py.File(f, "r") as hdf:
            hdf_path = "Analyses/RawGenomeCorrected_000/"
            hdf_events_path = '{hdf_path}BaseCalled_template/Events'.format(hdf_path=hdf_path)
            event_lengths = hdf[hdf_events_path]["length"]
#            minimals.append(min(event_lengths))
            medians.append(statistics.median(event_lengths))

#print("Minimal length of event: ", min(minimals))
#print("Median of medians: ", statistics.median(medians))
print("Average of medians: ", sum(medians) / n_folders)