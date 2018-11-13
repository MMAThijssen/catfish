#!usr/bin/env python3
"""
Created on Mon Oct  8 11:49:32 2018

@author: thijs030
"""
import h5py
import os
from sys import argv

###### FROM CARLOS
def clipped_bases_start(hdf):
    # Catches a version change!
    try:
        clipped_bases_start = hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
            'clipped_bases_start']
    except KeyError: #chage this part
        clipped_bases_start = hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
            'trimmed_obs_start']
    return clipped_bases_start

#### FROM CARLOS
def clipped_bases_end(hdf):
    # Catches a version change!
    try:
        clipped_bases_end = hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
            'clipped_bases_end']
    except KeyError: #chage this part
        clipped_bases_end = hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
            'trimmed_obs_end']
    return clipped_bases_end
    
def failed_reads(failed_file, dest_file="failed_fast5.txt"):
    """
    Retrieves reads unsuccessfully processed by Tombo.
    
    Args:
        failed_file -- str, name of Tombo-produced file that contains fails.
    
    Returns: file (str)
    """
    with open(failed_file, "r") as source:
        with open(dest_file, "w") as dest:
            for line in source:
                if not line.strip():
                    continue
                else:
                    word_list = line.split()
                    for word in word_list:
                        if word.startswith("BaseCalled_template"):
                            fast5 = word.split("/")[-1]
                            if fast5[-1] == ",":
                                fast5 = fast5[:-1]
                            dest.write(fast5)
                            dest.write("\n")
    return(dest_file)
    
def move_failed(failed_reads, fast5_dir, dest_dir="failed_tomb_"):
    # create directory to move failed reads to
    if not os.path.exists(dest_dir):    
        os.mkdir(dest_dir)
        print("Created directory {}.".format(dest_dir))
    else:
        print("Directory {} already exists.".format(dest_dir))
    with open(failed_reads, "r") as source:
        for line in source:
            if not line.strip():
                continue
            else:
                fast5 = line.strip()
                os.rename("{}/{}".format(fast5_dir, fast5), "{}/{}".format(dest_dir, fast5))
    print("Finished moving")

def correct_28(failed_file, dest_file="remove28"):
    with open(failed_file, "r") as source:
        with open(dest_file, "w") as dest:
            for line in source:
                if not line.strip():
                    continue
                else:
                    word_list = line.split()
                    for word in word_list:
                        if word.startswith("./Ecoli_MinKNOW"):
                            fast5 = word.split("/")[-1]
                            if fast5[-1] == ",":
                                fast5 = fast5[:-1]
                            dest.write(fast5)
                            dest.write("\n")
    return(dest_file)       
    
def get_bases(fast5):
    """
    Returns sequence
    """
    with h5py.File(fast5, "r") as hdf:
        try:
            return("".join(hdf["Analyses/RawGenomeCorrected_000/BaseCalled_template/Events"]["base"].astype(str)))
        except:
            return(1)            
            

if __name__ == "__main__":
# Get failed reads from file for one folder:
#    failed = argv[1]
#    failed_read = argv[2]
#    failed_reads(failed, failed_read)
# Get failed reads from file for all folders:
#    failed_folder = argv[1]
#    files = os.listdir(failed_folder)
#    for failed in files:
#        number = failed.split("_")[-1]
#        failed_reads(failed, "failed_{}".format(number))
#        print("Finished {}".format(number))
# Move failed reads to new dir:
#    loman_folder = "/mnt/nexenta/thijs030/data/Ecoli_MinKNOW_1.4_RAD002_Sambrook/"
#    folder_list = os.listdir(loman_folder)
#    folder_list.remove('35')
#    for folder in folder_list:
#        failed = "/mnt/nexenta/thijs030/data/failed_reads/tombo_failed/failed_{}".format(folder)
#        fast_dir = "{}/{}".format(loman_folder, folder)
#        failed_out = "failed_tombo_{}".format(folder)
#        move_failed(failed, fast_dir, failed_out)
# Get fasta on remaining files per folder:
#   TODO: Change to all folders
#    folder_all = "/mnt/nexenta/thijs030/data/Ecoli_MinKNOW_1.4_RAD002_Sambrook"
#    folder_list = os.listdir(folder_all)
#    folder_list.remove('35') 
#    done = [34,8,2,15,27,20,18,12,5,33,16,1,24,23,29,6,11,30,32,21,13,19,4,3,9,14,26,31]
#    for fld in done:
#        folder_list.remove(str(fld))
#    with open("fasta_all_rest.fa", "w") as dest:    
#        print("Start on extracting fasta.")
#        for folder in folder_list: 
#            path_to_folder = "{}/{}".format(folder_all, folder)
#            fast5_files = os.listdir(path_to_folder)
#            for fast5 in fast5_files:            
#                seq = get_bases("{}/{}".format(path_to_folder, fast5))
#                if not seq == 1:
#                    #sprint("{}/{}".format(path_to_folder, fast5))
#                    dest.write(">{}".format(fast5.split(".")[0]))
#                    dest.write("\n")
#                    dest.write(seq)
#                    dest.write("\n")
#            print("Finished folder {}".format(folder))
#    print("Finished making fasta.")
    move_failed(argv[1], argv[2], argv[3])

# 1. Append fasta_35.fa to fasta of all folders