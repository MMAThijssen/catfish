#!/usr/bin/env python3

import h5py
import os.path
from shutil import copyfile
from sys import argv

# TODO: adjust to leave basecalled information out
def split_signal(input_file, splits_hp, splits_nonhp, temp_dir, temp_dir_nonhp):
    """
    Splits raw signal of FAST5 file on specified entries. All other groups are copied.
    
    Args:
        input_file -- str, path to input file
        splits -- list of tuple, start and end value to split on, and label; end is exclusive
    
    Returns: list of new file names
    """    
    hp_list = []
    nonhp_list = []
    # get signal
    try:
        source = h5py.File(input_file, "r")
    except IOError:
        raise IOError("ERROR - could not open file, likely corrupted.")  
          
    try:
        read_name = list(source["Raw"]["Reads"])[0]
        read_attributes = list(source["Raw"]["Reads"][read_name].attrs)
        signal_dset = source["Raw"]["Reads"][read_name]["Signal"]
    except:
        raise RuntimeError("ERROR - no raw signal data was stored in file.")
    
    index = 0
    # split signal          # ADJUST TO IMMEDIATELY MAKE NEW FILE
    for s in splits_hp:
        new_signal = signal_dset[s[0] : s[1]]
        
        # make file
        dest_name = "{}/{}_{}.fast5".format(temp_dir, os.path.basename(input_file).split(".")[0], index)
        copyfile(input_file, dest_name)     # TODO: change dest_name to path within dest dir
        dest = h5py.File(dest_name, "r+")
        raw_reads = dest["Raw"]["Reads"][read_name]
        
        # replace signal with split signal
        del dest["Raw"]["Reads"][read_name]["Signal"]
        raw_reads.create_dataset("Signal", data=new_signal, dtype="int16", 
                                 compression="gzip",compression_opts=9)
        dest.close()
        index += 1
        
        hp_list.append(read_name)
        
    # split signal          # ADJUST TO IMMEDIATELY MAKE NEW FILE
    for s in splits_nonhp:
        new_signal = signal_dset[s[0] : s[1]]
        
        # make file
        dest_name = "{}/{}_{}.fast5".format(temp_dir_nonhp, os.path.basename(input_file).split(".")[0], index)
        copyfile(input_file, dest_name)     # TODO: change dest_name to path within dest dir
        dest = h5py.File(dest_name, "r+")
        raw_reads = dest["Raw"]["Reads"][read_name]
        
        # replace signal with split signal
        del dest["Raw"]["Reads"][read_name]["Signal"]
        raw_reads.create_dataset("Signal", data=new_signal, dtype="int16", 
                                 compression="gzip",compression_opts=9)
        dest.close()
        index += 1
        
        nonhp_list.append(read_name)    
    
    return hp_list, nonhp_list

    
if __name__ == "__main__":
    fast5 = argv[1]
    splits = [(13000, 13200, 1)]
    main_dir = "/mnt/scratch/thijs030/wrongread/findwrongout"
    print(split_signal(fast5, splits, main_dir))
    
    
    
    ### OUTPUTS ###
    # list(source.keys()) > ['Analyses', 'Raw', 'UniqueGlobalKey']
    # raw_group.shape, raw_group.dtype > 'Group' object has no attribute 'shape', 'dtype'
    # list(raw_group.keys()) > ['Reads']
    # source["Raw"]["Reads"] > ['Read_3141']
    # signal_dset.shape, signal_dset.dtype > ((226716,), dtype('int16'))




