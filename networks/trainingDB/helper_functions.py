import numpy as np
import tensorflow as tf
import os
import time
import warnings
import re
import shutil
import h5py
from math import nan
from statistics import median
from ExampleDb import ExampleDb


def parse_input_path(location):
    """
    Take path, list of files or single file. Add '/' if path. Return list of files with path name concatenated. 
    """
    if not isinstance(location, list):
        location = [location]

    all_files = []
    for loc in location:
        if os.path.isdir(loc):
            if loc[-1] != '/':
                loc += '/'
            file_names = os.listdir(loc)
            files = [loc + f for f in file_names]
            all_files.extend(files)
        elif os.path.exists(loc):
            all_files.extend(loc)
        else:
            warnings.warn('Given location %s does not exist, skipping' % loc, RuntimeWarning)

    if not len(all_files):
        ValueError('Input file location(s) did not exist or did not contain any files.')
    return all_files


def parse_output_path(location):
    """
    Take given path name. Add '/' if path. Check if exists, if not, make dir and subdirs. 
    """
    if location[-1] != '/':
        location += '/'
    if not os.path.isdir(location):
        os.makedirs(location)
    return location


def load_db(db_dir):
    if db_dir[-1] != '/':
        db_dir += '/'
    db = ExampleDb(db_name=db_dir + 'db.fs')
    return db
    

def load_squiggles(db_dir):
    if db_dir[-1] != '/':
        db_dir += '/'
    squiggles = parse_input_path(db_dir + 'test_squiggles')
    return squiggles




def retrieve_read_properties(raw_read_dir, read_name):
    read_name_grep = re.search('(?<=/)[^/]+_strand', read_name).group()
    # Reconstruct full read name + path
    fast5_name = raw_read_dir + read_name_grep + '.fast5'
    try:
        hdf = h5py.File(fast5_name, 'r')
    except OSError:
        warnings.warn('Read %s not found in raw data, skipping read property retrieval.' % fast5_name, RuntimeWarning)
        return [nan for _ in range(5)]

    # Get metrics
    qscore = hdf['Analyses/Basecall_1D_000/Summary/basecall_1d_template'].attrs['mean_qscore']
    alignment = hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment']
    alignment_metrics = [alignment.attrs[n] for n in ('num_deletions',
                                                      'num_insertions',
                                                      'num_mismatches',
                                                      'num_matches')]
    hdf.close()
    return [qscore] + alignment_metrics


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
