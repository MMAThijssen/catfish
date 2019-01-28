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
from trainingDB.ExampleDb import ExampleDb

from bokeh.models import ColumnDataSource, LinearColorMapper, LabelSet, Range1d
from bokeh.plotting import figure
# from bokeh.io import show

from math import pi

wur_colors = ['#E5F1E4', '#3F9C35']
categorical_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
continuous_colors = ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84',
                     '#fc8d59', '#ef6548', '#d7301f', '#990000']


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


def set_logfolder(brnn_object, param_base_name, parent_dir, epoch_index):
    """
    Create a folder to store tensorflow metrics for tensorboard and set it up for a specific session.
    Returns a filewriter object, which can be used to write info to tensorboard.
    """
    timedate = time.strftime('%y%m%d_%H%M%S')
    cur_tb_path = parent_dir + '%s_%s_ep%s/' % (
        timedate,
        param_base_name,
        epoch_index)
    if os.path.isdir(cur_tb_path):
        shutil.rmtree(cur_tb_path)
    os.makedirs(cur_tb_path)
    return tf.summary.FileWriter(cur_tb_path, brnn_object.session.graph)


def plot_roc_curve(roc_list):
    tpr, tnr, epoch = zip(*roc_list)
    roc_plot = figure(title='ROC')
    roc_plot.grid.grid_line_alpha = 0.3
    roc_plot.xaxis.axis_label = 'FPR'
    roc_plot.yaxis.axis_label = 'TPR'

    col_mapper = LinearColorMapper(palette=categorical_colors, low=1, high=max(epoch))
    source = ColumnDataSource(dict(
        TPR=tpr,
        FPR=[1-cur_tnr for cur_tnr in tnr],
        epoch=epoch
    ))
    roc_plot.scatter(x='FPR', y='TPR',
                     color={'field': 'epoch',
                            'transform': col_mapper},
                     source=source)
    roc_plot.ray(x=0, y=0, length=1.42, angle=0.25*pi, color='grey')
    roc_plot.x_range = Range1d(0, 1)
    roc_plot.y_range = Range1d(0, 1)
    roc_plot.plot_width = 500
    roc_plot.plot_height = 500
    return roc_plot


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
