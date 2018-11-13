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
    print("Loading npz files..")
    squiggles = parse_input_path(db_dir + 'test_squiggles')
    return db, squiggles


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


def plot_timeseries(raw, base_labels, y_hat, start=0):

    nb_points = y_hat.size

    # Main data source
    source = ColumnDataSource(dict(
        raw=raw[:nb_points],
        event=list(range(len(y_hat))),
        cat=y_hat,
        cat_height=np.repeat(np.mean(raw), len(y_hat))
    ))

    # Base labels stuff
    base_labels_condensed = [base_labels[0]]
    bl_xcoords = []
    event_ends = []
    new_range = []
    for idx, bl in enumerate(base_labels):
        if bl == base_labels_condensed[-1]:
            new_range.append(idx)
        else:
            base_labels_condensed.append(bl)
            bl_xcoords.append(median(new_range))
            event_ends.append(idx)
            new_range = [idx]
    bl_xcoords.append(median(new_range))
    event_ends.append(new_range[-1])

    bl_source = ColumnDataSource(dict(
        base_labels=base_labels_condensed,
        event_ends= event_ends,
        x=bl_xcoords,
        y= np.repeat(raw.max(), len(bl_xcoords))
    ))

    base_labels_labelset = LabelSet(x='x', y='y',text='base_labels',
                                    source=bl_source,
                                    text_baseline='middle',
                                    angle=0.25*pi)

    # Plotting
    ts_plot = figure(title='Classified time series')
    ts_plot.grid.grid_line_alpha = 0.3
    ts_plot.xaxis.axis_label = 'nb events'
    ts_plot.yaxis.axis_label = 'current signal'
    y_range = raw.max() - raw.min()

    colors=wur_colors
    col_mapper = LinearColorMapper(palette=colors, low=0, high=y_hat.max())
    ts_plot.rect(x='event', y='cat_height', width=1.05, height=y_range, source=source,
                 fill_color={
                     'field': 'cat',
                     'transform': col_mapper
                 },
                 line_color=None)

    # Event ending lines
    ts_plot.rect(x='event_ends',
                 y=raw.mean(),
                 height=y_range, width=0.01, line_color='black', source=bl_source)


    ts_plot.add_layout(base_labels_labelset)
    ts_plot.line(x='event', y='raw', source=source)
    ts_plot.plot_width = 1500
    ts_plot.plot_height = 500
    ts_plot.x_range = Range1d(start, start+100)
    return ts_plot

# def plot_timeseries(raw, base_labels, y_hat, brnn_object, categorical=False, start=0):
#     ts_plot = figure(title='Classified time series')
#     ts_plot.grid.grid_line_alpha = 0.3
#     ts_plot.xaxis.axis_label = 'nb events'
#     ts_plot.yaxis.axis_label = 'current signal'
#     y_range = raw.max() - raw.min()
#     if categorical:
#         colors = categorical_colors
#     else:
#         colors = continuous_colors
#     col_mapper = LinearColorMapper(palette=colors, low=1, high=brnn_object.num_classes)
#     # col_mapper = CategoricalColorMapper(factors=list(range(brnn_object.num_classes)), palette=colors)
#     source = ColumnDataSource(dict(
#         raw=raw,
#         event=list(range(len(y_hat))),
#         cat=y_hat,
#         cat_height=np.repeat(np.mean(raw), len(y_hat)),
#         base_labels=base_labels
#     ))
#     ts_plot.rect(x='event', y='cat_height', width=1, height=y_range, source=source,
#                  fill_color={
#                      'field': 'cat',
#                      'transform': col_mapper
#                  },
#                  line_color=None)
#     if categorical:
#         base_labels_labelset = LabelSet(x='event', y='cat_height',
#                                         y_offset=-y_range,
#                                         text='base_labels', text_baseline='middle',
#                                         source=source)
#     else:
#         base_labels_labelset = LabelSet(x='event', y='cat_height',
#                                         y_offset=-y_range, angle=-0.5 * pi,
#                                         text='base_labels', text_baseline='middle',
#                                         source=source)
#     ts_plot.add_layout(base_labels_labelset)
#     ts_plot.scatter(x='event', y='raw', source=source)
#     ts_plot.plot_width = 1000
#     ts_plot.plot_height = 500
#     ts_plot.x_range = Range1d(start, start+100)
#     return ts_plot


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


def clean_classifications(y_hat, threshold=15):
    """
    Remove any detected event lasting shorter than threshold
    """
    y_hat_compressed = [[y_hat[0],0]]
    for yh in y_hat:
        if yh == y_hat_compressed[-1][0]:
            y_hat_compressed[-1][1] += 1
        else:
            y_hat_compressed.append([yh, 1])

    for yhci, yhc, in enumerate(y_hat_compressed):
        if yhc[0] != 0 and yhc[1] < threshold:
            y_hat_compressed[yhci][0] = 0
    return np.concatenate([np.repeat(yhc[0], yhc[1]) for yhc in y_hat_compressed])


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
