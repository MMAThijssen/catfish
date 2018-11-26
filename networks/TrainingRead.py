# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:45:55 2018

@author: thijs030
"""
from helper_functions import normalize_raw_signal
from itertools import chain, repeat
from math import ceil
import numpy as np
from persistent import Persistent
from random import sample
import training_encodings

# A class for tombo-corrected MinION training reads, containing raw signal, and the derivation of classes on which
# a neural network can train.


class TrainingRead(Persistent):

    def __init__(self, hdf, normalization, hdf_path, clipped_bases, kmer_size, use_tombo=False):
        """Initialize a new training read.

        """
        self.lessen = 3
        self._raw = None
        self.condensed_events = None
        self._event_length_list = None
        self._hdf_path = None
        self.final_signal = None

        self.hdf = hdf
        self.normalization = normalization
        self.use_tombo = use_tombo
        self.clipped_bases = clipped_bases
        self.hdf_path = hdf_path
        self.kmer_size = kmer_size

        self.raw = None
        self.events = None
      
        self._classified = None
        self.classified = None


    def expand_sequence(self, sequence, length_list=None):
        """
        Expand a 1-event-per-item list to a one-raw-data-point-per-item list, using the event lengths derived from
        the basecaller. Uses event length list stored in object if none provided
        """
        if length_list is None:
            return list(chain.from_iterable(repeat(item, duration) for item, duration in zip(sequence,
                                                                                             self._event_length_list)))
        return list(chain.from_iterable(repeat(item, duration) for item, duration in zip(sequence, length_list)))

    @property
    def clipped_bases_start(self):
        # Catches a version change!
        if 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment' in self.hdf:
            if 'clipped_bases_start' in self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs:
                return self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                    'clipped_bases_start']
            elif 'trimmed_obs_start' in self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs:
                return self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                    'trimmed_obs_start']
        return self.clipped_bases

    @property
    def clipped_bases_end(self):
        # Catches a version change!
        if 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment' in self.hdf:
            if 'clipped_bases_end' in self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs:
                return self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                    'clipped_bases_end']
            elif 'trimmed_obs_end' in self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs:
                return self.hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs[
                    'trimmed_obs_end']
        return self.clipped_bases

    @property
    def raw(self):
        return self._raw

    @property
    def hdf_path(self):
        return self._hdf_path

    @property
    def events(self):
        events, _, _ = zip(*self.condensed_events)
        return self.expand_sequence(events)

    @property
    def start_idx(self):
        _, start_idx, _ = zip(*self.condensed_events)
        return self.expand_sequence(start_idx)

    @property
    def event_length_list(self):
        return self._event_length_list

    @property
    def classified(self):
        return(self._classified)

    @hdf_path.setter
    def hdf_path(self, pth):
        if pth[-1] != '/':
            pth += '/'
        if pth not in self.hdf:
            raise ValueError('hdf path not in hdf file!')
        if pth + 'BaseCalled_template' not in self.hdf:
            raise ValueError('hdf path in hdf file, but does not contain BaseCalled_template results!')
        self._hdf_path = pth

    @raw.setter
    def raw(self, _):
        if self.use_tombo:
            first_sample = self.hdf['{hdf_path}BaseCalled_template/Events'.format(hdf_path=self.hdf_path)].attrs[
                'read_start_rel_to_raw']
        else:
            first_sample = self.hdf['Analyses/Segmentation_000/Summary/segmentation'].attrs["first_sample_template"]
        raw_varname = self.hdf['Raw/Reads/'].visit(str)
        raw = self.hdf['Raw/Reads/'+ raw_varname + '/Signal'][()]
        raw = raw[first_sample:]
        self._raw = normalize_raw_signal(raw, self.normalization)

    @events.setter
    def events(self, _):
        """
        Retrieve k-mers and assign to corresponding raw data points.
        """
        hdf_events_path = '{hdf_path}BaseCalled_template/Events'.format(hdf_path=self.hdf_path)
        event_lengths = self.hdf[hdf_events_path]["length"]
        if self.use_tombo:
            # retrieve complete base sequence:
            event_states_sl = self.hdf[hdf_events_path]["base"]
            event_states_sl = event_states_sl.astype(str)
            # creates event list from base sequences: 
            kmer_overhang = self.kmer_size // 2     # append Ns to get middle of event from start
            event_states_sl = np.insert(event_states_sl, 0, ['N'] * kmer_overhang)
            event_states_sl = np.append(event_states_sl, ['N'] * kmer_overhang)
            # event_states = np.array([''.join(event_states_sl[i:i + self.kmer_size])
            #                          for i in range(0, event_states_sl.size - self.kmer_size + 1)])
            event_list = [''.join(event_states_sl[i:i + self.kmer_size])
                          for i in range(0, event_states_sl.size - self.kmer_size + 1)]
            # DONE: Requires adding 0 to start_idx_list as zip will otherwise cut off last value!
            start_idx_list = np.concatenate((self.hdf[hdf_events_path]["start"], np.array([0])))
            # this gets beginning and endings of each corrected event
            event_raw_list = [self.raw[b:e] for b, e in zip(start_idx_list[:-1], start_idx_list[1:])]
            self.final_signal = start_idx_list[-2] + event_lengths[-1]          # TODO: is this correct?
            event_raw_list[-1] = self.raw[start_idx_list[-2] : self.final_signal] # add signals to final event
            event_length_list = list(event_lengths)

        else:
            frequency = 4000
            event_states = self.hdf[hdf_events_path]["model_state"]
            event_states = event_states.astype(str)
            event_move = self.hdf[hdf_events_path]["move"]
            event_move = event_move.astype(int)

            event_move[0] = 0  # Set first move to 0 in case it was not 0; used to set first-base index
            start_idx = 0
            # outputs
            event_list = [event_states[0]]  # list of k-mers assigned to events
            start_idx_list = [0]  # 0-based index of first base in event k-mer in fasta file
            event_length_list = []  # List of lengths of events as number of raw data points
            event_raw_list = []  # List of lists containing raw data points per-event

            cur_event_length = event_lengths[0]
            temp_raw = list(self.raw)
            for n in range(event_move.size):
                # save an event if there has been a new event
                n_signals = ceil(cur_event_length * frequency)
                if event_move[n] != 0:
                    event_length_list.append(cur_event_length)
                    event_raw_list.append(temp_raw[ : n_signals])
                    del temp_raw[ : n_signals]
                    cur_event_length = event_lengths[n]
                    start_idx += event_move[n]
                    event_list.append(event_states[n])
                    start_idx_list.append(start_idx)
                else:
                    cur_event_length += event_lengths[n]
            event_length_list.append(cur_event_length)  # Last event length
            n_signals = ceil(cur_event_length * frequency)
            event_raw_list.append(temp_raw[ : n_signals]) # Last event raw data points
            del temp_raw[ : n_signals]

        ########## removed commented part: Set clipped bases to 'NNNNN'
        kmer_placeholder = ['N' * self.kmer_size]
        if self.clipped_bases_start != 0:
            event_list[:self.clipped_bases_start] = kmer_placeholder * self.clipped_bases_start
        if self.clipped_bases_end != 0:
            event_list[-self.clipped_bases_end:] = kmer_placeholder * self.clipped_bases_end
        self.condensed_events = list(zip(event_list,  
                                         start_idx_list,
                                         event_raw_list))
        self._event_length_list = event_length_list

    @classified.setter
    def classified(self, _):
        self._classified = self.classify_events()

        
    def classify_events(self, encoding_type='hp_class'):
        """
        Return event labels as specified by encoding_type.
        """
        kmers, _, _ = zip(*self.condensed_events)
        try:
            class_numbers = [training_encodings.hp_class_number(km) for km in kmers]
            extended_classes = training_encodings.extend_classification(class_numbers)
            labels = self.label_raws(extended_classes)
        except IndexError:
            print("IndexError: likely due to empty k-mer")
            return None    
        return labels
        
        
    def label_raws(self, classified_events):
        """
        Assign label to raw data points belonging to a classified event.
        """
        labels = [labels.append(classified_events[ev]) for ev in range(len(self.condensed_events)) for p in self.condensed_events[ev][2]]
        return labels
        # made list comprehension and removed parentheses labels - UNTESTED
            

    def get_pos(self, width, nb=None):
        width_l = width // 2  # if width is odd, pick point RIGHT of middle
        width_r = width - width_l
        condensed_hits = [idx for idx in range(width_l, self.final_signal - width_r + 1) if self.classified[idx] == 1]   
        raw_points_out = []
        raw_labels_out = []
        for ch in range(0, len(condensed_hits), self.lessen):
            start = condensed_hits[ch] - width_l
            end = condensed_hits[ch] + width_r + 1
            raw_labels_out.append(self.classified[start : end])
            raw_points_out.append(self.raw[start : end])
        return(raw_points_out, raw_labels_out)


    def get_neg(self, width, nb):
        width_l = width // 2  # if width is odd, pick point RIGHT of middle
        width_r = width - width_l
        idx_list = [idx for idx in range(width_l, self.final_signal - width_r + 1) if self.classified[idx] == 0]   
        raw_points_out = []
        raw_labels_out = []
        for cur_idx in sample(idx_list, nb):
            start = cur_idx - width_l
            end = cur_idx + width_r + 1
            raw_labels_out.append(self.classified[start : end])
            raw_points_out.append(self.raw[start : end])  
        return(raw_points_out, raw_labels_out)
