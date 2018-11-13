import numpy as np
from helper_functions import normalize_raw_signal
from itertools import chain, repeat
from random import choice
import training_encodings
from persistent import Persistent

# A class for nanoraw-corrected MinION training reads, containing raw signal, and the derivation of classes on which
# a neural network can train.


class TrainingRead(Persistent):

    def __init__(self, hdf, normalization, hdf_path, clipped_bases, kmer_size, use_nanoraw=False):
        """Initialize a new training read.

        """
        self._raw = None
        self.condensed_events = None
        self._event_length_list = None
        self._hdf_path = None

        self.hdf = hdf
        self.normalization = normalization
        self.use_nanoraw = use_nanoraw
        self.clipped_bases = clipped_bases
        self.hdf_path = hdf_path
        self.kmer_size = kmer_size

        self.raw = None
        self.events = None


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
        if self.use_nanoraw:
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
        #event_lengths = self.hdf[hdf_events_path]["length"]
        if self.use_nanoraw:
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
            event_length_list = list(self.hdf[hdf_events_path]["length"])
        
        ########### moved "else" to unused
        ########## removed commented part: Set clipped bases to 'NNNNN'

        kmer_placeholder = ['N' * self.kmer_size]
        if self.clipped_bases_start != 0:
            event_list[:self.clipped_bases_start] = kmer_placeholder * self.clipped_bases_start
        if self.clipped_bases_end != 0:
            event_list[-self.clipped_bases_end:] = kmer_placeholder * self.clipped_bases_end
        self.condensed_events = list(zip(event_list,  # k-mers
                                         start_idx_list,  # index of first base in fasta (1 point per event)
                                         event_raw_list))  # raw data points in event (1 list per event)
#        print("Event list: {}".format(event_list))
#        print("Start idx list: {}".format(start_idx_list))
#        print("Event raw list: {}".format(event_raw_list))
        self._event_length_list = event_length_list     # duration of events

    def classify_events(self, encoding_type='hp_class'):
        """
        Return event labels as specified by encoding_type.
        """
        class_number_vec = np.vectorize(training_encodings.class_number)
        kmers, _, _ = zip(*self.condensed_events)
        try:
            class_numbers = class_number_vec(kmers, encoding_type)
        except IndexError:
            print('Index error, likely due to empty k-mer')
            return None
        return class_numbers
        
    def label_seq(classified_seq):
        """
        Args:
            classified_seq -- list of ints
        """
        # eg: 0001000 for [CCTAA, CTAAA, TAAAA, AAAAA, AAAAG, AAAGG, AAGGT]
         # go forwards through seq    
        half_kmer = 2 # for now
        for label in range(len(classified_seq)):
            if classified_seq[label] == 1:
                count = 0
                while count < half_kmer:
                    classified_seq[label - half_kmer + count] = 1
                    count += 1
            else:
                continue
        # go backwards through seq
        for label in reversed(range(len(classified_seq))):
            if classified_seq[label] == 1:
                count = 0
                while count < half_kmer:
                    classified_seq[label + half_kmer - count] = 1
                    count += 1
            else:
                continue
        return(classified_seq)
                

    def get_pos(self, kmer, width, nb=None):
        # collects events that are positive k-mers
        condensed_hits = [(ce, idx) for idx, ce in enumerate(self.condensed_events) if ce[0] == kmer]       
        width_l = width // 2  # if width is even, pick point RIGHT of middle
        width_r = width - width_l
        raw_hits_out = []
#        raw_kmers_out = []
        for ch in condensed_hits:                  
            # get the raw data point of the events in the window of the target
            raw_signals = []
            count = 0
            while count < width_l: 
#                raw_signal_array = self.condensed_events[ch[1] - width_l + count][2]
#                for raw in raw_signal_array:
#                    raw_signals.append(raw)
                raw_signals.append(self.condensed_events[ch[1] - width_l + count][2])
#                raw_kmers_out.append(self.condensed_events[ch[1] - width_l + count][0])
                count += 1
            count = 0
            while count <= width_r:
#                raw_signal_array = self.condensed_events[ch[1] + count][2]
#                for raw in raw_signal_array:
#                    raw_signals.append(raw)
                raw_signals.append(self.condensed_events[ch[1] + count][2])
#                raw_kmers_out.append(self.condensed_events[ch[1] + count][0])
                count += 1 
            print("\n\n\n\n--------Positive-------------")
            raw_hits_out.append(raw_signals)
            print(raw_signals)
        return raw_hits_out
#        return raw_hits_out, raw_kmers_out


    def get_neg(self, kmer, width, nb):
        idx_list = list(range(len(self.condensed_events)))
        width_l = width // 2  # if width is even, pick point RIGHT of middle
        width_r = width - width_l
        raw_hits_out = []
        raw_kmers_out = []
        while len(raw_hits_out) < nb:
            cur_idx = choice(idx_list)
            cur_condensed_event = self.condensed_events[cur_idx]
            if cur_condensed_event[0] != kmer:
                raw_signals = []
                count = 0
                while count < width_l:
#                    raw_signal_array = self.condensed_events[cur_idx - width_l + count][2]
#                    for raw in raw_signal_array:
#                        raw_signals.append(raw)  
                    raw_signals.append(self.condensed_events[cur_idx - width_l + count][2])
#                    raw_kmers.append(self.condensed_events[cur_idx - width_l + count][0])
                    count += 1
                count = 0
                while count <= width_r:
#                    raw_signal_array = self.condensed_events[cur_idx + count][2]
#                    for raw in raw_signal_array:
#                        raw_signals.append(raw)
                    raw_signals.append(self.condensed_events[cur_idx + count][2])
#                    raw_kmers.append(self.condensed_events[cur_idx + count][0])
                    count += 1  
                raw_hits_out.append(raw_signals)
                raw_kmers_out.append(cur_condensed_event[0])
            idx_list.remove(cur_idx)
        return raw_hits_out, raw_kmers_out
