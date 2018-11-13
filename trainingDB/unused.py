# -*- coding: utf-8 -*-
"""
Unused scripts from Carlos' scripts for generation of training reads.

@author: thijs030
"""
## BACKUP FROM TRAININGREAD.PY  -- USED TO TAKE SIGNALS PER EVENT 
    def get_pos(self, width, nb=None):
        # collects events that are positive k-mers
        width_l = width // 2  # if width is even, pick point RIGHT of middle
        width_r = width - width_l
        condensed_hits = [(ce, idx) for idx, ce in enumerate(self.condensed_events[width_l : -width_r]) if self.classified[idx] == 1]       
        raw_hits_out = []
        raw_labels_out = []
#        raw_kmers_out = []
        for ch in condensed_hits:                  
            # get the raw data point of the events in the window of the target
            raw_signals = []
            raw_labels_out.append(self.classified[ch[1] - width_l : ch[1] + width_r + 1])
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
            raw_hits_out.append(raw_signals)
#            print("\n\n\n\n--------Positive-------------")
#            print(raw_hits_out) 
        return(raw_hits_out, raw_labels_out)


    def get_neg(self, width, nb):
        width_l = width // 2  # if width is even, pick point RIGHT of middle
        width_r = width - width_l
        idx_list = list(range(len(self.condensed_events)))[width_l : -width_r]      
        raw_hits_out = []
        raw_labels_out = []
        raw_kmers_out = []
        while len(raw_hits_out) < nb:
            cur_idx = choice(idx_list)
            cur_condensed_event = self.condensed_events[cur_idx]
            if self.classified[cur_idx] == 0:
                raw_labels_out.append(self.classified[cur_idx - width_l : cur_idx + width_r + 1])
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
#            print("\n\n\n\n ---------------Negatives --------------")
#            print(raw_hits_out)
        return(raw_hits_out, raw_labels_out, raw_kmers_out)
## from TrainingRead.py
    
@property
def positives(self):
    return self._positives        
    
@positives.setter
def positives(self, _):
    pos_events = []
#        neg_events = []
    self._positives = [self.]
    for lbl in range(len(labelled)):
        if labelled[lbl] == 1:
            pos_events.append((self.condensed_events[lbl][1], 1)) # creates list of pos event idx
#            else:
#                neg_events.append((self.condensed_events[lbl][1], 0))
    self._positives = pos_events 
#        return(self.positives)
#        self.negatives = neg_events
        
# --------- for saving all signals to one list per ch
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
                raw_signal_array = self.condensed_events[ch[1] - width_l + count][2]
                for raw in raw_signal_array:
                    raw_signals.append(raw)
#                raw_signals.append(self.condensed_events[ch[1] - width_l + count][2])
                print("\n\n\n\n--------Positive-------------")
                print(self.condensed_events[ch[1] - width_l + count][2])
#                raw_kmers_out.append(self.condensed_events[ch[1] - width_l + count][0])
                count += 1
            count = 0
            while count <= width_r:
                raw_signal_array = self.condensed_events[ch[1] + count][2]
                for raw in raw_signal_array:
                    raw_signals.append(raw)
#                raw_signals.append(self.condensed_events[ch[1] + count][2])
#                raw_kmers_out.append(self.condensed_events[ch[1] + count][0])
                count += 1 
            raw_hits_out.append(raw_signals)
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
                    raw_signal_array = self.condensed_events[cur_idx - width_l + count][2]
                    for raw in raw_signal_array:
                        raw_signals.append(raw)  
#                    raw_signals.append(self.condensed_events[cur_idx - width_l + count][2])
#                    raw_kmers.append(self.condensed_events[cur_idx - width_l + count][0])
                    count += 1
                count = 0
                while count <= width_r:
                    raw_signal_array = self.condensed_events[cur_idx + count][2]
                    for raw in raw_signal_array:
                        raw_signals.append(raw)
#                    raw_signals.append(self.condensed_events[cur_idx + count][2])
#                    raw_kmers.append(self.condensed_events[cur_idx + count][0])
                    count += 1  
                raw_hits_out.append(raw_signals)
                raw_kmers_out.append(cur_condensed_event[0])
            idx_list.remove(cur_idx)
        return raw_hits_out, raw_kmers_out

## FROM TRAINING_ENCODINGS.PY
    # retrieve list of booleans on first base being on every position:
    #    pattern = re.compile(kmer[0])
    #    pattern_index = [m.start(0) for m in pattern.finditer(kmer)] # base positions in kmer    
    #    bool_list = [i in pattern_index for i in range(k_length)]
    #    if bool_list.count(True) == len(kmer):
    #        return(1)
    #    else:
    #        return(0)
            
        ##### this must be adjusted further to classify not only from middle on but all
        #### meaning: NNNAAAAA = 00011111 / is now: 00010000
        ## TODO: after retrieving this sequence, every one should be extended 
        ## four to the left should also become 1
        ## pay attention to 11 that follow: so 
            
def trimer_class_number(kmer):
    mid = len(kmer)//2 + 1
    trimer = kmer[mid-2:mid+1]
    if trimer in cl1:
        return 1
    if trimer in cl2:
        return 2
    if trimer in cl3:
        return 3
    if trimer in cl4:
        return 4
    raise ValueError('trimer not recognized.')


def pu_py_class_number(kmer):
    mid_base = kmer[len(kmer) // 2]
    if mid_base in ['A', 'G']:
        return 1
    if mid_base in ['T', 'C']:
        return 2

def dt_class_number(kmer):
    """
    Classify dinucleotides-containing k-mers using a len(kmer)-class system
    """
    # TODO not finished
    k_length = len(kmer)
    class_list = []
    for base in [kmer[0], kmer[-1]]:
        pat = re.compile(base)
        pat_index = [m.start(0) for m in pat.finditer(kmer)]
        lst = [i in pat_index for i in range(k_length)]
        ccf = 0
        ccr = 0
        boolf = True
        boolr = True
        for i in range(k_length):
            if not lst[i]:
                boolf = False  # If series of trues stops in fwd direction, stop adding
            if not lst[-i-1]:
                boolr = False  # If series of trues stops in bwd direction, stop adding
            if not boolf and not boolr:
                break  # If both series are discontinued, stop iterating
            ccf += boolf
            ccr += boolr
        class_list += [ccf, ccr]
    return max(class_list + [1])  # return Nb in range 1( = no dimer at start) - k( = homopolymer)

# ------ Originals
# From get_training_read in ExampleDb.py:
if len(includes):
            forced_includes = []
            forced_includes_list = []
            for k in includes:
                if k in self.neg_kmers:
                    forced_includes_list.append(deepcopy(self.neg_kmers[k]))

            # limit to 20% of neg examples
            nb_neg_forced = nb_neg // 5
            nf_idx = 0
            while len(forced_includes) < nb_neg_forced and len(forced_includes_list):
                forced_includes.append(forced_includes_list[nf_idx].pop())
                if not len(forced_includes_list[nf_idx]):
                    forced_includes_list.remove([])
                nf_idx += 1
                if nf_idx == len(forced_includes_list):
                    nf_idx = 0
            ns = random.sample(range(self.nb_neg), nb_neg - len(forced_includes))
            ns.extend(forced_includes)
        else:
            pass # pass is not original
#
            
def hp_class_number(kmer):
    """
    Classify homopolymer-containing k-mers using a len(kmer)-class system
    """

    k_length = len(kmer)
    class_list = []
    if 'N' in kmer:  # always classify as 1 if kmer contains unknown base
        return 1
    for base in [kmer[0], kmer[-1]]:
        pattern = re.compile(base)
        pattern_index = [m.start(0) for m in pattern.finditer(kmer)]    
        lst = [i in pattern_index for i in range(k_length)]
        ccf = 0
        ccr = 0
        boolf = True
        boolr = True
        for i in range(k_length):
            if not lst[i]:
                boolf = False  # If series of trues stops in fwd direction, stop adding
            if not lst[-i-1]:
                boolr = False  # If series of trues stops in bwd direction, stop adding
            if not boolf and not boolr:
                break  # If both series are discontinued, stop iterating
            ccf += boolf
            ccr += boolr
        class_list += [ccf, ccr]
    return max(class_list + [1])  # return Nb in range 1( = no dimer at start) - k( = homopolymer)

valid_encoding_types = ['trimer',
                        'pupy',
                        'hp_5class']
                        
def class_number(kmer, encoding_type):
    if encoding_type == 'trimer':
        return trimer_class_number(kmer)
    if encoding_type == 'pupy':
        return pu_py_class_number(kmer)
    if encoding_type == 'hp_5class':
        return hp_class_number(kmer)
    ValueError('encoding type not recognized')
    
    
# Subdivision of all trimers in four ordinal classes
cl1 = ['GGT', 'GGA', 'AGT', 'GGG', 'AGG', 'GAT', 'AGA', 'GAG', 'GAA', 'CGT', 'CGA', 'AAT', 'TGA', 'CGG', 'AAG', 'TGT']
cl2 = ['GGC', 'AAA', 'GAC', 'CAT', 'CAG', 'AGC', 'TGG', 'TAT', 'CAA', 'TAG', 'AAC', 'CGC', 'TAA', 'TGC', 'CAC', 'TAC']
cl3 = ['GCT', 'CCT', 'TCT', 'ACT', 'CCG', 'TTT', 'GTT', 'GCG', 'TCG', 'CTT', 'GCA', 'ACG', 'CCA', 'TCA', 'ATT', 'ACA']
cl4 = ['CCC', 'TTG', 'TCC', 'GTA', 'TTA', 'GTG', 'GCC', 'CTG', 'ACC', 'CTA', 'ATG', 'ATA', 'TTC', 'GTC', 'CTC', 'ATC']

# from TrainingRead.py from def events:
        else:
            # TODO: adapt to new Albacore file format
            event_states = self.hdf[hdf_events_path]["model_state"]
            event_states = event_states.astype(str)
            event_move = self.hdf[hdf_events_path]["move"]
            event_move = event_move.astype(int)

            event_move[0] = 0  # Set first move to 0 in case it was not 0; used to set first-base index
            start_idx = 0
            # outputs
            event_list = [event_states[0]]  # list of k-mers assigned to events
            start_idx_list = [0]  # 0-based index of first base in event k-mer in fasta file
            event_length_list = []  # List of lengths of events in terms of raw data points
            event_raw_list = []  # List of lists containing raw data points per-event

            cur_event_length = event_lengths[0]
            temp_raw = list(self.raw)
            for n in range(event_move.size):
                if event_move[n] != 0:
                    event_length_list.append(cur_event_length)
                    event_raw_list.append(temp_raw[:cur_event_length])
                    del temp_raw[:cur_event_length]
                    cur_event_length = event_lengths[n]
                    start_idx += event_move[n]
                    event_list.append(event_states[n])
                    start_idx_list.append(start_idx)
                else:
                    cur_event_length += event_lengths[n]
            event_length_list.append(cur_event_length)  # Last event length
            event_raw_list.append(temp_raw[:cur_event_length]) # Last event raw data points
            del temp_raw[:cur_event_length]