#!/usr/bin/env python3 

from base_to_signal import get_base_new_signal
import h5py
from sys import argv

# 1. Implement majority voting to work back to corrected read and improve further
def correct_events(read, scores, start=30000, length=4970, use_tombo=True):
    """
    Corrects events based on majority voting by measurements belong to that event.
    
    Args:
        read -- str, path to read
        scores -- list of floats, score per measurement
        start -- int, start point in read
        length -- int, length to correct output / length of read from start
        use_tombo -- bool, use of corrected or uncorrected reads [default: True]
    
    Returns: corrected base sequence
    """
    # 0. Get event length and base sequence from read
    with h5py.File(read, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        # get list of event lengths:
        event_lengths = hdf[hdf_events_path]["length"]                         # indicates number of measurements belong to base/event
        if use_tombo:
            # get list of base sequence:
            event_bases = hdf[hdf_events_path]["base"].astype(str)
            
            # start at correct point:
            summed_len = 0
            voted_bases = 0
            found_start = False
            classes = []
            for n in range(len(event_lengths)):
                print(n)
                if not found_start and summed_len >= start:
                    start_event = n
                    if summed_len > start:                     
                        print("Could not classify first base")                  # print statements are not necessary
                    start_len = summed_len
                    found_start = True
                summed_len += event_lengths[n]
                if summed_len > length:
                    final_event = n - 1                                         # -1 so last one is last really used one
                    break
                elif found_start:
                    # do majority voting
                    voted_bases += 1
                    avg_score = sum(scores[start_len : summed_len]) / len(scores[start_len : summed_len])
                    classes.append(round(avg_score))
                    start_len = summed_len
                    if start_len == start + length:
                        final_event = n
                        break
            
        else:
            raise Exception("Not yet implemented for uncorrected reads")
    
    print(classes)
    print(event_lengths[:10])
    print(event_bases[:10])
    print("Classified {} bases from event {} to {}".format(voted_bases, start_event, final_event))

    # 2. Adjust sequence if needed
    # you know the first and last event numbers and have an equal length of labels
    # but how to correct?       look at both sides.. 
    if not len(classes) == len(event_lengths[start_event: final_event + 1]):
        raise ValueError("Number of events is not equal to number of classified events")
    
    # 3. Return corrected sequence
    


        

# 2. Try to get perfect measurement sequence (but is this needed)

if __name__ == "__main__":
    fast5 = argv[1]
    #~ predictions = argv[2]
    predictions = [0.91, 0.92, 0.93, 0.91, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.45]
    start = int(argv[3])
    length = int(argv[4])
    correct_events(fast5, predictions, start, length)
