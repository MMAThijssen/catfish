#!/usr/bin/env python3

from base_to_signal import get_base_new_signal
from collections import Counter
import matplotlib
#~ matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reader import load_npz_raw
from sklearn.cluster import KMeans
from statistics import median
from sys import argv


def main(output_file, main_dir, npz_dir):
    """
    Outputs information on all (true and false) positives in predicted output. 
    
    Args:
        output_file -- str, file outputted by neural network validation
        main_dir -- str, path to main directory containing FAST5 files
        npz_dir -- str, path to directory containing .npz files
        
    Returns: None
    """
    show_plots = False
    
    # 0. Save all predicted HPs with positions to dict
    predicted_labels = list_predicted(output_file, types="predicted_labels")
    predicted_hp = hp_loc_dict(predicted_labels)
    
    # 1. Make a distribution on measurement lengths of HPs
    count_predictions = get_prevalence(predicted_hp)
    
    # 2. Find underlying base sequence
    with open(output_file, "r") as source:
        for line in source:
            if not (line.startswith("#") or line.startswith("*") or line.startswith("@")):
                read_name = line.strip()
                print(read_name)
                # using directory       # FASTER! 0m1.437s
                bases, new = get_base_new_signal("{}/{}.fast5".format(main_dir, read_name))
                base_dict = {k: get_bases(bases, new, predicted_hp[k][0], predicted_hp[k][1]) for k in predicted_hp}
                
    # 3. Make a distrubtion on base lengths of HPs
    count_basehp = get_prevalence(base_dict, types="base_lengths")
    
    # 4a. Save all true HPs with positions to dict
    true_labels = list_predicted(output_file, types="true_labels")
    true_hp = hp_loc_dict(true_labels)
    count_truehp = get_prevalence(true_hp)
    
    # 4b. Compare "predicted" lengths to real HP lengths
        # this can be done with length plots
    base_dict_true = {k: get_bases(bases, new, true_hp[k][0], true_hp[k][1]) for k in true_hp}
    count_basetrue = get_prevalence(base_dict_true, types="base_lengths")
    detected_from_true(predicted_hp, true_hp)                                   # unnecessary
            
    # 5. Count HP compositions and prevalence and compare to truth
    count_seq = get_prevalence(base_dict, "bases")
    count_seqtrue = get_prevalence(base_dict_true, "bases")
    seq_list = dict_to_ordered_list(count_seq)
    seqtrue_list = dict_to_ordered_list(count_seqtrue)
    
    if show_plots:
        plot_prevalence(count_predictions, name="HP_length_measurements")       # 1
        plot_prevalence(count_basehp, name="HP_length_bases")                   # 3
        plot_prevalence(count_truehp, name="True HP lengths_measurements")      # 4a
        plot_prevalence(count_basetrue, name="True_HP_length_bases")            # 4b
        plot_list(seq_list, "Predicted_HP_sequences", rotate=True)              # 5
        plot_list(seqtrue_list, "True_HP_sequences")                            # 5
        
    
    # 6. Check finding back true HP
            # or with counting number of measurements found back
            # compare true labels with predicted labels
        # TODO:
    states = [check_true_hp(hp, predicted_labels) for hp in true_hp.values()] 
    print(states)       
    
    #~ # count how many completely found back
    #~ complete_st = [1 for st in states if st[0] == "complete"]
    #~ complete_st = sum(complete_st)
    #~ overleft = [st[1][0] for st in states if st[0] == "complete"]
    #~ overright = [st[1][1] for st in states if st[0] == "complete"]
    #~ perfectcomplete = [1 for st in states if (st[0] == "complete" and (st[1][0] == 0 and st[1][1] == 0))]
    #~ avg_overleft = sum(overleft) / complete_st
    #~ avg_overright = sum(overright) / complete_st
    #~ perfectcomplete = sum(perfectcomplete)
    #~ print(avg_overleft, avg_overright, perfectcomplete)
    
    #~ # count interruptions
    #~ incomplete_st = len(states) - complete_st
    #~ print("incomplete: ", incomplete_st)
    #~ list_inters = [len(st[2]) for st in states]
    #~ nr_inters = sum(list_inters)
    #~ avg_inters = nr_inters / incomplete_st                                      # if less than 0: usually no inters
    #~ median_inters = median(list_inters)  
    
    #~ print("interruptions: ", nr_inters)
    #~ print("average: ", avg_inters)
    #~ print("median: ", median_inters)
    
    #~ # what length of inters? only if inter is present
    #~ length_inters = [(st[2][0][1] - st[2][0][0] + 1) for st in states if st[2] != []]
    #~ print(length_inters)

    # something with positions of interruptions?
    
    # 7. Check surroundings
        # when do I consider it close?
            
    ### LATER - I want to do something with the raw signal ###
    # maybe something with some kind of clustering as well #
    # check def LATER #


    return None


def list_predicted(pred_file, types="predicted_labels"):
    """
    Creates list from predicted labels saved in file.
    
    Args:
        pred_file -- str, file that contains predicted labels as '[.., .., ..]'
        types -- str, type of line to retrieve: predicted labels, true labels or predicted scores [default: 'predicted_labels']
        
    Returns: predicted labels (list of ints)
    """
    if types == "predicted_labels":
        sign = "#"
        datatype = int
    elif types == "true_labels":
        sign = "*"
        datatype = int
    elif types == "predicted_scores":
        sign = "@"
        datatype = float       
        
    with open(pred_file, "r") as source:
        for line in source:
            if not line.strip():
                continue
            elif line.startswith(sign):
                predicted = line.strip()[3:-1].split(", ")
                predicted = list(map(datatype, predicted))

    return predicted
    
    
def list_basenew(pred_file, read_name, types="bases"):
    """
    Creates list from predicted labels saved in file.
    
    Args:
        pred_file -- str, file that contains predicted labels as [.., .., ..]
        read_name -- str, name of read
        types -- str, type of line to retrieve: bases or new [default: 'bases']
        
    Returns: predicted labels (list of ints)
    """
    if types == "bases":
        sign = "$"
    elif types == "new":
        sign = "!"

    search_for_read = True
    found_read = False
    with open(pred_file, "r") as source:
        while search_for_read:
            for line in source:
                if line.startswith(read_name):
                    found_read = True
                elif found_read and line.startswith(sign):
                    predicted = line.strip()[2:].split(", ")[0]
                    search_for_read = False
                
    return predicted


def hp_loc_dict(predicted_labels):
    # adjusted from save_hp_loc in info_hp
    """
    Creates dict to save positions of homopolymers.
    
    Args:
        predicted_labels -- list of int, predicted HP labels
        
    Returns: dict {id: start position, end position (inclusive)}
    """
    predicted_hp = {}
    counter = 0
    is_hp = False
    for i in range(len(predicted_labels)):
        if not is_hp and predicted_labels[i] == 1:
            is_hp = True
            start = i
            end = i
        #~ elif is_hp and predicted_labels[i] == 1:
            #~ end = i
        elif is_hp:
            if (i + 1 == len(predicted_labels)):
                end = i
                predicted_hp[counter] = (start, end)
            elif predicted_labels[i] == 0: 
                end = i - 1
                is_hp = False
                predicted_hp[counter] = (start, end)
                counter += 1
            
    
    return predicted_hp
            
    
def get_prevalence(hp_dict, types="measurements"):
    """
    Retrieves prevalance of occurence in dict.
    
    Args:
        hp_dict -- dict {id: start, end}
        
    Returns: Counter dict {length: prevalence}
    """
    if types == "measurements":
        length_list = [k[1] - k[0] + 1 for k in hp_dict.values()]
    elif types == "base_lengths":
        length_list = [len(k) for k in hp_dict.values()]
    elif types == "bases":
        length_list = hp_dict.values()
    count_dict = Counter(length_list)
    
    return count_dict   
    

def plot_prevalence(count_dict, name="Prevalence", rotate=False):
    """
    Plots prevalence.
    
    Args:
        count_dict -- dict {value: count}
        
    Returns: None
    """
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 9)) 
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.ylabel("count")
    plt.xlabel("length")
    plt.title(name)
    if rotate:
        plt.xticks(rotation=75)
    plt.show()
    #~ plt.savefig("{}.png".format(name), bbox_inches="tight")
    #~ plt.close()
    

def plot_list(count_list, name="Prevalence", rotate=False):
    """
    Plots prevalence.
    
    Args:
        count_dict -- dict {value: count}
        
    Returns: None
    """
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 9)) 
    b_list = [i[0] for i in count_list]
    c_list = [i[1] for i in count_list]
    plt.bar(b_list, c_list)
    plt.ylabel("count")
    plt.xlabel("length")
    plt.title(name)
    if rotate:
        plt.xticks(rotation=75)
    plt.show()
    #~ plt.savefig("{}.png".format(name), bbox_inches="tight")
    #~ plt.close()
    

def plot_multi_bars(x, bar1, bar2):     # not yet tested
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    ax.bar(x-0.2, bar1,width=0.2,color='b',align='center')
    ax.bar(x, bar2,width=0.2,color='g',align='center')

    plt.show()
    
    
def get_bases(base_seq, new_seq, hp_start, hp_end):
    """
    Retrieves underlying base sequences from measurement positions.
    
    Args:
        base_seq -- str, sequence of bases
        new_seq -- str, indication of new base
        hp_start -- int, start position of homopolymer 
        hp_end -- int, end position of homopolymer
        
    Returns: base sequence (str)
    """
    first_base = [base_seq[hp_start]]
    bases = [base_seq[i] for i in range(hp_start + 1, hp_end + 1) if new_seq[i] == "n"]
    bases = first_base + bases
    
    return "".join(bases)
    

def detected_from_true(predicted_hp, true_hp):
    predicted_list = [list(range(k[0], k[1] + 1)) for k in predicted_hp.values()]
    predicted_list = [j for i in predicted_list for j in i]
    true_list = [list(range(k[0], k[1] + 1)) for k in true_hp.values()]
    true_list = [j for i in true_list for j in i]    
    counter = [1 for c in range(len(true_list)) if true_list[c] in predicted_list]
    count_tp = sum(counter)
    print("Found {} of {} true measurements".format(count_tp, len(true_list)))  # this part is double: is like main stats on recall etc.


def dict_to_ordered_list(dict_in, sort_on=0):
    """
    Creates list of keys and values ordered.
    
    Args:
        dict_in -- dict {key: value}
        sort_on -- int, element of tuple to sort on [default: 0 (key)]
        
    Returns: list of tuples
    """
    ordered_list = []
    for k, v in dict_in.items():
        temp = (k, v)
        ordered_list.append(temp)

    return sorted(ordered_list, key=lambda x: x[sort_on])


def check_true_hp(true_hp, predicted_labels):
    """
    Checks how well a true homopolymer is detected by network.
    
    Args:
        true_hp -- tuple of ints, start and end position of hp
        predicted_labels -- list of ints, predicted labels
        
    Returns: [state, (left, right), [length of interruption, ...]]
            left and right are the number of over- or underestimated measurements
    """
    inter_list = []
    predicted_stretch = predicted_labels[true_hp[0] : true_hp[1] + 1]
    # check if true homopolymer is completely or partly detected
    if (true_hp[1] - true_hp[0] + 1) == predicted_stretch.count(1):
        state = "complete"
    else:
        state = "incomplete" 
        # check for interruptions: where, how long, how often  
        new_inter = False
        for i in range(len(predicted_stretch)):
            if not new_inter and predicted_stretch[i] == 0: 
                new_inter = True 
                start = i
            elif new_inter:
                if predicted_stretch[i] == 1:
                    end = i - 1
                    temp = (start, end)
                    if not (start == 0 or end == len(predicted_stretch) - 1):
                        inter_list.append(temp)
                    new_inter = False
                if len(predicted_stretch) == (i + 1):
                    end = i
                    temp = (start, end)
                    if not (start == 0 or end == len(predicted_stretch) - 1):
                        inter_list.append(temp)               

    # check if prediction is overestimated on either side:   
    l = 0
    if predicted_labels[true_hp[0]] == 1:
        check_left = True
        while check_left:
            position = true_hp[0] + l - 1
            if position != 0 and predicted_labels[position] == 1:
                l -= 1
            else:
                check_left = False
    r = 0 
    if predicted_labels[true_hp[1]] == 1:
        check_right = True
        while check_right:
            position = true_hp[1] + r + 1
            if position != len(predicted_labels) and predicted_labels[position] == 1:
                r += 1
            else:
                check_right = False
            
    # check if prediction is underestimated on either side:         
    if l == 0:
        check_left = True
        while check_left:
            if predicted_labels[true_hp[0] + l] == 0:
                l += 1
            else:
                check_left = False
    if r == 0: 
        check_right = True
        while check_right:
            if predicted_labels[true_hp[1] + r] == 0:
                r -= 1
            else:
                check_right = False    
    
    return state, (l, r), inter_list
    


def kmeans_clustering(x, n):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    """
    K-means clustering.
    
    Args:
        x -- array, input
        n -- int, number of clusters
    
    Returns: list of cluster to which input belongs (list of ints)
    """
    # perform k-means clustering:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)
    
    # visualize:
    plt.style.use("seaborn")
    plt.scatter(x, x, c=y_kmeans, s=50)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    
    return kmeans.labels_
    
    #~ # Then get the indices of points for each cluster
    #~ {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
    #~ # If you want to use array of points in X as values rather than the array of indices:
    #~ {i: X[np.where(estimator.labels_ == i)] for i in range(estimator.n_clusters)}
    

def hierarchical_clustering():
    """
    Hierarchical clustering.
    """
    pass
    

def LATER():
        # 7. Cluster HP compositions on raw measurements
        # maybe also as measurement long base sequences..
    # 7a. Get raw measurements for each predicted HP
    #~ raw = load_npz_raw("{}/{}.npz".format(npz_dir, read_name))
    # id in dicts is equal across dicts: so 1 in predicted is same part as 1 in base_dict
    # Create dict or something else so raw measurements are plotted but you know what sequence is underneath
    # 7b. K-means cluster
        # on base A, G, T, C > make an [count A, count C, count G, count T] and cluster 
        # or as A:1, C:2, T:3, G:4, nothing:0
    #~ length_dict = {k: len(base_dict[k]) for k in base_dict}
    #~ print(length_dict)
    #~ kmc_labels = kmeans_clustering(x, 2)
    #~ print(len(kmc_labels))
        # on size
        # on true HP and non HP
        
    # 7c. Hierarchical clustering
    pass

    
if __name__ == "__main__":
    read_file = argv[1]
    main_fast5_dir = argv[2]
    main_npz_dir = argv[3]
    predicted_dict = main(read_file, main_fast5_dir, main_npz_dir)


