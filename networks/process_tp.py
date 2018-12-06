#!/usr/bin/env python3

from base_to_signal import get_base_new_signal
from collections import Counter
import matplotlib
#~ matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reader import load_npz_raw
from sklearn.cluster import KMeans
from sys import argv

#TODO: start for a single read at the moment
# later do multiple samplings of eg. 100 reads
def main(output_file, main_dir, npz_dir):
    # ASSUMES OUTPUT FILE ON A SINGLE READ FOR NOW
    """
    Outputs information on all (true and false) positives in predicted output. 
    
    Args:
        output_file -- str, file outputted by neural network validation
        main_dir -- str, path to main directory containing FAST5 files
        npz_dir -- str, path to directory containing .npz files
        
    Returns: None
    """
    show_plots = True
    
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
    
        # or with counting number of measurements found back
        # TODO:
        # check if it is continuous or interrupted:
        # a. Check if true HP is completely found, short or too long
        # b. Check if true HP is interrupted

        
    # 5. Count HP compositions and prevalence and compare to truth
    count_seq = get_prevalence(base_dict, "bases")
    count_seqtrue = get_prevalence(base_dict_true, "bases")
    seq_list = dict_to_ordered_list(count_seq)
    seqtrue_list = dict_to_ordered_list(count_seqtrue)
    
    
    if show_plots:
        plot_prevalence(count_predictions, name="HP_length_measurements")       # 1
        plot_prevalence(count_basehp, name="HP_length_bases")                   # 3
        plot_prevalence(count_truehp, name="True HP lengths")                   # 4a
        plot_prevalence(count_basetrue, name="True_HP_length_bases")            # 4b
        plot_list(seq_list, "Predicted_HP_sequences", rotate=True)              # 5
        plot_list(seqtrue_list, "True_HP_sequences")                            # 5
        
        
        
    ### LATER - I want to do something with the raw signal ###
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

    
    return predicted_hp


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
        elif is_hp and predicted_labels[i] == 1:
            end = i
        elif is_hp and (predicted_labels[i] == 0 or i == len(predicted_labels) - 1):
            predicted_hp[counter] = (start, end)
            counter += 1
            is_hp = False
    
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
    
    # Then get the indices of points for each cluster
    {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
    # If you want to use array of points in X as values rather than the array of indices:
    {i: X[np.where(estimator.labels_ == i)] for i in range(estimator.n_clusters)}
    

def hierarchical_clustering():
    """
    Hierarchical clustering.
    """
    pass
    
if __name__ == "__main__":
    read_file = argv[1]
    main_fast5_dir = argv[2]
    main_npz_dir = argv[3]
    predicted_dict = main(read_file, main_fast5_dir, main_npz_dir)


