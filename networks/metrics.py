#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics
from sys import argv

def confusion_matrix(true_labels, predicted_labels):
    """
    Returns precision and recall
    """

    if len(true_labels) != len(predicted_labels):
        print("Len true labels: ", len(true_labels))
        print("Len pred labels: ", len(predicted_labels))
        print("True labels: ", true_labels)
        print("Pred labels: ", predicted_labels)
        raise ValueError("Length of labels to compare is not equal.")
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
    for i in range(len(true_labels)):
        if predicted_labels[i] == 1:
            if true_labels[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
        elif predicted_labels[i] == 0:
            if true_labels[i] == 0:
                true_neg += 1
            else:
                false_neg += 1
    print("TP: {}\tFP: {}\tTN: {}\tFN: {}".format(true_pos, false_pos, true_neg, false_neg))
    #~ return true_pos, false_pos, true_neg, false_neg


#~ def false_positive_rate(false_pos, true_neg):
    #~ return false_pos / (false_pos + true_neg)

#~ def precision_recall(true_pos, false_pos, false_neg):
    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0
        print("Precision could not be calculated.")
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0
        print("Recall could not be calculated.")
    return precision, recall
    

def calculate_auc(true_labels, predicted_labels, pos_label=1):
    # tpr is recall 
    tpr, fpr, thresholds = sklearn.metrics.roc_curve(y_true=true_labels, 
                                                     y_score=predicted_labels,
                                                     pos_label=pos_label)
    roc_auc = sklearn.metrics.auc(true_labels, predicted_labels)
    print("AUC: ", roc_auc)
    return tpr, fpr, roc_auc

def draw_roc(tpr, fpr, roc_auc):
    plt.title("Receiver Operator Characteristic")
    plt.plt(fpr, tpr, "b", label="AUC = {:.2f}".format(roc_auc))
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    

def weighted_f1(precision, recall, n, N):
    """
    Calculates weighted F1 score for a single class.
    
    Args:
        precision -- int
        recall -- int
        n -- int, number of samples belong to class
        N -- int, total number of samples
    
    Returns int
    """
    try:
        wF1 = 2 * n  / N * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        print("Precision, recall or both are zero. Unable of calculating weighted F1.")
        wF1 = 0
    return wF1
    

def parse_txt(cprofile, measure):
    """
    Parses cProfile text files to retrieve values per epoch.
    
    Args:
        cProfile -- str, text file should have "Epoch" on same line as metric
        measure -- str, either "Accuracy" or "Loss"
        
    Returns list
    """
    measure_list = []
    with open(cprofile, "r") as source:
        for line in source:
            if line.startswith("Epoch"):
                line = line.split()
                for i in range(len(line)):
                    if measure in line[i]:
                        if line[i + 1][-1] == "%":
                            line[i + 1] = line[i + 1][:-1]                   # -1 to lose the % sign
                        measure_list.append(float(line[i + 1]))       
        if len(measure_list) == 0:
            raise ValueError("Given measure has not been found in file.") 
    return measure_list
    
    
def plot_settings(measure):
    """
    Setting to create pretty plots. 
    
    Returns tableau colors as list
    """
    # scaled tableau20 colors:    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
         
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)
        
    # set size:                             common - (10, 7.5) and (12, 14)    
    plt.figure(figsize=(12, 9))    
      
    # remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_linestyle("--")
    ax.spines["bottom"].set_color("grey")  
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False) 
    
    # set tick lines for readability:
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    if measure == "Accuracy":      
        plt.yticks(range(0, 101, 10), [str(x) + "%" for x in range(0, 101, 10)], fontsize=14) 
        ymax = 100
        for y in range(0, 101, 10):    
            plt.plot(range(1, 11), [y] * len(range(1, 11)), "--", lw=0.5, color="black", alpha=0.3) 
    else:
        plt.yticks(np.arange(0, 1.1, 0.1), ["{:.1f}".format(x) for x in np.arange(0, 1.1, 0.1)], fontsize=14)
        ymax = 1
        for y in np.arange(0, 1.1, 0.1):    
            plt.plot(np.arange(1, 11), [y] * len(np.arange(1, 11)), "--", lw=0.5, color="black", alpha=0.3) 
    plt.xticks(fontsize=14)
        
    # limit range of plot:       
    plt.ylim(0, ymax)    
    plt.xlim(1, 10) 
  
    # remove tick marks because of tick lines    
    plt.tick_params(axis="both", which="both", bottom=False, top=False,    
                labelbottom=True, left=False, right=False, labelleft=True)
    
    return tableau20
    
def plot_networks_on_metric(network_list, metric):
    colours = plot_settings(measure)

    for i in range(len(network_list)):
        network = file_list[i].split("_")[0]
        plt.plot(range(1, len(network_list[i]) + 1), network_list[i],lw=2.5, color=colours[i], label=network)
    plt.title(measure, loc="center")
    plt.legend(loc="lower right")
    plt.savefig("{}.png".format(measure), bbox_inches="tight")
    #~ plt.show()


def generate_heatmap(predicted_list, label_list, title):
    sns.heatmap(predicted_list, vmin=0.0, vmax=1.0, cmap="GnBu",      # PiYG - YlGnBu
                 xticklabels=False, yticklabels=label_list, 
                 cbar_kws={"orientation": "horizontal"})
    plt.savefig("{}.png".format(title), bbox_inches="tight")
    #~ plt.show()

    
                
if __name__ == "__main__":
    measure = argv[1]
    file_list = argv[2:]
    measure_list = []
    for fl in file_list:
        accuracy = parse_txt(fl, measure)
        measure_list.append(accuracy)
    
    plot_networks_on_metric(measure_list, accuracy)