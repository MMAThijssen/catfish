#!/usr/bin/env python3
import matplotlib
#~ matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as sklmet
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
    return true_pos, false_pos, true_neg, false_neg
    

def precision_recall(true_pos, false_pos, false_neg):
    """
    Returns precision and recall
    """
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
    

def calculate_accuracy(true_pos, false_pos, true_neg, false_neg):
    try:
        accuracy = (true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg)
    except ZeroDivisionError:
        accuracy = 0
    return accuracy
    

def calculate_auc(true_labels, predicted_scores, pos_label=1):
    """
    Calculates the area under the receiver operator curve
    
    Args:
        true_labels -- list of ints (0 - neg, 1 - pos)
        predicted_scores -- list of floats, confidence of predictions
        pos_label -- int (0 or 1)
        
    Returns: TPR, NPR, AUC
    """
    # tpr is recall 
    tpr, fpr, thresholds = sklmet.roc_curve(y_true=true_labels, 
                                            y_score=predicted_scores,
                                            pos_label=pos_label)
    roc_auc = sklmet.auc(fpr, tpr)
    return tpr, fpr, roc_auc
    
    
def calculate_pr(true_labels, predicted_scores, pos_label=1):
    precision, recall, thresholds = sklmet.precision_recall_curve(true_labels, 
                                                                  predicted_scores,
                                                                  pos_label)  
    return precision, recall, thresholds
    
    
def compute_auc(tpr, fpr):
    # tpr and fpr should be arrays!
    return sklmet.auc(fpr, tpr)


def draw_roc_t(tpr, fpr, roc_auc, title, thresholds=[0.5]):
    colors = set_sns_style()
    
    plt.title("Receiver Operator Characteristic")
    for t in range(len(thresholds)):
        plt.plot(fpr[t], tpr[t], "b", label="AUC {} = {:.2f}".format(thresholds[t], roc_auc[t]), c=colors[t])
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1],'r--', c="r")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("ROC_{}-t.png".format(title), bbox_inches="tight")
#    plt.show()
    plt.close()
    
def draw_pr_t(precision, recall, title, thresholds=[0.5]):
    """
    Plots precision recall curve.
    
    Args:
        precision -- list of floats, precision
        recall -- list of floats, precision
        title -- str, name of plot
        thresholds -- list of floats, threshold for positive label [default: [0.5]]
        
    Returns: None
    """
    colors = set_sns_style()
    
    plt.title("Precision Recall Curve")
    for t in range(len(thresholds)):
        plt.plot(precision[t], recall[t], c=colors[t], label=thresholds[t])
    plt.plot([0, 1], [0.5, 0.5], 'r--', c="r")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend()
    plt.savefig("PR_{}-t.png".format(title), bbox_inches="tight")
#    plt.show()
    plt.close()
    
    return None
    

def class_from_threshold(predicted_scores, threshold):
    """
    Assigns classes on input based on given threshold.
    
    Args:
        predicted_scores -- list of floats, scores outputted by neural network
        threshold -- float, threshold
    
    Returns: list of class labels (ints)
    """
    return [1 if y >= threshold else 0 for y in predicted_scores]
    

def set_sns_style():
    plt.style.use("seaborn")
    c1 = "hotpink"
    c2 = "turquoise"
    c3 = "mintcream"
    c4 = "springgreen"
    c5 = "orange"
    c6 = "mediumvioletred"
    c7 = "midnightblue"
    c8 = "maroon"
    c9 = "gold"
    colors = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    
    return colors
    
    
def draw_roc(tpr, fpr, roc_auc, title):
    plt.style.use("seaborn")
    plt.title("Receiver Operator Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = {:.2f}".format(roc_auc), c="darkblue")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1],'r--', c="r")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("ROC_{}.png".format(title), bbox_inches="tight")
#    plt.show()
    plt.close()
    
def draw_pr(precision, recall, title):
    plt.style.use("seaborn")
    color = "hotpink"
    plt.title("Precision Recall Curve")
    plt.plot(precision, recall, c=color)
    plt.plot([0, 1], [0.5, 0.5], 'r--', c="r")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig("PR_{}.png".format(title), bbox_inches="tight")
#    plt.show()
    plt.close()

    
    
def weighted_f1(precision, recall, n, N):
    """
    Calculates balanced (not weighted!) F1 score for a single class.
    
    Args:
        precision -- int
        recall -- int
        n -- int, number of samples belong to class
        N -- int, total number of samples
    
    Returns: f1-score as int
    """
    try:
        f1 = 2 * n  / N * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        print("Precision, recall or both are zero. Unable of calculating weighted F1.")
        f1 = 0
    
    return f1
    

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


def plot_squiggle(signal, title):
    plt.figure(figsize=(30, 10)) 
    plt.plot(signal, color="black", linewidth=0.5) 
    plt.ylabel("Normalized signal", fontsize=14)
    plt.xlim(left=0, right=len(signal))
    plt.savefig("{}.png".format(title))
    plt.close()
    
    
    
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
    colours = plot_settings(metric)

    for i in range(len(network_list)):
        network = network_list[i].split("_")[0]
        network_list[i].insert(0, 0)    # prepend starting at 0
        plt.plot(range(0, len(network_list[i]) + 1), network_list[i],lw=2.5, color=colours[i], label=network)
    plt.title(metric, loc="center")
    plt.legend(loc="lower right")
    plt.savefig("{}.png".format(metric), bbox_inches="tight")
    plt.close()
    #~ plt.show()


def generate_heatmap(predicted_list, label_list, title):
    sns.heatmap(predicted_list, vmin=0.0, vmax=1.0, cmap="GnBu",      # PiYG - YlGnBu
                 xticklabels=False, yticklabels=label_list, 
                 cbar_kws={"orientation": "horizontal"})
    #~ plt.show()
    plt.savefig("{}.png".format(title), bbox_inches="tight")
    plt.close()


def draw_roc_and_pr_from_file(predout_file, name_network, thresholds=[0.5]):
    """
    Args:
        predout_file -- str, file outputted after validation
        name_network -- str, name of network
        thresholds -- list of floats, thresholds [default: 0.5]
        
    Returns: None
    """
    true_labels = []
    predicted_scores = []
    with open(predout_file, "r") as source1:
        for line in source1:
            if not line.strip():
                continue
            elif line.startswith("*"):
                labels = line.strip()[3:-1].split(", ")
                labels = list(map(int, labels))
                true_labels.extend(labels)
            elif line.startswith("@"):
                preds = line.strip()[3:-1].split(", ")
                preds = list(map(float, preds))
                predicted_scores.extend(preds)  

    tpr, fpr, auc = calculate_auc(true_labels, predicted_scores)
    draw_roc(tpr, fpr, auc, name_network)
    
    prec, rec, thres = calculate_pr(true_labels, predicted_scores)
    draw_pr(prec, rec, name_network)

    plot_pr_threshold(prec, rec, thres, name_network)
               
    adjusted_scores = [class_from_threshold(predicted_scores, t) for t in thresholds]
    tpr_list = []
    fpr_list = []
    auc_list = []
    prec_list = []
    rec_list = []
    
    for a in adjusted_scores:                    
        tpr, fpr, auc = calculate_auc(true_labels, a)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        auc_list.append(auc)
        
        prec, rec, thres = calculate_pr(true_labels, a)
        prec_list.append(prec)
        rec_list.append(rec)           
    
    draw_roc_t(tpr_list, fpr_list, auc_list, name_network, thresholds)
    draw_pr_t(prec_list, rec_list, name_network, thresholds)
    
    return None
    

def plot_pr_threshold(precision, recall, thresholds, name_network):
    """
    Plots precision and recall against threshold.
    
    """
    plt.style.use("seaborn")
    c1 = "crimson"
    c2 = "navajowhite"
    plt.figure(figsize=(8, 8))
    plt.title("Precision and recall as a function of threshold")
    plt.plot(thresholds, precision[:-1], "--", c=c1, label="precision")
    plt.plot(thresholds, recall[:-1], "-", c=c2, label="recall")
    plt.ylabel("score")
    plt.xlabel("decision threshold")
    plt.legend(loc='best')
    plt.savefig("PRT_{}.png".format(name_network), bbox_inches="tight")
    plt.close()
    
#def draw_pr_curve(in_files, outname):
#    """
#    Draws precision recall curve.
#    
#    Args:
#        in_file -- str, path to file
#        outname -- str, name of figure
#        
#    Returns: list (precision), list (recall)
#    """
#    plt.style.use("seaborn")
#    
#    # get precision and recall:
#    precision_list = []
#    recall_list = []
#    with open(in_files, "r") as source:
#        for line in source:
#            if line.strip().startswith("Precision"):
#                precision_list.append(float(line.strip()[:-1].split(": ")[1]) / 100) # -1 to get rid of %
#            elif line.strip().startswith("Recall"):
#                recall_list.append(float(line.strip()[:-1].split(": ")[1]) / 100) # -1 to get rid of %
#    
#    # draw plot
#    color = "hotpink"
#    plt.plot(precision_list, recall_list, c=color)   
#    plt.title("Precision Recall Curve")
#    plt.xlim(left=0)    
#    plt.ylim([0.0, 1.0])
#    plt.ylabel('Precision')
#    plt.xlabel('Recall')
#    plt.savefig("PR_{}.png".format(outname), bbox_inches="tight")
##    plt.show()
#    plt.close()

# for f1 plots: see draw_curves.py
    
    
                
if __name__ == "__main__":
    input_file = argv[1]
    output_name = argv[2]       # usually choose network: eg RNN92
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    draw_roc_and_pr_from_file(input_file, output_name, thresholds)
    

