import matplotlib.pyplot as plt
from trainingDB.metrics import set_sns_style
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sys import argv

colors = set_sns_style()

def draw_roc_curve(true_labels, predicted_scores, output_name, ts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]):
    auc = roc_auc_score(true_labels, predicted_scores)
    print("AUC: {:.3f}".format(auc))
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    print(len(fpr), len(tpr), len(thresholds))
    plt.title("Receiver Operator Characteristic")
    plt.plot([0, 1], [0, 1],'r--', c="crimson")
    plt.plot(fpr, tpr, "b", label="AUC = {:.3f}".format(auc), c="darkblue")
    for t in range(len(ts)):
        #~ adjusted_predictions = class_from_threshold(predicted_scores, t)
        close_point = np.argmin(np.abs(thresholds - ts[t]))
        plt.plot(fpr[close_point], tpr[close_point], marker=".", label=ts[t], markersize=10, c=colors[t])
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("ROC_{}.png".format(output_name), bbox_inches="tight")
    #~ plt.show()
    plt.close()
    
    
def draw_precision_recall_curve(precision, recall, thresholds, output_name):
    auc_score = auc(recall, precision)
    print(auc_score)
    
    plt.title("Precision Recall Curve")
    plt.plot(precision, recall, c="darkblue")
    plt.plot([0, 1], [0.022, 0.022], 'r--', c="crimson")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig("PR_{}.png".format(output_name), bbox_inches="tight")
    #~ plt.show()
    plt.close()
    

def draw_precision_recall_threshold(precision, recall, thresholds, output_name, ts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]):
    plt.title("Precision Recall Curve")
    plt.plot([0, 1], [0.022, 0.022], 'r--', c="crimson")
    plt.step(recall, precision, alpha=0.2, where="post", color=colors[1])
    plt.fill_between(recall, precision, step="post", alpha=0.2, color=colors[1])
    for t in range(len(ts)):
        #~ adjusted_predictions = class_from_threshold(predicted_scores, t)
        close_point = np.argmin(np.abs(thresholds - ts[t]))
        plt.plot(recall[close_point], precision[close_point], marker=".", label=t, c=colors[t])
    #~ plt.show()
    plt.savefig("PR-t_{}.png".format(output_name), bbox_inches="tight")
    plt.close()
    
    
def plot_pr_threshold(precision, recall, thresholds, name_network):
    """
    Plots precision and recall against threshold.
    """
    c1 = "mediumseagreen"
    c2 = "navajowhite"
    plt.figure(figsize=(8, 8))
    plt.title("Precision and recall as a function of threshold")
    plt.plot(thresholds, precision[:-1], "--", c=c1, label="precision")
    plt.plot(thresholds, recall[:-1], "-", c=c2, label="recall")
    plt.ylabel("score")
    plt.xlabel("decision threshold")
    plt.legend(loc='best')
    #~ plt.show()
    plt.savefig("PRT_{}.png".format(name_network), bbox_inches="tight")
    plt.close()

    
def compute_f1(true_labels, predicted_scores, thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]):
    return [f1_score(true_labels, class_from_threshold(predicted_scores, t)) for t in thresholds]


# from process_output
def class_from_threshold(predicted_scores, threshold):
    """
    Assigns classes on input based on given threshold.
    
    Args:
        predicted_scores -- list of floats, scores outputted by neural network
        threshold -- float, threshold
    
    Returns: list of class labels (ints)
    """
    return [1 if y >= threshold else 0 for y in predicted_scores]
    

# from metrics draw_from_file..
def predictions_from_file(in_file):
    """
    Retrieve true labels and predicted scores from file generated after validation.
    
    Args:
        in_file -- str, path to file
    
    Returns: labels, scores
    """
    true_labels = []
    predicted_scores = []
    with open(in_file, "r") as source1:
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
    
    return true_labels, predicted_scores


if __name__ == "__main__":   
    # 0. Get input
    input_file = argv[1]
    output_name = argv[2]                   # usually choose network: eg RNN92
    
    # 1. Get labels and scores
    labels, scores = predictions_from_file(input_file)
    
    # 2. Draw curves
    draw_roc_curve(labels, scores, output_name)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    draw_precision_recall_curve(precision, recall, thresholds, output_name)
    draw_precision_recall_threshold(precision, recall, thresholds, output_name)
    plot_pr_threshold(precision, recall, thresholds, output_name)
    
    # 3. Calculate F1 scores
    print(compute_f1(labels, scores))
