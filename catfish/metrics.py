#!/usr/bin/env python3

def confusion_matrix(true_labels, predicted_labels):
    """
    Returns precision and recall.
    
    Args:
        true_labels -- list of ints, true labels
        predicted_labels -- list of int, predicted labels
        
    Returns: number of true positives, false positives, true_negatives, false_negatives
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
