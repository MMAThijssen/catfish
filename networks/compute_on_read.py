#!/usr/bin/env python3 

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


def average_between_predictions(read):
    """
    Computes the average distance in measurements within a read.
    """
    search_read = True
    with open(output_file, "r") as source:
        for line in source:
            if search_read and not (line.startswith("#") or line.startswith("*") or line.startswith("@")):
                read_name = line.strip()
                if read_name in neg_list:
                    search_read = False
                    print(read_name)
            elif not search_read: 
                # get belonging predicted labels and true labels
                if predicted_labels == None:
                    predicted_labels = list_predicted(line, types="predicted_scores")
                    if predicted_labels != None:
                        predicted_scores = predicted_labels
                        predicted_labels = class_from_threshold(predicted_labels, threshold)
                        predicted_labels = list(correct_short(predicted_labels))
                        predicted_hp = hp_loc_dict(predicted_labels)
                        
                        # compute average length between reads
                        predicted_list = dict_to_ordered_list(predicted_hp)
                        
                    
