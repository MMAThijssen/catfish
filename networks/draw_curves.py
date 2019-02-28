#!/usr/bin/env python3 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics
import re
from sys import argv

def draw_f1(input_file, screenlog_file):
    """
    Draw F1 curve based on information logged to screen.
    
    Args:
        input_file -- str, path to input file
        screenlog_file -- str, path to logged screen file
        
    Returns: None
    """
    size_list = []
    # get network from file
    with open(screenlog_file, "r") as source:
        get_size = False
        for line in source:
            if get_size:
                size_list.append(int(line.strip()))
                get_size = False
            elif line.startswith("Saving network checkpoints to "):
                network_name = line.strip().rsplit("/", 1)[1]
                print(network_name)
            elif line.startswith("Training loss"):
                get_size = True
                
    # get validation measure
    confusion_lists = parse_txt_confusion(input_file)
    c_lists = []
    # correct for adding the numbers each round
    for i in reversed(range(1, len(confusion_lists))):
        clist = [confusion_lists[i][c] - confusion_lists[i -1][c] for c in range(len(confusion_lists[i]))]
        c_lists.append(clist)
    c_lists.append(confusion_lists[0])
    confusion_lists = list(reversed(c_lists))

    precision_recall_lists = [metrics.precision_recall(cl[0], cl[1], cl[3]) for cl in confusion_lists]
    n_list = [cl[0] + cl[3] for cl in confusion_lists]
    N_list = [sum(cl) for cl in confusion_lists]
    f1_list = [metrics.weighted_f1(precision_recall_lists[prl][0], precision_recall_lists[prl][1],
                                      n_list[prl], N_list[prl]) for prl in range(len(precision_recall_lists))]
    

    # draw plot
    draw_F1_plot(f1_list, size_list, network_name)
    
    print("Finished drawing plot.")                

def draw_accuracies(input_file, screenlog_file, measure="loss"):
    """
    Draws learning curve for defined measure based on logged screen file.
    
    Args:
        input_file -- str, path to input file
        screenlog_file -- str, path to logged screen file
        measure -- str, measure to depict [default: loss]
        
    Returns: None
    """    
    training = []
    validation = []
    sizes = []
    # get network from file
    with open(screenlog_file, "r") as source:
        get_size = False
        for line in source:
            if get_size:
                sizes.append(int(line.strip()))
                get_size = False
            elif line.startswith("Saving network checkpoints to "):
                network_name = line.strip().rsplit("/", 1)[1]
                print(network_name)
            # get training measures and sizes
            elif line.startswith("Training {}".format(measure)):
                train_acc = float(line.strip().split(": ")[1])
                training.append(train_acc)
            elif line.startswith("Validation {}".format(measure)):
                val_acc = float(line.strip().split(": ")[1])
                validation.append(val_acc)
            if line.startswith("Training loss"):
                get_size = True

    draw_learning_curves(training, validation, sizes, network_name, measure)
    print("Finished drawing plot.") 

def parse_txt_training(in_file, measure="accuracy"):
    """
    Parse screen log file.
    """
    training = []

    with open(in_file, "r") as source:
        for line in source:
                if line.startswith("Training {}".format(measure)):
                    train_acc = float(line.strip().split(": ")[1])
                    training.append(train_acc)

    return training
     

def parse_txt_confusion(in_file):
    """
    Extracts the number of TP, FP, TN, FN from .txt file.
    
    Args:
        in_file -- str, path to network output file
    
    Returns: list of list of TP, FP, TN, FN as ints
    """
    confusion_list = []
    
    re_digit = re.compile("\d")
    with open(in_file, "r") as source:
        for line in source:
            if "Detected" in line:
                line_list = line.strip().split()
                number_list = [int(l) for l in line_list if re_digit.search(l)]
                confusion_list.append(number_list)               

    return confusion_list      

def draw_F1_plot(f1_list, size_list, outname):
    plt.style.use("seaborn")
    matplotlib.rc("image", cmap="Pastel2")
    
    # draw plot
    plt.plot(size_list, f1_list, label="F1 score", c="gold")    
    plt.title("F1 score")  
    plt.ylim(bottom=0)
    plt.ylabel('F1 score')
    plt.xlabel("number of training examples")          # should adjust!
    plt.xticks(rotation=75)
    plt.savefig("F1_{}.png".format(outname), bbox_inches="tight")
#    plt.show()   
    plt.close()
    

def draw_learning_curves(training_scores, validation_scores, train_sizes, 
                         network_name, measure="accuracy"):
    """
    Plots learning curve. 
    
    Args:
        training_score -- list of lists of float/ints
        validation_score -- list of lists of float/ints
        train_sizes -- list of ints
        img_title -- str, name and title of figure
        network_name -- str, name of network; should contain biGRU-RNN or ResNetRNN
        measure -- str, 'accuracy' or 'loss' [default: accuracy]
        
    """    
    plt.style.use("seaborn")
    
    if "biGRU-RNN" in network_name:
        c = "darkcyan"
        c2 = "paleturquoise"
    if "ResNet-RNN" in network_name:
        c = "forestgreen"
        c2 = "lightgreen"
    
    #~ train_means, train_mins, train_maxs = compute_lines(training_scores)
    #~ val_means, val_mins, val_maxs = compute_lines(validation_scores)
    
    plt.plot(train_sizes, training_scores, label = 'training', color=c)
    plt.plot(train_sizes, validation_scores, label = 'validation', color=c2)
    #~ plt.fill_between(train_sizes, train_mins, train_maxs, alpha=0.3)
    #~ plt.fill_between(train_sizes, val_mins, val_maxs, alpha=0.3)

    plt.ylabel(measure, fontsize = 14)
    plt.xlabel('number of training examples', fontsize = 14)
    title = "Learning curve"
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.savefig("LC_{}-{}.png".format(network_name, measure[:2]), bbox_inches="tight")
    plt.close()
    #~ plt.show()


if __name__ == "__main__":
    #~ input_file = argv[1]        # eg  biGRU-RNN_89.txt
    #~ screenlog_file = argv[2]
    #~ sizes_file = argv[3]
    #~ output_name = argv[4]
    #~ measure = argv[5]
    #~ main(input_file, screenlog_file, sizes_file, output_name, measure)
    in_file = argv[1]
    screenlog_file = argv[2]
    #~ draw_f1(in_file, screenlog_file)
    draw_accuracies(in_file, screenlog_file)
