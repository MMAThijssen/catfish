#!/usr/bin/env python3 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics
import re
from sys import argv

def draw_f1(input_file, screenlog_file):
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
    
    #~ # search for according sizes in size file
    #~ with open(sizes_file, "r") as size_source:
        #~ found_network = False
        #~ for line in size_source:
            #~ if network_name in line:
                #~ found_network = True
            #~ elif found_network:
                #~ size_list = line.strip().split(", ")
                #~ break
                
    # for RNN 1 size list:
    #~ size_list = [102400, 204800, 307200, 409600, 512000, 614400, 716800, 819200, 921600, 1024000, 1126400, 1228800, 1331200, 1433600 , 1469440, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624, 27648, 28672, 29696, 30720, 31744, 32768, 33792, 34816, 35840, 36864, 37888, 38912, 39936, 40960, 41984, 43008, 44032, 45056, 46080, 47104, 48128, 49152, 50176, 51200, 52224, 53248, 54272, 55296, 56320, 57344, 58368, 59392, 60416, 61440, 62464, 63488, 64512, 5120000, 10240000, 15277568, 10240, 20480, 30720, 40960, 51200, 61440, 71680, 81920, 92160, 102400, 112640, 122880, 133120, 143360, 153600, 163840, 174080, 184320, 194560, 204800, 215040, 225280, 235520, 245760, 256000, 266240, 276480, 286720, 296960, 307200, 317440, 327680, 337920, 348160, 358400, 368640, 378880, 389120, 399360, 409600, 419840, 430080, 440320, 450560, 460800, 471040, 481280, 491520, 501760, 512000, 522240, 532480, 542720, 552960, 563200, 573440, 583680, 593920, 604160, 614400]
    
    # draw plot
    draw_F1_plot(f1_list, size_list, network_name)
                

def draw_accuracies(input_file, screenlog_file, measure="loss"):
    
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
    
    #~ with open(input_file, "r") as network_source:
        #~ get_accuracy = False
        #~ for line in network_source:
            #~ if not get_accuracy and "Average performance" in line:
                #~ get_accuracy = True
            #~ elif get_accuracy:
                #~ if measure[1:] in line:
                    #~ validation.append(float(line.strip().split()[1][:-1]) / 100)
                    #~ get_accuracy = False

    draw_learning_curves(training, validation, sizes, network_name, measure)


def parse_txt_training(in_file, measure="accuracy"):
    """
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



#~ # ORIGINAL ONE - could be used if screenlog file had correct output..    
#~ def parse_txt(in_file, measure="accuracy"):
    #~ """
    #~ """
    #~ training = []
    #~ validation = []
    #~ sizes = []
    
    #~ new_round = True
    #~ get_size = False
    #~ with open(in_file, "r") as source:
        #~ for line in source:
            #~ if new_round:
                #~ if get_size:
                    #~ size = int(line.strip())
                    #~ get_size = False
                #~ if line.startswith("Training {}".format(measure)):
                    #~ train_acc = float(line.strip().split(": ")[1])
                #~ elif line.startswith("Validation {}".format(measure)):
                    #~ val_acc = float(line.strip().split(": ")[1])
                    #~ new_round = False
                #~ if line.startswith("Training loss"):
                    #~ get_size = True
            #~ else:
                #~ training.append(train_acc)
                #~ validation.append(val_acc)
                #~ sizes.append(size)
                #~ new_round = True
    #~ return training, validation, sizes 
    

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
