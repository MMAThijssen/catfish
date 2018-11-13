#!/usr/bin/env python3

"""
Module to parse cProfile output and generate plots
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os.path
from sys import argv

def parse_txt(cprofile, measure):
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
    # These are the "Tableau 20" colors as RGB.    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
      
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)
        
    # Common sizes: (10, 7.5) and (12, 14)    
    plt.figure(figsize=(12, 9))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_linestyle("--")
    ax.spines["bottom"].set_color("grey")  
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False) 
    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()
    # Make sure your axis ticks are large enough to be easily read.  
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
        
    # Limit the range of the plot to only where the data is.        
    plt.ylim(0, ymax)    
    plt.xlim(1, 10) 
  
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
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
    plt.savefig("{}.png".format(os.path.basename(measure)), bbox_inches="tight")
    #~ plt.show()
    
                
if __name__ == "__main__":
    measure = argv[1]
    file_list = argv[2:]
    measure_list = []
    for fl in file_list:
        accuracy = parse_txt(fl, measure)
        measure_list.append(accuracy)
    
    plot_networks_on_metric(measure_list, accuracy)


    
