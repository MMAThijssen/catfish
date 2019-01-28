#!/usr/bin/env python3

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv

plt.style.use("seaborn")
color1 = "mediumaquamarine"
color2 = "slateblue"
color3 = "forestgreen"
color4 = "indianred"
color5 = "darkorange"
color6 = "gold"
color7 = "palevioletred"
color8 = "midnightblue"
color9 = "lightslategray"
color10 = "m"
color11 = "y"
color12 = "lightpink"
colors = [color1, color2, color3, color4, color5, color6, color7, color8, color9, color10, color11, color12]

def scatterplot(x, y, clr, df, xlabel, ylabel, plot_name):
    """
    Creates scatter plot.
    
    Args:
        clr -- str, column name to color on
    """
    unique = list(set(df[clr]))
    use_colors = {unique[u]: colors[u] for u in range(len(unique))}
    clr_column = [use_colors[i] for i in df[clr]]
    plt.scatter(df[x], df[y], c=clr_column, alpha=0.7)
    #~ plt.plot([85.00, 85.00], [-0.01, 0.26], color="grey", linewidth=1, alpha=0.7)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(plot_name.replace("_", " "))
    colors_list = list(use_colors.keys())
    colors_list.sort()
    patchList = []
    for lbl in colors_list:
        data_key = mpatches.Patch(color=use_colors[lbl], label=lbl)
        patchList.append(data_key)
    plt.legend(handles=patchList, loc="center left", bbox_to_anchor=(1, 0.5))
    #~ plt.show()
    plt.savefig("{}.png".format(plot_name), bbox_inches="tight")
    plt.close()   
    

if __name__ == "__main__":
    input_file = argv[1]
    df = pd.read_csv(input_file, sep=",", header=0, index_col=1)
    
    hyperparams = ["type", "batch_size", "optimizer", "learning_rate", "layer_size", 
                   "keep_probability", "n_layers", "layer_size_res", "n_layers_res"]
                   
    hps_res = ["layer_size_res", "n_layers_res"]
    
    #~ for hp in hyperparams:
        # hyperparams vs f1:
        #~ scatterplot(x=hp, y="F1", clr="type", df=df,
                    #~ xlabel=hp.replace("_", " "), ylabel="F1 score", 
                    #~ plot_name="{}_vs_F1".format(hp))  
    
    #~ for hp in hps_res:
        # validation acc vs f1 colored on hp:
        #~ scatterplot(x="accuracy", y="F1", clr=hp, df=df,
                #~ xlabel="accuracy", ylabel="F1 score", 
                #~ plot_name="accuracy_vs_F1_on_{}".format(hp))
        
        #~ # hyperparams on val acc:
        #~ scatterplot(x=hp, y="validation_accuracy", clr="type", df=df,
                    #~ xlabel=hp.replace("_", " "), ylabel="validation accuracy", 
                    #~ plot_name="{}_vs_accuracy".format(hp)) 
                    
        # precision vs recall
        #~ scatterplot(x="recall", y="precision", clr=hp, df=df,
                    #~ xlabel="recall", ylabel="precision", 
                    #~ plot_name="precision_vs_recall_on_{}".format(hp)) 
    
    #~ # optimizers:
    scatterplot(x="type", y="F1", clr="optimizer", df=df,
                xlabel="network architecture", ylabel="F1 score", 
                plot_name="type_vs_F1_on_optimizer") 
    
    #~ # for layers + size:
    #~ scatterplot(x="n_layers", y="layer_size", clr="F1_corrected", df=df,
            #~ xlabel="n layers", ylabel="layer size", cmap="YlGnBu",
            #~ plot_name="n_layers_vs_layer_size") 
    
    print("Finished all plots")
    
    #~ # validation acc vs f1:
    #~ scatterplot(x="accuracy", y="F1", clr="type", df=df,
                #~ xlabel="accuracy", ylabel="F1 score", 
                #~ plot_name="accuracy_vs_F1_score")
