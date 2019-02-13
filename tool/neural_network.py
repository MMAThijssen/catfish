#!/usr/bin/env python3

from models.resnet_class import ResNetRNN
from models.rnn_class import RNN

# TODO: adjust this to correct model! -- also network_type in second line  -- make default hpm_dict -- change path to point to be dependent on user

def build_model(network_type, saving=False, **kwargs):
    """
    Constructs neural network from given arguments.
    
    Args:
        network_type -- str, type of network: "RNN" or "ResNetRNN"
        
    Returns: network object
    """
    if network_type == "RNN":
        network = RNN(save=saving, **kwargs)
    
    elif network_type == "ResNetRNN":                   
        network = ResNetRNN(save=saving, **kwargs)
                        
    return network  
    
    
def load_network(network_type, path_to_network, checkpoint):
    hpm_dict = retrieve_hyperparams(path_to_network + "/ResNetRNN.txt")                # CHANGE later!
    #~ hpm_dict = {"batch_size": 128, "optimizer_choice": "RMSProp", "learning_rate": 0.001, 
                #~ "layer_size": 256, "n_layers": 4, "keep_prob": 0.2, "layer_size_res": 32, "n_layers_res": 4}
    model = build_model(network_type, **hpm_dict)    
    model.restore_network("{}/checkpoints".format(path_to_network), ckpnt="ckpnt-{}".format(checkpoint))

    return model
    
    
def retrieve_hyperparams(model_file, split_on=": "):
    """
    Retrieve hyperparameters from model file.
    
    Args:
        model_file -- str, path to file on model created by train_validate.build_model
        split_on -- str, combination of characters to split on [default: ": "]
    
    Returns: dict of hyperparameters
    """
    hpm_dict = {}
    with open(model_file, "r") as source:
        for line in source:
            if line.startswith("batch_size"):
                hpm_dict["batch_size"] = int(line.strip().split(split_on)[1])
            elif line.startswith("optimizer_choice"):
                hpm_dict["optimizer_choice"] = line.strip().split(split_on)[1]
            elif line.startswith("learning_rate"):
                hpm_dict["learning_rate"] = float(line.strip().split(split_on)[1])
            elif line.startswith("layer_size:"):
                hpm_dict["layer_size"] = int(line.strip().split(split_on)[1])
            elif line.startswith("n_layers:"):
                hpm_dict["n_layers"] = int(line.strip().split(split_on)[1])
            elif line.startswith("keep_prob"):
                hpm_dict["keep_prob"] = float(line.strip().split(split_on)[1])
            elif line.startswith("layer_size_res"):
                hpm_dict["layer_size_res"] = int(line.strip().split(split_on)[1])
            elif line.startswith("n_layers_res"):   
                hpm_dict["n_layers_res"] = int(line.strip().split(split_on)[1])
                
    return hpm_dict
