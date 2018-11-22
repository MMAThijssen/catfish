import datetime
import random
import numpy as np

def generate_random_hyperparameters(network_type,
                                    learning_rate_min=-4,      
                                    learning_rate_max=0,
                                    optimizer_list=["Adam", "RMSProp"],
                                    layer_size_list=[16, 32, 64, 128, 256],
                                    n_layers_min=1,
                                    n_layers_max=6,   
                                    batch_size_list=[64, 128, 256],
                                    dropout_min=0.2,
                                    dropout_max=0.8,
                                    n_layers_res_min=1,
                                    n_layers_res_max=12,
                                    size_layers_res_list=[16, 32, 64, 128, 256]):
    """
    Generates random hyperparameters
    
    Args:
        network_type -- str, type of network (empty (RNN) or "ResNetRNN")
        learning_rate_min -- int, minimal negative number in exponential for learning rate [default: -4]
        learning_rate_max -- int, maximal positive number in exponential for learning rate [default: 0]
    Returns: dict of hyperparameters
    """   
    # pick random hyperparameter:
    learning_rate = 10 ** np.random.randint(learning_rate_min, learning_rate_max)
    optimizer = np.random.choice(optimizer_list)
    layer_size = np.random.choice(layer_size_list)
    n_layers = np.random.randint(n_layers_min, n_layers_max)
    batch_size = np.random.choice(batch_size_list)
    dropout = round(np.random.uniform(dropout_min, dropout_max), 1)
    
    # create dict:
    hpm_dict = {"batch_size": batch_size, "optimizer_choice": optimizer, 
                "learning_rate": learning_rate, "layer_size": layer_size, 
                "n_layers": n_layers, "keep_prob": dropout}
    
    # extend dict if network is ResNetRNN:            
    if network_type == "ResNetRNN":
        n_layers_res = np.random.randint(n_layers_res_min, n_layers_res_max)
        size_layers_res = np.random.choice(size_layers_res_list)
        
        hpm_dict["layer_size_res"] = size_layers_res
        hpm_dict["n_layers_res"] = n_layers

    return hpm_dict

print(random.seed())
for i in range(50):
    print(generate_random_hyperparameters("RNN"))