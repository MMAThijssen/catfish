from sys import argv 
import matplotlib
import matplotlib.pyplot as plt

fn_truehp = {0: (2, 10), 1: (99, 112)}
    
all_fnpos = []
[all_fnpos.extend(range(k[0], k[1] + 1)) for k in fn_truehp.values()] 
print(all_fnpos)
    
print(all_fnpos.extend(fn_positions))