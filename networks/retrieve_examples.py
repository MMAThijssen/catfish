# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:40:05 2018

@author: thijs030
"""

import helper_functions
import reader
from sys import argv

if __name__ == "__main__":
    db_dir = argv[1]
    
    print("Start loading db.")
    db, squiggles = helper_functions.load_db(db_dir)
    print(squiggles)
    for squiggle in squiggles:
        x, y = reader.load_npz(squiggle)
        print(x[:100])
        print(y[:100])
        print(type(x))
        print(type(y))
   
    #~ train_set_x, train_set_y = db.get_training_set(5)
#    print(type(train_set_x[0]), type(train_set_y[0]))
    
#    test_x, test_y = db.get_training_set(1000, sets="test")

#~ #    print(type(train_set_x), type(train_set_y)) # tuple, tuple
    #~ print("Train set x")
    #~ print(train_set_x) # raw signal
    #~ print("Train set y ")
    #~ print(train_set_y) # 0 - neg; 1 - pos
