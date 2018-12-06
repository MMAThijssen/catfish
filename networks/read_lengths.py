# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:55:45 2018

@author: thijs030
"""

import helper_functions
from sys import argv
import reader
import datetime
import numpy as np
import statistics

#db = helper_functions.load_db(argv[1])
t1 = datetime.datetime.now()
squiggles = helper_functions.load_squiggles(argv[1])
t2 = datetime.datetime.now()

print("Loaded squiggels in {}".format(t2 - t1))
#target = argv[2]
t3 = datetime.datetime.now()
lengths = []
for sgl in squiggles:
    _, labels = reader.load_npz(sgl)
    lengths.append(len(labels))
t4 = datetime.datetime.now()
print("Loaded all npzs in {}".format(t4 - t3))
lengths.sort()

under5000 = [l for l in lengths if l >= 1000]
under10000 = [l for l in lengths if l >= 10000] 
under15000 = [l for l in lengths if l >= 100000]
under20000 = [l for l in lengths if l >= 200000]
under25000 = [l for l in lengths if l >= 300000]
under30000 = [l for l in lengths if l >= 400000]

print(len(under5000))
print(len(under10000))
print(len(under15000))
print(len(under20000))
print(len(under25000))
print(len(under30000))   


#print(sum(lengths) / len(squiggles))
#print(statistics.median(lengths))

