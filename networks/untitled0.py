# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:55:45 2018

@author: thijs030
"""
#
import helper_functions
from sys import argv
import reader
import datetime

#db = helper_functions.load_db(argv[1])
t1 = datetime.datetime.now()
squiggles = helper_functions.load_squiggles(argv[1])
t2 = datetime.datetime.now()

print("Loaded squiggels in {}".format(t2 - t1))
#target = argv[2]


for sgl in squiggles:
    t3 = datetime.datetime.now()
    _, labels = reader.load_npz(sgl)
    t4 = datetime.datetime.now()
    print("Loaded one npzs in {}".format(t4 - t3))