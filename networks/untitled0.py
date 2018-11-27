# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:55:45 2018

@author: thijs030
"""
#
#import helper_functions
#from sys import argv
##from ExampleDb import check_lengths
#
#db = helper_functions.load_db(argv[1])

window = 4      # actually 5

final = 10
width_l = window // 2
width_r = window - width_l

l = range(final)
print(l)

signals = [idx for idx in range(width_l, final - width_r + 1)]
print(signals)
print(signals[0], signals[-1])
cur_idx = 8
start = cur_idx - width_l
end = cur_idx + width_r + 1
print(start, end, l[start : end])