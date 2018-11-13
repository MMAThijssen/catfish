#! usr/bin/env python2
"""
Explore fast5 data using h5py.

Created on Wed Sep 12 09:27:12 2018
@author: thijs030
"""

import h5py
import numpy as np
import sys

def Process(fast5_path, runname):
    # Collate the attribute list
    hdf = h5py.File(fast5_path, 'r')
    # Get the names of all groups and subgroups in the file
    list_of_names = []
    hdf.visit(list_of_names.append)
    attribute = []
    for name in list_of_names:
        # Get all the attribute name and value pairs
        itemL = hdf[name].attrs.items()
        for item in itemL:
            attr, val = item
            if type(hdf[name].attrs[attr]) == np.ndarray:
                val = ''.join(hdf[name].attrs[attr])
            val = str(val).replace('\n', '')
            attribute.append([runname, name+'/'+attr, val])
    hdf.close()
    # Print the header
    print('{0}'.format('\t'.join(['runname', 'attribute', 'value'])))
    # Print the attribute list
    print('{0}'.format('\n'.join(['\t'.join([str(x) for x in item]) for item in attribute])))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage:   extractattr.py runname fast5_path')
        print('         Extract a table of attributes and values from a fast5 file.')
        print('')
        sys.exit(1)
    
    runname, fast5_path = sys.argv[1:]
    Process(fast5_path, runname)
