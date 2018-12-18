import tensorflow as tf

def write_metadata(filename, labels):
    """
    Create a metadata file image consisting of indices and labels.
    
    Args:
        filename -- str, name of the file to save on disk
        labels -- list of ints, labels
    """
    with open(filename, 'w+') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("{}\t{}\n".format(index, label))

    print('Metadata file saved in {}'.format(filename))
    

