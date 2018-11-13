valid_encoding_types = ['hp_class']

def is_valid_encoding_type(encoding_type):
    if encoding_type in valid_encoding_types:
        return True
    return False


def class_number(kmer, encoding_type):
    if encoding_type == 'hp_class':
        return hp_class_number(kmer)
    else:
        raise ValueError('encoding type not recognized')
        

def hp_class_number(kmer):
    """
    Classify homopolymer-containing k-mers using a binary class system.
    Non-homopolymer: 0; homopolymer: 1
    """

    k_length = len(kmer)
    # return 0 if kmer contains unknown base:
    if 'N' in kmer:
        return(0)
    elif kmer.count(kmer[0]) == k_length:
        return(1)
    else:
        return(0)


def extend_classification(classified_seq):
    """
    Extends classification to all bases part of homopolymer 
    as opposed to only middle base in 5-mer. 
    
    Args:
        classified_seq -- list of ints
    """
     # go forwards through seq  
    #TODO: adjust width = 4
    width = 4
    width_l = width // 2  # if width is even, pick point RIGHT of middle
    width_r = width - width_l
    for label in range(len(classified_seq)):
        if classified_seq[label] == 1:
            count = 0
            while count < width_l:
                idx = label - width_l + count
                if idx >= 0:
                    classified_seq[idx] = 1
                count += 1
    # go backwards through seq
    for label in reversed(range(len(classified_seq))):
        if classified_seq[label] == 1:
            count = 0
            while count < width_r:
                idx = label + width_r - count
                if idx < len(classified_seq):
                    classified_seq[idx] = 1
                count += 1
    return(classified_seq)


if __name__ == "__main__":
    print(hp_class_number("ABCDE"))
    print("------------------------------")
    print( hp_class_number("AAAAA"))
    print(hp_class_number("NNNNN"))