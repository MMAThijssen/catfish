#/usr/bin/env python3

"""
Retrieves information on homopolymers from FASTA or txt file.

@author: Marijke Thijssen
"""

from sys import argv


def get_sequence(fasta_file):
    """ 
    Retrieves sequence from FASTA or txt file.
    
    Args:
        fasta_file -- string, name of FASTA file
    
    Returns: dict {id: list of sequences}
    """
    seq_dict = {}
    seq_nr = 0
    with open(fasta_file, "r") as dest_file:
        for line in dest_file:
            if not line.strip():
                continue
            elif line.startswith(">"):
                seq_nr += 1
                seq_dict[seq_nr] = []
            else:
                seq_dict[seq_nr].append(line.strip())
    return seq_dict

# 2. Get homopolymer content and other info
def check_hp(seq, length = 5):
    """
    Checks if sequence of defined length is a homopolymer. 
    
    Args:
        seq -- str, DNA sequence
        length -- int, number of equal bases that defines 
                        homopolymer stretch [default:5]
    
    Returns: boolean (true if homopolymer, false if not homopolymer)
    """
    if len(seq) != length:
        raise ValueError("Provided sequence must have specified length.")
    if seq.count(seq[0]) == length:
            return(True)
    else:
        return(False)

def save_hp_loc(seq, threshold = 5):
    """
    Saves locations of homopolymers in a dictionary.
    
    Args:
        seq -- str, DNA sequence
        threshold -- int, minimal number of equal bases to define 
                        homopolymer stretch [default:5]
    
    Returns: dict {hp (str): positions (list of tuples)}
    """
    hp_dict = {}
    is_hp = False
    for i in range(len(seq) - threshold + 1):
        stretch = seq[i:i + threshold]
        if not is_hp and check_hp(stretch):
            is_hp = True            
            hp_start = i
            hp_end = i + threshold
        elif is_hp and check_hp(stretch):
            hp_end = i + threshold
        if is_hp and ((not check_hp(stretch)) or i == len(seq) - threshold):
            is_hp = False
            hp = seq[hp_start: hp_end]
            if hp in hp_dict:
                hp_dict[hp] += [(hp_start, hp_end)]
            else:
                hp_dict[hp] = [(hp_start, hp_end)] #copy parts as seq[start:end]
    return(hp_dict)

def calc_hp_bases(hp_dict):
    """
    Calculates homopolymer content as number of bases part of a homopolymer
    based on dict.
    
    Args:
        hp_dict -- dict {str (hp): list of tuples (loc)}
        seq -- str, DNA sequence
    
    Returns: total HP bases (int)
    """
    total_hps = 0    
    for hp in hp_dict:
        length = len(hp)
        number = len(hp_dict[hp])
        hps = length * number
        total_hps += hps
    return(total_hps)    # hp content

def calc_hpcontent(nr_hp_bases, seq):
    """
    Returns HP content as float. 
    """
    return(nr_hp_bases / float(len(seq)))

def len_hp(hp_dict):
    """
    Orders lengths of hp on length. 
    
    Args:
        hp_dict -- dict, {seq (str), ...}
        
    Returns: list of ints
    """
    
    hp_list = list(hp_dict.keys())
    size_list = [len(hp) for hp in hp_list]
    return(sorted(set(size_list))) 

def get_hp_count(hp_loc_dict):
    # to get sorted list (HP, count)    
    hp_nr_list = [(hp, len(hp_loc_dict[hp])) for hp in hp_loc_dict]
    sorted_hp_nr = sorted(hp_nr_list)
    return(sorted_hp_nr)  

def get_info(seq):
    """
    Gets information on homopolymers.
    
    Returns: total number of HP bases (int), total number of HP stretches (int),
             HP content (float), number of different HPs (int)
    """
    hp_loc_dict = save_hp_loc(seq)
    # get HP content and number of HP bases
    total_hps = calc_hp_bases(hp_loc_dict) 
    hp_content = calc_hpcontent(total_hps, seq)
    total_hp_stretches = sum([len(hp) for hp in hp_loc_dict.values()])
    # get HP lengths and number of different HPs
    hp_lengths = len_hp(hp_loc_dict)
    different_hps = len(hp_loc_dict.keys())
    hp_count_tuple = get_hp_count(hp_loc_dict)
    return(total_hps, total_hp_stretches,
                     hp_content, hp_lengths, different_hps, hp_count_tuple)

# 3. Print info to screen
def print_info(seq_file, total_hps, total_hp_stretches,
                     hp_cont, hp_lengths, different_hps, sorted_hp_nr, extra=False):
#def print_hp_info(seq_file):
    """
    Prints general information on homopolymers in a sequence.
    
    Args:
        seq_file -- str, full (!) path to file
        total_hps -- int
        total_hp_stretches -- int
        hp_cont -- float
        hp_lengths -- int
        different_hps -- int
        sorted_hp_nr -- list of tuples (hp, nr)
        extra -- bool, prints HP and prevelance [default:True]
    
    Returns: -
    """
    hp_info = "SEQUENCE: {}\n" \
        "Total number of bases in homopolymers: {}\n" \
        "Total number of homopolymer stretches: {}\n" \
        "Homopolymer content (0 - 1): {}\n" \
        "Homopolymer lengths: {}\n" \
        "Max number of different homopolymers: {}" 
    if "/" in seq_file:
        seq_file = seq_file.split("/")[-1]
    print(hp_info.format(seq_file, total_hps, total_hp_stretches,
                         hp_cont, hp_lengths, different_hps))
    if extra:
        print("Homopolymer: \tPrevelance:")                     
        for hpnr in sorted_hp_nr:
            print(hpnr[0] + "\t" + str(hpnr[1]))  

if __name__ == "__main__":
    # 1. Get sequence from file
#    if len(argv) == 3:
#        if argv[2] == "False":
#            extra = False
#        else:
#            extra = True

    seq_dict = get_sequence(argv[1])
    n_seq = len(seq_dict)
    sequences = list(seq_dict.values())
    seq_length = 0
    for seqs in sequences:
        for seq in seqs:
            seq_length += len(seq)
#    print("Total sequence length: {}".format(seq_length))

    # 2. Get homopolymer content and other info
    all_hps = 0
    all_stretches = 0
    all_content = 0
    all_lengths = [] #TODO
    all_different_hps = [] #TODO
    all_hp_count_tuple = [] #TODO
    for sq in seq_dict:
        seq = "".join(seq_dict[sq])
        total_hps, total_hp_stretches, hp_content, hp_lengths, different_hps, hp_count_tuple = get_info(seq)
        all_hps += total_hps
        all_stretches += total_hp_stretches        
        all_content += hp_content
        for lengths in hp_lengths:
            all_lengths.append(lengths)
        all_different_hps.append(different_hps)
    
    final_content = all_content / n_seq
    lengths = list(set(all_lengths))
    diff_hp = max(all_different_hps)
  
    # 3. Print info to screen
    print_info(argv[1], all_hps, all_stretches, final_content, lengths, 
               diff_hp, all_hp_count_tuple)      # no extra for now
    print("Total sequence length: {}\nHP count: {}".format(seq_length, 
          all_hps / seq_length))
