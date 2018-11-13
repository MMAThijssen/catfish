#!/usr/bin/env python3
"""
Module to enhance reference genome and print information of homopolymers.

@author: Marijke Thijssen
"""
from info_hp import print_info, get_sequence, save_hp_loc, get_info, check_hp
import numpy as np
import random
from sys import argv

# 2. Enhance reference
def get_pos_hps(hp_loc_dict):
    """
    Returns list of homompolymer positions per base + adjacent bases
    """
    pos_list = []
    for locs in hp_loc_dict.values():
        for loc in range(len(locs)):
            pos_list += list(range(locs[loc][0], locs[loc][1]))
    return(pos_list)

def calc_hp(seq, threshold = 5):
    """
    Calculates homopolymer content as number of bases part of a homopolymer
    in total DNA sequence.
    
    seq -- str, DNA sequence
    threshold -- int, minimal number of equal bases to define 
                    homopolymer stretch [default:5]
    
    Returns: float (0.00 to 1.00)
    """
    # does not check if sequence is DNA
    hp_nr = 0  
    hp = False
    for i in range(len(seq) - threshold + 1):
        stretch = seq[i:i + threshold]
        if not hp and check_hp(stretch):
            hp = True
            hp_nr += 5
        elif hp and check_hp(stretch):
            hp_nr += 1
        else:
            hp = False
    hp_content = float(hp_nr) / float(len(seq))
    return(hp_content)
    

def substitute(seq, subst_seq, pos_list):
    """
    Substitutes part of sequence with a homopolymer at random position.
    
    Args:
        seq -- str, sequence
        subst_seq -- str, sequence to substitute with
        pos_list -- list of hp positions
        
    Returns: seq
    """
    if not subst_seq.count(subst_seq[0]) == len(subst_seq):
        raise ValueError("The substituting sequence is not a homopolymer.")
    not_substituted = True        
    while not_substituted:
        rand_pos = random.randint(1, len(seq) - len(subst_seq) - 1)  # to assure to have one base at begin and end      
        # assure that the hp is not equal to side            
        if not (subst_seq[0] == seq[rand_pos - 1] or subst_seq[0] == seq[rand_pos + len(subst_seq)]):
            # assure that HP will not be placed in other HP OR next to a HP
            if (rand_pos not in pos_list) or ((rand_pos + len(subst_seq) -1) not in pos_list):
                new_seq = seq[:rand_pos] + subst_seq + seq[rand_pos + len(subst_seq):]                
                not_substituted = False
    return(new_seq, rand_pos, rand_pos + len(subst_seq))
    
def create_progfile(file_name, prog, txt, ext="fasta"):
    """
    Creates progress file.

    Args:
        file_name -- str, name of output file
        prog -- str / int, progress
        txt -- str / int, text to be save in file
        ext -- str, file extension

    Returns: name of file (str)
    """
    out_file = "{}_{}.{}".format(file_name, prog, ext)
    with open(out_file, "w") as dest_file:
        dest_file.write(txt)
    return(out_file)
    
def enhance_seq(seq_file, perc, enhanced_name):
    """
    Enhances reference to specified percentage.
    
    Args:
        seq -- str, DNA sequence
        perc -- float, desired percentage of homopolymers
    
    Returns: file (str)
    """
    # check if enhancement is needed
    seq = get_sequence(seq_file)
    hp_loc_dict = save_hp_loc(seq)
    total_hps, total_hp_stretches, hp_cont, hp_lengths, different_hps, sorted_hp_nr = get_info(seq)
    if perc <= hp_cont * 100:
        print("Enhancement is not needed. Homopolymer content is {}% already. \
        (Requested was {}%)".format(round(hp_cont * 100, 2), perc))
        return(seq_file)
    # get needed stretches and number to copy:
    else:
        # to get HPs and proportions
        hp_list = [hpl[0] for hpl in sorted_hp_nr] # to get hps seperately
        nr_list = [hpr[1] for hpr in sorted_hp_nr] # to get according numbers
        prop_list = [float(nr)/total_hp_stretches for nr in nr_list]

        # to get enhancement properties
        bases_per_perc = len(seq) / 100
        stretch_per_perc = total_hp_stretches / (hp_cont * 100)
        needed_stretches = round(perc * stretch_per_perc)
        samples = np.random.choice(hp_list, size=needed_stretches, p=prop_list)  
        #new_stretches = needed_stretches - total_hp_stretches     
        
        # substitute original reference to over/undersample:
        #TODO: calculate percentage to be sure it reached defined percentage
        added_5perc = 5 * bases_per_perc
        added_hps = 0
        progress = 0
        hp_pos = get_pos_hps(hp_loc_dict)
        for smp in samples:
            seq, start, end = substitute(seq, smp, hp_pos)
            hp_pos += list(range(start, end))
            added_hps += len(smp)
            if added_hps >= added_5perc:
                progress += 1
                prog_file = create_progfile(enhanced_name, progress, seq)
                print("Created file {}.".format(prog_file))
                print("HP content: {}".format(calc_hp(get_sequence(prog_file))))
                added_hps -= added_5perc
        progress = "finished"
        prog_file = create_progfile(enhanced_name, progress, seq)
        print("Created file {}.".format(prog_file))
        return(prog_file)
        

if __name__ == "__main__":
    # 1. Get command line arguments
    if len(argv) != 4:
        print("Arguments is missing. Enter reference file, desired percentage, name of outfile.")
    else:
        seq_file = argv[1]
        wanted_perc = float(argv[2])
        enhanced_name = argv[3]

        # 2. Print HP info on reference
        print_info(seq_file, *get_info(get_sequence(seq_file)), extra=False)

        # 3. Enhance reference
        out_file = enhance_seq(seq_file, wanted_perc, enhanced_name) 

        # 4. Print HP info on enhanced sequence
        print_info(out_file, *get_info(get_sequence(out_file)), extra=False)   