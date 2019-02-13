#!/usr/bin/env python3

from base_to_signal import get_base_new_signal
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trainingDB.metrics import generate_heatmap
import numpy as np
from reader import load_npz_labels
from statistics import median
from sys import argv

def check_for_neg(output_file, main_dir, npz_dir, out_name, threshold=0.8):
    """
    Outputs information on all (true and false) positives in predicted output. 
    
    Args:
        output_file -- str, file outputted by neural network validation
        main_dir -- str, path to main directory containing FAST5 files
        out_name -- str, name of file to write output to
        start -- int, start position taken at validation
        length -- int, length of stretch that was validated
        max_nr -- int, maximum number of reads to use [default: 12255]
        
    Returns: None
    """
    show_plots = True
    is_confusion = True
    predicted_labels = None
    true_labels = None
    
    read_counter = 0
    
    n_bases = 0    # is a check
    
    search_read = True
    with open(output_file, "r") as source:
        for line in source:
            if search_read and not (line.startswith("#") or line.startswith("*") or line.startswith("@")):
                read_name = line.strip()
                search_read = False
                print(read_name)
            elif not search_read: 
                # get belonging predicted labels and true labels
                if predicted_labels == None:
                    predicted_labels = list_predicted(line, types="predicted_scores")
                    if predicted_labels != None:
                        predicted_scores = predicted_labels
                        predicted_labels = class_from_threshold(predicted_labels, threshold)
                        length = len(predicted_labels)
                if true_labels == None:    
                    #~ true_labels = list_predicted(line, types="true_labels")
                    true_labels = list(load_npz_labels("{}/{}.npz".format(npz_dir, read_name)))
                    #~ print(true_labels)
                elif predicted_labels != None and true_labels != None:
                    bases, new = get_base_new_signal("{}/{}.fast5".format(main_dir, read_name))
                    bases = bases[start: start + length]
                    new = new[start: start + length]
                    count_basen = [1 for w in new if w == "n"]
                    n_bases += sum(count_basen) 

                    true_labels = true_labels[start: start + length]
                    #~ predicted_labels = list(correct_short(predicted_labels))
                    
                    #~ # generate heatmap
                    if read_counter < 10:
                        generate_heatmap([correct_short(predicted_labels), true_labels, predicted_scores], ["predicted", "truth", "confidences"],
                                        "Comparison_{}_corrected".format(read_name)) 
                    
                    read_counter += 1
                    if read_counter == 10:
                        break
                    search_read = True
                    predicted_labels = None
                    true_labels = None

def main(output_file, main_dir, npz_dir, out_name, threshold=0.5, start=0, max_nr=12255):
    """
    Outputs information on all (true and false) positives in predicted output. 
    
    Args:
        output_file -- str, file outputted by neural network validation
        main_dir -- str, path to main directory containing FAST5 files
        out_name -- str, name of file to write output to
        threshold -- float, threshold for classification labels [default: 0.5]
        start -- int, start position taken at validation [default: 0]
        length -- int, length of stretch that was validated [default: 4970]
        max_nr -- int, maximum number of reads to use [default: 12255]
        
    Returns: None
    """       
    show_plots = False
    is_confusion = True                                                         # was on measurements
    on_measurements = False
    if on_measurements:
        on_hps = False
    else:
        on_hps = True
        print("Results on HPs instead of measurements")
    predicted_labels = None
    true_labels = None
    
    read_counter = 0
    
    states = []
    false_states = []
    neg_states = []

    # initialize for all predicted and true positives                           # TODO: do not calculate but take TP + FP together for this / TP + FN
    all_count_predictions = Counter({})
    all_count_basehp = Counter({})
    all_count_truehp = Counter({})
    all_count_basetrue = Counter({})
    all_count_seq = Counter({})
    all_count_seqtrue = Counter({})
    all_truepositions = []
    all_predictedpositions = []

    if is_confusion:
        # initialize for TP
        all_count_tptrue = Counter({})
        all_count_tpbasestrue = Counter({})
        all_count_tpseqtrue = Counter({})
        all_tppositions = []
        
        # initialize for FN
        all_count_fntrue = Counter({})
        all_count_fnbasestrue = Counter({})
        all_count_fnseqtrue = Counter({})
        all_fnpositions = []

        # initialize for FP
        all_count_fptrue = Counter({})
        all_count_fpbasestrue = Counter({})
        all_count_fpseqtrue = Counter({})
        all_fppositions = []
        
        # initialize for TN
        all_count_tntrue = Counter({})
        all_count_tnbasestrue = Counter({})
        all_count_tnseqtrue = Counter({})
        all_tnpositions = []
    
    n_bases = 0                                                                 # is a check
    
    search_read = True
    with open(output_file, "r") as source:
        for line in source:
            if search_read and not (line.startswith("#") or line.startswith("*") or line.startswith("@")):
                read_name = line.strip()
                search_read = False
                print(read_name)
            elif not search_read: 
                # get belonging predicted labels and true labels
                if predicted_labels == None:
                    predicted_labels = list_predicted(line, types="predicted_scores")
                    if predicted_labels != None:
                        predicted_scores = predicted_labels
                        predicted_labels = class_from_threshold(predicted_labels, threshold)
                        length = len(predicted_labels)
                if true_labels == None:    
                    #~ true_labels = list_predicted(line, types="true_labels")
                    true_labels = list(load_npz_labels("{}/{}.npz".format(npz_dir, read_name)))
                    #~ print(true_labels)
                elif predicted_labels != None and true_labels != None:
                    bases, new = get_base_new_signal("{}/{}.fast5".format(main_dir, read_name))
                    bases = bases[start: start + length]
                    new = new[start: start + length]
                    count_basen = [1 for w in new if w == "n"]
                    n_bases += sum(count_basen) 

                    true_labels = true_labels[start: start + length]
                    predicted_labels = list(correct_short(predicted_labels))

                    #~ # generate heatmap
                    #~ if read_counter == 1:
                        #~ generate_heatmap([predicted_labels, true_labels, predicted_scores], ["predicted", "truth", "confidences"],
                                            #~ "Comparison_{}".format(read_name))

                        #~ generate_heatmap([correct_short(predicted_labels), true_labels, predicted_scores], ["predicted", "truth", "confidences"],
                                            #~ "Comparison_{}_corrected".format(read_name)) 
                    
                    # save information to dict
                    predicted_hp = hp_loc_dict(predicted_labels)
                    true_hp = hp_loc_dict(true_labels)
                    detected_from_true(predicted_hp, true_hp)
                    predicted_vs_true(predicted_labels, true_labels, read_name)
                    
                    # check all positives:      real positives + predicted positives
                    if true_hp:
                        count_truehp, count_basetrue, count_seqtrue = prediction_information(true_hp, bases, new)       
                        [all_truepositions.extend(range(k[0], k[1] + 1)) for k in true_hp.values()]
                        all_count_truehp = all_count_truehp + count_truehp
                        all_count_basetrue = all_count_basetrue + count_basetrue
                        all_count_seqtrue = all_count_seqtrue + count_seqtrue  
                    if predicted_hp:
                        count_predictions, count_basehp, count_seq = prediction_information(predicted_hp, bases, new)        
                        all_count_predictions = all_count_predictions + count_predictions
                        all_count_basehp = all_count_basehp + count_basehp
                        all_count_seq = all_count_seq + count_seq
                        [all_predictedpositions.extend(range(k[0], k[1] + 1)) for k in predicted_hp.values()]
                    
                    # check finding back true HP - gives TP and FP (+ partials)
                    read_states = [check_hp(predicted_hp[hp], true_labels, hp) for hp in predicted_hp] 
                    if read_states != []:
                        states.extend(read_states)    
                    
                    # check all true negatives - gives FN and TN (+ partials)
                    fake_states = [check_hp(true_hp[hp], predicted_labels, hp) for hp in true_hp] 
                    if fake_states != []:
                        neg_states.extend(fake_states) 

                    # add dictionaries to make common:                    
                    if is_confusion:
                        if on_measurements:
                        # generate dicts that contain only TP, FN, TN, FN:   
                            tp_truehp = generate_dict(predicted_labels, true_labels, pred_label=1, true_label=1)
                            fn_truehp = generate_dict(predicted_labels, true_labels, pred_label=0, true_label=1)
                            fp_truehp = generate_dict(predicted_labels, true_labels, pred_label=1, true_label=0)
                            tn_truehp = generate_dict(predicted_labels, true_labels, pred_label=0, true_label=0)
                            
                        if on_hps:                            
                            tp_truehp = create_confdict(predicted_hp, read_states, "tp")
                            fn_truehp = create_confdict(true_hp, fake_states, "fn")
                            fp_truehp = create_confdict(predicted_hp, read_states, "fp")
                        
                        # calculate lengths, composition and positions
                        if tp_truehp != 0:
                            [all_tppositions.extend(range(k[0], k[1] + 1)) for k in tp_truehp.values()]
                            count_tptrue, count_tpbasestrue, count_tpseqtrue = prediction_information(tp_truehp, bases, new)          # true positives
                            #~ for hp in tp_truehp:
                                #~ print(tp_truehp[hp])
                                #~ print(true_labels[tp_truehp[hp][0]: tp_truehp[hp][1] + 1])
                                #~ print(predicted_labels[tp_truehp[hp][0] : tp_truehp[hp][1] + 1])
                                #~ print(bases[tp_truehp[hp][0] : tp_truehp[hp][1] + 1])
                                #~ print(new[tp_truehp[hp][0] : tp_truehp[hp][1] +1])
                                #~ fake_new = new[tp_truehp[hp][0] : tp_truehp[hp][1] +1]
                                #~ fake_bases = bases[tp_truehp[hp][0] : tp_truehp[hp][1] +1]
                                #~ seq = [fake_bases[i] for i in range(len(fake_new)) if fake_new[i] == "n"]
                                #~ print("".join(seq))
                                    
                        if fn_truehp != 0:
                            [all_fnpositions.extend(range(k[0], k[1] + 1)) for k in fn_truehp.values()]                             # check positions
                            count_fntrue, count_fnbasestrue, count_fnseqtrue = prediction_information(fn_truehp, bases, new)          # false negatives
                            #~ for hp in fn_truehp:
                                #~ print(fn_truehp[hp])
                                #~ print(true_labels[fn_truehp[hp][0]: fn_truehp[hp][1] + 1])
                                #~ print(predicted_labels[fn_truehp[hp][0] : fn_truehp[hp][1] + 1])
                                #~ print(bases[fn_truehp[hp][0] : fn_truehp[hp][1] + 1])
                                #~ print(new[fn_truehp[hp][0] : fn_truehp[hp][1] +1])
                                #~ fake_new = new[fn_truehp[hp][0] : fn_truehp[hp][1] +1]
                                #~ fake_bases = bases[fn_truehp[hp][0] : fn_truehp[hp][1] +1]
                                #~ seq = [fake_bases[i] for i in range(len(fake_new)) if fake_new[i] == "n"]
                                #~ print("".join(seq))
                                                            
                        if fp_truehp != 0:
                            count_fptrue, count_fpbasestrue, count_fpseqtrue = prediction_information(fp_truehp, bases, new)         # false positives
                            [all_fppositions.extend(range(k[0], k[1] + 1)) for k in fp_truehp.values()]
                            #~ print(count_fpseqtrue)
                            #~ print(fp_truehp)
                        # to check:
                            #~ st = 2522
                            #~ ed = 2554
                            #~ if read_name == "nanopore2_20170301_FNFAF09967_MN17024_sequencing_run_170301_MG1655_PC_RAD002_62645_ch194_read5177_strand":
                                #~ print(fp_truehp)
                                #~ st_hp = 11341
                                #~ end_hp = 11401
                                #~ print(true_labels[st_hp: end_hp])
                                #~ print(predicted_labels[st_hp : end_hp])
                                #~ print(bases[st_hp : end_hp])
                                #~ print(new[st_hp : end_hp])
                                #~ fake_new = new[st_hp : end_hp]
                                #~ fake_bases = bases[st_hp : end_hp]
                                #~ seq = [fake_bases[i] for i in range(len(fake_new)) if fake_new[i] == "n"]
                                #~ print("".join(seq))
                                #~ 3/0

                        # add to dictionaries to make common:
                        # for TP:
                        if tp_truehp != 0: 
                            all_count_tptrue = all_count_tptrue + count_tptrue
                            all_count_tpbasestrue = all_count_tpbasestrue + count_tpbasestrue
                            all_count_tpseqtrue = all_count_tpseqtrue + count_tpseqtrue
                        
                        # for FN:
                        if fn_truehp != 0: 
                            all_count_fntrue = all_count_fntrue + count_fntrue
                            all_count_fnbasestrue = all_count_fnbasestrue + count_fnbasestrue
                            all_count_fnseqtrue = all_count_fnseqtrue + count_fnseqtrue

                        # for FP:
                        if fp_truehp != 0: 
                            all_count_fptrue = all_count_fptrue + count_fptrue
                            all_count_fpbasestrue = all_count_fpbasestrue + count_fpbasestrue
                            all_count_fpseqtrue = all_count_fpseqtrue + count_fpseqtrue

                    read_counter += 1
                    if read_counter == max_number:
                        break
                    search_read = True
                    predicted_labels = None
                    true_labels = None
                    
            else:
                continue
    
    # 6b. Check from TP perspective:                    
    # count how many completely found back
    print(len(states), len(neg_states))
    print(n_bases)
    complete_st = [1 for st in states if st[0] == "complete"]                   # TP
    complete_st = sum(complete_st)
        
    # count how many absent:                                                    # FP
    absent_st = [1 for st in states if st[0] == "absent"]
    absent_st = sum(absent_st)
    
    # count number of partials:
    incomplete_st = len(states) - complete_st - absent_st                       # partial TPs
    
    # count interruptions in true hps:
    list_inters = [len(st[2]) for st in neg_states]
    nr_inters = sum(list_inters)
    if nr_inters != 0:
        avg_inters = nr_inters / incomplete_st                                 
        median_inters = median(list_inters)  
        length_inters = [(st[2][0][1] - st[2][0][0] + 1) for st in neg_states if st[2] != []]       # only take interruptions into account
        avg_len_inters = sum(length_inters) / nr_inters
        median_len_inters = median(length_inters)
    
        # TODO: something with positions of interruptions?

    if incomplete_st + complete_st != 0:
        overleft = [st[1][0] for st in states if st[0] != "absent"]
        overright = [st[1][1] for st in states if st[0] != "absent"]
        perfectcomplete = [1 for st in states if (st[0] == "complete" and (st[1][0] == 0 and st[1][1] == 0))]
        avg_overleft = sum(overleft) / (incomplete_st + complete_st)
        avg_overright = sum(overright) / (incomplete_st + complete_st)
        perfectcomplete = sum(perfectcomplete)
    
    full_false_negs = [1 for st in neg_states if st[0] == "absent"]             # FN
    full_false_negs = sum(full_false_negs)
    
    #~ partial_false_negs = len(neg_states) - true_negs  - full_false_negs                 # partial FPs

    #~ # generate size list to calc avg and median length of measurements and bases:
    #~ hp_size_list = []
    #~ base_size_list = []
    #~ all_truehp_list = dict_to_ordered_list(all_count_truehp)
    #~ for t in all_truehp_list:
        #~ hp_size_list.extend(t[1] * [t[0],])
    #~ all_basetrue_list = dict_to_ordered_list(all_count_basetrue)
    #~ for t in all_basetrue_list:
        #~ base_size_list.extend(t[1] * [t[0],])
    
    # order all counts to get nice graphs:
    all_seq_list = dict_to_ordered_list(all_count_seq)
    all_seqtrue_list = dict_to_ordered_list(all_count_seqtrue)
    prev_bases = count_bases(all_seq_list)
    prev_basestrue = count_bases(all_seqtrue_list)
    all_count_truepositions = dict_to_ordered_list(Counter(all_truepositions))
    
    if is_confusion:
        all_fnseqtrue = dict_to_ordered_list(all_count_fnseqtrue)
        all_count_fnpositions = dict_to_ordered_list(Counter(all_fnpositions))
        
        all_fpseqtrue = dict_to_ordered_list(all_count_fpseqtrue)
        all_count_fppositions = dict_to_ordered_list(Counter(all_fppositions))
        
        all_tpseqtrue = dict_to_ordered_list(all_count_tpseqtrue)
        all_count_tppositions = dict_to_ordered_list(Counter(all_tppositions))

        fn = 0
        for a in all_count_fntrue:
            fn += all_count_fntrue[a]
        fp = 0
        for a in all_count_fptrue:
            fp += all_count_fptrue[a]
        tp = 0
        for a in all_count_tptrue:
            tp += all_count_tptrue[a]

    # 8. Write outputs to file:
    with open("{}.txt".format(out_name), "w") as dest:
        dest.write("PROCESSED VALIDATED READS\n\n")
        dest.write("\tbased on {} reads\n".format(read_counter))
        # part on predicted HPs
        dest.write("\ntotal real HPs: {}\n".format(sum(all_count_truehp.values())))
        dest.write("\ntotal predicted HPs {}\n".format(sum(all_count_predictions.values())))
        #~ dest.write("\taverage number of measurements per HP: {}\n".format(sum(hp_size_list) / len(hp_size_list)))
        #~ dest.write("\tmedian number of measurements per HP: {}\n".format(median(hp_size_list)))
        #~ dest.write("\tmin number of measurements per HP: {}\n".format(min(hp_size_list)))
        #~ dest.write("\tmax number of measurements per HP: {}\n".format(max(hp_size_list)))
        #~ dest.write("\taverage number of bases per HP: {}\n".format(sum(base_size_list) / len(hp_size_list)))
        #~ dest.write("\tmedian number of bases per HP: {}\n".format(median(base_size_list)))
        #~ dest.write("\tmin number of bases per HP: {}\n".format(min(base_size_list)))
        #~ dest.write("\tmax number of bases per HP: {}\n".format(max(base_size_list)))
        dest.write("TPs: {} of which {} partial TPs\n".format(complete_st + incomplete_st, incomplete_st))
        if complete_st + incomplete_st != 0:
            dest.write("\taverage underestimated to the left: {}\n".format(avg_overleft))
            dest.write("\taverage underestimated to the right: {}\n".format(avg_overright))
            dest.write("\tperfectly found HPs: {}\n".format(perfectcomplete))
        dest.write("\ttotal interruptions over all positives: {}\n".format(nr_inters))
        if nr_inters != 0:
            dest.write("\taverage interruptions: {}\n".format(avg_inters))
            dest.write("\tmedian interruptions: {}\n".format(median_inters))
            dest.write("\taverage length interruptions: {}\n".format(avg_len_inters))
            dest.write("\tmedian length interruptions: {}\n".format(median_len_inters))
        dest.write("FPs: {}\n".format(absent_st))
        #~ dest.write("TNs: {}\n".format(true_negs))
        dest.write("FNs: {}\n".format(full_false_negs))
        
        
        if is_confusion:
            # part on TP, FP, TN, FN
            if all_tppositions != []:
                dest.write("\nTPs: {}\tTP measurements: {}\tTP bases: {}\n".format(tp, sum_dict(all_count_tptrue), sum_dict(all_count_tpbasestrue)))
                dest.write("\tLength of TP in measurements\n")
                dest.write("{}\n".format(all_count_tptrue))
                dest.write("\tLength of TP in bases\n")
                dest.write("{}\n".format(all_count_tpbasestrue))
                dest.write("\tTP composition\n")
                dest.write("{}\n".format(all_count_tpseqtrue))
                dest.write("\tTP positions\n")
                dest.write("{}\n".format(all_count_tppositions))     

            if all_fppositions != []:
                dest.write("\nFPs: {}\tFP measurements: {}\tFP bases: {}\n".format(fp, sum_dict(all_count_fptrue), sum_dict(all_count_fpbasestrue)))
                dest.write("\tLength of FP in measurements\n")
                dest.write("{}\n".format(all_count_fptrue))
                dest.write("\tLength of FP in bases\n")
                dest.write("{}\n".format(all_count_fpbasestrue))
                dest.write("\tFP composition\n")
                dest.write("{}\n".format(all_count_fpseqtrue))
                #~ dest.write("\tFP positions\n")
                #~ dest.write("{}\n".format(all_count_fppositions))  

            if all_fnpositions:
                dest.write("\nFNs: {}\tFN measurements: {}\tFN bases: {}\n".format(fn, sum_dict(all_count_fntrue), sum_dict(all_count_fnbasestrue)))
                dest.write("\tLength of FN in measurements\n")
                dest.write("{}\n".format(all_count_fntrue))
                dest.write("\tLength of FN in bases\n")
                dest.write("{}\n".format(all_count_fnbasestrue))
                dest.write("\tFN composition\n")
                dest.write("{}\n".format(all_count_fnseqtrue))
                dest.write("\tFN positions\n")
                dest.write("{}\n".format(all_count_fnpositions))

        # part on all predictions
        dest.write("\nComparison on length and composition        (length / seq, count)\n")
        dest.write("\tPredicted HP length in measurements\n")
        dest.write("{}\n".format(dict_to_ordered_list(all_count_predictions)))
        dest.write("\tPredicted HP length in bases\n")
        dest.write("{}\n".format(dict_to_ordered_list(all_count_basehp)))
        dest.write("\tPredicted HP sequences\n")
        dest.write("{}\n".format(all_seq_list))
        dest.write("\tPrevalence of bases underlying predictions\n")
        dest.write("{}\n\n".format(prev_bases))
        dest.write("\tTrue HP length in measurements\n")
        dest.write("{}\n".format(dict_to_ordered_list(all_count_truehp)))
        dest.write("\tTrue HP length in bases\n")
        dest.write("{}\n".format(dict_to_ordered_list(all_count_basetrue)))
        dest.write("\tTrue HP sequences\n")
        dest.write("{}\n".format(all_seqtrue_list))
        dest.write("\tPrevalence of true bases\n")
        dest.write("{}\n".format(prev_basestrue))
        #~ dest.write("{}\n".format())
        if show_plots:
            dest.write("\nSaved plots on prevalences.\n")
                            
    if show_plots:
        # plots on all positives:
        plot_prevalence(all_count_predictions, name="Predicted_HP_length_measurements")       # 1
        plot_prevalence(all_count_basehp, name="Predicted_HP_length_bases")                   # 3
        plot_prevalence(all_count_truehp, name="True_HP_lengths_measurements")      # 4a
        plot_prevalence(all_count_basetrue, name="True_HP_length_bases")            # 4b
        plot_list(all_seq_list, name="Predicted_HP_sequences", rotate=True)              # 5
        plot_list(all_seqtrue_list, name="True_HP_sequences", rotate=True)                            # 5
        
        plot_list(all_count_truepositions, name="True_positions")
    
        if is_confusion:
            # plots on false negatives:
            if all_fnpositions != []: 
                plot_prevalence(all_count_fntrue, name="FN_measurements")    
                plot_prevalence(all_count_fnbasestrue, name="FN_bases")  
                plot_list(all_count_fnpositions, name="FN_positions")        
                plot_list(all_fnseqtrue, name="FN_sequences", rotate=True)    
   
            # plots on false positives:
            if all_fppositions != []: 
                plot_prevalence(all_count_fptrue, name="FP_measurements")    
                plot_prevalence(all_count_fpbasestrue, name="FP_bases")  
                plot_list(all_count_fppositions, name="FP_positions")        
                plot_list(all_fpseqtrue, name="FP_HP_sequences", rotate=True) 
            
            # plots on true positives:
            if all_tppositions != []: 
                plot_prevalence(all_count_tptrue, name="TP_measurements")    
                plot_prevalence(all_count_tpbasestrue, name="TP_bases")  
                plot_list(all_count_tppositions, name="TP_positions")        
                plot_list(all_tpseqtrue, name="TP_HP_sequences", rotate=True)   
    
    # 7. Check surroundings
        # when do I consider it close?

    return all_tppositions, all_fppositions, all_fnpositions, all_tnpositions
  
  
def class_from_threshold(predicted_scores, threshold):
    """
    Assigns classes on input based on given threshold.
    
    Args:
        predicted_scores -- list of floats, scores outputted by neural network
        threshold -- float, threshold
    
    Returns: list of class labels (ints)
    """
    return [1 if y >= threshold else 0 for y in predicted_scores]
    

def list_predicted(line, types="predicted_labels"):
    """
    Creates list from predicted labels saved in file.
    
    Args:
        pred_file -- str, file that contains predicted labels as '[.., .., ..]'
        types -- str, type of line to retrieve: predicted labels, true labels or predicted scores [default: 'predicted_labels']
        
    Returns: predicted labels (list of ints)
    """
    if types == "predicted_labels":
        sign = "#"
        datatype = int
    elif types == "true_labels":
        sign = "*"
        datatype = int
    elif types == "predicted_scores":
        sign = "@"
        datatype = float       
    
    if line.startswith(sign):
        predicted = line.strip()[3:-1].split(", ")
        predicted = list(map(datatype, predicted))
    else:
        predicted = None

    return predicted


def hp_loc_dict(predicted_labels, pos=1, neg=0):
    # adjusted from save_hp_loc in info_hp
    """
    Creates dict to save positions of homopolymers.
    
    Args:
        predicted_labels -- list of int, predicted HP labels
        
    Returns: dict {id: start position, end position (inclusive)}
    """
    predicted_hp = {}
    counter = 0
    is_hp = False
    for i in range(len(predicted_labels)):
        if not is_hp and predicted_labels[i] == pos:
            is_hp = True
            start = i
            end = i
        if is_hp:
            if (i + 1 == len(predicted_labels)):
                end = i
                predicted_hp[counter] = (start, end)
                is_hp = False
            elif predicted_labels[i] == neg: 
                end = i - 1
                is_hp = False
                predicted_hp[counter] = (start, end)
                counter += 1
    
    return predicted_hp
            
    
def get_prevalence(hp_dict, types="measurements"):
    """
    Retrieves prevalance of occurence in dict.
    
    Args:
        hp_dict -- dict {id: start, end}
        
    Returns: Counter dict {length: prevalence}
    """
    if types == "measurements":
        length_list = [k[1] - k[0] + 1 for k in hp_dict.values()]
    elif types == "base_lengths":
        length_list = [len(k) for k in hp_dict.values()]
    elif types == "bases":
        length_list = hp_dict.values()        
    count_dict = Counter(length_list)
    
    return count_dict   
    

def plot_prevalence(count_dict, name="Prevalence", rotate=False):
    """
    Plots prevalence.
    
    Args:
        count_dict -- dict {value: count}
        
    Returns: None
    """
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 9)) 
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.ylabel("count")
    plt.xlabel("length")
    plt.title(name)
    if rotate:
        plt.xticks(rotation=75)
    #~ plt.show()
    plt.savefig("{}.png".format(name), bbox_inches="tight")
    plt.clf()
    plt.close()
    

def plot_list(count_list, name="Prevalence", rotate=False):
    """
    Plots prevalence.
    
    Args:
        count_dict -- dict {value: count}
        
    Returns: None
    """
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 9)) 
    b_list = [i[0] for i in count_list]
    c_list = [i[1] for i in count_list]
    plt.bar(b_list, c_list)
    plt.ylabel("count")
    plt.xlabel("length")
    plt.title(name)
    if rotate:
        plt.xticks(rotation=75)
    plt.xticks(fontsize=6)
    #~ plt.show()
    plt.savefig("{}.png".format(name), bbox_inches="tight")
    plt.close()
    

def prediction_information(hp_dict, bases, new):
    """
    Args:
        read_states -- list, based on true HPs (FN, TP); assumes containing only positive examples
        types -- str, from confusion matrix: 'P', 'N'

    Returns: length in measurements (dict), length in bases (dict), composition (dict)
    """          
    count_measurements = get_prevalence(hp_dict)
    basedict = base_count_dict(hp_dict, bases, new)
    count_bases = get_prevalence(basedict, types="base_lengths")
    count_seq = get_prevalence(basedict, "bases")

    return count_measurements, count_bases, count_seq
    
    
def get_bases(base_seq, new_seq, hp_start, hp_end):
    """
    Retrieves underlying base sequences from measurement positions.
    
    Args:
        base_seq -- str, sequence of bases
        new_seq -- str, indication of new base
        hp_start -- int, start position of homopolymer 
        hp_end -- int, end position of homopolymer
        
    Returns: base sequence (str)
    """
    #~ first_base = [base_seq[hp_start]]
    bases = [base_seq[i] for i in range(hp_start, hp_end + 1) if new_seq[i] == "n"]
    #~ bases = first_base + bases
    
    return "".join(bases)
    

def base_count_dict(indict, bases, new):
    return {k: get_bases(bases, new, indict[k][0], indict[k][1]) for k in indict}
    

def detected_from_true(predicted_hp, true_hp):
    """
    Counts number of predicted labels that are really positive. 
    """
    predicted_list = [list(range(k[0], k[1] + 1)) for k in predicted_hp.values()]
    predicted_list = [j for i in predicted_list for j in i]
    true_list = [list(range(k[0], k[1] + 1)) for k in true_hp.values()]
    true_list = [j for i in true_list for j in i]    
    counter = [1 for c in range(len(true_list)) if true_list[c] in predicted_list]
    count_tp = sum(counter)
    print("Found {} of {} true measurements".format(count_tp, len(true_list)))  # this part is double: is like main stats on recall etc.
    

def predicted_vs_true(predicted_labels, true_labels, read_name):
    """
    Counts number of predicted labels vs true labels.
    Prints reads that do not have positive measurements, but were labelled to have one.
    
    Args:
        predicted_labels -- list, predicted labels: either 0 or 1
        true_labels -- list, true labels: either 0 or 1
        
    Returns: None
    """
    count_predicted = predicted_labels.count(1)
    count_true = true_labels.count(1)
    if count_true == 0:
        print("Found {}, but no true measurements present in read {}.".format(count_predicted, read_name))    
    else:
        print("Predicted {}, but in reality had {} positive measurements.".format(count_predicted, count_true) )


def dict_to_ordered_list(dict_in, sort_on=0):
    """
    Creates list of keys and values ordered.
    
    Args:
        dict_in -- dict {key: value}
        sort_on -- int, element of tuple to sort on [default: 0 (key)]
        
    Returns: list of tuples
    """
    ordered_list = []
    for k, v in dict_in.items():
        temp = (k, v)
        ordered_list.append(temp)

    return sorted(ordered_list, key=lambda x: x[sort_on])

# IMPLEMENT LATER
#~ def combine_dicts(dict1, dict2):
    #~ """
    #~ Combines two dicts based on keys by adding values.
    
    #~ Args:
        #~ dict1 -- dict, {k: v}
        #~ dict2 -- dict, {k: v}
        
    #~ Returns: dict
    #~ """
    #~ return dict1 + dict2
    

def check_hp(hp, labels, ids, pos=1, neg=0):
    """
    Checks how well a homopolymer is detected by network.
    
    Args:
        hp -- tuple of ints, start and end position of hp
        labels -- list of ints, predicted labels
        ids -- str / int, id for hp to recognize
        
    Returns: [state, (left, right), [length of interruption, ...]]
            left and right are the number of over- or underestimated measurements
    """    
    inter_list = []
    predicted_stretch = labels[hp[0] : hp[1] + 1]
    # check if true homopolymer is completely or partly detected
    if len(predicted_stretch) == predicted_stretch.count(pos):
        state = "complete"                                                      
    elif len(predicted_stretch) == predicted_stretch.count(neg):
        state = "absent"
        l = None
        r = None
    else:
        state = "incomplete" 
        # check for interruptions: where, how long, how often  
        new_inter = False
        for i in range(len(predicted_stretch)):
            if not new_inter and predicted_stretch[i] == neg: 
                new_inter = True 
                start = i
            elif new_inter:
                if predicted_stretch[i] == 1:
                    end = i - 1
                    temp = (start, end)
                    if not (start == 0 or end == len(predicted_stretch) - 1):
                        inter_list.append(temp)
                    new_inter = False
                if len(predicted_stretch) == (i + 1):
                    end = i
                    temp = (start, end)
                    if not (start == 0 or end == len(predicted_stretch) - 1):
                        inter_list.append(temp)               

    if state != "absent":
        # check if prediction is overestimated on either side:   
        l = 0
        if labels[hp[0]] == pos:
            check_left = True
            while check_left:
                position = hp[0] + l - 1
                if (not position <= 0 and labels[position] == pos):
                    l -= 1
                else:
                    check_left = False
        r = 0 
        if labels[hp[1]] == pos:
            check_right = True
            while check_right:
                position = hp[1] + r + 1
                if (not position >= len(labels) and labels[position] == pos):
                    r += 1
                else:
                    check_right = False
                
        # check if prediction is underestimated on either side:         
        if l == 0:
            check_left = True
            while check_left:
                position = hp[0] + l
                if (not position >= len(labels) and labels[position] == neg):
                    l += 1
                else:
                    check_left = False
        if r == 0: 
            check_right = True
            while check_right:
                position = hp[1] + r
                if (not position >= len(labels) and labels[position] == neg):
                    r -= 1
                else:
                    check_right = False    
    
    return state, (l, r), inter_list, ids
   
   
def generate_dict(predicted_labels, true_labels, pred_label=1, true_label=1):
    """
    Generates dict with specified conditions for predictions and truth.
    pred:1 - true:1 > TP
    pred:0 - true:1 > FN
    pred:1 - true:0 > FP
    pred:0 - true:0 > TN
    
    Args:
        predicted_labels -- list of ints
        true_labels -- list of ints
        pred_label -- int, label of predictions to find [default: 1]
        true_label -- int, label of truth to compare to [default: 1]
        
    Returns: dict {id: start, end}
    """
    is_target = False
    pos_dict = {}
    counter = 0
    
    for i in range(len(predicted_labels)):
        if not is_target and predicted_labels[i] == pred_label and true_labels[i] == true_label:
                start = i
                is_target = True
        if is_target:
            if ((i + 1) == len(predicted_labels)) or true_labels[i + 1] != true_label or predicted_labels[i + 1] != pred_label:
                end = i
                is_target = False
                pos_dict[counter] = (start, end)
                counter += 1
    
    if not pos_dict:
        pos_dict = 0
        
    return pos_dict

    
def count_bases(counter_tuple):
    """
    Counts the number of bases from Counter dict.
    
    Args:
        counter_tuple -- tuple (seq: count)
        
    Returns: dict {A: count, C: count, G: count, T: count}
    """
    count_A = 0
    count_C = 0
    count_G = 0 
    count_T = 0
    
    count_dict = {"A": count_A, "C": count_C, "G": count_G, "T": count_T} 
    
    for i in range(len(counter_tuple)):
        for b in counter_tuple[i][0]:
            count_dict[b] += counter_tuple[i][1]
                               
    return count_dict
    

def filter_dict(indict, include):
    """
    Filter dict to contain key-value pairs specified by key.
    
    Args: 
        indict -- dict to filter
        include -- list, keys to include
    
    Returns: dict    
    """
    return {k: v for k, v in indict.items() if k in include}
    

def reduce_dict(hp_dict, read_states, types):
    """
    Reduces a dictionary to only contain positive or negative examples.  
    
    Args: 
        hp_dict -- dict {id: (start, end)}, based on true HPs or predicted HPs
        read_states -- tuple, checks if given HP can be found back in labels, eg trueHP given but not found back > absent > False Negative
    
    Returns: dict    
    """
    if types == "N":
        keyword = "absent"
    elif types == "P":
        keyword = "complete"        # complete or incomplete - solved by using in
    
    searches = [st[3] for st in read_states if keyword in st[0]]        # searches is either negatives or positives
    if searches != []:                                     
        hp_dict = filter_dict(hp_dict, searches)
    else:
        hp_dict = None
        
    return hp_dict
    

def sum_dict(my_dict):
    """
    Sums the total length in a dict. 
    
    Args:
        my_dict -- (Counter) dict {length : count}
        
    Returs: int
    """
    return sum([k * my_dict[k] for k in my_dict])
    
    
def cut_from_output(sourcefile, destfile, rounds, reads=856, lines=4):
    """
    Copies lines from a file to another file. Based on files that contain output
    of multiple rounds of validation.
    """
    counter = 0
    with open(destfile, "w") as dest:
        with open(sourcefile, "r") as source:
            for line in source:
                if counter in list(range(rounds * lines * reads, (rounds + 1) * lines * reads + 1)):
                   dest.write(line)
                counter += 1
                   
    print("Finished writing")
    
    
def thresholded_labels(thres_file, thres):
    """
    Gets labels according to given threshold.
    
    Args:
        thres_file -- str, path to thresholded file
        thres -- float, threshold
    
    Returns: list of labels (ints)
    """
    # check for threshold in file
    get_thres = True
    with open(thres_file, "r") as source:
        for line in source:
            if get_thres:
                thresholds = line.strip()[1:-1].split(", ")
                thresholds = list(map(float, thresholds))
                print(thresholds)
                get_thres = False
                line_counter = 0
                for i in range(len(thresholds)):
                    if thresholds[i] == thres:
                        number = i
                        print("number is: ", number)
                    elif thres not in thresholds:
                        raise ValueError("Given threshold not in file.")
            # return belonging labels
            elif not get_thres:
                if line_counter == number:
                    print("line used is: ", line_counter)
                    labels = line.strip()[1:-1].split(", ")
                    labels = list(map(int, labels))
                    print(len(labels))
                    break
                else:
                    line_counter += 1
    
    return labels
    
def create_confdict(predicted_hp, states, confusion):
    """
    Creates dict like confusion matrix.
    
    Args:
        predicted_hp -- dict
        states -- list
        confusion -- choose "tp", "fp", "tn", "fn"
        
    Returns: reduced dict
    """
    hpdict = {}
    if confusion == "tp":
        for i in range(len(states)):
            if states[i][0] != "absent":
                hpdict[states[i][3]] = predicted_hp[states[i][3]]
                #~ print("id: ", states[i][3])
                #~ print("predicted: ", predicted_hp[states[i][3]])
                #~ print("hp content: ", hpdict[states[i][3]])
                #~ print("hp dict: ", hpdict)
        #~ hpdict = {[states[i][3]] : predicted_hp[states[i][3]] 
                    #~ for i in range(len(states)) if states[i][0] != "absent"}
    elif confusion == "fp" or confusion == "fn":
        for i in range(len(states)):
            if states[i][0] == "absent":
                hpdict[states[i][3]] = predicted_hp[states[i][3]]
    
    if not hpdict:
        hpdict = 0
    #~ print(hpdict)
    return hpdict


# adapted from Carlos:                      
def correct_short(predictions, threshold=15):
    """
    Corrects class prediction to negative label if positive stretch is shorter than threshold.
    
    Args:
        predictions -- list of int, predicted class labels
        threshold -- int, threshold to correct stretch [default: 15]
    
    Returns: corrected predictions
    """
    compressed_predictions = [[predictions[0],0]]

    for p in predictions:
        if p == compressed_predictions[-1][0]:
            compressed_predictions[-1][1] += 1
        else:
            compressed_predictions.append([p, 1])

    for pred_ci, pred_c, in enumerate(compressed_predictions):
        if pred_c[0] != 0:
            if pred_c[1] < threshold:
                # remove predictions shorter than threshold
                compressed_predictions[pred_ci][0] = 0
            #~ else: 
                #~ # extend predictions longer than threshold
                #~ compressed_
            
    return np.concatenate([np.repeat(pred_c[0], pred_c[1]) for pred_c in compressed_predictions])    
    
    
    

def LATER():
        # 7. Cluster HP compositions on raw measurements
        # maybe also as measurement long base sequences..
    # 7a. Get raw measurements for each predicted HP
    #~ raw = load_npz_raw("{}/{}.npz".format(npz_dir, read_name))
    # id in dicts is equal across dicts: so 1 in predicted is same part as 1 in base_dict
    # Create dict or something else so raw measurements are plotted but you know what sequence is underneath
    # 7b. K-means cluster
        # on base A, G, T, C > make an [count A, count C, count G, count T] and cluster 
        # or as A:1, C:2, T:3, G:4, nothing:0
    #~ length_dict = {k: len(base_dict[k]) for k in base_dict}
    #~ print(length_dict)
    #~ kmc_labels = kmeans_clustering(x, 2)
    #~ print(len(kmc_labels))
        # on size
        # on true HP and non HP
        
    # 7c. Hierarchical clustering
    pass

    
if __name__ == "__main__":   
    # Output results:
    if len(argv) < 5:
        print("Required input is:\n\t-file containing true labels and predicted scores" +
              "\n\t-directory containing original FAST5 files\n\t-directory containing .npz files" +
              "\n\t-name of output file\n\t-threshold\nOPTIONAL:\n\t-maximum number of validation reads" +
              "\n\t-maximum length of validation reads\n\t-start position in validation reads")
    
    read_file = argv[1]
    main_fast5_dir = argv[2]
    output_name = argv[3]
    threshold = float(argv[4])
    npz_dir = argv[5]
    max_number = 12256
    #~ max_seq_length = 14980 #4970   #9975 
    start = 0  # 30000
    
    #~ if len(argv) >= 6:
        #~ max_number = int(argv[5])
    #~ if len(argv) >= 7:
        #~ max_seq_length = int(argv[6])
    #~ if len(argv) >= 8:
        #~ start = int(argv[7])
    # ~ print("start at: ", start)
    tp, fp, fn, tn = main(read_file, main_fast5_dir, npz_dir, output_name, 
                          threshold, start, max_number)

    #~ check_for_neg(read_file, main_fast5_dir, npz_dir, output_name)
                          

