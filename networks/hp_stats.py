#!/usr/bin/env python3

def main(output_file, main_dir, npz_dir, out_name, threshold=0.5, max_nr=12255):
    predicted_labels = None
    true_labels = None
    
    read_counter = 0
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
                            predicted_labels = class_from_threshold(predicted_labels, threshold)
                    if true_labels == None:    
                        true_labels = list_predicted(line, types="true_labels")
                    elif predicted_labels != None and true_labels != None:
                        bases, new = get_base_new_signal("{}/{}.fast5".format(main_dir, read_name))
                        count_basen = [1 for w in new[:4970] if w == "n"]
                        n_bases += sum(count_basen)
                        # save information to dict
                        predicted_hp = hp_loc_dict(predicted_labels)
                        true_hp = hp_loc_dict(true_labels)
                        detected_from_true(predicted_hp, true_hp)
                        
                        # check all positives:      real positives + predicted positives
                        count_truehp, count_basetrue, count_seqtrue = prediction_information(true_hp, bases, new)  
                        stats_hp(count_truehp) 

def stats_hp(hp_dict):
    hp_list = list(hp_dict.keys())
    size_list = [len(hp) for hp in hp_list]
    avg_len = sum(size_list) / len(size_list)
    median_len = median(size_list)
    print("HOMOPOLYMER STATISTICS")
    print("Average length of HPs: {}".format(avg_len))
    print("Median length of HPs: {}".format(median_len))
    print("Minimal length: {}".format(min(size_list)))
    print("Maximal length: {}".format(max(size_list)))
    
def write_stats(avg_len, median_len):
    with open("hp_stats.txt", "w") as dest:
        dest.write("HOMOPOLYMER STATISTICS")
        dest.write("Average length of HPs: {} (avg)\n".format(avg_len))
        dest.write("Median length of HPs: {} (avg)\n".format(median_len))
        dest.write("Minimal length: {}\n".format(min(size_list)))
        dest.write("Maximal length: {}\n".format(max(size_list)))
        
        
if __name__ == "__main__":
    # get all reads
 
    # calc hp dict per read
    
    # calc stats; return avg, median, min, max
    
    # return avg / median on avg, median per read
    
