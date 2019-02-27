#!/usr/bin/env python3

import click
import datetime
import infer
import neural_network
import os
import os.path
#~ from read_quality import read_quality
import shutil
import split_f5
import subprocess
from sys import argv

"""
Main script to run tool from 
"""

@click.command()
@click.option("--input-dir", "-i", help="Path to input dir in FAST5 format")
@click.option("--output-file", "-o", help="Name of read quality output file")
@click.option("--save-dir", "-s", help="Directory to save basecalled reads to")
@click.option("--chunk-size", "-c", help="Chunk size for homopolymer containing stretches", default=1000, show_default=True)
#~ @click.option("--perfect", "-p", help="FOR TESTING PURPOSES ONLY: evaluates performance on labeled test set", is_flag=False)
def main(input_dir, output_file, save_dir, chunk_size, threshold=0.9, network_path="/home/thijs030/thesis/scripts/tool/ResNetRNN", network_type="ResNetRNN"):
    """
    A tool with a neural network as basis to predict the presence of 
    homopolymers in the raw signal from a MinION sequencer. 
    """
    counter = 0

    temp_dir = "{}/TEMP".format(os.path.abspath(save_dir))
    temp_dir_hp = "{}/HP".format(os.path.abspath(temp_dir))
    temp_dir_nonhp = "{}/nonHP".format(os.path.abspath(temp_dir))
    os.makedirs(temp_dir_hp)
    os.mkdir(temp_dir_nonhp)
    save_dir_hp = save_dir + "/HP"
    save_dir_nonhp = save_dir + "/nonHP"

    # 0. Load model
    t1 = datetime.datetime.now()
    model = neural_network.load_network(network_type, network_path, checkpoint=30000)
    print("Loaded model in {}".format(datetime.datetime.now() - t1))
    
    # 1. Load files in directory
    input_dir = os.path.abspath(input_dir)
    input_files = os.listdir(input_dir)

    #~ # ONLY for HPs right now ##################################################
    # 2. Get predicted HPs for every read
    print("Checking for homopolymers in raw signal..")
    t2 = datetime.datetime.now()
    for fast5_file in input_files:
        files_hp = []
        files_nonhp = []
        #~ # ! input_dir should point to dir containing NPZs; fst5_file is .npz file
        #~ if perfect:
            #~ #print("FOR TESTING PURPOSES: evaluation of tool on perfect set")
            #~ hp_positions, len_read = infer_class_from_npz("{}/{}".format(input_dir, fast5_file), model)
        #~ else:
        hp_positions, len_read = infer.infer_class_from_signal("{}/{}".format(input_dir, fast5_file), model, label=1)
            
    # TODO: catch short reads (<1000) and no_hppositions

        merged_positions = [hp_positions[0]]
        for i in range(len(hp_positions)):
            if hp_positions[i][1] >= chunk_size + merged_positions[-1][0]:
                merged_positions[-1][-1] = hp_positions[i - 1][1]
                center_hp(merged_positions, len_read, chunk_size)
                merged_positions.append(hp_positions[i])
        center_hp(merged_positions, len_read, chunk_size)
        
        #~ number_hp = len(hp_positions)
        #~ number_hpstretches = len(merged_positions)
        #~ print(number_hp, number_hpstretches)
        
        # 2b. Add non-homopolymers to separate dict
        m_start = 0
        nonhp_reads = []
        for m in range(len(merged_positions)):
            if merged_positions[m][0] > m_start:
                m_end = merged_positions[m][0] - 1
                nonhp_reads.append([m_start, m_end])
            m_start = merged_positions[m][1]
        if merged_positions[-1][1] != len_read:
            nonhp_reads.append([merged_positions[-1][1], len_read])
            
        # 3. Split FAST5 in multiple FAST5s
        hp_files, nonhp_files = split_f5.split_signal("{}/{}".format(input_dir, fast5_file), merged_positions, nonhp_reads, temp_dir_hp, temp_dir_nonhp)
        files_hp.extend(hp_files)                                               # HP files are absolute paths
        files_nonhp.extend(nonhp_files)
    
    #~ print("Number of files containing HPs after split: ", len(files_hp))
    #~ print("Number of files containing non-HPs after split: ", len(files_nonhp))
    #~ ## TODO: change later to not return anything. Is not necessary.

        #~ if counter % 10 == 0 or counter == len(input_files):
            #~ # 4. Process each FAST5 belong to HP or non-HP group
            #~ THREADS = 4
            #~ CONFIG = "/mnt/nexenta/thijs030/data/r94_450bps_linear.cfg"                 # TODO: adjust!
            #~ FLOW = "FLO-MIN106"
            #~ KIT = "SQK-RAD002"            
            #~ OUTPUT = "fastq"                                                            # can be FAST5 or FASTQ or both ; this for testing - maybe make variable?
            #~ subprocess.run("source activate basecall", shell=True)
                #~ #TODO: a. Register number of failed and number of passes (in total and per group)
                #~ # maybe use fast5seek or better: sequencing_summary.txt on passes_filtering?
            #~ subprocess.run("read_fast5_basecaller.py -i {} -t {} -s {} -o {} -r -f {} -k {}".format(
                            #~ temp_dir_nonhp, THREADS, save_dir_nonhp, OUTPUT, FLOW, KIT), shell=True)
            #~ subprocess.run("read_fast5_basecaller.py -i {} -t {} -s {} -o {} -r -f {} -k {} --basecaller.homopolymer_correct=1".format(
                            #~ temp_dir_hp, THREADS, save_dir_hp, OUTPUT, FLOW, KIT), shell=True)
            #~ # close conda env
            #~ subprocess.run("source deactivate", shell=True) 
                   
                #~ # b. Remove split files
            #~ shutil.rmtree(temp_dir_hp, ignore_errors=True)
            #~ shutil.rmtree(temp_dir_nonhp, ignore_errors=True)
            
            #~ if os.path.isdir(save_dir_hp + "/workspace/pass"):
                #~ subprocess.run("cat {}/workspace/pass/*.fastq > {}/homopolymer_{}.fastq".format(save_dir_hp, save_dir, counter))
                #~ shutil.rmtree(save_dir_hp, ignore_errors=True)
            
            #~ if os.path.isdir(save_dir_nonhp + "/workspace/pass"):
                #~ subprocess.run("cat {}/workspace/pass/*.fastq > {}/nonhomopolymer_{}.fastq".format(save_dir_nonhp, save_dir, counter))
                #~ shutil.rmtree(save_dir_nonhp, ignore_errors=True)
            
            #~ os.mkdir(temp_dir_hp)
            #~ os.mkdir(temp_dir_nonhp)        
            
            #~ counter += 1 
            
            #~ if counter % 100 == 0:
                #~ print("Basecalled {} reads".format(counter))
            
            #~ if counter >= 20:
                #~ break
            
    #~ # 5. Extract FASTAs from FASTQs in passed dir / Concatenate reads
    #~ subprocess.run("cat {}/*.fastq > {}/basecalled.fastq".format(save_dir)) 
    #~ subprocess.run("cat {}/workspace/pass/*.fastq > {}/homopolymer.fastq".format(save_dir_hp, save_dir))
    #~ subprocess.run("cat {}/workspace/pass/*.fastq > {}/nonhomopolymer.fastq".format(save_dir_nonhp, save_dir))
    #~ subprocess.run("cat {}/*.fastq > {}/basecalled.fastq".format(save_dir))
    
    #~ os.remove("{}/homopolymer.fastq".format(save_dir))
    #~ os.remove("{}/nonhomopolymer.fastq".format(save_dir))
    
    #~ shutil.rmtree(save_dir_hp, ignore_errors=True)
    #~ shutil.rmtree(save_dir_nonhp, ignore_errors=True)

    #~ shutil.rmtree(temp_dir_hp, ignore_errors=True)
    #~ shutil.rmtree(temp_dir_nonhp, ignore_errors=True)
    
    #~ print("Finished basecalling. FASTQ containing all reads can be found at {}.".format(os.path.abspath(save_dir)))
        
        #~ # maybe use fast5seek
        #~ # does albacore use duration? (or start time?) because I just copy this to all subfiles
    
    
    #~ # 6. Check read quality
    #~ REF_FASTA = "/mnt/nexenta/thijs030/data/reference/ecoli_ref.fasta"          # TODO: change
    #~ read_quality("{}/workspace/pass".format(save_dir), REF_FASTA, output_file)      # reads can be folder of FASTQs or FASTAs - but first concat per read
    
    # 7. MAYBE - produce nice graphs on read quality and CPU speed
    
    
    #~ # basecall all reads with Albacore:



    
    # skip failed reads from Albacore
    
    # polish reads with HPs:
    
        
        # IDEE: code zo aanpassen dat basecaller het meteen inneemt?
            
            # let non-HP regions go through fast basecaller
            
            # let (extended) HP reads go through polisher / specialized basecaller
            
            # write basecalls to FAST5
            
            # if something goes wrong:
            #~ except Error as e:
    # write unsuccessfull reads to file
    #~ with open(output_file, "w") as dest:                                        # output_file is error file
        #~ dest.write("Unable to basecall the following reads: \n\n")            
        #~ failed_reads = []
        #~ dest.write("{}: {}\n".format(e, read_name))            
        
    #~ print("Successfully basecalled {} of {} reads.".format(successes, len()) # is not necessary: albacore splits this up already
    
    #~ return None

def center_hp(merged_positions, len_read, chunk_size=1000):
    len_hp = merged_positions[-1][-1] - merged_positions[-1][0]
    if len_hp < chunk_size:
        left_padding = (chunk_size - len_hp) // 2
        right_padding = (chunk_size - len_hp) - left_padding
        merged_positions[-1][0] = merged_positions[-1][0] - left_padding
        merged_positions[-1][1] = merged_positions[-1][1] + right_padding
        if merged_positions[-1][0] < 0:
            merged_positions[-1][1] -= merged_positions[-1][0]
            merged_positions[-1][0] = 0
        if merged_positions[-1][1] > len_read: 
            merged_positions[-1][0] -= len_read - merged_positions[-1][1]     
            merged_positions[-1][1] = len_read
    
    return merged_positions

    
if __name__ == "__main__":
    main()
    




