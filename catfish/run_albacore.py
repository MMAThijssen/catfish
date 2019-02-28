#!/usr/bin/env python3
import subprocess
from sys import argv

def run_albacore(input_dir, save_dir, hp_correct=0, chunk_size=1000, threads=4, output="fastq", flow="FLO-MIN106", kit="SQK-RAD002"):
    """
    Python implementation to activate conda environment and run albacore.
    """
    subprocess.run("source activate basecall", shell=True)
    subprocess.run("read_fast5_basecaller.py -i {} -t {} -s {} -o {} -r -f {} -k {} --basecaller.homopolymer_correct={} --basecaller.max_events={}".format(
                    input_dir, threads, save_dir, output, flow, kit, hp_correct, chunk_size), shell=True)
    subprocess.run("source deactivate", shell=True) 

if __name__ == "__main__":
    print("Basecalling with Albacore v2.3.3  ..")
    run_albacore(argv[1], argv[2], chunk_size=10000)
    print("Finished basecalling")

