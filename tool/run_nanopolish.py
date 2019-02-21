#!/usr/bin/env python3
import subprocess
from sys import argv 

# 1. Data preprocessing
def preprocess_data(fast5_dir, output_name="reads.fa", seq_summ="sequencing_summary.txt"):
    subprocess.run("nanopolish index -d {} -s {} {}".format(fast5_dir, seq_summ, output_name), 
                    shell=True) 
    # -d can be specified more than once
    # -f can be passed file containing the path to sequencing summaries
    return output_name
    

# 2. Generate draft consensus
def draft_canu(output_dir, albacore_fastq):
    subprocess.run("canu -p ecoli -d {} genomeSize=4.6m -nanopore-raw {}".format(
                    output_dir, albacore_fastq))
    
    return output_name
 
# 3. Compute new consensus
def align_to_draft(draft, preprocessed_fasta, output_name="reads"):
    subprocess.run("minimap2 -ax map-ont -t 8 {} {} | samtools sort -o {}.sorted.bam -T {}.tmp".format(
                    draft, preprocessed_fasta, output_name, output_name), shell=True)
    subprocess.run("samtools index {}.bam".format(output_name), shell=True)
    
    return output_name + ".sorted.bam"    

def compute_consensus(draft, preprocessed_fasta, sorted_bam, output_name="polished_genome"):
    subprocess.run("python nanopolish_makerange.py {d} | parallel --results nanopolish.results -P 8 \
                    nanopolish variants --consensus -o polished.{1}.vcf -w {1} -r {pf} -b {sb} -g {d} -t 4".format(
                    d=draft, pf=preprocessed_fasta, sb=sorted_bam), 
                    shell=True)
                    # add --fix-homopolymers (flag)
                    # removed: --min-candidate-frequency 0.1
                    # I think {1} indicates automatic incrementation
    subprocess.run("nanopolish vcf2fasta --skip-checks -g {} polished.*.vcf > {}.fa".format(draft, output_name), shell=True)
    
    return output_name + ".fa"
    
#~ nanopolish vcf2fasta -g draft.fa polished.*.vcf > polished_genome.fa

if __name__ == "__main__":
    main_dir = argv[1]
    draft = argv[2]         # make in this script too?
    
    fasta_reads = preprocess_data(main_dir)
    sorted_bam_reads = align_to_draft(draft, fasta_reads)
    polished = compute_consensus(draft, fasta_reads, sorted_bam_reads)
