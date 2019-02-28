# catfish
-------------------------------------------------------------------------------
Pre-processing tool with a neural network as basis, designed to predict the presence of homopolymers
in the raw signal of a MinION sequencer, to select stretches with homopolymers and split
the raw nanopore signal on those. The purpose is to basecall homopolymer containing stretches with
a specialised basecaller while regularly processing reads containing no homopolymers, which would increase
accuracy while saving computational power on the reads that do not need the extra care.

Tool is designed as part of MSc thesis "Calling homopolymers in nanopore sequencing data"
by Marijke Thijssen.


## Installation
pip install git+https://git.wur.nl/thijs030/thesis/tree/master/catfish

## Dependencies
Tool is dependent on the packages h5py, matplotlib, numpy, seaborn and tensorflow.

`pip install click` <br />
`pip install h5py`  <br />
`pip install numpy`  <br />
`pip install --upgrade tensorflow`

Additionally, Albacore installed in a conda environment called *basecall* is a prequisite. Albacore v2.3.3
was used in the research, but other version might work as well.
1. Anacoda or Miniconda, available at https://conda.io/miniconda.html
2. Albacore v2.3.3

### Build a virtual environment for basecalling with Albacore
`conda create -n basecall python=[python version compatible with Albacore]`  <br />
`source activate basecall` <br />
`pip install ont_albacore-[version].whl` <br />
`source deactivate`


## Usage
`cd catfish`
```
catfish [-h] [--input-dir] [--split-dir]

Required arguments:
    -i, --input-dir         Path to input directory of reads in FAST5 format
    -s, --split-dir         Path to directory to save split reads to
    
Options:
    -c, --chunk-size        Length of chunks containing homopolymers for basecalling
    -h                      Shows help message and exit
```


## Output 
Marijke-tool outputs a list of paths to the split reads containing homopolymers and
a list of path to the split reads containing no homopolymers.

### Example usage
`Marijke-tool -i <fast5_directory> -s <split_reads> -c <size_of_homopolymer_chunks>`


## Useful links
The name is a reference to the fish as well as to the act of catfishing, in which it is 
an art to recognise a 



