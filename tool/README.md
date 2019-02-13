# Title
-------------------------------------------------------------------------------
Tool with neural network as basis, designed to predict the presence of homopolymers
in the raw signal of a MinION sequencer, to select stretches with homopolymers for
basecalling with a specialized basecaller while keeping fast basecalling with Albacore for
general reads.

Tool is designed as part of MSc thesis "Calling homopolymers in nanopore sequencing data"
by Marijke Thijssen.


## Installation
pip install git+https://git.wur.nl/thijs030/thesis/tree/master/tool

## Dependencies
Tool is dependent on the packages h5py, matplotlib, numpy, seaborn and tensorflow.
(? remove matplotlib and seaborn -- only needed for metrics, can be adjusted ?)

`pip install h5py`
`pip install matplotlib`
`pip install numpy`
`pip install seaborn`
`pip install --upgrade tensorflow`

Additionally, Albacore installed in a conda environment called *basecall* is a prequisite. Albacore v2.3.3
was used in the research, but other version might work as well.
1. Anacoda or Miniconda, available at https://conda.io/miniconda.html
2. Albacore v2.3.3

### Build a virtual environment for basecalling with Albacore
`conda create -n basecall python=[python version compatible with Albacore]`
`source activate basecall`
`pip install ont_albacore-[version]`
`source deactivate`


## Usage
`Marijke-tool [-h] [--input-file] [--output-file]`

`Required arguments:`
    `-i, --input-dir         Path to input directory of FAST5 files`
    `-s, --save-dir          Path to directory for saving basecalls in FASTQ format`
    
`Options:`
    `-c, --chunk-size        Length of chunks containing homopolymers for basecalling`
    `-h                      Shows help message and exit`
    `-o, --output-file       Name of output file for failed reads`


## Output 
Marijke-tool outputs a FASTQ file of all basecalled reads.

### Example usage
`Marijke-tool -i fast5_directory -o name_failed_reads_file -s basecalled -c 1000`

## Test
Maybe include test to check if installation went well

## Useful links
\ link to thesis \



###### to do:
add network checkpoint to folder data

take parts from metrics that you need, move others to reduce dependencies

make requirements file
