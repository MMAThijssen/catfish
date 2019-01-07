# Title
-------------------------------------------------------------------------------
Tool with neural network as basis, designed to predict the presence of homopolymers
in the raw signal from a MinION sequencer. 

Tool is designed as part of MSc thesis "Calling homopolymers in nanopore sequencing data"
by Marijke Thijssen.


## Installation
pip install git+https://git.wur.nl/thijs030/thesis/tree/master/tool

## Dependencies
Marijke-tool is dependent on the packages h5py, matplotlib, numpy, seaborn and tensorflow.
(? remove matplotlib and seaborn -- only needed for metrics, can be adjusted ?)

pip install h5py
pip install matplotlib
pip install numpy
pip install seaborn
pip install --upgrade tensorflow


## Usage
Marijke-tool [-h] [--input-file] [--output-file]

Required arguments:
    -i, --input-file        Path to input file in FAST5 format
    -o, --output-file       Name of output file
    
Options:
    -h                      Shows help message and exit

## Output 
Marijke-tool outputs file with predicted scores for each measurement in the raw signal.

### Examples
Marijke-tool -i fast5_input -o output_name

## Test
Maybe include test to check if installation went well

## Useful links
\ link to thesis \



###### to do:
add network checkpoint to folder data

maybe remove dependence of metrics or take only those parts that you need
