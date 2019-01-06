#!/bin/bash

# NAME OF THE JOB
#SBATCH --job-name=ResNetRNN_randomsearch_all

# MAIL ADDRESS
#SBATCH --mail-user=marijke.thijssen@wur.nl
#SBATCH --mail-type=ALL

# OUTPUT FILES
#SBATCH --output=out_trainingResNetRNN_%j.txt
#SBATCH --error=error_trainingResNetRNN_%j.txt

# COMMENTS
#SBATCH --comment="master_thesis_Marijke_Thijssen"

# REQUIRED RESOURCE 
#SBATCH --partition=BIOINF_Std
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --time=30-00:00:00
#SBATCH --array=1-5

# ENVIRONMENT, OPERATIONS AND JOB STEPS 
#load modules
module load python/3.5.2

# display task id
echo "Start random search on ResNetRNNs"
SAVE_PATH=/lustre/scratch/WUR/BIOINF/thijs030
HOME_PATH=/home/WUR/thijs030

# activate env
source $HOME_PATH/networks/bin/activate

# execute python script
python3 $HOME_PATH/scripts/slurm_randomsearch_all.py ResNetRNN \
$SAVE_PATH/train57192_3/ 302922566 1 $SAVE_PATH/val12255/ 5000
