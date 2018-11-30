#!/bin/bash

# NAME OF THE JOB
#SBATCH --job-name=install_psutil

# MAIL ADDRESS
#SBATCH --mail-user=marijke.thijssen@wur.nl
#SBATCH --mail-type=ALL

# OUTPUT FILES
#SBATCH --output=out_%j.txt
#SBATCH --error=error_%j.txt

# COMMENTS
#SBATCH --comment="master_thesis_Marijke_Thijssen"

# REQUIRED RESOURCE
#SBATCH --partition=BIOINF_Std
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --time=0-0:10:00

# ENVIRONMENT, OPERATIONS AND JOB STEPS
#load modules
module load python/3.5.2

# set variables
echo "Install psutil"
HOME_PATH=/home/WUR/thijs030

# activate env
source $HOME_PATH/networks/bin/activate

pip install psutil
