#!/bin/bash 
#SBATCH --job-name=exp3
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --mail-type=END
#SBATCH --array=0
#SBATCH --output=./output/%j_%x_%N.out
#SBATCH --error=./error/%j_%x_%N.err
#SBATCH --export=ALL

bash /scratch/sh6317/research/run-singularity.sh \
     /scratch/sh6317/research/pytorch-visual-learning/config_script.sh
