#!/bin/bash

#SBATCH --job-name=cpu_task
#SBATCH --error=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/slurm_%j.err
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/slurm_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=1-0:00:00
 
module load 2023 python/3.9.4


source /lustre/scratch/WUR/AIN/nedun001/climaX-air-pollution/climax/bin/activate


python -u /lustre/scratch/WUR/AIN/nedun001/climaX-air-pollution/compute_freq.py 