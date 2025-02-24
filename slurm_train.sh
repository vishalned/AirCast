#!/bin/bash

#SBATCH --job-name=climax
#SBATCH --error=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/slurm_%j.err
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/slurm_%j.out
#SBATCH --constraint='nvidia&A100'
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-0:00:00


nvidia-smi

module load 2023 OpenBLAS/0.3.23-GCC-12.3.0

conda activate aircast


python -u src/climax/regional_forecast/train.py \
        --config configs/regional_forecast_climax2_5.yaml \
        --data.batch_size=32 \
        --trainer.max_epochs=50 \
        --trainer.devices=4 \
        --model.lr=5e-4 \
        --data.root_dir /lustre/scratch/WUR/AIN/nedun001/climaX-air-pollution/aircast_data >> /lustre/scratch/WUR/AIN/nedun001/climaX-air-pollution/exp_logs/seed_104.txt



