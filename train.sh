#!/bin/bash
#SBATCH -A kcis
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --partition=lovelace
#SBATCH --time=3-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BBDM

python3 main.py --config /home2/aniruth.suresh/BBDM-RRC/configs/Template-LBBDM-f4.yaml --train --sample_at_start --save_top --gpu_ids 0 

python3 main.py --config /home2/aniruth.suresh/BBDM-RRC/configs/Template-LBBDM-f4.yaml --sample_to_eval --gpu_ids 0 --resume_model /home2/aniruth.suresh/BBDM-RRC/results/xarm2panda/LBBDM-f4/checkpoint/top_model_epoch_78.pth


