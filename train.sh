#!/bin/bash
#SBATCH -A kcis
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --partition=lovelace
#SBATCH --time=3-00:00:00

user=varun.edachali
env=bbdm

source ~/.bashrc
conda activate $env

python3 main.py --config /home2/$user/BBDM-RRC/configs/Template-LBBDM-f4-v.yaml --train --sample_at_start --save_top --gpu_ids 0

# python3 main.py --config /home2/$user/BBDM-RRC/configs/Template-LBBDM-f4-v.yaml --sample_to_eval --gpu_ids 0 --resume_model /home2/varun.edachali/BBDM-RRC/results/xarm2panda/LBBDM-f4/checkpoint/top_model_epoch_78.pth

