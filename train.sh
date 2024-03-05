#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p normal
#SBATCH --gres=gpu:a100:1
#SBATCH -t 3:00:00
#SBATCH --array=1-1
#SBATCH --mem=30GB
#SBATCH --job-name=3_3_train_sine_seq
#SBATCH --output=outputs/logs/%x_%a.log
#SBATCH -e outputs/errs/%x_%a.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=qiyao@mit.edu	

python train.py 