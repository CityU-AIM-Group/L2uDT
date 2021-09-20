#!/bin/bash
#SBATCH -J ProstateX
#SBATCH -o error.err    
#SBATCH -e L2uDT.err
#SBATCH --gres=gpu:1
#SBATCH -w node30
#SBATCH --partition=team1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch

cd /home/xiaoqiguo2/L2uDT/L2uDT_Prostate/
python ./train.py --gpu=0 --cfg=L2uDT --batch_size=2