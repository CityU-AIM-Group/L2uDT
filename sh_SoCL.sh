#!/bin/bash
#SBATCH -J SoCL
#SBATCH -o CVC_SoCL.out    
#SBATCH -e error.err
#SBATCH --gres=gpu:1
#SBATCH -w hpc-gpu007
#SBATCH --partition=gpu_7d1g 
#SBATCH --ntasks-per-node=2 
#SBATCH --nodes=1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

module load cuda/10.2.89
conda activate torch

cd /home/xiaoqiguo2/L2uDT/experiment/Ours/
python ./train_SoCL.py
