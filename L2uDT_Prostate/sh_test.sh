#!/bin/bash
#SBATCH -J ProstateX
#SBATCH -o test.err    
#SBATCH -e test_L2uDT.out
#SBATCH --gres=gpu:1
#SBATCH -w hpc-gpu006
#SBATCH --partition=gpu_7d1g 
#SBATCH --ntasks-per-node=2 
#SBATCH --nodes=1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

module load cuda/10.2.89
conda activate torch

cd /home/xiaoqiguo2/L2uDT/L2uDT_Prostate/
python ./test.py --gpu=5 --cfg=DMFNet --mode=2 --is_out=True --save_format='nii' --snapshot=True