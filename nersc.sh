#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J nskt-para
#SBATCH --mail-user=gaoyang29@berkeley.edu
#SBATCH --mail-type=all
#SBATCH -t 23:00:00
#SBATCH -A m4633

module load pytorch
source /pscratch/sd/y/yanggao/SuperDiffusion/venv/bin/activate

cd /pscratch/sd/y/yanggao/SuperDiffusion
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set visible GPUs
python3 trainFC_nstk.py --run-name full1021 --sampling-fre 50 --epochs 1000