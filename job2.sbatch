#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J KT
#SBATCH --mail-user=gaoyang29@berkeley.edu
#SBATCH --mail-type=all
#SBATCH -t 23:00:00
#SBATCH -A m4633


module load pytorch
source /pscratch/sd/y/yanggao/SuperDiffusion/venv/bin/activate

cd /pscratch/sd/y/yanggao/SuperDiffusion
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 trainSR_nstk.py --run-name full0925 --sampling-fre 25 --epochs 250
