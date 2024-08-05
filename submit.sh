#!/bin/bash -l
#SBATCH --time=10:10:00
#SBATCH -C "gpu&hbm80g"
#SBATCH --account=m4663
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J SR-test
#SBATCH -o job_log_%j.out

# load libs
module load pytorch

## you may also do the following if you want:
# module load conda
# conda activate your_env

# for DDP
export MASTER_ADDR=$(hostname)
cmd="python trainSR_nstk.py --run-name ddim_v --prediction-type v --sampler ddim --time-steps 5 --multi-node"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    " 
