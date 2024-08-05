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
