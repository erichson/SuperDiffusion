# load libs
module load pytorch

## you may also do the following if you want:
# module load conda
# conda activate your_env

# for DDP
export MASTER_ADDR=$(hostname)

cmd="python train.py --run-name ddim_v_multinode_$@ --prediction-type v --sampler ddim --time-steps 2 --multi-node --batch-size 8  --base-width $@"
#cmd="python train.py --run-name ddim_v_multinode_$@ --prediction-type v --sampler ddim --time-steps 2 --multi-node --batch-size 8 --base-width $@  --dataset climate"
#cmd="python train.py --run-name ddim_v_multinode_$@ --prediction-type v --sampler ddim --time-steps 2 --multi-node --batch-size 8 --base-width $@  --dataset simple"

#cmd="python train.py --run-name ddim_v_multinode_clip_$@ --prediction-type v --sampler ddim --time-steps 2 --multi-node --batch-size 8 --base-width $@  --clip_loss --dataset climate"
#cmd="python train.py --run-name ddim_v_multinode_clip_$@ --prediction-type v --sampler ddim --time-steps 2 --multi-node --batch-size 8 --base-width $@  --clip_loss --dataset simple"

#cmd="python train.py --run-name ddim_v_multinode_sample_$@ --prediction-type v --sampler ddim --time-steps 2 --multi-node --batch-size 8 --base-width $@  --sample_loss --dataset climate"
#cmd="python train.py --run-name ddim_v_multinode_sample_$@ --prediction-type v --sampler ddim --time-steps 2 --multi-node --batch-size 8 --base-width $@  --sample_loss --dataset simple"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    " 
